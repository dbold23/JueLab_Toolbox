#!/usr/bin/env python3
"""
Train ML classifiers for growth curve quality classification.

Trains two models:
  1. Pre-fit gate: rejects obvious junk before expensive fitting
  2. Post-fit classifier: three-class (GOOD/BORDERLINE/BAD) after Gompertz fit

Usage:
    python 09_train_classifier.py                     # Train on synthetic data
    python 09_train_classifier.py --compare            # Compare GBT vs RF
    python 09_train_classifier.py --validate-real      # Also validate on real audit data
    python 09_train_classifier.py --output models/     # Custom output directory
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, make_scorer,
)

# Add scripts dir to path for ml_classifier imports
sys.path.insert(0, str(Path(__file__).parent))
from ml_classifier import (
    PREFIT_FEATURES, ALL_POSTFIT_FEATURES, METADATA_FEATURES,
    compute_derived_features, extract_prefit_features, extract_metadata_features,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def specificity_score(y_true, y_pred):
    """Specificity = TN / (TN + FP)."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


specificity_scorer = make_scorer(specificity_score)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(
    ground_truth_csv: str,
    processing_results_csv: str,
) -> pd.DataFrame:
    """Merge ground truth labels with pipeline-extracted features."""
    gt = pd.read_csv(ground_truth_csv)
    pr = pd.read_csv(processing_results_csv)

    # Standardize strain matching
    gt['strain_match'] = gt['strain_name'].str.upper().str.strip()
    pr['strain_match'] = pr['strain'].str.upper().str.strip().apply(
        lambda s: m.group(1) if (m := re.search(r'(CURVE\d+)', s)) else s
    )

    merged = gt.merge(pr, on='strain_match', how='inner', suffixes=('_gt', '_pr'))
    merged['label'] = (merged['expected_class'] == 'GOOD').astype(int)

    print(f"Loaded {len(merged)} matched curves "
          f"({merged['label'].sum()} GOOD, {(~merged['label'].astype(bool)).sum()} BAD)")

    return merged


def load_real_audit_data(audit_csv: str, results_dir: str) -> pd.DataFrame:
    """Load real audit data as labeled training examples.

    Returns DataFrame matching the format of load_training_data output,
    with 'label' column and all post-fit feature columns.
    """
    audit = pd.read_csv(audit_csv)

    # Load all group results
    results_base = Path(results_dir)
    all_results = []
    for group_dir in sorted(results_base.glob('Group*_Results')):
        csv_path = group_dir / 'processing_results.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_results.append(df)
    if not all_results:
        consolidated = results_base / 'all_groups_results.csv'
        if consolidated.exists():
            all_results.append(pd.read_csv(consolidated))
    if not all_results:
        return pd.DataFrame()

    real_df = pd.concat(all_results, ignore_index=True)

    # Merge
    real_df['strain_match'] = real_df['strain'].str.strip()
    audit['strain_match'] = audit['strain'].str.strip()
    merged = audit.merge(real_df, on='strain_match', how='inner', suffixes=('_audit', '_pr'))

    if len(merged) == 0:
        return pd.DataFrame()

    # Derive ground truth label from audit
    is_good_col = 'is_good' if 'is_good' in merged.columns and not merged['is_good'].isna().all() else 'is_good_pr'
    pipeline_good = merged[is_good_col].astype(bool)
    audit_correct = merged['audit_result'] == 'correct'
    # True label: pipeline was right AND audit confirmed, OR pipeline wrong AND audit says wrong
    merged['label'] = ((pipeline_good & audit_correct) | (~pipeline_good & ~audit_correct)).astype(int)

    # Drop "unsure" audit results — don't train on ambiguous labels
    unsure_mask = merged['audit_result'] == 'unsure'
    n_unsure = unsure_mask.sum()
    if n_unsure > 0:
        print(f"  Dropping {n_unsure} 'unsure' audit curves from training")
        merged = merged[~unsure_mask].reset_index(drop=True)

    # Use real strain name for metadata
    if 'strain' not in merged.columns:
        merged['strain'] = merged['strain_match']

    n_good = merged['label'].sum()
    n_bad = (merged['label'] == 0).sum()
    print(f"  Real audit data: {len(merged)} curves ({n_good} GOOD, {n_bad} BAD)")

    return merged


def prepare_prefit_features(df: pd.DataFrame, raw_data_dir: str = None) -> tuple:
    """Extract pre-fit features from raw data files (not post-pipeline CSV).

    If raw_data_dir is provided, reads actual raw OD data and computes features
    using extract_prefit_features(). This ensures training features match
    inference features exactly — no train/inference mismatch.

    Falls back to CSV-proxy extraction if raw_data_dir is not available.
    """
    y = df['label'].values

    if raw_data_dir is not None:
        raw_dir = Path(raw_data_dir)
        data_files = sorted(raw_dir.glob('*.csv'))
        if not data_files:
            print(f"  Warning: no CSV files found in {raw_dir}, falling back to CSV proxy")
            return _prepare_prefit_from_csv(df)

        # Build lookup: strain_name → (time, od) from raw data
        print(f"  Extracting pre-fit features from {len(data_files)} raw data files...")
        raw_curves = {}
        for fpath in data_files:
            raw_df = pd.read_csv(fpath)
            time_col = [c for c in raw_df.columns if 'TIME' in c.upper()][0]
            time = raw_df[time_col].values
            for col in raw_df.columns:
                if col == time_col:
                    continue
                # Extract curve name: e.g. "SYNTHETIC_400pts_99h_CURVE0005_blanked" → "CURVE0005"
                import re as _re
                m = _re.search(r'(CURVE\d+)', col)
                if m:
                    curve_name = m.group(1)
                    od = raw_df[col].values
                    # Remove NaN values
                    valid = ~np.isnan(od)
                    if valid.sum() > 5:
                        raw_curves[curve_name] = (time[valid], od[valid])

        print(f"  Found {len(raw_curves)} curves in raw data")

        # Extract features for each training curve
        rows = []
        matched = 0
        for _, row_data in df.iterrows():
            strain = row_data.get('strain_match', '')
            # Try matching
            import re as _re
            m = _re.search(r'(CURVE\d+)', str(strain))
            curve_key = m.group(1) if m else strain

            if curve_key in raw_curves:
                t, od = raw_curves[curve_key]
                # Pass strain name for metadata extraction (synthetic names
                # like CURVE0005 won't match control patterns → NaN metadata)
                original_strain = row_data.get('strain_name', row_data.get('strain', ''))
                features = extract_prefit_features(t, od, strain_name=str(original_strain))
                rows.append(features)
                matched += 1
            else:
                # Missing curve — fill with NaN
                rows.append({f: np.nan for f in PREFIT_FEATURES})

        print(f"  Matched {matched}/{len(df)} curves to raw data")

        X = pd.DataFrame(rows)[PREFIT_FEATURES]
        return X, y

    return _prepare_prefit_from_csv(df)


def _prepare_prefit_from_csv(df: pd.DataFrame) -> tuple:
    """Fallback: extract pre-fit features from CSV columns (proxy, not ideal)."""
    print("  WARNING: Using CSV-proxy features (post-pipeline values).")
    print("  Pass --raw-data-dir for honest pre-fit features.")
    feature_map = {
        'raw_delta_od': 'delta_od',
        'raw_max_od': 'max_od',
        'raw_snr': 'snr',
        'raw_monotone_fraction': 'monotone_fraction',
        'raw_baseline_std': 'baseline_std',
        'raw_baseline_mean': None,
        'n_points': 'points_used',
        'time_span': 'truncation_time',
    }

    X = pd.DataFrame()
    for feat, col in feature_map.items():
        if col and col in df.columns:
            X[feat] = df[col].astype(float)
        else:
            X[feat] = np.nan

    if 'initial_od' in df.columns:
        X['raw_baseline_mean'] = df['initial_od'].astype(float)

    # Add metadata features from strain name
    strain_col = 'strain' if 'strain' in df.columns else 'strain_name'
    for feat in METADATA_FEATURES:
        X[feat] = np.nan
    if strain_col in df.columns:
        for idx, strain in enumerate(df[strain_col]):
            meta = extract_metadata_features(str(strain))
            for feat in METADATA_FEATURES:
                X.loc[X.index[idx], feat] = meta[feat]

    y = df['label'].values
    return X, y


def prepare_postfit_features(df: pd.DataFrame) -> tuple:
    """Extract post-fit feature matrix and labels from merged data."""
    X = pd.DataFrame()

    # Direct and secondary features from CSV
    for feat in ALL_POSTFIT_FEATURES:
        if feat in df.columns:
            X[feat] = df[feat].astype(float)
        else:
            X[feat] = np.nan

    # Compute derived features row by row
    derived_cols = ['mu_over_a', 'lambda_over_trunc_time', 'rmse_over_delta_od',
                    'err_product', 'points_per_hour']
    if not all(c in X.columns and X[c].notna().any() for c in derived_cols):
        derived_rows = []
        for _, row in X.iterrows():
            derived_rows.append(compute_derived_features(row.to_dict()))
        derived_df = pd.DataFrame(derived_rows)
        for col in derived_cols:
            X[col] = derived_df[col].values

    # Add metadata features from strain name
    strain_col = 'strain' if 'strain' in df.columns else 'strain_name'
    if strain_col in df.columns:
        for idx, strain in enumerate(df[strain_col]):
            meta = extract_metadata_features(str(strain))
            for feat in METADATA_FEATURES:
                X.loc[X.index[idx], feat] = meta[feat]

    y = df['label'].values
    return X[ALL_POSTFIT_FEATURES], y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    model_type: str = 'hgb',
    n_folds: int = 5,
) -> tuple:
    """Train with stratified K-fold CV. Returns (model, metrics_dict)."""
    if model_type == 'hgb':
        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=5,
            min_samples_leaf=5,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42,
        )
    elif model_type == 'rf':
        # RF can't handle NaN/inf, so we wrap with a pipeline that imputes
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=3,
                class_weight='balanced_subsample',
                random_state=42,
            )),
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Replace inf with NaN (HistGradientBoosting handles NaN natively;
    # RF pipeline uses SimpleImputer to fill NaN with median)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'specificity': specificity_scorer,
    }

    cv_results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=False
    )

    metrics = {
        k: float(np.mean(cv_results[f'test_{k}']))
        for k in scoring
    }
    metrics_std = {
        f'{k}_std': float(np.std(cv_results[f'test_{k}']))
        for k in scoring
    }
    metrics.update(metrics_std)

    # Train final model on all data
    model.fit(X, y)

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = dict(zip(X.columns, model.feature_importances_))
        metrics['feature_importance'] = dict(
            sorted(importances.items(), key=lambda x: -x[1])
        )

    return model, metrics


def analyze_borderline(model, X, y, good_thresh=0.7, bad_thresh=0.3):
    """Analyze three-class distribution at given thresholds."""
    probs = model.predict_proba(X)[:, 1]

    good_mask = probs >= good_thresh
    bad_mask = probs <= bad_thresh
    borderline_mask = ~good_mask & ~bad_mask

    n_good = good_mask.sum()
    n_bad = bad_mask.sum()
    n_border = borderline_mask.sum()

    # Accuracy treating borderline as BAD (conservative)
    y_pred_conservative = good_mask.astype(int)
    acc = accuracy_score(y, y_pred_conservative)

    # What's in borderline?
    border_true_good = (y[borderline_mask] == 1).sum()
    border_true_bad = (y[borderline_mask] == 0).sum()

    return {
        'good_threshold': good_thresh,
        'bad_threshold': bad_thresh,
        'n_good': int(n_good),
        'n_borderline': int(n_border),
        'n_bad': int(n_bad),
        'borderline_true_good': int(border_true_good),
        'borderline_true_bad': int(border_true_bad),
        'accuracy_conservative': float(acc),
    }


# ---------------------------------------------------------------------------
# Validation on real data
# ---------------------------------------------------------------------------

def validate_on_real_data(model, feature_names, audit_csv, results_dir,
                          prefit_model=None):
    """Validate trained models on real data with manual audit labels."""
    audit = pd.read_csv(audit_csv)

    # Load all group results
    results_base = Path(results_dir)
    all_results = []
    for group_dir in sorted(results_base.glob('Group*_Results')):
        csv_path = group_dir / 'processing_results.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_results.append(df)

    if not all_results:
        consolidated = results_base / 'all_groups_results.csv'
        if consolidated.exists():
            all_results.append(pd.read_csv(consolidated))

    if not all_results:
        print("Warning: No real data results found for validation")
        return None

    real_df = pd.concat(all_results, ignore_index=True)

    # Merge with audit
    real_df['strain_match'] = real_df['strain'].str.strip()
    audit['strain_match'] = audit['strain'].str.strip()
    merged = audit.merge(real_df, on='strain_match', how='inner', suffixes=('_audit', '_pr'))

    if len(merged) == 0:
        print("Warning: No matching strains between audit and results")
        return None

    # Derive ground truth labels from audit
    is_good_col = 'is_good' if 'is_good' in merged.columns else 'is_good_pr'
    pipeline_good = merged[is_good_col].astype(bool)
    audit_correct = merged['audit_result'] == 'correct'
    y_true = ((pipeline_good & audit_correct) | (~pipeline_good & ~audit_correct)).astype(int)

    n_good = y_true.sum()
    n_bad = (y_true == 0).sum()
    print(f"\nReal data: {len(merged)} curves ({n_good} GOOD, {n_bad} BAD by audit)")

    # --- Pre-fit gate validation ---
    if prefit_model is not None:
        print(f"\n  Pre-fit gate (on raw features from CSV — approximate):")
        X_pre = pd.DataFrame()
        pre_map = {
            'raw_delta_od': 'delta_od', 'raw_max_od': 'max_od',
            'raw_snr': 'snr', 'raw_monotone_fraction': 'monotone_fraction',
            'raw_baseline_std': 'baseline_std', 'raw_baseline_mean': None,
            'n_points': 'points_used', 'time_span': 'truncation_time',
        }
        for feat, col in pre_map.items():
            if col and col in merged.columns:
                X_pre[feat] = merged[col].astype(float)
            else:
                X_pre[feat] = np.nan
        # Add metadata features
        for idx, row in merged.iterrows():
            strain = row.get('strain_match', '')
            meta = extract_metadata_features(str(strain))
            for feat in METADATA_FEATURES:
                X_pre.loc[idx, feat] = meta[feat]
        X_pre = X_pre.replace([np.inf, -np.inf], np.nan)

        probs_pre = prefit_model.predict_proba(X_pre[PREFIT_FEATURES])[:, 1]
        for thresh in [0.05, 0.10, 0.20]:
            reject = probs_pre <= thresh
            n_reject = reject.sum()
            false_reject = (y_true[reject] == 1).sum() if n_reject > 0 else 0
            true_reject = (y_true[reject] == 0).sum() if n_reject > 0 else 0
            print(f"    threshold {thresh:.2f}: reject {n_reject}/{len(y_true)} "
                  f"({true_reject} true BAD, {false_reject} false rejects)")

    # --- Post-fit classifier validation ---
    X = pd.DataFrame()
    for feat in feature_names:
        if feat in merged.columns:
            X[feat] = merged[feat].astype(float)
        else:
            X[feat] = np.nan

    derived_cols = ['mu_over_a', 'lambda_over_trunc_time', 'rmse_over_delta_od',
                    'err_product', 'points_per_hour']
    derived_rows = []
    for _, row in X.iterrows():
        derived_rows.append(compute_derived_features(row.to_dict()))
    derived_df = pd.DataFrame(derived_rows)
    for col in derived_cols:
        if col in X.columns:
            X[col] = derived_df[col].values

    # Add metadata features
    for idx, row in merged.iterrows():
        strain = row.get('strain_match', '')
        meta = extract_metadata_features(str(strain))
        for feat in METADATA_FEATURES:
            X.loc[idx, feat] = meta[feat]

    X = X.replace([np.inf, -np.inf], np.nan)

    if len(X) > 0:
        probs = model.predict_proba(X[feature_names])[:, 1]
        y_pred = (probs >= 0.5).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        print(f"\n  Post-fit classifier:")
        print(f"    Accuracy:    {accuracy_score(y_true, y_pred):.3f}")
        print(f"    Precision:   {precision_score(y_true, y_pred, zero_division=0):.3f}")
        print(f"    Recall:      {recall_score(y_true, y_pred, zero_division=0):.3f}")
        print(f"    F1:          {f1_score(y_true, y_pred, zero_division=0):.3f}")
        print(f"    Specificity: {specificity_score(y_true, y_pred):.3f}")
        print(f"\n    Confusion matrix:")
        print(f"                     Predicted GOOD  Predicted BAD")
        print(f"      Actual GOOD:        {cm[1,1]:4d}           {cm[1,0]:4d}")
        print(f"      Actual BAD:         {cm[0,1]:4d}           {cm[0,0]:4d}")

        # Show misclassified strains
        misclassified = merged[(y_pred != y_true)]
        if len(misclassified) > 0:
            print(f"\n    Misclassified strains ({len(misclassified)}):")
            for _, row in misclassified.iterrows():
                strain = row.get('strain_match', '?')
                true_label = 'GOOD' if y_true[row.name] == 1 else 'BAD'
                pred_label = 'GOOD' if y_pred[row.name] == 1 else 'BAD'
                p = probs[row.name]
                print(f"      {strain}: true={true_label}, pred={pred_label} (p={p:.3f})")

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train ML classifiers for growth curve quality'
    )
    parser.add_argument(
        '--ground-truth',
        default='synthetic_data/output/comprehensive_test/test_data/ground_truth.csv',
        help='Path to ground truth CSV'
    )
    parser.add_argument(
        '--processing-results',
        default='synthetic_data/output/comprehensive_test/validation_latest/processing_results.csv',
        help='Path to processing_results.csv from pipeline'
    )
    parser.add_argument(
        '--raw-data-dir',
        default='synthetic_data/output/comprehensive_test/test_data/DATA',
        help='Directory with raw OD CSV files for pre-fit feature extraction'
    )
    parser.add_argument(
        '--output', default='models/',
        help='Output directory for serialized models'
    )
    parser.add_argument(
        '--compare', action='store_true',
        help='Compare GBT vs RF'
    )
    parser.add_argument(
        '--validate-real', action='store_true',
        help='Also validate on real audit data'
    )
    parser.add_argument(
        '--audit-csv',
        default='results/tables/validation_audit.csv',
        help='Path to manual audit CSV'
    )
    parser.add_argument(
        '--results-dir',
        default='results/tables',
        help='Directory with real data processing results'
    )
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of CV folds'
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=" * 60)
    print("LOADING TRAINING DATA")
    print("=" * 60)
    df = load_training_data(args.ground_truth, args.processing_results)

    # Mix in real audit data for training (if available)
    if Path(args.audit_csv).exists() and Path(args.results_dir).exists():
        print("\nLoading real audit data for mixed training...")
        real_df = load_real_audit_data(args.audit_csv, args.results_dir)
        if len(real_df) > 0:
            # Tag source for diagnostics
            df['source'] = 'synthetic'
            real_df['source'] = 'real'
            df = pd.concat([df, real_df], ignore_index=True)
            print(f"Combined training set: {len(df)} curves "
                  f"({df['label'].sum()} GOOD, {(df['label'] == 0).sum()} BAD)")

    # -----------------------------------------------------------------------
    # 70/30 Stratified train/test split
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAIN/TEST SPLIT (70/30 stratified)")
    print("=" * 60)

    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['label']
    )
    print(f"  Train: {len(train_df)} ({train_df['label'].sum()} GOOD, "
          f"{(train_df['label'] == 0).sum()} BAD)")
    print(f"  Test:  {len(test_df)} ({test_df['label'].sum()} GOOD, "
          f"{(test_df['label'] == 0).sum()} BAD)")
    if 'source' in df.columns:
        for split_name, split_df in [('Train', train_df), ('Test', test_df)]:
            n_synth = (split_df['source'] == 'synthetic').sum()
            n_real = (split_df['source'] == 'real').sum()
            print(f"  {split_name}: {n_synth} synthetic + {n_real} real")

    # -----------------------------------------------------------------------
    # Train pre-fit gate (on train split only)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING PRE-FIT GATE")
    print("=" * 60)

    X_pre_train, y_pre_train = prepare_prefit_features(train_df, raw_data_dir=args.raw_data_dir)
    X_pre_test, y_pre_test = prepare_prefit_features(test_df, raw_data_dir=args.raw_data_dir)
    print(f"Pre-fit features: {list(X_pre_train.columns)}")
    print(f"Train: {X_pre_train.shape}, Test: {X_pre_test.shape}")

    prefit_model, prefit_metrics = train_model(X_pre_train, y_pre_train, 'hgb', args.n_folds)

    print(f"\nPre-fit gate CV results ({args.n_folds}-fold on train set):")
    for k in ['accuracy', 'precision', 'recall', 'f1', 'specificity']:
        print(f"  {k:12s}: {prefit_metrics[k]:.3f} ± {prefit_metrics[f'{k}_std']:.3f}")

    # Evaluate on held-out test set
    X_pre_test_clean = X_pre_test.replace([np.inf, -np.inf], np.nan)
    probs_pre_test = prefit_model.predict_proba(X_pre_test_clean[PREFIT_FEATURES])[:, 1]
    print(f"\nPre-fit gate HELD-OUT TEST results:")
    for thresh in [0.05, 0.10, 0.20, 0.30]:
        reject_mask = probs_pre_test <= thresh
        n_reject = reject_mask.sum()
        false_reject = (y_pre_test[reject_mask] == 1).sum() if n_reject > 0 else 0
        true_reject = (y_pre_test[reject_mask] == 0).sum() if n_reject > 0 else 0
        print(f"  threshold {thresh:.2f}: reject {n_reject}/{len(y_pre_test)} "
              f"({true_reject} true BAD, {false_reject} false rejects)")

    joblib.dump(prefit_model, output_dir / 'prefit_gate.joblib')
    print(f"\nSaved pre-fit gate to {output_dir / 'prefit_gate.joblib'}")

    # -----------------------------------------------------------------------
    # Train post-fit classifier (on train split only)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING POST-FIT CLASSIFIER")
    print("=" * 60)

    X_post_train, y_post_train = prepare_postfit_features(train_df)
    X_post_test, y_post_test = prepare_postfit_features(test_df)
    print(f"Post-fit features: {list(X_post_train.columns)}")
    print(f"Train: {X_post_train.shape}, Test: {X_post_test.shape}")

    postfit_model, postfit_metrics = train_model(X_post_train, y_post_train, 'hgb', args.n_folds)

    print(f"\nPost-fit classifier CV results ({args.n_folds}-fold on train set):")
    for k in ['accuracy', 'precision', 'recall', 'f1', 'specificity']:
        print(f"  {k:12s}: {postfit_metrics[k]:.3f} ± {postfit_metrics[f'{k}_std']:.3f}")

    # Feature importance
    if 'feature_importance' in postfit_metrics:
        print("\nTop 10 features:")
        for i, (feat, imp) in enumerate(postfit_metrics['feature_importance'].items()):
            if i >= 10:
                break
            print(f"  {feat:30s}: {imp:.4f}")

    # -----------------------------------------------------------------------
    # HELD-OUT TEST SET evaluation (the honest numbers)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET RESULTS (30% never seen during training)")
    print("=" * 60)

    X_post_test_clean = X_post_test.replace([np.inf, -np.inf], np.nan)
    probs_test = postfit_model.predict_proba(X_post_test_clean[ALL_POSTFIT_FEATURES])[:, 1]
    y_pred_test = (probs_test >= 0.5).astype(int)

    print(f"\nPost-fit classifier on {len(y_post_test)} held-out curves:")
    print(f"  Accuracy:    {accuracy_score(y_post_test, y_pred_test):.3f}")
    print(f"  Precision:   {precision_score(y_post_test, y_pred_test, zero_division=0):.3f}")
    print(f"  Recall:      {recall_score(y_post_test, y_pred_test, zero_division=0):.3f}")
    print(f"  F1:          {f1_score(y_post_test, y_pred_test, zero_division=0):.3f}")
    print(f"  Specificity: {specificity_score(y_post_test, y_pred_test):.3f}")

    cm = confusion_matrix(y_post_test, y_pred_test)
    print(f"\n  Confusion matrix (held-out test):")
    print(f"                   Predicted GOOD  Predicted BAD")
    print(f"    Actual GOOD:        {cm[1,1]:4d}           {cm[1,0]:4d}")
    print(f"    Actual BAD:         {cm[0,1]:4d}           {cm[0,0]:4d}")

    # Three-class analysis on test set
    print(f"\n  Three-class analysis (held-out test):")
    for good_t, bad_t in [(0.7, 0.3), (0.6, 0.4), (0.8, 0.2)]:
        good_mask = probs_test >= good_t
        bad_mask = probs_test <= bad_t
        border_mask = ~good_mask & ~bad_mask
        y_cons = good_mask.astype(int)
        bg = (y_post_test[border_mask] == 1).sum()
        bb = (y_post_test[border_mask] == 0).sum()
        print(f"    [{bad_t:.1f}, {good_t:.1f}]: GOOD={good_mask.sum()}, "
              f"BORDERLINE={border_mask.sum()} ({bg}G+{bb}B), "
              f"BAD={bad_mask.sum()}, acc={accuracy_score(y_post_test, y_cons):.3f}")

    # Show misclassified test strains
    test_strains = test_df['strain'].values if 'strain' in test_df.columns else test_df.get('strain_match', test_df.get('strain_name', pd.Series(['?'] * len(test_df)))).values
    misclassified = np.where(y_pred_test != y_post_test)[0]
    if len(misclassified) > 0:
        print(f"\n  Misclassified ({len(misclassified)}):")
        for idx in misclassified:
            true_label = 'GOOD' if y_post_test[idx] == 1 else 'BAD'
            pred_label = 'GOOD' if y_pred_test[idx] == 1 else 'BAD'
            strain = test_strains[idx] if idx < len(test_strains) else '?'
            source = test_df.iloc[idx].get('source', '?') if 'source' in test_df.columns else '?'
            print(f"    {strain}: true={true_label}, pred={pred_label} "
                  f"(p={probs_test[idx]:.3f}, {source})")

    # -----------------------------------------------------------------------
    # Now retrain final model on ALL data for deployment
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RETRAINING FINAL MODEL ON ALL DATA (for deployment)")
    print("=" * 60)

    X_pre_all, y_pre_all = prepare_prefit_features(df, raw_data_dir=args.raw_data_dir)
    X_post_all, y_post_all = prepare_postfit_features(df)

    X_pre_all = X_pre_all.replace([np.inf, -np.inf], np.nan)
    X_post_all = X_post_all.replace([np.inf, -np.inf], np.nan)

    prefit_model.fit(X_pre_all, y_pre_all)
    postfit_model.fit(X_post_all, y_post_all)

    joblib.dump(prefit_model, output_dir / 'prefit_gate.joblib')
    joblib.dump(postfit_model, output_dir / 'postfit_classifier.joblib')
    print(f"\nSaved post-fit classifier to {output_dir / 'postfit_classifier.joblib'}")

    # -----------------------------------------------------------------------
    # Compare GBT vs RF (optional)
    # -----------------------------------------------------------------------
    if args.compare:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON (GBT vs RF)")
        print("=" * 60)

        rf_model, rf_metrics = train_model(X_post, y_post, 'rf', args.n_folds)
        print(f"\n{'Metric':<15s} {'GBT':>10s} {'RF':>10s}")
        print("-" * 37)
        for k in ['accuracy', 'precision', 'recall', 'f1', 'specificity']:
            print(f"  {k:<13s} {postfit_metrics[k]:>9.3f}  {rf_metrics[k]:>9.3f}")

    # -----------------------------------------------------------------------
    # Validate on real data (optional)
    # -----------------------------------------------------------------------
    if args.validate_real:
        print("\n" + "=" * 60)
        print("REAL DATA VALIDATION")
        print("=" * 60)
        validate_on_real_data(
            postfit_model, feature_names=ALL_POSTFIT_FEATURES,
            audit_csv=args.audit_csv, results_dir=args.results_dir,
            prefit_model=prefit_model,
        )

    # -----------------------------------------------------------------------
    # Save feature config
    # -----------------------------------------------------------------------
    feature_config = {
        'prefit_feature_names': PREFIT_FEATURES,
        'postfit_feature_names': ALL_POSTFIT_FEATURES,
        'prefit_metrics': {k: v for k, v in prefit_metrics.items() if k != 'feature_importance'},
        'postfit_metrics': {k: v for k, v in postfit_metrics.items() if k != 'feature_importance'},
        'postfit_feature_importance': postfit_metrics.get('feature_importance', {}),
        'training_date': datetime.now().isoformat(),
        'n_training_samples': len(df),
        'class_distribution': {'GOOD': int(df['label'].sum()), 'BAD': int((~df['label'].astype(bool)).sum())},
        'model_type': 'HistGradientBoostingClassifier',
    }

    config_path = output_dir / 'feature_config.json'
    with open(config_path, 'w') as f:
        json.dump(feature_config, f, indent=2)
    print(f"\nSaved feature config to {config_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
