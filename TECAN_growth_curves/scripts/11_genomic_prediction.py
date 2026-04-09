"""
Genotype-to-phenotype prediction for TECAN growth curves.

Predicts Gompertz growth parameters (mu, lambda, A) from genomic features
extracted by genomic_features.py. Generates informative priors for the
Bayesian hierarchical models in 06_advanced_fitting.py.

Three models:
  A) Elastic Net — primary predictor of Gompertz mu (growth rate)
  B) Bayesian Ridge — generates prior distributions for Bayesian integration
  C) Random Forest — binary degradation classifier for rapid screening

Usage:
    python 11_genomic_prediction.py [--validate] [--pesticide Malathion]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
GENOMIC_DIR = PROJECT_DIR / 'data' / 'genomic'
RESULTS_DIR = PROJECT_DIR / 'results' / 'tables'
OUTPUT_DIR = RESULTS_DIR / 'Genomic_Analysis'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_phenotype_data(results_csv: Path,
                        pesticide_filter: Optional[str] = None
                        ) -> pd.DataFrame:
    """
    Load phenotypic Gompertz parameters from pipeline results.

    Filters to pesticide+LB strains (the primary degradation assay) and
    adds the resolved biological strain ID.
    """
    from genomic_features import resolve_strain_id, extract_pesticide_from_strain

    df = pd.read_csv(results_csv)

    # Add biological strain ID and pesticide columns
    df['bio_strain_id'] = df['strain'].apply(resolve_strain_id)
    df['pesticide'] = df['strain'].apply(extract_pesticide_from_strain)

    # Filter to treatment conditions only (exclude controls)
    df = df[df['pesticide'].notna()].copy()

    # Filter to pesticide+LB conditions (the primary assay)
    # These have "ANDLB" or "ANDG" in the strain name
    andlb_mask = df['strain'].str.upper().str.contains('AND')
    df = df[andlb_mask].copy()

    if pesticide_filter:
        df = df[df['pesticide'] == pesticide_filter].copy()

    # Keep only strains with successful Gompertz fits
    required_cols = ['gompertz_mu', 'gompertz_lambda', 'gompertz_a']
    for col in required_cols:
        if col in df.columns:
            df = df[df[col].notna()].copy()

    logger.info(f"Loaded {len(df)} phenotypic records"
                f"{f' for {pesticide_filter}' if pesticide_filter else ''}")
    return df


def load_genomic_data(genomic_csv: Path) -> pd.DataFrame:
    """Load pre-computed genomic features."""
    df = pd.read_csv(genomic_csv, index_col='strain_id')
    logger.info(f"Loaded genomic features for {len(df)} strains")
    return df


def merge_genotype_phenotype(phenotype_df: pd.DataFrame,
                             genomic_df: pd.DataFrame
                             ) -> Tuple[pd.DataFrame, List[str]]:
    """
    Merge genomic features with phenotypic data on strain ID.

    Returns:
        merged DataFrame, list of genomic feature column names used
    """
    # Identify genomic feature columns (exclude metadata)
    genomic_cols = [c for c in genomic_df.columns
                    if c not in ('feature_source',)]

    # Normalize case for matching (TECAN uses uppercase, genomes may use mixed case)
    phenotype_df = phenotype_df.copy()
    phenotype_df['bio_strain_id_upper'] = phenotype_df['bio_strain_id'].str.upper()
    genomic_upper = genomic_df[genomic_cols].copy()
    genomic_upper.index = genomic_upper.index.str.upper()

    # Merge on biological strain ID (case-insensitive)
    merged = phenotype_df.merge(
        genomic_upper,
        left_on='bio_strain_id_upper',
        right_index=True,
        how='inner'
    )
    merged = merged.drop(columns=['bio_strain_id_upper'])

    # Drop rows where all genomic features are NaN
    genomic_all_nan = merged[genomic_cols].isna().all(axis=1)
    n_dropped = genomic_all_nan.sum()
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped} rows with no genomic data")
        merged = merged[~genomic_all_nan].copy()

    logger.info(f"Merged dataset: {len(merged)} strain x pesticide combinations, "
                f"{len(genomic_cols)} genomic features")
    return merged, genomic_cols


# ---------------------------------------------------------------------------
# Model A: Elastic Net for Gompertz parameter prediction
# ---------------------------------------------------------------------------

def train_elastic_net(X: np.ndarray, y: np.ndarray,
                      feature_names: List[str],
                      target_name: str = 'gompertz_mu',
                      n_alphas: int = 50) -> Dict:
    """
    Train ElasticNetCV for Gompertz parameter prediction.

    Uses leave-one-out CV internally when n < 50, otherwise 5-fold.
    """
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    n_samples = X.shape[0]
    cv_folds = min(n_samples, 5) if n_samples >= 10 else n_samples

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('enet', ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
            n_alphas=n_alphas,
            cv=cv_folds,
            max_iter=10000,
            random_state=42,
        )),
    ])

    pipeline.fit(X, y)
    enet = pipeline.named_steps['enet']

    # Feature importances (coefficients)
    coefs = enet.coef_
    importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs,
        'abs_coefficient': np.abs(coefs),
    }).sort_values('abs_coefficient', ascending=False)

    result = {
        'model': pipeline,
        'alpha': enet.alpha_,
        'l1_ratio': enet.l1_ratio_,
        'r_squared': pipeline.score(X, y),
        'n_nonzero': np.sum(coefs != 0),
        'feature_importance': importance,
        'target': target_name,
    }

    logger.info(f"  ElasticNet ({target_name}): R²={result['r_squared']:.3f}, "
                f"alpha={result['alpha']:.4f}, l1_ratio={result['l1_ratio']:.2f}, "
                f"{result['n_nonzero']}/{len(coefs)} features selected")
    return result


# ---------------------------------------------------------------------------
# Model B: Bayesian Ridge for prior generation
# ---------------------------------------------------------------------------

def train_bayesian_ridge(X: np.ndarray, y: np.ndarray,
                         feature_names: List[str],
                         target_name: str = 'gompertz_mu') -> Dict:
    """
    Train BayesianRidge for generating posterior predictive priors.

    Returns mean + sigma predictions that can be used as informative
    priors in the Bayesian Gompertz model.
    """
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('bridge', BayesianRidge(
            max_iter=1000,
            compute_score=True,
        )),
    ])

    pipeline.fit(X, y)
    bridge = pipeline.named_steps['bridge']

    # Predict with uncertainty
    scaler = pipeline.named_steps['scaler']
    X_scaled = scaler.transform(X)
    y_pred, y_std = bridge.predict(X_scaled, return_std=True)

    result = {
        'model': pipeline,
        'r_squared': pipeline.score(X, y),
        'predictions': y_pred,
        'uncertainties': y_std,
        'target': target_name,
        'alpha_posterior': bridge.alpha_,
        'lambda_posterior': bridge.lambda_,
    }

    logger.info(f"  BayesianRidge ({target_name}): R²={result['r_squared']:.3f}, "
                f"mean_uncertainty={np.mean(y_std):.4f}")
    return result


# ---------------------------------------------------------------------------
# Model C: Random Forest degradation classifier
# ---------------------------------------------------------------------------

def train_degradation_classifier(X: np.ndarray, y_binary: np.ndarray,
                                 feature_names: List[str]) -> Dict:
    """
    Train a Random Forest to predict binary degradation capacity.

    y_binary: 1 = degrader (GOOD classification), 0 = non-degrader (BAD)
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, f1_score

    # HistGBT handles NaN natively — no imputation needed
    model = HistGradientBoostingClassifier(
        max_depth=3,
        min_samples_leaf=5,
        max_iter=200,
        random_state=42,
    )

    model.fit(X, y_binary)
    y_pred = model.predict(X)

    result = {
        'model': model,
        'accuracy': accuracy_score(y_binary, y_pred),
        'f1': f1_score(y_binary, y_pred, zero_division=0),
        'feature_importances': pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else np.zeros(len(feature_names)),
        }).sort_values('importance', ascending=False),
    }

    logger.info(f"  Degradation classifier: accuracy={result['accuracy']:.3f}, "
                f"F1={result['f1']:.3f}")
    return result


# ---------------------------------------------------------------------------
# Leave-one-strain-out cross-validation
# ---------------------------------------------------------------------------

def losocv(X: np.ndarray, y: np.ndarray,
           strain_ids: np.ndarray,
           feature_names: List[str],
           target_name: str = 'gompertz_mu') -> pd.DataFrame:
    """
    Leave-one-strain-out cross-validation.

    Groups by biological strain ID so all conditions for a strain
    are in the same fold (prevents data leakage).
    """
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    unique_strains = np.unique(strain_ids)
    results = []

    for held_out in unique_strains:
        mask_test = strain_ids == held_out
        mask_train = ~mask_test

        X_train, X_test = X[mask_train], X[mask_test]
        y_train, y_test = y[mask_train], y[mask_test]

        if len(X_train) < 3:
            logger.warning(f"  Skipping {held_out}: only {len(X_train)} training samples")
            continue

        # Standardize
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train
        cv_folds = min(len(X_train), 5)
        enet = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 0.99],
            n_alphas=20,
            cv=cv_folds,
            max_iter=10000,
            random_state=42,
        )
        enet.fit(X_train_s, y_train)
        y_pred = enet.predict(X_test_s)

        for i, idx in enumerate(np.where(mask_test)[0]):
            results.append({
                'strain': held_out,
                'actual': y_test[i],
                'predicted_genomic': y_pred[i],
                'predicted_baseline': np.mean(y_train),
                'error_genomic': np.abs(y_pred[i] - y_test[i]),
                'error_baseline': np.abs(np.mean(y_train) - y_test[i]),
                'target': target_name,
            })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        mae_genomic = results_df['error_genomic'].mean()
        mae_baseline = results_df['error_baseline'].mean()
        improvement = (mae_baseline - mae_genomic) / mae_baseline * 100

        # R-squared
        ss_res = np.sum((results_df['actual'] - results_df['predicted_genomic'])**2)
        ss_tot = np.sum((results_df['actual'] - results_df['actual'].mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        logger.info(f"  LOSOCV ({target_name}): R²={r2:.3f}, "
                    f"MAE_genomic={mae_genomic:.4f}, MAE_baseline={mae_baseline:.4f}, "
                    f"improvement={improvement:+.1f}%")

    return results_df


# ---------------------------------------------------------------------------
# Prior generation for Bayesian integration
# ---------------------------------------------------------------------------

def generate_bayesian_priors(phenotype_df: pd.DataFrame,
                             genomic_df: pd.DataFrame,
                             merged_df: pd.DataFrame,
                             genomic_cols: List[str]) -> pd.DataFrame:
    """
    Generate per-strain Bayesian priors from genomic predictions.

    For each strain x pesticide combination, generates:
    - mu_prior_mean, mu_prior_sigma: prior on growth rate
    - lam_prior_mean, lam_prior_sigma: prior on lag time
    - A_prior_mean, A_prior_sigma: prior on max OD

    These are formatted for direct use by build_gompertz_model() in
    06_advanced_fitting.py.
    """
    priors = []

    for target in ['gompertz_mu', 'gompertz_lambda', 'gompertz_a']:
        if target not in merged_df.columns:
            continue

        valid = merged_df[merged_df[target].notna()].copy()
        if len(valid) < 5:
            logger.warning(f"Too few samples for {target} prior generation ({len(valid)})")
            continue

        X = valid[genomic_cols].values.copy()
        y = valid[target].values.copy()

        # Replace NaN features with column medians
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            nan_mask = np.isnan(X[:, j])
            X[nan_mask, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0.0

        # Train Bayesian Ridge
        bridge_result = train_bayesian_ridge(X, y, genomic_cols, target)

        # Generate predictions for all strains (including those without phenotype data)
        for strain_id in genomic_df.index:
            X_strain = genomic_df.loc[[strain_id], genomic_cols].values.copy()
            # Replace NaN with medians
            for j in range(X_strain.shape[1]):
                if np.isnan(X_strain[0, j]):
                    X_strain[0, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0.0

            scaler = bridge_result['model'].named_steps['scaler']
            bridge = bridge_result['model'].named_steps['bridge']
            X_scaled = scaler.transform(X_strain)
            pred_mean, pred_std = bridge.predict(X_scaled, return_std=True)

            param_short = target.replace('gompertz_', '')
            priors.append({
                'strain_id': strain_id,
                f'{param_short}_prior_mean': pred_mean[0],
                f'{param_short}_prior_sigma': pred_std[0],
            })

    if not priors:
        return pd.DataFrame()

    # Combine priors for all parameters
    priors_df = pd.DataFrame(priors)
    # Pivot: one row per strain with columns for each parameter's mean/sigma
    priors_combined = priors_df.groupby('strain_id').first().reset_index()

    # Merge rows from different targets
    dfs = []
    for target in ['mu', 'lambda', 'a']:
        cols = ['strain_id', f'{target}_prior_mean', f'{target}_prior_sigma']
        subset = priors_df[priors_df.columns.intersection(cols)].copy()
        if len(subset.columns) > 1:
            dfs.append(subset)

    if dfs:
        result = dfs[0]
        for df in dfs[1:]:
            common = result.columns.intersection(df.columns)
            if 'strain_id' in common:
                result = result.merge(df, on='strain_id', how='outer')

    # Simpler approach: pivot by strain_id
    priors_pivot = priors_df.groupby('strain_id').first().reset_index()
    logger.info(f"Generated priors for {len(priors_pivot)} strains")
    return priors_pivot


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_losocv_results(cv_df: pd.DataFrame, output_dir: Path):
    """Generate LOSOCV diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Predicted vs Actual
    ax = axes[0]
    ax.scatter(cv_df['actual'], cv_df['predicted_genomic'],
               alpha=0.7, edgecolor='k', linewidth=0.5, s=60, c='steelblue')
    lims = [min(cv_df['actual'].min(), cv_df['predicted_genomic'].min()),
            max(cv_df['actual'].max(), cv_df['predicted_genomic'].max())]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect prediction')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted (genomic)')
    ax.set_title(f'LOSOCV: {cv_df["target"].iloc[0]}')
    ax.legend()

    # 2. Error comparison
    ax = axes[1]
    ax.bar(['Genomic\nModel', 'Population\nMean'],
           [cv_df['error_genomic'].mean(), cv_df['error_baseline'].mean()],
           color=['steelblue', 'lightcoral'], edgecolor='k')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Prediction Error Comparison')

    # 3. Per-strain errors
    ax = axes[2]
    strain_errors = cv_df.groupby('strain')[['error_genomic', 'error_baseline']].mean()
    x = range(len(strain_errors))
    width = 0.35
    ax.bar([i - width/2 for i in x], strain_errors['error_genomic'],
           width, label='Genomic', color='steelblue', edgecolor='k')
    ax.bar([i + width/2 for i in x], strain_errors['error_baseline'],
           width, label='Baseline', color='lightcoral', edgecolor='k')
    ax.set_xticks(list(x))
    ax.set_xticklabels(strain_errors.index, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('MAE')
    ax.set_title('Per-Strain Error')
    ax.legend()

    plt.tight_layout()
    plot_path = output_dir / 'losocv_diagnostics.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved LOSOCV plot to {plot_path}")


def plot_feature_importance(importance_df: pd.DataFrame, output_dir: Path,
                            target: str):
    """Plot feature importance bar chart."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Top 15 features
    top = importance_df.head(15)
    if len(top) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['steelblue' if c > 0 else 'lightcoral'
              for c in top['coefficient']]
    ax.barh(range(len(top)), top['abs_coefficient'].values,
            color=colors, edgecolor='k')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'].values, fontsize=9)
    ax.set_xlabel('|Coefficient|')
    ax.set_title(f'Elastic Net Feature Importance ({target})')
    ax.invert_yaxis()

    plt.tight_layout()
    plot_path = output_dir / f'feature_importance_{target}.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved feature importance plot to {plot_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_genomic_prediction(config: Optional[Dict] = None,
                           pesticide_filter: Optional[str] = None,
                           validate: bool = True) -> Dict:
    """
    Run the full genomic prediction pipeline.

    Args:
        config: Pipeline config dict (from config.yaml)
        pesticide_filter: Restrict to one pesticide (e.g., 'Malathion')
        validate: If True, run LOSOCV

    Returns:
        Dict with model results, predictions, and validation metrics
    """
    # Resolve paths
    genomic_cfg = (config or {}).get('genomic', {})
    genomic_csv = PROJECT_DIR / genomic_cfg.get(
        'features_csv', 'data/genomic/genomic_features.csv')
    results_csv = RESULTS_DIR / 'all_groups_results.csv'

    if not genomic_csv.exists():
        logger.error(f"Genomic features not found: {genomic_csv}")
        logger.error("Run genomic_features.py first to extract features from BLAST/GFF data")
        return {}

    if not results_csv.exists():
        logger.error(f"Pipeline results not found: {results_csv}")
        return {}

    # Load data
    phenotype_df = load_phenotype_data(results_csv, pesticide_filter)
    genomic_df = load_genomic_data(genomic_csv)

    if len(phenotype_df) == 0:
        logger.error("No phenotype data after filtering")
        return {}

    # Merge
    merged, genomic_cols = merge_genotype_phenotype(phenotype_df, genomic_df)
    if len(merged) < 5:
        logger.error(f"Too few merged samples ({len(merged)}) for modeling")
        return {}

    # Prepare feature matrix
    X = merged[genomic_cols].values.copy()
    # Replace NaN with column medians for sklearn
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        nan_mask = np.isnan(X[:, j])
        X[nan_mask, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0.0

    strain_ids = merged['bio_strain_id'].values

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUTPUT_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)

    results = {}

    # --- Model A: Elastic Net for Gompertz mu ---
    logger.info("\n=== Model A: Elastic Net (Gompertz mu) ===")
    y_mu = merged['gompertz_mu'].values
    enet_result = train_elastic_net(X, y_mu, genomic_cols, 'gompertz_mu')
    results['elastic_net_mu'] = enet_result

    enet_result['feature_importance'].to_csv(
        OUTPUT_DIR / 'feature_importance.csv', index=False)
    plot_feature_importance(enet_result['feature_importance'], plots_dir, 'gompertz_mu')

    # Also train for lambda and A if enough data
    for target in ['gompertz_lambda', 'gompertz_a']:
        if target in merged.columns and merged[target].notna().sum() >= 5:
            y_target = merged[target].values
            valid = ~np.isnan(y_target)
            if valid.sum() >= 5:
                logger.info(f"\n=== Model A: Elastic Net ({target}) ===")
                enet_extra = train_elastic_net(
                    X[valid], y_target[valid], genomic_cols, target)
                results[f'elastic_net_{target}'] = enet_extra

    # --- Model B: Bayesian Ridge for priors ---
    logger.info("\n=== Model B: Bayesian Ridge (prior generation) ===")
    priors_df = generate_bayesian_priors(phenotype_df, genomic_df, merged, genomic_cols)
    if len(priors_df) > 0:
        priors_df.to_csv(OUTPUT_DIR / 'genomic_priors.csv', index=False)
        results['priors'] = priors_df
        logger.info(f"Saved genomic priors to {OUTPUT_DIR / 'genomic_priors.csv'}")

    # --- Model C: Degradation classifier ---
    logger.info("\n=== Model C: Degradation Classifier ===")
    if 'is_good' in merged.columns:
        y_binary = merged['is_good'].astype(int).values
        if len(np.unique(y_binary)) > 1:
            clf_result = train_degradation_classifier(X, y_binary, genomic_cols)
            results['classifier'] = clf_result

            # Save classification predictions
            merged_copy = merged[['strain', 'bio_strain_id', 'pesticide', 'is_good']].copy()
            merged_copy['predicted_degrader'] = clf_result['model'].predict(X)
            merged_copy['predicted_proba'] = clf_result['model'].predict_proba(X)[:, 1]
            merged_copy.to_csv(OUTPUT_DIR / 'degradation_classification.csv', index=False)

    # --- Validation: LOSOCV ---
    if validate and len(merged) >= 5:
        logger.info("\n=== Leave-One-Strain-Out Cross-Validation ===")
        cv_mu = losocv(X, y_mu, strain_ids, genomic_cols, 'gompertz_mu')
        if len(cv_mu) > 0:
            cv_mu.to_csv(OUTPUT_DIR / 'cross_validation.csv', index=False)
            results['cv_mu'] = cv_mu
            plot_losocv_results(cv_mu, plots_dir)

    # --- Save all predictions ---
    predictions = merged[['strain', 'bio_strain_id', 'pesticide',
                          'gompertz_mu', 'gompertz_lambda', 'gompertz_a']].copy()
    predictions['predicted_mu'] = enet_result['model'].predict(X)
    predictions.to_csv(OUTPUT_DIR / 'genomic_predictions.csv', index=False)

    # --- Summary ---
    logger.info("\n" + "="*60)
    logger.info("GENOMIC PREDICTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Strains with genomic data: {len(genomic_df)}")
    logger.info(f"Merged strain x pesticide records: {len(merged)}")
    logger.info(f"Elastic Net R² (mu): {enet_result['r_squared']:.3f}")
    logger.info(f"Features selected: {enet_result['n_nonzero']}/{len(genomic_cols)}")
    if 'cv_mu' in results and len(results['cv_mu']) > 0:
        cv = results['cv_mu']
        improvement = ((cv['error_baseline'].mean() - cv['error_genomic'].mean())
                       / cv['error_baseline'].mean() * 100)
        logger.info(f"LOSOCV improvement over baseline: {improvement:+.1f}%")
    logger.info(f"Outputs saved to: {OUTPUT_DIR}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Genotype-to-phenotype prediction for TECAN growth curves'
    )
    parser.add_argument('--pesticide', type=str, default=None,
                        help='Restrict to one pesticide (e.g., Malathion)')
    parser.add_argument('--validate', action='store_true', default=True,
                        help='Run LOSOCV validation (default: True)')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip LOSOCV validation')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    # Load config
    import yaml
    config_path = SCRIPT_DIR / 'config.yaml'
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    validate = not args.no_validate

    results = run_genomic_prediction(
        config=config,
        pesticide_filter=args.pesticide,
        validate=validate,
    )

    if not results:
        logger.error("Genomic prediction failed — check logs above")
        sys.exit(1)

    logger.info("Done.")


if __name__ == '__main__':
    main()
