"""Tests for genotype-to-phenotype prediction module."""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_genomic_features():
    """Create synthetic genomic features for testing."""
    rng = np.random.default_rng(42)
    n_strains = 15
    strain_ids = [f'STRAIN{i}' for i in range(n_strains)]

    df = pd.DataFrame({
        'strain_id': strain_ids,
        'n_degradation_genes': rng.integers(0, 5, n_strains).astype(float),
        'has_carboxylesterase': rng.choice([0.0, 1.0], n_strains),
        'has_opd_mpd': rng.choice([0.0, 1.0], n_strains),
        'has_pyrethroid_hydrolase': rng.choice([0.0, 1.0], n_strains),
        'has_cytochrome_p450': rng.choice([0.0, 1.0], n_strains),
        'has_nitroreductase': rng.choice([0.0, 1.0], n_strains),
        'has_nitrilase': rng.choice([0.0, 1.0], n_strains),
        'has_efflux_pump': rng.choice([0.0, 1.0], n_strains),
        'max_pident_carboxylesterase': rng.uniform(40, 99, n_strains),
        'max_pident_opd_mpd': rng.uniform(30, 95, n_strains),
        'pesticide_gene_relevance_score': rng.uniform(0, 1, n_strains),
    })
    df = df.set_index('strain_id')
    return df


@pytest.fixture
def synthetic_phenotype_data():
    """Create synthetic phenotype data matching the genomic strains."""
    rng = np.random.default_rng(42)
    n_strains = 15
    records = []
    for i in range(n_strains):
        strain_id = f'STRAIN{i}'
        # Create a pesticide+LB condition
        records.append({
            'strain': f'MalathionANDLB-{strain_id}',
            'bio_strain_id': strain_id,
            'pesticide': 'Malathion',
            'is_good': rng.choice([True, False]),
            'gompertz_mu': 0.1 + rng.uniform(0, 0.5),
            'gompertz_lambda': 1.0 + rng.uniform(0, 5),
            'gompertz_a': 0.8 + rng.uniform(0, 1),
        })
    return pd.DataFrame(records)


@pytest.fixture
def merged_data(synthetic_phenotype_data, synthetic_genomic_features):
    """Merge genomic and phenotypic data."""
    genomic_cols = [c for c in synthetic_genomic_features.columns]
    merged = synthetic_phenotype_data.merge(
        synthetic_genomic_features,
        left_on='bio_strain_id',
        right_index=True,
        how='inner'
    )
    return merged, genomic_cols


# ---------------------------------------------------------------------------
# Elastic Net tests
# ---------------------------------------------------------------------------

class TestElasticNet:
    def test_train_elastic_net_runs(self, merged_data):
        merged, genomic_cols = merged_data
        X = merged[genomic_cols].values.copy()
        y = merged['gompertz_mu'].values.copy()

        # Replace NaN
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            nan_mask = np.isnan(X[:, j])
            X[nan_mask, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0.0

        from importlib import import_module
        gp = import_module('11_genomic_prediction')

        result = gp.train_elastic_net(X, y, genomic_cols, 'gompertz_mu')
        assert 'model' in result
        assert 'r_squared' in result
        assert 'feature_importance' in result
        assert isinstance(result['r_squared'], float)

    def test_elastic_net_produces_predictions(self, merged_data):
        merged, genomic_cols = merged_data
        X = merged[genomic_cols].values.copy()
        y = merged['gompertz_mu'].values

        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            nan_mask = np.isnan(X[:, j])
            X[nan_mask, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0.0

        from importlib import import_module
        gp = import_module('11_genomic_prediction')

        result = gp.train_elastic_net(X, y, genomic_cols, 'gompertz_mu')
        predictions = result['model'].predict(X)
        assert len(predictions) == len(y)
        assert not np.any(np.isnan(predictions))


# ---------------------------------------------------------------------------
# Bayesian Ridge tests
# ---------------------------------------------------------------------------

class TestBayesianRidge:
    def test_train_bayesian_ridge_runs(self, merged_data):
        merged, genomic_cols = merged_data
        X = merged[genomic_cols].values.copy()
        y = merged['gompertz_mu'].values

        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            nan_mask = np.isnan(X[:, j])
            X[nan_mask, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0.0

        from importlib import import_module
        gp = import_module('11_genomic_prediction')

        result = gp.train_bayesian_ridge(X, y, genomic_cols, 'gompertz_mu')
        assert 'model' in result
        assert 'predictions' in result
        assert 'uncertainties' in result
        assert len(result['predictions']) == len(y)
        assert len(result['uncertainties']) == len(y)
        # Uncertainties should be positive
        assert np.all(result['uncertainties'] > 0)


# ---------------------------------------------------------------------------
# LOSOCV tests
# ---------------------------------------------------------------------------

class TestLOSOCV:
    def test_losocv_runs(self, merged_data):
        merged, genomic_cols = merged_data
        X = merged[genomic_cols].values.copy()
        y = merged['gompertz_mu'].values
        strain_ids = merged['bio_strain_id'].values

        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            nan_mask = np.isnan(X[:, j])
            X[nan_mask, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0.0

        from importlib import import_module
        gp = import_module('11_genomic_prediction')

        cv_df = gp.losocv(X, y, strain_ids, genomic_cols, 'gompertz_mu')
        assert len(cv_df) > 0
        assert 'actual' in cv_df.columns
        assert 'predicted_genomic' in cv_df.columns
        assert 'predicted_baseline' in cv_df.columns
        assert 'error_genomic' in cv_df.columns
        assert 'error_baseline' in cv_df.columns

    def test_losocv_all_strains_held_out(self, merged_data):
        merged, genomic_cols = merged_data
        X = merged[genomic_cols].values.copy()
        y = merged['gompertz_mu'].values
        strain_ids = merged['bio_strain_id'].values

        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            nan_mask = np.isnan(X[:, j])
            X[nan_mask, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0.0

        from importlib import import_module
        gp = import_module('11_genomic_prediction')

        cv_df = gp.losocv(X, y, strain_ids, genomic_cols, 'gompertz_mu')
        # Each unique strain should appear as held-out
        held_out_strains = set(cv_df['strain'].unique())
        all_strains = set(np.unique(strain_ids))
        # At least most strains should be held out (some may be skipped if < 3 training)
        assert len(held_out_strains) >= len(all_strains) - 2


# ---------------------------------------------------------------------------
# Degradation classifier tests
# ---------------------------------------------------------------------------

class TestDegradationClassifier:
    def test_train_classifier_runs(self, merged_data):
        merged, genomic_cols = merged_data
        X = merged[genomic_cols].values.copy()
        y_binary = merged['is_good'].astype(int).values

        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            nan_mask = np.isnan(X[:, j])
            X[nan_mask, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0.0

        from importlib import import_module
        gp = import_module('11_genomic_prediction')

        result = gp.train_degradation_classifier(X, y_binary, genomic_cols)
        assert 'model' in result
        assert 'accuracy' in result
        assert 0 <= result['accuracy'] <= 1
        assert 'f1' in result


# ---------------------------------------------------------------------------
# ML classifier integration test
# ---------------------------------------------------------------------------

class TestMLClassifierIntegration:
    def test_genomic_features_in_classifier(self, synthetic_genomic_features):
        from ml_classifier import (
            extract_genomic_features_for_classifier,
            GENOMIC_FEATURES,
        )

        features = extract_genomic_features_for_classifier(
            'MalathionANDLB-STRAIN0',
            genomic_df=synthetic_genomic_features,
        )
        assert len(features) == len(GENOMIC_FEATURES)
        # STRAIN0 should have data
        assert not all(np.isnan(v) for v in features.values())

    def test_genomic_features_missing_strain(self, synthetic_genomic_features):
        from ml_classifier import (
            extract_genomic_features_for_classifier,
            GENOMIC_FEATURES,
        )

        features = extract_genomic_features_for_classifier(
            'MalathionANDLB-UNKNOWN',
            genomic_df=synthetic_genomic_features,
        )
        # Unknown strain should return all NaN
        assert all(np.isnan(v) for v in features.values())

    def test_genomic_features_no_df(self):
        from ml_classifier import (
            extract_genomic_features_for_classifier,
            GENOMIC_FEATURES,
        )

        features = extract_genomic_features_for_classifier(
            'MalathionANDLB-MAL8',
            genomic_df=None,
        )
        assert all(np.isnan(v) for v in features.values())


# ---------------------------------------------------------------------------
# Prior format tests
# ---------------------------------------------------------------------------

class TestPriorFormat:
    def test_prior_csv_columns(self, merged_data, synthetic_genomic_features):
        """Verify genomic_priors.csv has the columns Bayesian model expects."""
        merged, genomic_cols = merged_data

        from importlib import import_module
        gp = import_module('11_genomic_prediction')

        # Mock phenotype_df with required columns
        phenotype_df = merged[['strain', 'bio_strain_id', 'pesticide',
                               'gompertz_mu', 'gompertz_lambda', 'gompertz_a',
                               'is_good']].copy()

        priors_df = gp.generate_bayesian_priors(
            phenotype_df, synthetic_genomic_features, merged, genomic_cols
        )

        if len(priors_df) > 0:
            assert 'strain_id' in priors_df.columns
            # Should have at least mu priors
            assert 'mu_prior_mean' in priors_df.columns
            assert 'mu_prior_sigma' in priors_df.columns
