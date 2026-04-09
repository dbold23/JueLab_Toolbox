"""Tests for genomic feature extraction module."""

import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from genomic_features import (
    resolve_strain_id,
    extract_pesticide_from_strain,
    build_strain_lookup,
    parse_blast_results,
    classify_hit_gene_family,
    extract_genomic_features,
    extract_genomic_features_from_gff,
    parse_prokka_gff,
    load_all_genomic_features,
    get_genomic_feature_names,
    get_compact_genomic_feature_names,
    _empty_features,
    BLAST_COLUMNS,
    GENE_FAMILIES,
)


# ---------------------------------------------------------------------------
# Strain ID resolver tests
# ---------------------------------------------------------------------------

class TestResolveStrainId:
    def test_pesticide_and_lb(self):
        assert resolve_strain_id('BifenthrinANDLB-BIF2') == 'BIF2'

    def test_pesticide_and_lb_uppercase(self):
        assert resolve_strain_id('MALATHIONANDLB-MAL8') == 'MAL8'

    def test_lb_control(self):
        assert resolve_strain_id('LB-BIF2') == 'BIF2'

    def test_h2o_control(self):
        assert resolve_strain_id('H2O-IMID2') == 'IMID2'

    def test_pesticide_only(self):
        assert resolve_strain_id('Bifenthrin-BIF2') == 'BIF2'

    def test_lambda_cyhalothrin(self):
        assert resolve_strain_id('LAMBDACYHALOTHRIN-LCY1') == 'LCY1'

    def test_case_insensitive_strain(self):
        assert resolve_strain_id('BifenthrinANDLB-BiF7') == 'BiF7'

    def test_glucose_media(self):
        assert resolve_strain_id('MALATHIONANDG-MAL8') == 'MAL8'

    def test_permethrin_strain(self):
        assert resolve_strain_id('PermethrinANDLB-CISPERM1') == 'CISPERM1'

    def test_year_suffix(self):
        assert resolve_strain_id('H2O-IMID1Y') == 'IMID1Y'

    def test_no_prefix(self):
        # If no pattern matches, return as-is
        assert resolve_strain_id('RAW_STRAIN') == 'RAW_STRAIN'


class TestExtractPesticide:
    def test_bifenthrin(self):
        assert extract_pesticide_from_strain('BifenthrinANDLB-BIF2') == 'Bifenthrin'

    def test_malathion(self):
        assert extract_pesticide_from_strain('MALATHIONANDLB-MAL8') == 'Malathion'

    def test_lb_control(self):
        assert extract_pesticide_from_strain('LB-BIF2') is None

    def test_h2o_control(self):
        assert extract_pesticide_from_strain('H2O-IMID2') is None

    def test_imidacloprid(self):
        assert extract_pesticide_from_strain('IMIDACLOPRID-IMID3') == 'Imidacloprid'

    def test_lambda_cyhalothrin(self):
        assert extract_pesticide_from_strain('LAMBDACYHALOTHRINANDLB-LCY1') == 'LambdaCyhalothrin'


# ---------------------------------------------------------------------------
# BLAST parsing tests
# ---------------------------------------------------------------------------

@pytest.fixture
def blast_file(tmp_path):
    """Create a synthetic BLAST format-6 output file."""
    content = (
        "MAL8_contig1\tcarboxylesterase_EstA\t95.5\t300\t14\t0\t1\t300\t1\t300\t1e-50\t500\n"
        "MAL8_contig1\tcarboxylesterase_EstB\t78.2\t250\t55\t0\t1\t250\t1\t250\t1e-30\t350\n"
        "MAL8_contig2\topd_Pseudomonas\t45.0\t200\t110\t2\t1\t200\t1\t200\t1e-15\t200\n"
        "MAL8_contig3\trandom_gene\t25.0\t100\t75\t0\t1\t100\t1\t100\t0.5\t50\n"  # below thresholds
        "MAL8_contig4\tnitroreductase_nfl1\t55.0\t180\t81\t1\t1\t180\t1\t180\t1e-12\t180\n"
    )
    f = tmp_path / 'MAL8_blast.txt'
    f.write_text(content)
    return f


class TestBlastParsing:
    def test_parse_blast_results_filters(self, blast_file):
        df = parse_blast_results(blast_file, evalue_threshold=1e-10, min_identity=30.0)
        # Should filter out random_gene (evalue=0.5) and opd (pident=45 > 30 but evalue=1e-15 > 1e-10? no 1e-15 < 1e-10)
        # Actually 1e-15 < 1e-10, so opd should be kept
        # random_gene: evalue=0.5 > 1e-10, so filtered out
        assert len(df) == 4  # EstA, EstB, opd, nitroreductase

    def test_parse_blast_results_columns(self, blast_file):
        df = parse_blast_results(blast_file)
        assert list(df.columns) == BLAST_COLUMNS

    def test_parse_missing_file(self, tmp_path):
        df = parse_blast_results(tmp_path / 'nonexistent.txt')
        assert len(df) == 0
        assert list(df.columns) == BLAST_COLUMNS

    def test_parse_empty_file(self, tmp_path):
        empty = tmp_path / 'empty.txt'
        empty.write_text('')
        df = parse_blast_results(empty)
        assert len(df) == 0


class TestClassifyHit:
    def test_carboxylesterase(self):
        assert classify_hit_gene_family('carboxylesterase_EstA_Pseudomonas') == 'carboxylesterase'

    def test_opd(self):
        assert classify_hit_gene_family('opd_organophosphate_degradation') == 'opd_mpd'

    def test_nitroreductase(self):
        assert classify_hit_gene_family('nitroreductase_nfl1') == 'nitroreductase'

    def test_unknown(self):
        assert classify_hit_gene_family('hypothetical_protein_12345') is None

    def test_pyrethroid_hydrolase(self):
        assert classify_hit_gene_family('pytH_pyrethroid_hydrolase') == 'pyrethroid_hydrolase'


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_blast_hits():
    """Create a sample filtered BLAST results DataFrame."""
    return pd.DataFrame({
        'qseqid': ['c1', 'c1', 'c2'],
        'sseqid': ['carboxylesterase_EstA', 'carboxylesterase_EstB', 'opd_Pseudomonas'],
        'pident': [95.5, 78.2, 45.0],
        'length': [300, 250, 200],
        'mismatch': [14, 55, 110],
        'gapopen': [0, 0, 2],
        'qstart': [1, 1, 1],
        'qend': [300, 250, 200],
        'sstart': [1, 1, 1],
        'send': [300, 250, 200],
        'evalue': [1e-50, 1e-30, 1e-15],
        'bitscore': [500, 350, 200],
    })


class TestFeatureExtraction:
    def test_extract_returns_all_features(self, sample_blast_hits):
        features = extract_genomic_features('MAL8', sample_blast_hits)
        expected_names = get_genomic_feature_names()
        for name in expected_names:
            assert name in features, f"Missing feature: {name}"

    def test_carboxylesterase_detected(self, sample_blast_hits):
        features = extract_genomic_features('MAL8', sample_blast_hits)
        assert features['has_carboxylesterase'] == 1.0
        assert features['max_pident_carboxylesterase'] == 95.5
        assert features['best_bitscore_carboxylesterase'] == 500

    def test_opd_detected(self, sample_blast_hits):
        features = extract_genomic_features('MAL8', sample_blast_hits)
        assert features['has_opd_mpd'] == 1.0
        assert features['max_pident_opd_mpd'] == 45.0

    def test_undetected_family_is_nan(self, sample_blast_hits):
        features = extract_genomic_features('MAL8', sample_blast_hits)
        assert features['has_pyrethroid_hydrolase'] == 0.0
        assert np.isnan(features['max_pident_pyrethroid_hydrolase'])

    def test_n_degradation_genes(self, sample_blast_hits):
        features = extract_genomic_features('MAL8', sample_blast_hits)
        assert features['n_degradation_genes'] == 3  # 2 carboxylesterase + 1 opd

    def test_relevance_score_for_malathion(self, sample_blast_hits):
        features = extract_genomic_features('MAL8', sample_blast_hits, pesticide='Malathion')
        score = features['pesticide_gene_relevance_score']
        assert score > 0  # Has carboxylesterase + opd, both relevant for malathion

    def test_relevance_score_for_imidacloprid(self, sample_blast_hits):
        # These hits are organophosphate genes, not relevant for neonicotinoids
        features = extract_genomic_features('MAL8', sample_blast_hits, pesticide='Imidacloprid')
        score_imid = features['pesticide_gene_relevance_score']
        features_mal = extract_genomic_features('MAL8', sample_blast_hits, pesticide='Malathion')
        score_mal = features_mal['pesticide_gene_relevance_score']
        # Malathion relevance should be higher than imidacloprid relevance
        assert score_mal > score_imid

    def test_empty_blast_hits(self):
        empty = pd.DataFrame(columns=BLAST_COLUMNS)
        features = extract_genomic_features('BIF2', empty)
        assert features['n_degradation_genes'] == 0
        for family in GENE_FAMILIES:
            assert features[f'has_{family}'] == 0.0

    def test_empty_features_all_nan(self):
        features = _empty_features()
        assert np.isnan(features['n_degradation_genes'])
        for family in GENE_FAMILIES:
            assert np.isnan(features[f'has_{family}'])


# ---------------------------------------------------------------------------
# Feature name lists
# ---------------------------------------------------------------------------

class TestFeatureNames:
    def test_genomic_feature_names_not_empty(self):
        names = get_genomic_feature_names()
        assert len(names) > 0
        assert 'n_degradation_genes' in names
        assert 'pesticide_gene_relevance_score' in names

    def test_compact_feature_names_subset(self):
        full = set(get_genomic_feature_names())
        compact = set(get_compact_genomic_feature_names())
        assert compact.issubset(full)
        assert len(compact) < len(full)

    def test_all_families_represented(self):
        names = get_genomic_feature_names()
        for family in GENE_FAMILIES:
            assert f'has_{family}' in names
            assert f'max_pident_{family}' in names


# ---------------------------------------------------------------------------
# Batch loading tests
# ---------------------------------------------------------------------------

class TestBatchLoading:
    def test_load_with_blast_dir(self, tmp_path):
        # Create strain mapping
        mapping = pd.DataFrame({
            'strain_id': ['MAL8', 'BIF2'],
            'genome_accession': ['', ''],
            'genome_fasta_path': ['', ''],
            'annotation_gff_path': ['', ''],
            '16S_taxonomy': ['', ''],
            'notes': ['', ''],
        })
        mapping_path = tmp_path / 'strain_mapping.csv'
        mapping.to_csv(mapping_path, index=False)

        # Create BLAST dir with one file
        blast_dir = tmp_path / 'blast_results'
        blast_dir.mkdir()
        blast_content = (
            "c1\tcarboxylesterase_EstA\t95.5\t300\t14\t0\t1\t300\t1\t300\t1e-50\t500\n"
        )
        (blast_dir / 'MAL8_blast.txt').write_text(blast_content)

        df = load_all_genomic_features(blast_dir, mapping_path)
        assert len(df) == 2
        assert 'MAL8' in df.index
        assert 'BIF2' in df.index
        # MAL8 has data, BIF2 doesn't
        assert df.loc['MAL8', 'feature_source'] == 'blast'
        assert df.loc['BIF2', 'feature_source'] == 'none'

    def test_load_empty_dir(self, tmp_path):
        mapping = pd.DataFrame({
            'strain_id': ['MAL8'],
            'genome_accession': [''],
            'genome_fasta_path': [''],
            'annotation_gff_path': [''],
            '16S_taxonomy': [''],
            'notes': [''],
        })
        mapping_path = tmp_path / 'strain_mapping.csv'
        mapping.to_csv(mapping_path, index=False)

        blast_dir = tmp_path / 'blast_results'
        blast_dir.mkdir()

        df = load_all_genomic_features(blast_dir, mapping_path)
        assert len(df) == 1
        assert df.loc['MAL8', 'feature_source'] == 'none'


# ---------------------------------------------------------------------------
# GFF parsing tests
# ---------------------------------------------------------------------------

class TestGffParsing:
    def test_parse_prokka_gff(self, tmp_path):
        gff_content = (
            "##gff-version 3\n"
            "contig1\tProkka\tCDS\t100\t400\t.\t+\t0\tID=gene1;product=carboxylesterase type B\n"
            "contig1\tProkka\tCDS\t500\t800\t.\t+\t0\tID=gene2;product=hypothetical protein\n"
            "contig2\tProkka\tCDS\t100\t350\t.\t-\t0\tID=gene3;product=nitroreductase family protein\n"
        )
        gff_file = tmp_path / 'MAL8.gff'
        gff_file.write_text(gff_content)

        df = parse_prokka_gff(gff_file)
        assert len(df) == 3
        # carboxylesterase should be classified
        assert df.iloc[0]['gene_family'] == 'carboxylesterase'
        # hypothetical protein should not be classified
        assert pd.isna(df.iloc[1]['gene_family'])
        # nitroreductase should be classified
        assert df.iloc[2]['gene_family'] == 'nitroreductase'

    def test_gff_feature_extraction(self, tmp_path):
        gff_content = (
            "##gff-version 3\n"
            "c1\tProkka\tCDS\t100\t400\t.\t+\t0\tID=g1;product=carboxylesterase\n"
        )
        gff_file = tmp_path / 'MAL8.gff'
        gff_file.write_text(gff_content)

        gff_df = parse_prokka_gff(gff_file)
        features = extract_genomic_features_from_gff('MAL8', gff_df)
        assert features['has_carboxylesterase'] == 1.0
        assert features['n_degradation_genes'] == 1
