"""
Genomic feature extraction for genotype-to-phenotype prediction.

Extracts degradation gene features from BLAST results or Prokka annotations
and maps them to strain IDs used in the TECAN growth curve pipeline.

Usage:
    # As a module
    from genomic_features import load_all_genomic_features, resolve_strain_id

    # As a standalone script
    python genomic_features.py --blast-dir data/genomic/blast_results \
                               --strain-mapping data/genomic/strain_mapping.csv \
                               --output data/genomic/genomic_features.csv
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gene family definitions — target degradation genes per pesticide class
# ---------------------------------------------------------------------------

GENE_FAMILIES = {
    'carboxylesterase': {
        'keywords': ['carboxylesterase', 'carboxyl esterase', 'estA', 'estB',
                     'carE', 'CE', 'esterase'],
        'relevant_pesticides': ['Malathion', 'Diazinon'],
        'weight': 1.0,  # high relevance for organophosphates
    },
    'opd_mpd': {
        'keywords': ['organophosphate degradation', 'organophosphorus hydrolase',
                     'opd', 'mpd', 'methyl parathion hydrolase',
                     'phosphotriesterase', 'OPH', 'OpdA', 'PTE'],
        'relevant_pesticides': ['Malathion', 'Diazinon'],
        'weight': 0.8,
    },
    'pyrethroid_hydrolase': {
        'keywords': ['pyrethroid hydrolase', 'pytH', 'pytZ', 'Est3385',
                     'pyrethroid esterase', 'permethrinase'],
        'relevant_pesticides': ['Bifenthrin', 'Permethrin', 'LambdaCyhalothrin'],
        'weight': 1.0,
    },
    'cytochrome_p450': {
        'keywords': ['cytochrome P450', 'CYP', 'monooxygenase', 'cypA',
                     'cyp6', 'P450'],
        'relevant_pesticides': ['Bifenthrin', 'Permethrin', 'LambdaCyhalothrin',
                                'Imidacloprid', 'Flupyradifurone'],
        'weight': 0.5,  # general — present in many organisms
    },
    'nitroreductase': {
        'keywords': ['nitroreductase', 'nfl', 'nitro reductase'],
        'relevant_pesticides': ['Imidacloprid', 'Flupyradifurone'],
        'weight': 1.0,
    },
    'nitrilase': {
        'keywords': ['nitrilase', 'nitrile hydratase', 'amidase'],
        'relevant_pesticides': ['Imidacloprid', 'Flupyradifurone'],
        'weight': 0.7,
    },
    'efflux_pump': {
        'keywords': ['efflux pump', 'acrA', 'acrB', 'tolC', 'multidrug',
                     'drug resistance', 'MFS transporter'],
        'relevant_pesticides': ['Bifenthrin', 'Permethrin', 'LambdaCyhalothrin',
                                'Malathion', 'Diazinon', 'Imidacloprid',
                                'Flupyradifurone'],
        'weight': 0.3,  # general tolerance, not degradation
    },
}

# All known pesticide names (normalized)
ALL_PESTICIDES = [
    'Bifenthrin', 'Flupyradifurone', 'LambdaCyhalothrin',
    'Malathion', 'Imidacloprid', 'Permethrin', 'Diazinon',
]

# BLAST format 6 column names
BLAST_COLUMNS = [
    'qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen',
    'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore',
]


# ---------------------------------------------------------------------------
# Strain ID resolver
# ---------------------------------------------------------------------------

# Pattern: everything after the last hyphen is the biological strain ID
# e.g., "BifenthrinANDLB-BIF2" -> "BIF2"
#        "MALATHIONANDLB-MAL8" -> "MAL8"
#        "LB-BIF2" -> "BIF2"
#        "H2O-BIF2" -> "BIF2"
_STRAIN_SUFFIX_RE = re.compile(r'-([A-Za-z]+\d+[A-Za-z]*)$')

# Known media/condition prefixes to strip
_CONDITION_PREFIXES = [
    'BifenthrinANDLB', 'BIFENTHRINANDLB', 'Bifenthrin', 'BIFENTHRIN',
    'FlupyradifuroneANDLB', 'FLUPYRADIFURONEANDLB', 'Flupyradifurone',
    'FLUPYRADIFURONE',
    'LambdaCyhalothrinANDLB', 'LAMBDACYHALOTHRINANDLB',
    'LambdaCyhalothrin', 'LAMBDACYHALOTHRIN',
    'MalathionANDLB', 'MALATHIONANDLB', 'Malathion', 'MALATHION',
    'ImidaclopridANDLB', 'IMIDACLOPRIDANDLB', 'Imidacloprid', 'IMIDACLOPRID',
    'PermethrinANDLB', 'PERMETHRINANDLB', 'Permethrin', 'PERMETHRIN',
    'DiazinonANDLB', 'DIAZINONANDLB', 'Diazinon', 'DIAZINON',
    # Glucose variants (Group5)
    'MalathionANDG', 'MALATHIONANDG',
    'ImidaclopridANDG', 'IMIDACLOPRIDANDG',
    'DiazinonANDG', 'DIAZINONANDG',
    'LB', 'H2O',
]


def resolve_strain_id(strain_name: str) -> str:
    """
    Extract the biological strain ID from a full pipeline strain name.

    Examples:
        'BifenthrinANDLB-BIF2' -> 'BIF2'
        'MALATHIONANDLB-MAL8' -> 'MAL8'
        'LB-BIF2' -> 'BIF2'
        'H2O-IMID2' -> 'IMID2'
        'Bifenthrin-BIF2' -> 'BIF2'
        'LAMBDACYHALOTHRIN-LCY1' -> 'LCY1'
    """
    match = _STRAIN_SUFFIX_RE.search(strain_name)
    if match:
        return match.group(1)
    # Fallback: return as-is if no pattern matches
    return strain_name


def extract_pesticide_from_strain(strain_name: str) -> Optional[str]:
    """
    Extract the pesticide/condition from a full strain name.

    Returns normalized pesticide name or None for controls (LB, H2O).
    """
    upper = strain_name.upper()
    if upper.startswith('H2O-') or upper.startswith('LB-'):
        return None

    pesticide_map = {
        'BIFENTHRIN': 'Bifenthrin',
        'FLUPYRADIFURONE': 'Flupyradifurone',
        'LAMBDACYHALOTHRIN': 'LambdaCyhalothrin',
        'MALATHION': 'Malathion',
        'IMIDACLOPRID': 'Imidacloprid',
        'PERMETHRIN': 'Permethrin',
        'DIAZINON': 'Diazinon',
    }
    for key, val in pesticide_map.items():
        if key in upper:
            return val
    return None


def build_strain_lookup(results_csv: Path) -> Dict[str, List[str]]:
    """
    Build a mapping from biological strain ID -> list of full strain names.

    Args:
        results_csv: Path to all_groups_results.csv

    Returns:
        Dict mapping e.g. 'BIF2' -> ['BifenthrinANDLB-BIF2', 'LB-BIF2', ...]
    """
    df = pd.read_csv(results_csv)
    lookup = {}
    for full_name in df['strain'].unique():
        bio_id = resolve_strain_id(full_name)
        lookup.setdefault(bio_id, []).append(full_name)
    return lookup


# ---------------------------------------------------------------------------
# BLAST result parsing
# ---------------------------------------------------------------------------

def parse_blast_results(blast_file: Path,
                        evalue_threshold: float = 1e-10,
                        min_identity: float = 30.0) -> pd.DataFrame:
    """
    Parse a BLAST format-6 output file and filter by thresholds.

    Args:
        blast_file: Path to BLAST tabular output (format 6)
        evalue_threshold: Maximum e-value to keep
        min_identity: Minimum percent identity to keep

    Returns:
        Filtered DataFrame with BLAST hits
    """
    if not blast_file.exists():
        logger.warning(f"BLAST file not found: {blast_file}")
        return pd.DataFrame(columns=BLAST_COLUMNS)

    try:
        df = pd.read_csv(blast_file, sep='\t', header=None, names=BLAST_COLUMNS,
                         comment='#')
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=BLAST_COLUMNS)

    # Filter
    mask = (df['evalue'] <= evalue_threshold) & (df['pident'] >= min_identity)
    filtered = df[mask].copy()
    logger.info(f"  {blast_file.name}: {len(df)} raw hits -> {len(filtered)} after filtering")
    return filtered


_ACCESSION_MAP = None

def _load_accession_map() -> Dict[str, str]:
    """Load accession-to-family mapping from JSON if available."""
    global _ACCESSION_MAP
    if _ACCESSION_MAP is not None:
        return _ACCESSION_MAP

    import json
    map_path = Path(__file__).parent.parent / 'data' / 'genomic' / 'reference_databases' / 'accession_to_family.json'
    if map_path.exists():
        with open(map_path) as f:
            data = json.load(f)
        _ACCESSION_MAP = {k: v['family'] for k, v in data.items()}
        logger.info(f"Loaded accession map: {len(_ACCESSION_MAP)} entries")
    else:
        _ACCESSION_MAP = {}
    return _ACCESSION_MAP


def classify_hit_gene_family(seq_id: str) -> Optional[str]:
    """
    Classify a BLAST hit into a gene family.

    First checks accession-to-family mapping (for NCBI accessions from tblastn).
    Falls back to keyword matching on the sequence ID string.
    """
    # Check accession mapping first (handles NCBI accessions like TFF53494.1)
    acc_map = _load_accession_map()
    if seq_id in acc_map:
        family = acc_map[seq_id]
        return family if family != 'other' else None

    # Fallback: keyword matching on sequence ID
    sid_lower = seq_id.lower()
    for family_name, family_info in GENE_FAMILIES.items():
        for keyword in family_info['keywords']:
            if keyword.lower() in sid_lower:
                return family_name
    return None


# ---------------------------------------------------------------------------
# Prokka GFF annotation parsing (alternative to BLAST)
# ---------------------------------------------------------------------------

def parse_prokka_gff(gff_file: Path) -> pd.DataFrame:
    """
    Parse a Prokka GFF annotation file and extract gene product annotations.

    Returns DataFrame with columns: gene_id, product, gene_family
    """
    if not gff_file.exists():
        logger.warning(f"GFF file not found: {gff_file}")
        return pd.DataFrame(columns=['gene_id', 'product', 'gene_family'])

    records = []
    with open(gff_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.startswith('>'):
                break  # FASTA section
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            attrs = parts[8]
            # Parse product from GFF attributes
            product = None
            gene_id = None
            for attr in attrs.split(';'):
                if attr.startswith('product='):
                    product = attr.split('=', 1)[1]
                elif attr.startswith('ID='):
                    gene_id = attr.split('=', 1)[1]

            if product:
                gene_family = _match_product_to_family(product)
                records.append({
                    'gene_id': gene_id or 'unknown',
                    'product': product,
                    'gene_family': gene_family,
                })

    return pd.DataFrame(records)


def _match_product_to_family(product: str) -> Optional[str]:
    """Match a Prokka product annotation to a gene family."""
    product_lower = product.lower()
    for family_name, family_info in GENE_FAMILIES.items():
        for keyword in family_info['keywords']:
            if keyword.lower() in product_lower:
                return family_name
    return None


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_genomic_features(strain_id: str,
                             blast_hits: pd.DataFrame,
                             pesticide: Optional[str] = None) -> Dict[str, float]:
    """
    Extract genomic features for a single strain from its BLAST results.

    Args:
        strain_id: Biological strain ID (e.g., 'MAL8')
        blast_hits: Filtered BLAST results DataFrame for this strain
        pesticide: If provided, compute pesticide-specific relevance score

    Returns:
        Dict of feature_name -> value (float). Missing features are NaN.
    """
    features = {}

    # Total degradation gene hits
    features['n_degradation_genes'] = 0

    for family_name in GENE_FAMILIES:
        # Classify hits by gene family using query ID (reference protein)
        # For tblastn: qseqid = reference protein, sseqid = genome contig
        # Try qseqid first, fall back to sseqid for standard blastn
        family_hits = blast_hits[
            blast_hits['qseqid'].apply(
                lambda s: classify_hit_gene_family(s) == family_name
            ) | blast_hits['sseqid'].apply(
                lambda s: classify_hit_gene_family(s) == family_name
            )
        ] if len(blast_hits) > 0 else pd.DataFrame()

        n_hits = len(family_hits)
        features[f'has_{family_name}'] = 1.0 if n_hits > 0 else 0.0
        features[f'n_hits_{family_name}'] = float(n_hits)

        if n_hits > 0:
            features[f'max_pident_{family_name}'] = family_hits['pident'].max()
            features[f'best_bitscore_{family_name}'] = family_hits['bitscore'].max()
            features[f'mean_pident_{family_name}'] = family_hits['pident'].mean()
            features['n_degradation_genes'] += n_hits
        else:
            features[f'max_pident_{family_name}'] = np.nan
            features[f'best_bitscore_{family_name}'] = np.nan
            features[f'mean_pident_{family_name}'] = np.nan

    # Pesticide-specific relevance score
    features['pesticide_gene_relevance_score'] = _compute_relevance_score(
        features, pesticide
    )

    return features


def extract_genomic_features_from_gff(strain_id: str,
                                      gff_df: pd.DataFrame,
                                      pesticide: Optional[str] = None
                                      ) -> Dict[str, float]:
    """
    Extract genomic features from parsed Prokka GFF annotations.

    Same output format as extract_genomic_features() for interchangeability.
    """
    features = {}
    features['n_degradation_genes'] = 0

    for family_name in GENE_FAMILIES:
        family_hits = gff_df[gff_df['gene_family'] == family_name]
        n_hits = len(family_hits)

        features[f'has_{family_name}'] = 1.0 if n_hits > 0 else 0.0
        features[f'n_hits_{family_name}'] = float(n_hits)

        # GFF doesn't have pident/bitscore — use presence count as proxy
        if n_hits > 0:
            features[f'max_pident_{family_name}'] = 100.0  # annotated = assumed high identity
            features[f'best_bitscore_{family_name}'] = np.nan  # not available from GFF
            features[f'mean_pident_{family_name}'] = 100.0
            features['n_degradation_genes'] += n_hits
        else:
            features[f'max_pident_{family_name}'] = np.nan
            features[f'best_bitscore_{family_name}'] = np.nan
            features[f'mean_pident_{family_name}'] = np.nan

    features['pesticide_gene_relevance_score'] = _compute_relevance_score(
        features, pesticide
    )
    return features


def _compute_relevance_score(features: Dict[str, float],
                             pesticide: Optional[str]) -> float:
    """
    Compute a weighted relevance score combining gene presence with
    pesticide-specific importance.

    Score = sum(has_family * weight * pident_normalized) for relevant families.
    If no pesticide specified, uses all families equally.
    """
    if pesticide is None:
        # General score: average across all families
        total = 0.0
        count = 0
        for family_name, family_info in GENE_FAMILIES.items():
            has_gene = features.get(f'has_{family_name}', 0.0)
            if has_gene:
                pident = features.get(f'max_pident_{family_name}', 0.0)
                if np.isnan(pident):
                    pident = 0.0
                total += family_info['weight'] * (pident / 100.0)
            count += 1
        return total / max(count, 1)

    # Pesticide-specific score: weight only relevant families
    total = 0.0
    max_possible = 0.0
    for family_name, family_info in GENE_FAMILIES.items():
        if pesticide in family_info['relevant_pesticides']:
            max_possible += family_info['weight']
            has_gene = features.get(f'has_{family_name}', 0.0)
            if has_gene:
                pident = features.get(f'max_pident_{family_name}', 0.0)
                if np.isnan(pident):
                    pident = 0.0
                total += family_info['weight'] * (pident / 100.0)

    return total / max(max_possible, 1e-10)


# ---------------------------------------------------------------------------
# Batch feature extraction
# ---------------------------------------------------------------------------

def load_all_genomic_features(blast_dir: Path,
                              strain_mapping_csv: Path,
                              annotations_dir: Optional[Path] = None,
                              evalue_threshold: float = 1e-10,
                              min_identity: float = 30.0) -> pd.DataFrame:
    """
    Load and extract genomic features for all strains.

    Tries BLAST results first; falls back to Prokka GFF annotations
    if BLAST results are not available for a strain.

    Args:
        blast_dir: Directory containing BLAST format-6 output files
        strain_mapping_csv: CSV mapping strain IDs to genome paths
        annotations_dir: Optional directory with Prokka GFF files
        evalue_threshold: BLAST e-value filter
        min_identity: BLAST minimum percent identity filter

    Returns:
        DataFrame indexed by strain_id with genomic feature columns
    """
    mapping = pd.read_csv(strain_mapping_csv)
    all_features = []

    for _, row in mapping.iterrows():
        strain_id = row['strain_id']
        features = {'strain_id': strain_id}

        # Try BLAST results first
        blast_file = _find_blast_file(blast_dir, strain_id)
        if blast_file is not None:
            hits = parse_blast_results(blast_file, evalue_threshold, min_identity)
            feat = extract_genomic_features(strain_id, hits)
            features.update(feat)
            features['feature_source'] = 'blast'
            logger.info(f"  {strain_id}: {len(hits)} BLAST hits -> "
                        f"{int(feat['n_degradation_genes'])} degradation genes")

        # Fall back to GFF annotations
        elif annotations_dir is not None:
            gff_file = _find_gff_file(annotations_dir, strain_id)
            if gff_file is not None:
                gff_df = parse_prokka_gff(gff_file)
                feat = extract_genomic_features_from_gff(strain_id, gff_df)
                features.update(feat)
                features['feature_source'] = 'gff'
                logger.info(f"  {strain_id}: {len(gff_df)} GFF annotations -> "
                            f"{int(feat['n_degradation_genes'])} degradation genes")
            else:
                feat = _empty_features()
                features.update(feat)
                features['feature_source'] = 'none'
                logger.warning(f"  {strain_id}: no BLAST or GFF data found")
        else:
            feat = _empty_features()
            features.update(feat)
            features['feature_source'] = 'none'
            logger.warning(f"  {strain_id}: no BLAST data found (GFF dir not specified)")

        all_features.append(features)

    df = pd.DataFrame(all_features)
    if len(df) > 0:
        df = df.set_index('strain_id')
    return df


def _find_blast_file(blast_dir: Path, strain_id: str) -> Optional[Path]:
    """Find BLAST output file for a strain (tries common naming patterns)."""
    patterns = [
        f'{strain_id}_vs_degradation_db.txt',
        f'{strain_id}_blast.txt',
        f'{strain_id}.blast',
        f'{strain_id}_blast_results.tsv',
        f'{strain_id}.tsv',
    ]
    for pat in patterns:
        candidate = blast_dir / pat
        if candidate.exists():
            return candidate

    # Try case-insensitive glob
    for f in blast_dir.glob(f'{strain_id}*'):
        if f.suffix in ('.txt', '.tsv', '.blast', '.out'):
            return f

    return None


def _find_gff_file(annotations_dir: Path, strain_id: str) -> Optional[Path]:
    """Find Prokka GFF file for a strain."""
    patterns = [f'{strain_id}.gff', f'{strain_id}.gff3']
    for pat in patterns:
        candidate = annotations_dir / pat
        if candidate.exists():
            return candidate
    for f in annotations_dir.glob(f'{strain_id}*'):
        if f.suffix in ('.gff', '.gff3'):
            return f
    return None


def _empty_features() -> Dict[str, float]:
    """Return a feature dict with all NaN values (no genomic data)."""
    features = {'n_degradation_genes': np.nan}
    for family_name in GENE_FAMILIES:
        features[f'has_{family_name}'] = np.nan
        features[f'n_hits_{family_name}'] = np.nan
        features[f'max_pident_{family_name}'] = np.nan
        features[f'best_bitscore_{family_name}'] = np.nan
        features[f'mean_pident_{family_name}'] = np.nan
    features['pesticide_gene_relevance_score'] = np.nan
    return features


# ---------------------------------------------------------------------------
# Genomic feature names (for ML classifier integration)
# ---------------------------------------------------------------------------

def get_genomic_feature_names() -> List[str]:
    """Return the list of genomic feature column names."""
    names = ['n_degradation_genes']
    for family_name in GENE_FAMILIES:
        names.extend([
            f'has_{family_name}',
            f'n_hits_{family_name}',
            f'max_pident_{family_name}',
            f'best_bitscore_{family_name}',
            f'mean_pident_{family_name}',
        ])
    names.append('pesticide_gene_relevance_score')
    return names


def get_compact_genomic_feature_names() -> List[str]:
    """
    Return a compact set of genomic features for the ML classifier.

    These are the most informative features without redundancy.
    """
    names = ['n_degradation_genes']
    for family_name in GENE_FAMILIES:
        names.extend([
            f'has_{family_name}',
            f'max_pident_{family_name}',
        ])
    names.append('pesticide_gene_relevance_score')
    return names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Extract genomic features from BLAST results or GFF annotations'
    )
    parser.add_argument('--blast-dir', type=Path, required=True,
                        help='Directory with BLAST format-6 output files')
    parser.add_argument('--strain-mapping', type=Path, required=True,
                        help='CSV mapping strain IDs to genome data')
    parser.add_argument('--annotations-dir', type=Path, default=None,
                        help='Directory with Prokka GFF files (fallback)')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output CSV path for genomic features')
    parser.add_argument('--evalue', type=float, default=1e-10,
                        help='BLAST e-value threshold (default: 1e-10)')
    parser.add_argument('--min-identity', type=float, default=30.0,
                        help='BLAST minimum percent identity (default: 30)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    logger.info("Extracting genomic features...")
    logger.info(f"  BLAST dir: {args.blast_dir}")
    logger.info(f"  Strain mapping: {args.strain_mapping}")
    logger.info(f"  Annotations dir: {args.annotations_dir}")

    df = load_all_genomic_features(
        blast_dir=args.blast_dir,
        strain_mapping_csv=args.strain_mapping,
        annotations_dir=args.annotations_dir,
        evalue_threshold=args.evalue,
        min_identity=args.min_identity,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output)
    logger.info(f"Saved genomic features for {len(df)} strains to {args.output}")

    # Summary
    if len(df) > 0 and 'n_degradation_genes' in df.columns:
        has_data = df['n_degradation_genes'].notna()
        logger.info(f"  Strains with genomic data: {has_data.sum()}/{len(df)}")
        if has_data.any():
            mean_genes = df.loc[has_data, 'n_degradation_genes'].mean()
            logger.info(f"  Mean degradation genes per strain: {mean_genes:.1f}")


if __name__ == '__main__':
    main()
