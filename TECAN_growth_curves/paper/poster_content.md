# UROC Poster Content

## Title
**From Plate Reader to Genome: An Automated Pipeline for Bacterial Growth Curve Analysis with Genomic Prediction of Pesticide Degradation Capacity**

Daniel Sambold, Mary Snook, Nate Walton, Dominique Scott, and Nathaniel Jue

Department of Marine Science, California State University, Monterey Bay

---

## Panel 1: Introduction & Motivation

Pesticide contamination threatens soil ecosystems worldwide. Bioremediation using soil bacteria that degrade pesticides is a sustainable alternative to chemical remediation, but screening candidate strains requires analyzing hundreds of growth curves from high-throughput plate reader experiments — a process that is labor-intensive and subjective when performed manually.

**This study:**
- 92 bacterial strains tested against 7 pesticides (3 chemical classes)
- 161 averaged growth curves from 717 raw wells
- 3 independent operators spanning 2018-2025
- Automated eleven-step computational pipeline (Python)
- Novel genomic prediction module to screen strains computationally

**Pesticides tested:**
- Pyrethroids: bifenthrin, permethrin, lambda-cyhalothrin
- Organophosphates: malathion, diazinon
- Neonicotinoids: imidacloprid, flupyradifurone

---

## Panel 2: Pipeline Overview

*Use Figure 1 (pipeline schematic)*

**Key methods:**

Modified Gompertz growth model:
y(t) = A * exp(-exp((mu*e/A)(lambda - t) + 1))

Haldane substrate inhibition ODE:
mu(S) = mu_max * S / (Ks + S + S^2/Ki)

**Pipeline steps:**
0. Preprocess raw TECAN data (blank subtract, average triplicates)
1. Train 2-stage ML classifier (HistGradientBoosting)
2. Gompertz fitting with adaptive truncation + ML classification
3. Combine results across 6 experimental groups
4. Haldane ODE inhibition modeling (AICc comparison)
5. Bayesian hierarchical models (NUTS + DEMetropolisZ)
6. Statistical analysis (ANOVA, pairwise tests)
7. Truncation method comparison + bad strain rescue
8. Export for collaboration
9. Synthetic validation (555 curves, independent holdout)
10. Inter-operator reproducibility analysis
11. Genomic prediction (genotype-to-phenotype) *NEW*

---

## Panel 3: Results

*Use Figure 6 (Ki forest plot) and Figure 8 (representative curves)*

### ML Classification
| Metric | Value |
|--------|-------|
| Held-out accuracy | 99.5% |
| Precision | 100% |
| Recall | 99.3% |
| Synthetic validation | 98.7% (555 curves, no data leakage) |

### Substrate Inhibition (Ki Rankings)
| Pesticide | Ki (median) | Interpretation |
|-----------|-------------|----------------|
| Imidacloprid | 4.83 | Most inhibitory |
| Flupyradifurone | 5.87 | Highly inhibitory |
| Diazinon | 9.32 | Moderate |
| Malathion | 11.84 | Moderate |
| Lambda-cyhalothrin | 12.28 | Moderate |
| Permethrin | 14.79 | Low inhibition |
| Bifenthrin | 17.55 | Least inhibitory |

- Haldane kinetics preferred over Gompertz in **30/42 (71%)** pesticide-strain combinations
- Lower Ki = stronger inhibition at high substrate concentration

---

## Panel 4: Genomic Prediction Module (HERO)

### "Predicting Degradation Capacity from Genotype"

**The problem:** Running plate reader experiments for every strain x pesticide combination is expensive. Can we predict which strains will degrade a given pesticide from their genome sequence alone?

**Our approach:**

```
Strain Genome
    |
    v
BLAST vs. curated degradation gene databases
(carboxylesterase, opd/mpd, pyrethroid hydrolase,
 CYP P450, nitroreductase)
    |
    v
Gene Features (presence/absence + % identity)
    |
    v
Elastic Net Regression --> Predicted Gompertz mu
    |
    v
Bayesian Ridge --> Prior mean + sigma
    |
    v
Informative priors for hierarchical Bayesian model
```

**What we tested:**
- Gene presence/absence via tblastn against 76 NCBI reference proteins (carboxylesterase, OPD/MPD, pyrethroid hydrolase, nitrilase, CYP P450)
- Codon usage bias (ENC, GC3s) — predictor of maximum growth rate (Phydon approach)
- Assembly statistics (genome size, N50, GC%, gene count)
- Combined model: 33 features, elastic net with L1/L2 regularization

**Results (n=13 overlapping strains, malathion + diazinon):**

| Metric | Value |
|--------|-------|
| LOSOCV R² | -0.056 |
| Features selected by elastic net | 0/33 |
| Permutation test p-value | 1.000 |
| Improvement over population mean | 0.0% |

**Why:** MAL strains are near-clonal *Pseudomonas putida* (100% carboxylesterase identity, ENC 38.5 +/- 0.3, GC 62%). No genomic variation to predict with. Growth rate variation (mu 0.004-0.38) is driven by pesticide condition, not genotype.

**Key insight:** Genomic gene content does not predict within-species growth rate variation under pesticide stress. This is consistent with the literature: phenotypic plasticity in clonal populations is environment-driven, not genotype-driven (James et al. 2025, Lees et al. 2020).

**Next experiment:** Run TECAN assays on 15 Chlorp + Dimeth strains already sequenced (GC 37-68%, ENC 31-56 — real phylogenetic diversity). Pan-genome GWAS with Scoary on the diverse panel. RNA-seq on a subset to capture expression-level variation.

---

## Panel 5: Conclusions

1. Developed a validated eleven-step automated pipeline that transforms raw TECAN plate reader data into publication-ready growth curve analysis without manual intervention

2. Two-stage ML classifier achieves 99.5% accuracy with 100% precision — zero false positives on held-out test data

3. Haldane substrate inhibition kinetics are preferred over Gompertz in 71% of pesticide-strain combinations, confirming that substrate inhibition is a significant factor in bioremediation screening

4. Imidacloprid (Ki=4.83) and flupyradifurone (Ki=5.87) are the most inhibitory pesticides, suggesting neonicotinoids pose the greatest challenge for bacterial degradation

5. Genomic gene content (presence/absence, codon usage) does not predict within-species growth rate under pesticide stress (permutation p=1.0) — phylogenetically diverse strain panels and transcriptomic data are needed to close the genotype-phenotype loop

**"The pipeline works. The genomic module works. The biology says: bring diverse strains."**

---

## References

1. Zwietering et al. (1990) Modeling of the bacterial growth curve. *Appl Environ Microbiol* 56:1875-1881
2. Andrews (1968) A mathematical model for the continuous culture of microorganisms utilizing inhibitory substrates. *Biotechnol Bioeng* 10:707-723
3. James et al. (2025) Whole-genome phenotype prediction with machine learning: open problems in bacterial genomics. *Bioinformatics* 41:btaf206
4. Lees et al. (2020) Improved prediction of bacterial genotype-phenotype associations using pangenome-spanning regressions. *mBio* 11:e01344-20
5. Ngara & Zhang (2022) mibPOPdb: An online database for microbial biodegradation of persistent organic pollutants. *iMeta* 1:e45
6. Jain et al. (2020) Omics approaches to pesticide biodegradation. *Current Microbiology* 77:3369-3386

---

## Figures to Include
- **Panel 2**: Figure1_pipeline.png (pipeline schematic)
- **Panel 3**: Figure6_Ki_forest_plot.png (Ki rankings — hero visual)
- **Panel 3**: Figure8_representative_curves.png (example fitted curves)
- **Panel 4**: Genomic prediction dataflow diagram + results table showing R²=-0.06, p=1.0
- **Panel 4**: Genome diversity plot — GC% vs ENC colored by strain group (MAL/DIAZ clustered, Chlorp/Dimeth spread)
