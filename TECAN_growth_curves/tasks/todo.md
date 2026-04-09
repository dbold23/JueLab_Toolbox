# Next Steps — Genomic Prediction Phase 3

## Completed
- [x] Genomic module built and tested (52/52 tests)
- [x] 30 genomes annotated with Pyrodigal
- [x] BLAST against 76 NCBI reference proteins
- [x] Codon usage features extracted (ENC, GC3s)
- [x] Combined model (33 features): R²=0.000, p=1.0 — honest negative result
- [x] Poster content updated with real results
- [x] Documentation updated (PIPELINE_METHODOLOGY.md, flowchart, accuracy fix)

## Next: Wet Lab
- [ ] Run TECAN plate reader assays on 15 Chlorp + Dimeth strains
  - These have GC 37-68%, ENC 31-56 — real phylogenetic diversity
  - Would bring n from 13 near-identical to 28 diverse strains
  - Pesticide conditions: malathion, diazinon, chlorpyrifos, dimethoate(?)

## Next: Computational (after TECAN data)
- [ ] Re-run genomic prediction pipeline with diverse strain panel
- [ ] Pan-genome GWAS with Scoary (accessory gene associations)
- [ ] Permutation test on diverse panel
- [ ] If signal found: generate poster figure (predicted vs actual growth rate)

## Stretch Goals
- [ ] RNA-seq on 3-5 strains under pesticide exposure
- [ ] Pfam domain annotation via HMMER (hmmsearch vs Pfam-A)
- [ ] DNABERT-2 embeddings as features (if n > 50)
