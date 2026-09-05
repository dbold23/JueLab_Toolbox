<!-- Produced by a three-designer, two-judge design panel. Implementation follows this spec. -->
# TECAN Growth-Curve App — Final Design Spec

**Stance.** Build teaching-first's product (BAD-first review, "why BAD" scorecard, tweak-and-see-what-flipped loop, lab-manual explanation) on analysis-first's engineering (one shared per-curve code path, `config.yaml` defaults, manifest provenance, rules as call-of-record) with bench-first's plate-grid front door (paint the key, read verdicts off the grid, example-plate button, ratio strip). Launch: `make app` → `streamlit run app/Home.py`. Offline, no accounts, no server.

**Decisions that resolve the three designs' contradictions**

| Conflict | Decision |
|---|---|
| Verdict view: grid vs table vs cards | All three, same data: plate grid (Home, after run), BAD-first card list (Curves), sortable table (Export). Grid click → card. |
| ML default | **Off**; rules are the call of record. When `models/*.joblib` exist, an Advanced toggle adds a `P(good)` second-opinion column only. Fresh clones (models gitignored) therefore match each other. |
| Adaptive (MCCV) truncation | Never a plate default. Per-curve button "Try adaptive on this curve" (~5–6.5 s measured) and an Advanced plate-wide option labelled "~3 min per plate". |
| Draggable cut handle, right-click/Backspace clearing, `st.data_editor` fill-selection | **Cut** — not implementable in Streamlit 1.63. Replaced by: truncation slider (hours, snapped to cycles) + click-a-point via `st.plotly_chart(on_select="rerun")`; key painting via Plotly box-select + chips + a "Clear selected" button. |
| Mandatory override notes | Truncation edits and replicate exclusions: note optional, always timestamped in the manifest. Hand-flipping a verdict GOOD↔BAD: note **required** (it changes the call of record). |
| Preview-and-confirm before Run | Cut. QC plots live on the Plate page; Run needs only zero blocking key errors. |
| Threshold change = refit? | Post-fit gate changes re-run only `classify_by_fit_quality` on cached `FitResult`s (instant); truncation/pre-fit changes re-run `analyze_curve` (~3 s per plate). Both raise the flip badge. |
| Shared per-curve function | Already present in the working tree: `analyze_curve()` extracted in `scripts/01` and the untracked `tecan_growth/` package (io, plate, preprocess, analyze, settings). Commit them once the parity test (§10) passes; the app builds on them. Fix `tecan_growth/settings.py` default `use_adaptive=True` → `False` to match `config.yaml`. |
| `== 'Blank'` bug in `02_preprocess` | Fix in the script (case-insensitive); app uses `tecan_growth.plate.is_blank_label`. |

## 1. Page map

| Page | Purpose |
|---|---|
| **Home** | Load export → key on the plate grid → Run. "Try the example plate" (Group 3). Recent runs. After Run, the grid recolors by verdict. |
| **1 Plate QC** | Raw-well sanity: ΔOD heatmap, replicate facets per media, blank flatness, temperature trace. |
| **2 Curves** | BAD-first cards → detail: 4-panel pipeline strip, residuals, gate scorecard, "Show the math", truncation override, replicate exclusion, review (correct/wrong/unsure). |
| **3 Compare** | A / μ / λ / doubling time across treatment types per strain; ratio strip; curve overlays. |
| **4 Export** | Tables, figures, methods paragraph, manifest, ZIP. |
| **5 How it works** | Lab-manual explanation with live thresholds, equations, annotated gallery. Every "?" in the app deep-links here. |
| Sidebar (all pages) | Run name; classifier pill ("Rules" / "Rules + ML opinion"); counts GOOD/BAD/not fitted; Settings drawer (3 sliders + locked Advanced); sticky "Results are stale — Re-run" banner. |

## 2. User flow

1. **Home.** Drop `GrowthRate_*.xlsx` or a values CSV, or click **Try the example plate**. `tecan_growth.io.read_tecan_export` finds the kinetic block by header signature (`Cycle Nr.`/`Time [s]` + A1…H12) in any sheet — handles `Values`, `GROUP2_DATA`, `Bif_Flup_LCY_Values`, and the row-63 block in *Result sheet* — and reads instrument metadata. Receipt line: "400 cycles × 15 min = 99.7 h · 96 wells · 600 nm · 37.0 °C · 2025-03-15".
2. **Key.** Auto-loads a `Cell,Media,Strain` sheet if present; else a dropped key CSV, a saved key from `keys/`, or paint from scratch. The 96-well Plotly grid pre-marks BLANK from the *Result sheet* layout labels (`BL*`). Paint: choose a Media chip and a Strain chip (type-to-add), box-select wells, click **Assign**. Validation badges update live. **Save key as…** writes CLI-compatible CSV.
3. **Run** (enabled only with zero blocking errors). `preprocess_plate` → `analyze_plate` with a per-curve progress bar; ~3 s for 24 curves. Grid recolors by verdict; app jumps to Curves.
4. **Curves.** Cards sorted BAD-first. Click a card → detail. Read the scorecard: which gate failed, the value, the threshold, one sentence why.
5. **Disagree?** Move the truncation slider or click a point → refit that curve via `analyze_curve` with `truncation_override`; exclude a replicate → re-preprocess + refit that curve. Or open Settings, nudge a slider → stale banner → **Re-run** → flip badge: "LB-IMID5: BAD → GOOD (R² gate 0.95 → 0.93)".
6. **Compare**, then **Export** the bundle. Reopening a run folder restores it without recomputing.

## 3. Input controls

| Control | Where | Default (source) | Validation |
|---|---|---|---|
| Data file | Home | — | `.xlsx`/`.csv`; kinetic header found; time strictly increasing; ≥15 cycles; wavelength ≠ 600 nm → warning |
| Key source | Home | key sheet → CSV → saved → paint | every non-empty well has Media+Strain |
| Media / Strain chips | Home | last used | uppercased, trimmed; **no `-` or `_`** (results IDs are `MEDIA-STRAIN`; scripts 03/04 `rsplit('-',1)`) — blocking with auto-fix suggestion; `Blank/BLANK/BL` → BLANK; `NA`/empty → unused |
| Well state | Home | EMPTY | SAMPLE / BLANK / EMPTY / EXCLUDED |
| Blank mode (per media with no blanks) | Home banner | per_media | choose "blanks from ▸ media", "all plate blanks", or "no subtraction"; recorded as `blank_source` |
| Run name | Sidebar | `<date>_<file stem>` | folder-safe |
| Truncation method | Settings | **First peak** (`config.yaml`) | First peak / Global max / Adaptive (labelled ~3 min/plate, confirm) |
| Min R² | Settings | 0.95 | 0.80–0.99, step 0.005 |
| Max parameter error % | Settings | 20 | 5–50 |
| Min ΔOD (absolute) | Settings | 0.15 | 0.02–0.50 |
| Reset to defaults | Settings | — | restores `config.yaml` |
| Advanced (locked by "I understand these change the call of record") | Settings | from `config.yaml`: smoothing 5, buffer 3, min points 20, excellent-R² 0.98, SNR 5.0, ΔOD-CI 0.10, monotone 0.55, autocorr 0.70, pre-fit SNR 3.0 / ΔOD 0.5, tail fraction 0.15, slope 0.005 | excellent-R² ≥ min R²; smoothing odd 3–15; every change written to manifest |
| ML second opinion | Advanced | off; disabled with tooltip "no models/*.joblib — run `make train-classifier`" | thresholds 0.7/0.3 read-only |
| Truncation override | Curve detail | auto value | slider in hours snapped to cycles, or click a point; must leave ≥20 points; "Try adaptive on this curve" button |
| Exclude replicate | Curve detail | none | ≥1 well must remain (warn if <2); note optional |
| Override verdict | Curve detail | none | GOOD/BAD + **required** note |
| Review | Curve detail | — | correct / wrong / unsure / skip + note (matches `validation_audit.csv`) |

## 4. Visualizations (all Plotly unless noted)

| View | Question | Encodings | Interactions |
|---|---|---|---|
| **Plate grid** (Home) | Is the layout right? Which wells worked? | 8×12 circles. Before run: fill = media (categorical), bold "B" = blank, dotted = empty, grey X = excluded, text = strain. After run: fill = GOOD green / BAD red-orange / BORDERLINE amber (ML only) / grey not fitted; text = max OD; ⚠ corner = replicate outlier | box-select → paint; click → Curves detail; hover: well, media, strain, verdict, reason |
| **ΔOD heatmap** (Plate QC) | Dead wells, edge effects, contamination? | color = raw max − mean(first 5); border = media; hatched blank | hover; click → detail |
| **Replicate facets** (Plate QC) | Do triplicates agree? Which is the outlier? | facet per media; 3 thin lines per strain; blank mean dashed | legend isolates; raw/blanked toggle |
| **Temperature trace** | Did the incubator hold? | line + target ±1 °C band | — |
| **Curve cards** | Where do I look first? | sparkline (blanked + fit), verdict chip, A·μ·λ·R² | filter media/strain/verdict; sort BAD-first |
| **Pipeline strip** (4 linked panels, shared x) | What did the model see? | ① raw wells + blank mean dashed ② blanked mean ± replicate SD ribbon, zero-clipped points marked ③ smoothed line, cut as vertical band, excluded region shaded, "INCOMPLETE — not truncated" badge ④ points used + Gompertz over full axis (extrapolation dashed), λ dotted vertical, μ tangent at inflection, A dashed horizontal | click point in ③ sets cut; shared zoom; colours: data grey, fit blue, cut amber |
| **Residuals** | Is the misfit structured or noise? | residual vs time, ±2·RMSE band, sign-runs shaded, lag-1 r annotated | hover |
| **Gate scorecard** (`st.dataframe`) | Why BAD, exactly? | gate · value · threshold · PASS/FAIL/BYPASSED/NOT APPLIED · one sentence | select row → evidence highlighted on strip (e.g. baseline window for SNR) |
| **Show the math** (expanders) | How was this number produced? | equation with actual numbers substituted (LaTeX) | link to How-it-works section |
| **Parameter dot plot** (Compare) | Does pesticide inhibit? Does +LB rescue? | x = treatment type (H2O_only, Pesticide_only, Pesticide_plus_LB, LB_only), colour = strain, ±1 SE bars, hollow = BAD | parameter tabs A/μ/λ/t_d; include-BAD toggle |
| **Ratio strip** (Compare) | Inhibition? Carbon source? | per strain: μ(pesticide+LB)/μ(LB), A(pesticide)/A(H2O); 1.0 reference line | hover |
| **Overlay per strain** (Compare) | Same question, shape-level | blanked curves + fits by media | strain selector |
| **Stats table** (Compare) | More than noise? | ANOVA + Kruskal on μ via `run_statistical_tests`; suppressed with note when n<3 per group | — |
| Export figures | — | matplotlib PNG via existing `plot_*`; Plotly HTML | — |

## 5. Errors and edge cases

| Situation | Detection | Surfacing |
|---|---|---|
| No kinetic block in any sheet | `read_tecan_export` | Blocking banner listing sheets found + expected header; manual sheet picker |
| Media with no BLANK | key validation | Blocking: wells pulse red, Run button reads "Add blanks for MALATHION"; escape hatch = blank-mode picker, recorded |
| Blank wells rising (max blank ΔOD > 0.1) | preprocess | Warning on blank wells: "Blank grew — contamination?" |
| Data well not in key | key vs data | Info count "6 wells unassigned", treated EMPTY |
| Key well not in data | key vs data | Blocking (CSV export dropped columns) |
| Duplicate media+strain | key | Blocking |
| `-`/`_` in names | key | Blocking with auto-fix |
| <3 replicates / outlier well (>3 SD from others at peak) | preprocess | ⚠ chip; exclusion offered |
| <15 usable points | `analyze_curve` stage `too_few_points` | Grey "not fitted: too few points" |
| Pre-fit rejection | stage `prefit_rejected` | "Not fitted: signal below noise (SNR 1.8 < 3, ΔOD 0.04 < 0.5)" |
| Truncation invalid | `validate_truncation` | BAD with verbatim reason; suggests manual cut or Global max |
| Incomplete curve | `detect_incomplete_curve` | Amber banner: no cut applied; autocorrelation gate BYPASSED; A/λ less certain |
| `curve_fit` fails | `FitResult.success=False` | BAD "Fit failed: <scipy message>"; panels ①–③ still drawn |
| Baseline clipped (≥50 % baseline points = 0 or `baseline_std` ≤ 1e-6) | scorecard | "SNR not meaningful (baseline clipped)"; flagged in manifest — never show 2.7×10⁷ |
| Non-monotonic time, NaN rows, temp excursion >1 °C | loader | Warnings; NaN rows dropped and counted |
| Settings changed, not re-run | hash mismatch | Sticky "Results are stale — Re-run" |
| ML models missing | startup | Pill "Rules"; toggle disabled; never an error |

## 6. "How it works" outline (each section: one figure + live thresholds)

1. **What the reader measures** — OD600 every 15 min; why blanks; why triplicates.
2. **Preprocessing** — t(h)=t(s)/3600; `OD(t) = mean(replicate wells) − mean(blank wells of that media)`, floored at 0; the flooring artefact and why SNR can be meaningless.
3. **Growth phases and truncation** — 5-point centred rolling mean (used only to find the peak); first-peak rule (derivative > 0.001 then ≤ 0.005 after OD > 0.1, peak ≥ 50 % of max) + 3 buffer points; Global max; Adaptive MCCV (one paragraph); incomplete rule (last 15 % still rising > 0.005 OD/h → keep everything).
4. **The model** — `y(t) = A · exp(−exp((μ·e/A)·(λ − t) + 1))` (Zwietering et al. 1990). **A**: the ceiling the culture reaches (OD). **μ**: the steepest slope, fastest the OD ever rises (OD/h). **λ**: hours the culture waits before that rise (where the steepest tangent crosses baseline). **e** makes μ the true tangent slope. Doubling time `t_d = ln2·A/(μ·e)`.
5. **Fitting** — least squares (`scipy.optimize.curve_fit`), initial guesses (max OD, max slope, time of max slope), bounds, SE = √diag(cov), relative error = SE/value.
6. **Fit quality** — `R² = 1 − Σ(y−ŷ)²/Σ(y−ȳ)²`: fraction of variation explained; RMSE; residuals should look like noise.
7. **The verdict** — decision tree figure; table: fit failed → BAD; R² < 0.95 "not S-shaped"; A or μ error > 20 % "number not trustworthy"; ΔOD < 0.15 "no growth" (always); A < 3·σ_baseline (always); when R² < 0.98: SNR < 5, ΔOD − 2σ < 0.1, monotone fraction < 0.55 "wanders instead of rising", lag-1 autocorrelation > 0.7 "systematic misfit — diauxic or contaminated" (skipped for incomplete curves). Pre-fit gate. ML opinion: what P(good) and BORDERLINE mean.
8. **Annotated gallery** — good; flat H2O; noisy low-OD; incomplete; early peak/contaminated; diauxic (first-peak vs global max) — from Group 3 + `synthetic_data/src/growth_models.py`.
9. **Comparing treatments** — what lower μ in pesticide+LB means; why BAD curves are excluded; the n=3 ANOVA caveat.
10. **What the software cannot tell you** — SE ≠ CI; BAD means "not fittable", not "no growth".
11. **Reproducibility and citation** — manifest fields, commit, Zwietering reference, this repo.

## 7. Exports

| File | Content |
|---|---|
| `processing_results.csv` | CLI columns via `legacy_results_frame` + `media, strain, treatment_type, wells, blank_source, excluded_wells, truncation_override, override_note, final_call, p_good` |
| `results_clean.csv` | `growth_rate_summary_for_collaboration.csv` columns (Treatment, Strain_ID, A±SE, μ±SE, λ±SE, doubling time, R²) — report-ready |
| `curves_long.csv` | `time_h, media, strain, well, od_raw, blank_mean, od_blanked, in_fit, od_predicted, residual` |
| `processed/<MEDIA>_DATA.csv` + `key.csv` | CLI input format (`make step-2` reproduces) |
| `run_manifest.json` | app/repo version + commit, package versions, input SHA-256, sheets used, instrument metadata, all settings, ML status/model hashes, overrides with notes and timestamps |
| `methods.md` | auto-filled paragraph with thresholds actually used |
| `validation_audit.csv` | existing schema (`audit_result` ∈ correct/wrong/unsure/skip) — feeds `09_train_classifier.py` |
| `figures/` | Plotly `.html` (always) + PNG from existing `plot_*` (kaleido optional) |
| `report.xlsx` | Results / Clean / Curves / Key / Settings / Overrides |
| `bundle.zip` | everything; runs live in `runs/<name>/`, manifest reload restores |

## 8. Code layout and how it calls validated code

```
tecan_growth/            # commit the existing untracked package
  legacy.py              # importlib-loads scripts/01 (unchanged numerics)
  io.py plate.py preprocess.py settings.py analyze.py   # exist
  analyze.py  + reclassify(cr, settings)  # classify_by_fit_quality on cached FitResult, same args as analyze_curve
  plots.py               # Plotly builders: plate_grid, pipeline_strip, residuals, compare_*, ratio_strip
  export.py              # results frames, curves_long, manifest, methods.md, xlsx, zip
  media.py               # LB / H2O / X / XANDLB → treatment_type (reuse parse_strain_column)
app/
  Home.py                # st.navigation, upload, key grid, Run, recent runs
  pages/1_Plate_QC.py 2_Curves.py 3_Compare.py 4_Export.py 5_How_it_works.py
  state.py               # session keys, run hash, stale flag, overrides
  content/how_it_works/*.md
keys/  runs/ (gitignored)  tests/test_app_parity.py
Makefile: app: streamlit run app/Home.py
```

Call chain: `read_tecan_export` → `PlateKey` → `preprocess_plate` → `analyze_plate` → per curve `scripts/01.analyze_curve` (pre-fit gate → `detect_incomplete_curve` → `truncate_at_max` → `validate_truncation` → `fit_gompertz` → `classify_by_fit_quality`). Overrides pass `truncation_override` into the same function; nothing numeric lives in `app/`. Caching: `st.cache_data` keyed on (file hash, key hash, settings hash).

## 9. Build order

| Priority | Deliverable |
|---|---|
| **P0 (MVP)** | Commit `analyze_curve` + `tecan_growth`; blank-case fix in 02; `settings.use_adaptive=False`; parity test; Home with loader, key from sheet/CSV, plate grid (display + click), Run; Curves cards + panels ①②④ + scorecard; sidebar 3 sliders; `processing_results.csv` + manifest; `make app` |
| **— MVP cut line —** | |
| **P1** | Paint-mode key + saved keys; Advanced lock; stale banner + flip badge (instant reclassify); residual panel + "Show the math"; Plate QC; How-it-works §1–7, 10; Compare dot plot + ratio strip |
| **P2** | Truncation slider/click + per-curve adaptive; replicate exclusion; verdict override; review → audit CSV; `curves_long`, `results_clean`, methods.md, xlsx, ZIP, manifest reload; example plate + gallery |
| **P3** | ML opinion column; ANOVA table; kaleido PNGs; batch mode (`all_groups_results.csv`) |

## 10. Acceptance checks

1. `make app` opens the browser in <10 s on a clean venv; no network calls (test with Wi-Fi off).
2. Example plate: open → Run → click a BAD well → read the failing gate, under 2 minutes.
3. `tests/test_app_parity.py`: Group 3 xlsx through the app path yields `gompertz_a/mu/lambda`, `fit_r_squared`, and classification identical (atol 1e-9) to `make step-2` rules-mode output for all 24 curves; loader succeeds on all four raw exports.
4. `reclassify` == `analyze_curve` classification for every Group 3 curve under default and three perturbed settings.
5. Remove all BLANK wells from MALATHION in the key → Run disabled, button text names the media.
6. Rename a strain to `MAL-8` → blocking error with fix suggestion.
7. Set Min R² to 0.93 → stale banner → Re-run → flip badge lists LB-IMID5 (or whichever flips) with the gate and old/new threshold.
8. Move truncation on one curve → only that curve refits; manifest records old/new cut with timestamp.
9. IMIDACLOPRID-IMID2 scorecard shows "SNR not meaningful (baseline clipped)", never a value > 1e6.
10. Export ZIP: `processed/*_DATA.csv` + `key.csv` rerun through the CLI reproduce `processing_results.csv`; `run_manifest.json` contains the input SHA-256 and commit; reloading the run folder restores every verdict and override without recomputation.
11. How-it-works renders every threshold from the current run's settings; changing Min R² changes the text.
12. With no `models/*.joblib`, the ML toggle is disabled with the tooltip and no BORDERLINE verdict appears anywhere.
