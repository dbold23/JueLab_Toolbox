# Lab protocol: from plate reader to results in ten minutes

This is the bench-side checklist for running one plate through the app.
For what the numbers mean, read [How it works](HOW_IT_WORKS.md).

## Before the run

1. **Lay out the plate with blanks and triplicates.** Every media type on
   the plate needs at least two, preferably three, wells of that media with
   no cells (blanks). Every strain × media condition needs three wells.
   Leave unused wells empty and mark them `NA` in the key.
2. **Write the key as you pipette.** One row per well, three columns:

   | Cell | Media | Strain |
   |---|---|---|
   | A1 | LB | Blank |
   | A2 | LB | IMID2 |
   | B7 | MALATHIONANDLB | MAL1 |
   | H12 | NA | NA |

   Use the strain name for `Strain` and `Blank` for blank wells. The media
   name is free text, but keep it identical for every well of that media
   (`LB` and `lb` are merged with a warning; `LB ` with a trailing space is
   not the same as `LB`). Pesticide-plus-broth wells are recognised from
   names like `MALATHIONANDLB`, `Malathion+LB` or `LB + malathion`.
3. **Set the reader to record every 15 minutes** for at least 24 h, with
   shaking between reads and the temperature you want logged. Longer runs
   are fine: the software cuts each curve where primary growth ends.

## Running the analysis

Start the app from the repository folder:

```bash
make app          # or: python -m streamlit run app/Home.py
```

### Home page: load and check the plate

1. Drop the SparkControl `.xlsx` export on the uploader (or a `Values`
   CSV). The reader's own metadata (start time, temperature) appears under
   the file name.
2. Drop your key CSV, or pick the key sheet if it is inside the workbook.
   The key opens in an editable table: fix typos there and press
   **Apply key**. Any problem that would make the run meaningless (a media
   with no blank, a well in the key that is not on the plate) is listed in
   red and the run button stays off until it is fixed.
3. Look at the **plate grid**. Colour is the condition; blanks are
   hatched. Click a well to see its raw trace. You can paint a condition
   onto wells straight from this grid if the key is wrong.
4. Give the run a name (the default is the file name and the date) and
   press **Run analysis**. A 96-well plate takes a few seconds with the
   default window rule and one to three minutes with the adaptive rule.

### Plate QC page: is the plate itself sound?

- The **Δ OD heat map** should show replicates of the same colour at the
  same brightness and blanks near zero. A bright blank or a dark replicate
  is a well to exclude (Curves page) or a note for the lab book.
- The **temperature trace** should be flat to within half a degree.
- The **verdict grid** shows GOOD, BAD and not-fitted wells at a glance.

### Curves page: one condition at a time

- Cards are sorted BAD-first, so the curves that need a human come first.
  Each card names the gate that failed and shows the measured value beside
  the threshold.
- Open a card to see the fit, the residuals, the gate table and **Show the
  math**. If the automatic window is wrong, drag the slider and press
  **Refit with this window**, or **Try adaptive on this curve**. Both are
  recorded as overrides with your note.
- To drop one bad replicate, tick it under **Exclude wells** and apply;
  the condition is re-averaged from the remaining wells.
- To overrule the verdict, choose the **Call of record**, write why, and
  save. The automatic verdict is kept alongside yours in every export.

### Compare page: what the pesticide did

Pick the parameter (μ, A, λ or doubling time). The strip shows each
strain's pesticide conditions relative to its own LB control. Only GOOD
curves are included; the number behind each bar is printed.

### Export page: take it with you

- **Results CSV**: one row per condition with parameters, errors, gates
  and your overrides.
- **Clean summary**: the report-ready table (A ± SE, μ ± SE, λ ± SE,
  doubling time, R²).
- **Full bundle (zip)**: everything above plus the processed curves, the
  key, a methods paragraph with the thresholds actually used, the figures,
  a `config.yaml` snapshot and `run_manifest.json` (which includes the
  command that reproduces the run from the command line). Keep the bundle
  with the lab notebook entry; it can be re-loaded into the app later.

## Sanity checks before you trust a plate

| Check | What you expect | If not |
|---|---|---|
| LB controls | GOOD, R² > 0.98, A between 0.6 and 1.5 OD | Blank subtraction or plate layout is wrong |
| Water controls | Little growth: a rise of 0.1 to 0.2 OD from broth carried over with the inoculum is normal | A water well as high as the LB control means contamination, or the key has the wells swapped |
| Replicate band | Narrow band around each curve | Pipetting error, edge effect, condensation |
| Temperature | Flat trace | Rates around the dip are not comparable |
| Manifest | Settings match what you meant to use | Reset settings in the sidebar and re-run |

## Batch mode for many plates

The same analysis runs from the command line and writes the same tables:

```bash
python scripts/run_full_pipeline.py --steps 0,2,3 --no-ml      # preprocess, fit, combine
python scripts/run_full_pipeline.py --dry-run                   # show what would run
```

Group folders and file names are listed in `scripts/config.yaml`.
