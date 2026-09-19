## What the software cannot tell you

- **A standard error is not a confidence interval.** The ± next to each
  parameter is the precision of the fit to this one averaged curve. It
  does not include the spread between replicate wells (shown as a band on
  the plot), between plates, or between days. For a publishable interval,
  fit each replicate separately or repeat the plate.
- **BAD means "not fittable", not "no growth".** A culture that grows in
  two phases, that lyses early, or that only rises 0.1 OD can be
  biologically interesting and still be BAD. Look at the curve before
  drawing a conclusion from the call.
- **GOOD means the model fits, not that the experiment was right.** A
  contaminated well can produce a beautiful sigmoid.
- **μ is an OD rate, not a specific growth rate.** It is the slope of the
  steepest tangent in OD per hour. Convert to per-cell terms with the
  doubling time, and remember OD600 stops being proportional to biomass
  above roughly 1 OD.
- **The ceiling A of an incomplete curve is an extrapolation.** Read μ and
  λ from such a curve, but not A.
- **Blanks are only as good as the wells you chose.** Media that change
  colour or precipitate over time make the blank subtraction drift; the
  Plate QC heat map will show blanks that are not flat.
- **Thresholds are conventions.** R² ≥ {min_r_squared} and 20 % errors are
  sensible defaults tuned on this project's plates. If you change them,
  the methods paragraph and the manifest record what you used.
