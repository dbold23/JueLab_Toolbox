## Preprocessing: from raw wells to one curve per condition

Every condition on the plate (one media, one strain) becomes a single curve
in three steps. Nothing else is done to the data before fitting; in
particular the raw points are **never smoothed** before the model sees them.

### 1. Time in hours

The reader records time in seconds since the start of the run:

$$t_{{\mathrm{{h}}}} = \frac{{t_{{\mathrm{{s}}}}}}{{3600}}$$

### 2. Blank subtraction, per media

At every time point, the mean of the blank wells **of the same media** is
subtracted from every culture well of that media:

$$\mathrm{{OD}}_{{\mathrm{{blanked}}}}(t) = \max\big(\,\overline{{\mathrm{{OD}}_{{\mathrm{{replicates}}}}}}(t) - \overline{{\mathrm{{OD}}_{{\mathrm{{blanks}}}}}}(t),\; 0\big)$$

The current setting is **`{blank_mode}`** blanks. `per_media` uses the blank
wells of that media only; `global` uses every blank well on the plate (use
it only when a media has no blank of its own); `none` skips the subtraction.

### 3. Replicates averaged

The replicate wells are averaged (`mean`). The standard deviation between
them is kept and shown as a band on the plots and in the `od_sd` column of
the long export.

### The clipping artefact, and why "signal-to-noise" needs care

Blank subtraction can produce slightly negative numbers when a culture well
reads a little below its blank, which is normal for a well that never grows.
Those values are set to zero (`clip_negative` = {clip_negative}). A well that
is clipped for its whole baseline then has **zero measured noise**, which
would make any signal-to-noise ratio infinite. The quality gates therefore
never use a noise estimate smaller than **0.002 OD**, the resolution of the
reader, and the results table records `baseline_clipped` so you can see
when that floor was applied.

### What you can check on the Plate QC page

- The **Δ OD heat map** shows every well's total rise. Replicates should look
  alike; blanks should be near zero. One replicate far from its siblings is
  the first thing to look at, and a single well can be excluded from a
  condition on the Curves page.
- The **temperature trace** should be flat. A dip means the lid was opened or
  the incubator lagged; growth rates around that time are not comparable.
