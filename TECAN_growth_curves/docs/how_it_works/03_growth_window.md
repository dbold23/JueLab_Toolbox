## Growth phases and the fitting window

![Growth window](images/growth_window.svg)

A culture goes through a **lag** (no visible growth), an **exponential**
phase (fastest rise), a **stationary** phase (plateau) and often a **decline**
(cells lyse, clumps settle, the medium changes colour). The Gompertz model
describes the first three phases only. Fitting it to a curve that also
contains a decline, or a second rise on a different nutrient (diauxic
growth), pulls the parameters toward nonsense. So the first job is to cut
the curve where primary growth ends.

### The rule applied by default: first peak

1. A **{smoothing_window}-point centred rolling mean** of the curve is computed.
   It is used only to locate the peak; the model is fitted to the raw points.
2. The rate of rise of that smoothed curve is computed in OD per hour. The
   curve's own fastest rate sets the scale, so the rule does not depend on
   how often the reader sampled or how fast the strain grows.
3. Scanning from the first point where the smoothed OD exceeds 0.1, the peak
   is where the rate crosses **25 %** of the curve's fastest rate on the way
   down: the mean rate over the previous 5 steps was above it, the mean rate
   over the next 10 steps is at or below it, and the OD there is at least
   half of the curve's maximum. On an ideal sigmoid that crossing sits at
   about 90 % of the ceiling *A*. If no such point exists the global maximum
   is used.

   The 25 % was calibrated on this project's four plates, against the
   verdicts of the committed results tables corrected by the visual audit
   (85 conditions): it agrees on 83 of them and cuts within a median of 3 h
   of the cross-validated search. A stricter 10 % left curves with a slow
   secondary drift uncut and agreed on only 75 of 85.
4. The true maximum of the raw data within ±{smoothing_window} points of that
   spot is found, and **{buffer_points} buffer points** past it are kept so
   the plateau is represented.

### The other choices

- **Global maximum**: cut at the highest smoothed point, plus the buffer. Use
  it for a curve with two rises when you want the whole envelope rather than
  the first phase.
- **Adaptive (cross-validated)**: try many candidate windows, both start and
  end. Each window is scored by fitting the model on random 80 % subsets of
  its points and measuring how well it predicts the 20 % it did not see, 20
  times over. The best-predicting window wins; windows that tie within 0.002
  of the best score are broken in favour of the one closest to the first
  peak, and any window whose fitted ceiling exceeds the highest OD actually
  observed by more than 15 % is discarded. This is the method used for the
  published tables. It is about a hundred times slower (one to three minutes
  per plate) and is fully deterministic: the same plate always gives the
  same answer.

### Curves that never reached a plateau

Before any cut, the last 15 % of the run is inspected. If the smoothed curve
is still rising there faster than **0.005 OD/h** and the final value is
within 5 % of the run's maximum, the culture was still growing when the run
ended. Such an **incomplete** curve is fitted whole, the parameter *A* is
flagged as extrapolated, and the residual-pattern gate is not applied to it,
because a curve without a plateau cannot be expected to fit like one.

### When the window is rejected

A window is unusable, and the condition is reported as **not fitted**, when it
holds fewer than **{min_points_after_truncation}** points or when the peak lies
within the first hour of the run (a spike at the start, not growth). With
readings every 15 minutes the point minimum is reached after five hours of
growth; with hourly readings, lower it in the sidebar.

### Overriding the cut by hand

On the Curves page you can drag the window yourself and refit. The override
is stored with a timestamp and a note, exported with the results, and shown
on the curve as "manual window". The automatic cut is never lost: "Back to
automatic" restores it.
