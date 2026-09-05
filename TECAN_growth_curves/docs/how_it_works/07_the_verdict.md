## The verdict: GOOD or BAD

![Decision tree](images/verdict_tree.svg)

Every condition ends up as one of three calls. The Curves page shows which
gate produced the call, with the measured value next to the threshold.

| Call | Meaning |
|---|---|
| **GOOD** | The model describes the curve and the three parameters can be reported |
| **BAD** | Something about the curve makes the parameters untrustworthy. It does **not** mean nothing grew |
| **Not fitted** | The fit was not attempted or no usable window existed (too flat, too short, peak at the start) |

### Before fitting

A curve is rejected without fitting when the rise above the baseline is
less than **{pre_fit_snr_threshold}×** the baseline noise **and** the total
rise is below **{pre_fit_delta_od_override} OD**. Both conditions must hold:
a small but clean rise is still fitted.

### After fitting: the gates

| Gate | Threshold (current settings) | Plain meaning |
|---|---|---|
| Fit converged | – | The optimiser found an answer inside the allowed ranges |
| R² | ≥ {min_r_squared} | The curve is S-shaped |
| Error of A | ≤ {max_param_error_pct} % | The ceiling is pinned down |
| Error of μ | ≤ {max_param_error_pct} % | The growth rate is pinned down |
| Total rise (Δ OD) | ≥ {min_absolute_delta_od} OD | Enough growth to measure at all |
| A above noise | A ≥ 3 × baseline noise | The fitted ceiling is not a noise artefact |

The last two are enforced on every curve. The next four are enforced only
when the fit is **not excellent**, that is when R² < {excellent_r2_threshold}.
A fit that good proves the growth is real on its own, and these gates exist
to catch flat, noisy curves that happen to pass R²:

| Gate | Threshold | Plain meaning |
|---|---|---|
| Signal-to-noise | ≥ {min_snr} | The rise is clearly larger than the baseline scatter |
| Δ OD confidence | Δ OD − 2 × noise ≥ {min_delta_od_ci} | The rise is significant, not a lucky wobble |
| Monotone fraction | ≥ {min_monotone_fraction} of steps rising | The curve climbs instead of wandering (noise gives about 0.5) |
| Residual autocorrelation | ≤ {max_residual_autocorr} at a lag of 1 h | No systematic misfit (diauxic shift, decline, contamination) |

The monotone fraction is measured inside the fitted window only, so a long
plateau or decline after the peak does not count against a good curve. The
autocorrelation gate is skipped for incomplete curves.

### The optional machine-learning opinion

When a trained classifier is present and switched on, it gives a
probability **P(good)** for every fitted curve, computed from 24 features
of the curve and the fit. P(good) ≥ {postfit_good_threshold} agrees with GOOD,
≤ {postfit_bad_threshold} agrees with BAD, and anything in between is
**BORDERLINE**: the two opinions disagree and the curve deserves a look.
In this app the call of record is always the rule-based one, and the
classifier is shown as a second opinion only. It is off by default
(`use_ml` = {use_ml}).

### Changing your mind

Every threshold above is a slider in the sidebar, and every curve can be
given a manual verdict on the Curves page. A manual verdict is stored with
your note and a timestamp and is exported as `final_call` next to the
automatic one, so a reader of the results can always tell which is which.
