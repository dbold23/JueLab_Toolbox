## Fit quality: does the model actually describe the curve?

Three diagnostics are computed for every fit and shown on the Curves page.

### R²: how much of the variation is explained

$$R^2 = 1 - \frac{{\sum_i (y_i - \hat y_i)^2}}{{\sum_i (y_i - \bar y)^2}}$$

R² = 1 means the curve passes through every point; R² = 0 means it does
no better than a flat line at the mean. For a growth curve with a clear
rise and plateau, a correct Gompertz fit typically gives R² above 0.98.
A value below **{min_r_squared}** means the shape is not the S-shape the
model assumes.

R² has a known blind spot: a flat, noisy curve and a real but small rise
can both give a low R², and a big rise with a wrong shape can still give a
high one. That is why R² is one gate among several rather than the only
one.

### RMSE: how far the points sit from the curve

$$\mathrm{{RMSE}} = \sqrt{{\frac{{1}}{{n}}\sum_i (y_i - \hat y_i)^2}}$$

in OD units. Compare it with the reader's noise (a few thousandths of an OD
for a well-mixed culture) and with the spread between replicates.

### Residuals: what is left after the model

The residual plot shows *data minus model* at each time point. For a fit
you can trust it looks like noise: points scattered above and below zero
with no pattern. A **wave** in the residuals (long runs above zero, then
long runs below) means the model is missing a feature of the curve: a
second growth phase, a decline, or a bump from condensation on the lid.

The software measures that pattern as the **autocorrelation of the residuals
at a lag of about one hour**: how similar each residual is to the one
recorded an hour earlier. Noise gives a value near 0; a wave gives a value
near 1. Above **{max_residual_autocorr}** the fit is marked as systematically
wrong. The lag is a span of time rather than one sample so that the
statistic does not depend on how often the reader sampled.
