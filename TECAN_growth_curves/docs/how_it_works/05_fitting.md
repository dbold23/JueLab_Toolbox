## Fitting: how the three numbers are found

The model is fitted to the raw points inside the window by **non-linear
least squares** (`scipy.optimize.curve_fit`, trust-region reflective
algorithm): the algorithm adjusts A, μ and λ until the sum of squared
differences between the data and the curve is as small as it can make it.

### Starting values

Non-linear fits need a starting guess, taken from the data itself:

| Parameter | Starting guess | Allowed range |
|---|---|---|
| A | The highest OD in the window (at least 0.02) | 0.01 to 3× the guess |
| μ | The steepest rise between consecutive points | 0.001 to 10× the guess + 1 |
| λ | The time of that steepest rise | 0 to the end of the window |

The ranges keep the optimiser away from physically impossible answers. A
fit that ends **on** a range boundary has not really converged, and its
uncertainty is unknown; the software treats it as failed rather than
reporting a spuriously precise number.

### Standard errors

The optimiser also returns the covariance matrix of the parameters. Its
diagonal gives the **standard error (SE)** of each parameter:

$$\mathrm{{SE}}_i = \sqrt{{\,\mathrm{{cov}}_{{ii}}}}, \qquad \text{{relative error}} = \frac{{\mathrm{{SE}}_i}}{{\hat\theta_i}}$$

The relative error of A and μ is one of the quality gates. Note what a
standard error is: the precision with which **this** curve pins down the
parameter, assuming the model is right. It is not a confidence interval
across replicates, and it says nothing about whether the model is the
right one; that is what the next section checks.

### When a fit is refused

The fit is not attempted, and the condition is reported as not fitted,
when the window has fewer than four points, contains non-finite values,
has time going backwards, or shows no measurable growth. Any of these
would otherwise produce a number that looks real and is not.
