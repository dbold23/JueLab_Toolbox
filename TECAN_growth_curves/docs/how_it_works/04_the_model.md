## The model: modified Gompertz

![Gompertz parameters](images/gompertz_parameters.svg)

Every fitted curve is described by three numbers, using the modified
Gompertz equation of Zwietering et al. (1990):

$$y(t) = A \exp\!\left(-\exp\!\left(\frac{{\mu e}}{{A}}(\lambda - t) + 1\right)\right)$$

| Symbol | Name | Unit | What it means on the plot |
|---|---|---|---|
| **A** | Maximum OD (carrying capacity) | OD | The ceiling the curve levels off at |
| **μ** | Maximum growth rate | OD / h | The slope of the steepest tangent, the fastest the OD ever rises |
| **λ** | Lag time | h | Where that steepest tangent crosses the baseline: how long the culture waited before taking off |

The constant *e* (2.718) is there so that μ is exactly the slope of the
tangent at the inflection point rather than a scaled version of it. The
inflection point is where the curve rises fastest; it sits at OD = A/e,
about 37 % of the way up.

### Doubling time

Because μ is an absolute rate in OD per hour, the per-cell (specific) rate
at the inflection point is μ divided by the OD there, μe/A. The fastest
**doubling time** of the culture is therefore

$$t_d = \frac{{\ln 2 \cdot A}}{{\mu e}}$$

reported in hours. It is a minimum: at any other moment the culture doubles
more slowly. Two strains with the same μ but different A do **not** have the
same doubling time; the one with the smaller ceiling doubles faster per
cell.

### What the model does not describe

- A decline after the plateau (the fitting window is cut before it).
- A second growth phase on a second nutrient (diauxic growth).
- A culture that is still rising at the end of the run: the fit is done,
  but *A* is then an extrapolation and is flagged as such.

For these the parameters are either meaningless or uncertain, which is what
the quality gates in the following sections are for.
