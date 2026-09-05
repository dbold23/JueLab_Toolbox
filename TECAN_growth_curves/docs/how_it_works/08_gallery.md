## Gallery: what the calls look like

![Gallery](images/gallery.png)

Six curves, each annotated with its call and the gate that decided it.
The first three are real conditions from the example plate (Group 3); the
last three are generated curves of the kinds the pipeline is validated
against with its synthetic data set.

1. **Good**: clean lag, steep rise, flat plateau. R² above 0.99, both
   parameter errors under 5 %. The dotted line is where the window was
   cut: the model only ever sees the primary growth.
2. **Flat: nothing grew**: the reading wanders by a few hundredths of an OD
   and never rises. Rejected before fitting, or BAD on the total-rise gate.
   This is the expected result for a well with no viable cells, and BAD
   here means "nothing to fit", not "something went wrong".
3. **Wrong shape**: a real condition from the same plate that grows in two
   steps and is still creeping upward when the run ends. The single sigmoid
   runs through the middle of both phases, the residuals form a wave, and
   the fit fails on R² (a curve like this that keeps R² above the line is
   caught by the residual-pattern gate instead). Its parameters would
   describe neither phase.
4. **Incomplete**: still climbing at the end of the run. Fitted whole; the
   ceiling *A* is an extrapolation and is flagged.
5. **Early peak, then decline**: the culture peaks early and falls. The
   first-peak rule cuts before the decline and the fit is GOOD; the
   parameters describe the primary growth only.
6. **Diauxic (two phases)**: growth on one nutrient, a pause, then growth on
   another. **First peak** fits the first phase only; **Global maximum**
   fits the envelope, misses both plateaus, and fails on R² or on the
   residual pattern. Pick the rule to match the question you are asking.

A note on water controls: on this project's plates the H2O wells often
rise by 0.1 to 0.2 OD and fit well, because the inoculum carries some
broth with it. That is normal. A water well that climbs as high as the LB
control is not.
