## Comparing treatments

The Compare page puts the fitted parameters of every **GOOD** condition side
by side, grouped by what was in the well:

| Treatment type | What it is | Role |
|---|---|---|
| **LB** | Rich broth only | Positive control: the strain's best growth |
| **H2O** | Water only | Negative control: should not grow |
| **Pesticide** | Pesticide in water | Can the strain grow on the pesticide alone? |
| **Pesticide + LB** | Pesticide added to broth | Does the pesticide slow growth on rich medium? |

### Reading the ratio strip

For each strain the strip shows μ, A and λ in a pesticide condition
**relative to the same strain in LB**. A ratio of 1 means no effect; μ at
0.6 means the strain grows 40 % slower with the pesticide present; λ at 2
means the lag doubled. Because each strain is compared with itself, the
strip removes differences between strains that have nothing to do with the
pesticide.

### Why BAD curves are left out

A BAD curve has parameters the model could not pin down. Averaging them in
would add noise at best and bias at worst (a flat curve fitted anyway tends
to give a tiny μ and a huge error). Leaving them out is the honest choice,
and the count of conditions behind every bar is printed so you can see how
much data each number rests on.

### The statistics, and their limits

With three replicates per condition the plate gives **n = 3** per group.
The Compare page runs a one-way ANOVA across treatment types (and a
Kruskal-Wallis test, which does not assume normal spread) and reports the
p-value. Treat that p-value as a screening number: with n = 3 the test has
low power, a single odd replicate moves it a lot, and it does not correct
for the several parameters and strains you are looking at simultaneously.
A difference worth reporting should be visible in the overlay plot and hold
up across plates.
