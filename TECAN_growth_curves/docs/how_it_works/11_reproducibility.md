## Reproducibility and citation

Every export includes `run_manifest.json`, which records what produced the
numbers:

| Field | What it holds |
|---|---|
| `inputs` | File names and SHA-256 checksums of the export and the key, and which sheets were used |
| `instrument` | Start and end time, temperature, and the metadata rows of the export |
| `settings` | Every threshold and rule exactly as used for this run |
| `software` | Repository commit, package versions (numpy, scipy, pandas) |
| `overrides` | Every manual window, exclusion or verdict, with note and timestamp |
| `ml` | Whether a classifier was consulted and the hashes of its model files |

Load a run folder back into the app and it restores the settings and the
overrides, so a colleague can see exactly what you did and continue from
it.

The folder also holds `config.yaml`, the thresholds of the run written in
the shape the batch scripts read, and the manifest's `reproduce.command`
is the one line that re-runs the analysis from the exported curves with
the command-line pipeline:

```
python <repo>/scripts/01_growth_curve_analysis.py processed --config config.yaml -o cli_results --no-plots
```

The test suite checks that this reproduces the app's numbers.

### How to cite

The model is the modified Gompertz equation of

> Zwietering, M. H., Jongenburger, I., Rombouts, F. M., & van 't Riet, K.
> (1990). Modeling of the bacterial growth curve. *Applied and Environmental
> Microbiology*, 56(6), 1875–1881.

The analysis software is this repository. `methods.md` in every export is a
paragraph, filled in with the thresholds actually used, that can be pasted
into a report.
