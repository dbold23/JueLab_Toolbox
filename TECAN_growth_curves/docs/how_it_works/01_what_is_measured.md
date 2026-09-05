## What the plate reader measures

![Pipeline](images/pipeline.svg)

The TECAN Spark shines light at 600 nm through each well of a 96-well plate and
reports how much is absorbed. Bacteria scatter that light, so the reading, the
**optical density at 600 nm (OD600)**, rises as the culture grows. The reader
takes one reading per well every cycle (typically every 15 minutes) for the
whole run, which gives a time series of about 100 to 400 points per well.

Three things matter for how the numbers are read later:

- **OD600 is not a cell count.** It is proportional to biomass only over
  roughly 0.05 to 1.0 OD and flattens above that. The analysis compares
  curves against each other, so a consistent reader setting matters more
  than the absolute value.
- **The medium absorbs light too.** Rich broth (LB) alone reads 0.03 to 0.10 OD,
  and a pesticide suspension can be cloudy. That is why every media type
  needs its own **blank wells**: medium with no cells. Their reading is
  subtracted from every culture well of that media (next section).
- **One well is one measurement.** Three wells of the same strain in the same
  media (**triplicates**) are averaged, and the spread between them tells you
  how repeatable the result is. A curve built from one well cannot show that.

### What the software needs from you

| Input | Where it comes from | What it is used for |
|---|---|---|
| The SparkControl export (`.xlsx`) or a `Values` CSV | The reader's software | Time, temperature and the OD600 of every well |
| A plate **key** (`Cell, Media, Strain` per well) | You, or the `KEY` sheet of the export | Which wells are replicates, which are blanks, what each well contains |

Wells whose strain is `Blank` (any capitalisation) are blanks. Wells marked
`NA`, `empty` or left out of the key are ignored. Two wells with the same
media and strain are treated as replicates of the same condition.
