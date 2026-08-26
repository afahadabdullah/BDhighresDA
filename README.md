# BRISHTI-05

**Bangladesh Rainfall Integration of Satellite, Hydrometeorological, and
Terrestrial Information at 0.05°.**

BRISHTI-05 is a 30-member, daily precipitation-analysis workflow for
Bangladesh on a 0.05° (about 5 km) grid. A conditional rectified-flow prior
learns fine-scale rainfall structure from historical CPC, ERA5, and CHIRPS.
At analysis time, the prior is guided jointly by BMD rain gauges and GPM IMERG
V07B satellite accumulations. The repository name, `BDhighresDA`, is retained
for continuity; the current public analysis product is BRISHTI-05.

The selected production configuration is the historical machine label
`v2_simul_s04_ig010`. It uses simultaneous BMD and 0.4° IMERG guidance and a
30-member ensemble. See the full, versioned description in
[the methodology](docs/METHODOLOGY.md).

## What each data source does

| Source | Role in BRISHTI-05 | Not a claim of |
|---|---|---|
| CPC 0.5° | Coarse conditioning and residual baseline for the learned prior | Independent verification |
| ERA5 | Atmospheric conditioning variables | Rainfall truth |
| CHIRPS 0.05° | Historical training target family | Verification of the generated analysis |
| BMD gauges | Analysis-time point observations | A spatially complete rainfall field |
| GPM IMERG V07B | Analysis-time 0.4° footprint observations | Independent verification when assimilated |

Consequently, held-out BMD station folds are the primary performance evidence.
All-station BMD scores measure assimilation fit. Comparisons with CPC, CHIRPS,
or IMERG describe agreement with those products; they are not independent-truth
scores. A native 0.05° grid does not by itself establish resolved 0.05°
physical skill.

## Production contract

- **Grid:** Bangladesh analysis domain, 128 × 128 cells at 0.05°.
- **Ensemble:** 30 stochastic members per day.
- **Satellite operator:** exact 8 × 8 fine-cell averages, giving 0.4° IMERG
  footprints (`S04`), with correlated-footprint error treatment.
- **Gauge treatment:** each member receives perturbed BMD observations; the
  established production gauge weight is `1.0`.
- **Time convention:** for public BMD label day `D`, BMD and IMERG are the BMD
  03 UTC end-date `D`; CPC/ERA5/CHIRPS background inputs are record `D-1`.
  This convention must be preserved in all comparisons.
- **Current archive:** May–September 2021–2023 and May–June 2024, with five
  rotated spatial BMD holdout folds plus an all-station gridded production
  archive.

The machine-learning and DA design, error assumptions, evaluation hierarchy,
and limitations are documented in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).
The reproducible historical production run is documented in
[docs/EXPERIMENT_v2_confirmatory_2021_2024.md](docs/EXPERIMENT_v2_confirmatory_2021_2024.md).

## Run the established production workflow

On the target HPC system, after selecting the supported Python environment:

```bash
git pull --ff-only origin main
bash slurm/submit_v2_confirmatory_2021_2024.sh
```

This produces the rotated BMD-fold evidence and the all-station archive under
`data/processed/v2_confirmatory_2021_2024/`. It is a substantial HPC job;
consult the experiment document before changing dates, folds, checkpoints, or
the BMD/IMERG time alignment.

## Evaluate the gridded archive

```bash
bash slurm/submit_v2_gridded_evaluation.sh
bash slurm/submit_brishti05_may_aug2023_eval.sh
```

The second command produces same-station, date-aligned diagnostics for May,
July, August, and May–August 2023 using original CPC 0.5° and native IMERG
0.1° as descriptive references. Maps are clipped to the Bangladesh boundary.
Read [the archive evaluation guide](docs/EVALUATION_v2_gridded_archive.md) and
[the native-reference evaluation](docs/EVALUATION_cpcv2_june2023_bangladesh.md)
before interpreting those comparisons.

## Repository layout

```text
configs/train_h100_cpc_v2.yaml   current CPCv2 / BRISHTI-05 prior configuration
docs/METHODOLOGY.md              product definition, date contract, and evidence hierarchy
docs/EXPERIMENT_v2_confirmatory_2021_2024.md
                                 reproducible historical production protocol
scripts/28_simultaneous_method_sweep.py
                                 controlled DA-arm and gauge-authority experiments
scripts/                          data preparation, training, DA, and evaluation tools
slurm/                            HPC submission wrappers
src/bdhires/                      grids, transforms, models, observations, DA, and metrics
```

## Development validation

The repository also contains observing-system simulation experiments (OSSEs)
and V7 development scripts. They are useful engineering diagnostics but do not
supersede the BRISHTI-05 production protocol or its held-out BMD evidence. In
particular, a perfect-observation OSSE gives an upper-bound test of the
assimilation machinery, not a real-world verification result.

## License

MIT for code. CHIRPS, ERA5, CPC, and GPM IMERG have their own data terms. BMD
station data are not redistributable without permission.
