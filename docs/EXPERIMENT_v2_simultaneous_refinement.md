# CPC-v2 simultaneous DA refinement

## Question

Can the corrected CPC-v2 simultaneous BMD + IMERG S04 analysis improve beyond
the current 2022-05-01–10 result without repeating configurations already
screened in the v1 ingestion and v2 gauges-only tournaments?

The completed matched controls are loaded from
`data/processed/v2_ingestion_triplet/ing2022_s04_g010_sqrtfix`:

- best v2 gauges-only: spread 6, gamma 0.01, temperature 1.0;
- best earlier IMERG resolution: S04 (0.4 degree), stride 1;
- current simultaneous: those two likelihood components together;
- matched v2 background.

The control folds are not rerun. The summarizer requires the new run to have
the same dates, checkpoint, station IDs, fold membership, ensemble size,
background offset and seed, then checks that its repeated cheap background is
numerically identical.

## New one-factor arms

| Axis | Arms | Reason |
|:--|:--|:--|
| IMERG likelihood weight | 0.50, 0.75 | Test whether S04 is slightly too authoritative. Observation perturbations retain the physical error variance; the weight changes only the posterior likelihood. |
| IMERG gamma | 0.003, 0.01 | Soften the satellite component early in the flow while keeping its final-time error model fixed. |
| Gauge weight | 1.25 | Test whether slightly more gauge authority improves the wet-event trade-off. |
| Robust likelihood | Huber delta 3 | Limit leverage from an inconsistent gauge or retrieval footprint. |
| Complementary coverage | IMERG footprint centres at least 50 or 100 km from assimilated gauges | Let gauges control observed neighbourhoods and use IMERG to fill gauge gaps. The mask is fold-specific and uses assimilated gauges only. |
| ODE discretisation | 25, 50, 100 Heun steps with zero correctors | Isolate numerical integration resolution from Langevin updates. |
| Operational compute | 100 Heun steps with the usual two correctors/level | Test whether roughly doubling the actual DA compute improves skill. This is a performance arm, not a clean ODE-convergence arm. |

S04 factor 8, stride 1, error-correlation length 0.75 footprint cells, gauge
spread 6, gauge gamma 0.01, temperature 1.0, checkpoint, dates, folds and seeds
remain fixed unless the arm explicitly changes one item above.

## How many steps does current DA use?

The operational guided sampler uses **50 Heun ODE steps**. Heun evaluates the
guided velocity twice per step except at the terminal step, giving 99
integration guidance evaluations. It also uses two Langevin correctors at each
of the 49 interior levels, adding 98 evaluations: **197 guided model/likelihood
evaluations per member trajectory**.

Changing 50 to 100 while leaving two correctors per level would nearly double
both the ODE resolution and the number of DA corrections. That cannot identify
which mechanism mattered. The `nc0_n025`, `nc0_n050`, and `nc0_n100` arms turn
correctors off and therefore provide the clean step-convergence comparison.
The 50-step no-corrector arm also compares directly with the operational
50-step/two-corrector arm to measure whether correctors earn their cost.
The additional `n100` arm retains the two correctors per level, costing 199
integration plus 198 corrector evaluations (397 total), and compares directly
with the 197-evaluation operational control.

## Submission

```bash
git pull --ff-only
bash slurm/submit_v2_simultaneous_refinement.sh
```

Optional Slurm flags can follow the wrapper, for example:

```bash
bash slurm/submit_v2_simultaneous_refinement.sh --account=g0609
```

The wrapper reuses completed candidate folds on resubmission. A lone NPZ or
JSON is treated as a partial fold and fails rather than silently mixing output.

## Outputs to compare

New fold data are written under:

`data/processed/v2_simultaneous_refinement/ing2022_s04`

The main decision artifacts are:

- `refinement_selection.md` — ranked metrics, paired intervals, ODE test and promotion decision;
- `refinement_selection.json` — all metrics, comparisons and sampler costs;
- `refinement_selection.png` — pooled six-panel tournament plot;
- `fold_plots/fold0_diagnostics.png` through `fold4_diagnostics.png` — spatial and score checks for every fold.

The ten-day tournament is a screen, not a publishable independent test. Promote
at most two arms, freeze the rule, and evaluate those on a longer period before
claiming a gain.
