# Improving the SURMA precipitation analysis

Keep the frozen CPCv2 checkpoint initially. Improve input quality, test small
assimilation changes, and demonstrate gains using gauges excluded from the
likelihood. A finer output grid and larger rainfall maxima do not establish a
better product. Architecture changes are justified only if these tests expose
persistent limitations in the learned rainfall distribution or spatial patterns.

## Ordered work list

1. **Review station quality.** Check suspicious BMD/BWDB observations against
   original sources and nearby gauges. Verify units, coordinates, duplicate
   station-days, missing tokens and date conventions. Remove only confirmed
   errors, with reasons; localized extremes can legitimately disagree with
   neighbors. The archived canonical tables already contain reader-level
   filtering, so the QC script cannot recover values discarded by those readers.
2. **Rebuild affected gauge inputs.** Recalculate super-observations and their
   error budgets after confirmed exclusions. Keep original gauges separately.
3. **Verify temporal support.** Preserve half-hourly IMERG reporting-window
   integration and background offset -1. BMD support is [D-1 00:00,D 00:00] UTC;
   BWDB is [D-1 03:00,D 03:00] UTC in the existing evaluated pipeline. Audit the
   underlying reporting conventions. Correct the calendar-D CHIRPS daily
   comparison before using it to judge reporting-day skill; exact 03:00 support
   cannot be recovered from calendar-day CHIRPS. Do not shift corrected BMD again.
4. **Check historical conditioning consistency.** Investigate CPC's large
   early/late-period rainfall change using original files and packing metadata.
   Record the two approved previous-day CPC substitutions. Do not silently
   change normalization or the checkpoint conditioning distribution.
5. **Freeze independent evaluation splits.** Withhold original stations before
   super-observation aggregation and error-budget estimation. Remove assimilated
   gauges in their 0.25-degree cells and within a spatial buffer. Withholding
   from SURMA does not establish independence from upstream IMERG/CPC/CHIRPS.
   Use separate tuning and assessment dates. A truly fresh benchmark is still
   needed for strong final claims: the current test years have been inspected.
6. **Run a small paired pilot.** Retain the checkpoint, observations, 30 members
   and initial/observation random seeds. Compare production settings, temperatures
   1.15 and 1.25, Euler without additional noise, and Euler with noise 0.15.
   The Euler control separates stochasticity from the solver change. Noise 0
   still includes configured Langevin correctors. Later, after QC, test gauge/
   IMERG error weighting or Huber robustness one aspect at a time.
7. **Judge several outcomes together.** Check common-sample withheld bias, MAE,
   RMSE, fair CRPS, field-interval coverage, heavy-rain probabilities, detection,
   misses and false alarms at 50/100 mm. Check monthly totals, storm footprints
   and spatial variability separately. More variance in square-root space can
   increase the mean in millimetres without improving rainfall reconstruction.
8. **Assess calibration.** If bias or insufficient spread persists, test
   lightweight calibration fitted only to development data. Account for gauge
   and representativeness error before interpreting field-interval coverage.
   Preserve spatial dependence; avoid arbitrary wet offsets or pixelwise spikes.
9. **Regenerate consistently.** Rerun affected quarters for station-only changes.
   Regenerate the full archive if assimilation settings change. Keep the current
   product as an immutable, versioned baseline.
10. **Validate and document release.** Check completeness, units, timestamps,
    climatology, variability, extremes and historical discontinuities. Supply
    mean rainfall, quantiles/probabilities and ensemble information. Evaluate
    member-wise extremes; annual member trajectories require checking temporal
    coherence because days are sampled independently. Seek independent
    hydrological validation. State effective resolution and uncertainty limits.

## Implemented next-step workflow

`scripts/106_surma_improvement.py` reads the existing production preparation
artifacts. It writes into a separate ignored directory:
`data/processed/surma_improvement_pilot`. It never edits source station files,
production outputs or the checkpoint. Station data and generated diagnostics
must remain outside Git.

The default configuration has eight 15-day windows (120 days): two historical
sensitivity cases, two tuning cases and four reserved assessment cases. Historical
cases overlap model training years; they do not measure unseen-year skill.
Cases sample wet and dry seasons and the 2007 CPC fallback, but are not a
comprehensive flood-event or climate validation. The original May 2022 selection
month is rejected. The same original station split is used in every window,
requiring at least 80% coverage per window. This favors long-running stations;
newer stations need additional later-period evaluation.

### 1. Print the experiment and run CPU QC

From the repository root, with the project's Python environment activated:

```bash
python scripts/106_surma_improvement.py plan
python scripts/106_surma_improvement.py qc
```

Review these local outputs:

- `qc/review_flags.csv`: daily and monthly wet disagreements against nearby BWDB.
- `qc/nearby_monthly_comparisons.csv`: monthly sums on paired valid days, at least
  80% coverage per neighbor and three usable BWDB neighbors within 50 km.
- `qc/station_coverage.csv`: dates, available observations and maximum rainfall.
- `qc/reviewed_exclusions.csv`: initially empty; fill only confirmed invalid
  intervals, using `station_id,start,end,reason` (dates inclusive).

These thresholds prioritize review; they are not a full QC system, and different
BMD/BWDB daily support matters. Confirmed date/unit/coordinate corrections must
be made in a new station-source version and re-prepared separately; this script
only masks explicitly reviewed intervals in its pilot copies.

### 2. Prepare after review

```bash
python scripts/106_surma_improvement.py prepare \
  --exclusions data/processed/surma_improvement_pilot/qc/reviewed_exclusions.csv
```

The script verifies selected production input checksums, builds the fixed
withheld split with a 20-km buffer, rebuilds super-observations and measured error
budgets using only retained assimilated stations, then freezes `pilot_plan.json`.
If no reviewed exclusions file is supplied, the plan explicitly records that QC
has not supplied exclusions. An empty file records zero reviewed exclusions; it
does not certify all data as valid.

The baseline is the current assimilation settings **on the pilot's QC/split
inputs**, rather than the previously assimilated all-station product. Buffering
measures performance across those gaps and may differ from operational
interpolation with the full network. Whole-quarter prepared IMERG files are
reused and the sampler selects the requested dates; no new downloads are needed.

### 3. Submit parallel GPU windows and a dependent CPU summary

```bash
bash slurm/submit_surma_improvement.sh
```

This requires the prepared plan. It submits one GPU array task per window,
defaults to two concurrent tasks, and queues the summary after successful
completion. To request four concurrent windows:

```bash
SURMA_IMPROVE_CONCURRENCY=4 bash slurm/submit_surma_improvement.sh
```

Optional environment variables: `SURMA_IMPROVE_OUT` (pilot directory) and
`PYTHON_BIN` (project Python). Create a new directory/plan when inputs, windows
or settings change. Completed tasks reuse checksum-verified outputs. Partial
outputs fail explicitly; inspect and move aside a failed task's partial output
before retrying that task.

For a direct local/GPU run and CPU summary:

```bash
python scripts/106_surma_improvement.py run --task 0
# Run all other tasks before summarizing, or omit --task to run serially.
python scripts/106_surma_improvement.py summarize
```

Read `summary/comparison.md`, `withheld_scores.csv`, and `paired_day_scores.csv`.
All variants use the same finite station-day sample. Scores are pooled separately
by historical/tune/test roles and BMD/BWDB networks. The summary reports descriptive sensitivity; it
does not supply uncertainty intervals, promote a winner, launch production, or
complete the climate/hydrology validation work. Few heavy events leave tail
skill unresolved. Pick a candidate using tuning cases, freeze it, then assess it
against the baseline on reserved cases.

## Small sampler correction included

Heun now applies the temperature drift at both evaluations. Temperature 1.0
(the frozen production setting) is unaffected. Older temperature >1 experiments
used the incomplete second evaluation and should not be mixed with this pilot.
The sampler/noise experiment is a hypothesis: broader tails need not improve
the rainfall mean, storm location or probabilistic reliability.

## Next round: diagnose heavy rainfall and gauge likelihood

The first 120-day pilot did not justify production changes: temperature/noise
did not improve ensemble-mean heavy-event detection, and temperature reduced
field coverage despite increasing spread. The following next steps reuse saved
results before another separate experiment. No architecture change is involved.

```bash
python scripts/107_surma_tail_diagnostics.py diagnose
```

Read `data/processed/surma_tail_pilot/diagnostics_before/`:

- `intensity_scores.csv`: common-sample withheld bias, CRPS, spread, interval
  width and below/above interval misses in dry, 1–10, 10–50, 50–100 and >=100 mm
  observed rainfall bins, separately by network and case role.
- `heavy_events.csv`: individual withheld events with the ensemble mean, 5/95%
  quantiles, maximum member and 50/100-mm probabilities. A maximum-member hit
  only shows that an event is possible in the ensemble, not that it is skillful.
- `aggregation_extremes.csv`: original assimilated heavy observations versus
  their actual super-observation or passthrough record. Withheld gauges are
  excluded from this aggregation comparison. Never compare a held-out gauge to
  an unrelated nearby super-observation as if it contained that gauge.
- `review_flags_for_pilot.csv`: only existing QC candidates affecting retained
  or withheld stations in these windows. Review original source records and
  nearby gauges before confirming an exclusion.

This postprocessing verifies the saved plan, data and completion hashes. It
permits old runtime code hashes to differ **for reading completed results only**,
recording those changes. New sampling still requires current code hashes. Do
not modify the old plan or its reviewed-exclusions file, which remain provenance
for the first experiment. After pulling the new code, use script 107 to inspect
the completed first experiment; script 106 deliberately rejects sampling with
changed code under its old plan.

Create a separate review file so the original pilot remains reproducible:

```bash
cp data/processed/surma_improvement_pilot/qc/reviewed_exclusions.csv \
   data/processed/surma_tail_pilot/reviewed_exclusions_round2.csv
```

Edit the copied CSV only for confirmed invalid observations. Leaving it empty
applies zero corrections and keeps the next experiment provisional. Source,
coordinate, unit or date corrections require preparing a new station-source
version; the next runner masks explicitly reviewed intervals only.

```bash
python scripts/107_surma_tail_diagnostics.py prepare \
  --exclusions data/processed/surma_tail_pilot/reviewed_exclusions_round2.csv

SURMA_IMPROVE_CONCURRENCY=4 bash slurm/submit_surma_tail.sh
```

The runner reuses exactly the first experiment's eligible original stations,
withheld IDs, spatial buffer, dates, checkpoint and random seeds. It rebuilds
super-observations and error budgets after new exclusions. It refuses to change
the held-out fold silently if exclusions leave inadequate held-out coverage.
Cases previously marked `test` become `retest` because their results are already
known; a fresh assessment remains necessary before claiming improvement.

The GPU group has a background and four analysis arms:

| Analysis | Change from current settings |
|---|---|
| `dense_s6_bwdb_r4` | Baseline on the same next-round QC inputs |
| `tail_gauge_huber5` | Gauge Huber threshold 3 -> 5; IMERG remains at 3 |
| `tail_gauge_w125` | Gauge likelihood weight 1 -> 1.25 |
| `tail_gauge_w075` | Gauge likelihood weight 1 -> 0.75 |

Temperature stays 1.0 and additional sampler noise stays 0. Observation
perturbations are shared across variants. A likelihood weight scales both gauge
R and early-time inflation in the cost; it does not claim that instrument errors
have changed. Do not combine changes in this first comparison.
Because Huber is nonlinear, a weight of 1.25 does not multiply every large
residual's robust cost by exactly 1.25.

The dependent CPU job writes `data/processed/surma_tail_pilot/summary/` with the
same detailed diagnostics. Compare within this round; if QC changes, a direct
comparison against first-round scores confounds QC and method changes. Require
improvements in heavy-rain probabilities and misses without worsening dry-day
false alarms, ordinary rainfall or uncertainty. The small observed extreme
sample does not establish a robust tail improvement by itself. No automatic
production rerun is submitted.

## Event-level attribution after the likelihood pilot

Run `scripts/108_surma_event_attribution.py review` to trace the downloaded
aggregation and heavy-event tables to the supplied corrected BMD/BWDB archives.
It checks downloaded prepared inputs and completed-result hashes, reports
existing QC overlaps, and lists both nearby raw gauges and actual assimilated
superobs. If the uncorrected BWDB workbook is available, it also checks that
compilation (accounting for the reviewed CL9 date correction). Archive agreement
is provenance corroboration, not independent verification of rainfall truth.
No station-day is automatically deleted or shifted.

The review writes `data/processed/surma_event_review/`. It selects candidates
with observed rainfall >=100 mm, matching source values, no existing event flag,
and at least two of the five nearest valid BWDB stations within 50 km reporting
>=50 mm. This support rule does not prove a point observation is a grid average;
BMD/BWDB still have different three-hour daily support. Nearby gauges excluded
by the withheld buffer can corroborate an event without constraining the model.

```bash
python scripts/108_surma_event_attribution.py review
python scripts/108_surma_event_attribution.py prepare
SURMA_IMPROVE_CONCURRENCY=4 bash slurm/submit_surma_events.sh
```

The default configuration has four short windows, nine days total: July 8-10
and July 12, 2019; July 1-3, 2021; August 1-2, 2024. The selected dates avoid
the pilot's currently flagged input station-days. Preparation refuses flagged
windows, changed review inputs, changed withheld observations, and overlapping
dates. Station identities, observations and amounts stay in ignored data/results
paths; the tracked configuration contains dates and parent-window labels only.

The four variants are background, production gauges alone, production S04 IMERG
alone, and unchanged `dense_s6_bwdb_r4`. The parent superobs, error budget,
checkpoint, retained/withheld split, physical observation perturbations and
date-based seeds remain matched. The parent error budget is deliberately frozen,
including any influence from flagged days elsewhere in the parent window, so
this experiment isolates stream removal rather than error-budget recalculation.
For the gauges-only arm, whole-gradient spreading equals the combined arm's
gauge-component spreading (6 cells); satellite spreading stays zero. Stream
gamma and BWDB error inflation remain identical to their combined-arm values.

The dependent summary writes `data/processed/surma_event_attribution/summary/`,
including `selected_event_attribution.csv`. The mean difference between combined
and gauges-only estimates the conditional effect of adding IMERG; combined minus
IMERG-only estimates the conditional effect of adding gauges. These effects need
not add linearly. The windows were selected after seeing results, so they diagnose
mechanisms and do not establish archive-wide skill or automatically select a
production method. A separate original-gauge versus superob experiment is still
needed to isolate aggregation itself.


## Consolidated ingestion comparison

Run `python scripts/109_surma_ingestion.py launch --concurrency 4` once on PRISM.
It prepares and submits three new ingestion recipes together, reuses the completed
baseline, and automatically writes a selection report after the array finishes.
The four total recipes compare 0.25/0.10-degree gauge averaging and six/three-cell
gauge influence, with conservative errors for finer cells. Development windows
choose one candidate; separate retest windows can reject it. The current method
is retained if the fixed scoring gates fail. No model retraining or production
rerun is performed. See [the workflow instructions](docs/surma_ingestion_selection.md).
