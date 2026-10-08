# One bounded ingestion-selection run

From the PRISM repository root:

```bash
git pull --ff-only origin main
python scripts/109_surma_ingestion.py launch --concurrency 4
```

This prepares all inputs, submits a GPU array using the existing GH200 environment,
and submits the CPU comparison with an `afterok` dependency. It uses the completed
`data/processed/surma_tail_pilot` archive (eight 15-day windows), not the short,
heavy-event-selected attribution experiment. It does not change the checkpoint.

| Recipe | Gauge averaging mesh | Gauge spread (cells) | Representativeness |
|---|---:|---:|---|
| `current_025_s6` | 0.25° | 6 | Existing budget; completed results reused |
| `local_025_s3` | 0.25° | 3 | Existing budget |
| `fine_010_s6` | 0.10° | 6 | At least the parent's point-gauge error |
| `fine_010_s3` | 0.10° | 3 | At least the parent's point-gauge error |

There are 16 GPU tasks: eight windows for the local recipe and eight for the two
fine recipes together. Three new recipes are sampled with the parent's member
count (normally 30). Background controls are regenerated to check agreement with
the cached baseline. The model, correct reporting-window half-hourly IMERG,
background date offset, sampler and likelihood weights are fixed. Prior and
satellite seeds match; gauge perturbations are not fully paired when the station
table changes. The fine recipes change both spatial aggregation and the matching
error treatment, and may change which gauges receive the unmerged BWDB multiplier.
They are ingestion recipes, not a pure one-factor test.

Historical and tuning windows choose one candidate. It must improve pooled fair
CRPS by at least 2% and heavy-rain (>=50 mm) CRPS by at least 5%. RMSE may increase
at most 1%, either network's CRPS at most 2%, dry-day absolute bias at most 0.5 mm,
and 50-mm Brier score at most 0.002. Heavy absolute bias must not worsen; field
coverage may decrease by at most three percentage points. At least 20 heavy and
30 dry station-days are required. At least five >=100 mm station-days must also
be available, with non-worsening extreme CRPS and absolute bias. The separate retest windows then accept or
reject that same candidate, requiring non-worsening overall and heavy CRPS and
the same remaining safeguards. A failed candidate does not trigger selection of
a runner-up on the retest data. These are practical thresholds, not significance
or independent calibration tests. All these windows have been inspected before.

Outputs in `data/processed/surma_ingestion_selection/summary/`:

- `comparison.md`: terminal-readable decision and scores.
- `scores.csv`: each method, window, split, network and rainfall bin.
- `selection.json`: selected recipe, error policy, per-window errors, reasons,
  provenance, and explicit production readiness status.

The recommendation can be the current method. The script does not promise a
universal best method, retrain the model, inject extreme rainfall, change station
dates, remove QC candidates, or launch a new production archive. Applying a
successful recipe to production requires preparation using its mesh and error
policy; the manifest is not a flag already understood by the production wrapper.

Run the same launch command to resume. Active tracked jobs are not duplicated;
finished task receipts are checked and only unfinished tasks are submitted.
Incomplete output folders are preserved under `.partial-TIMESTAMP` before their
task is retried. An obsolete `DependencyNeverSatisfied` summary from this workflow
is cancelled before resubmission. To regenerate only the final report:

```bash
python scripts/109_surma_ingestion.py summarize
cat data/processed/surma_ingestion_selection/summary/comparison.md
```

Use `--pilot PATH` / `--out-dir PATH` consistently for non-default locations.
Prepared plans are immutable and hash-checked. Keep the code used to prepare the
run until it finishes. Station inputs, prepared tables and outputs remain outside
Git. The GPU runs and Slurm submission require PRISM; CPU checks alone do not
establish runtime GPU performance or scientific improvement.
