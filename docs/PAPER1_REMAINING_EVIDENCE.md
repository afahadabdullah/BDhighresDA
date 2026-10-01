# Nine remaining evidence gaps after the first full-archive rerun

The returned `paper1_updated_evidence/` folder confirms a completed five-product
comparison on 29,401 original Bangladesh withheld station-days over 489 dates.
It contains separate results for 2021–2024, intensity and temporal product
comparisons, country CRPS intervals, calibration, retained-gauge IDW, network
geometry and the validation history. It has no generated gridded output directory.
The following gaps remain; a present input is not a completed result.

## Immediate saved-data recovery on PRISM

After pulling the updated scripts, run from the repository root:

```bash
mkdir -p logs
PAPER1_EVIDENCE_OUT=output/paper1_updated_evidence_v2 \
PAPER1_FULL_GRIDDED=1 \
sbatch slurm/cpcv2_updated_paper_evidence.sbatch
```

This keeps the first result folder intact, reuses all four saved periods and
adds the following evidence. It does not generate rainfall or submit GPU jobs.

| Item | What the updated code does | What still needs original evidence |
|---|---|---|
| 3: paired IDW gains | Script 92 exports `analysis_mean_mm` and `background_mean_mm` in `interpolation_station_days.csv`, then generates `paired_idw_intervals.csv`, `idw_intensity_scores.csv`, `tab_idw_paired.tex`, `tab_idw_intensity.tex` | Original withheld ensembles and retained raw gauge CSVs are on PRISM; old IDW exports cannot reconstruct daily model means |
| 4: sub-0.4-degree skill | The batch launcher now enables the full saved-grid evaluator by default. Script 96 separately records output filenames/hashes and input availability | Saved production stores plus original withheld arrays; copy `evaluation/superob-final/gridded/` back after the run |
| 6: checkpoint epoch | Script 96 writes `checkpoint_metadata.json`, reading the zero-based epoch and completed epoch from the hash-verified checkpoint | The actual `best.pt`; a plotted CRPS minimum alone cannot establish checkpoint epoch |
| 8: statistics provenance | Script 87 records the exact bytes' SHA-256, actual path and transform in `stats_provenance` for future preparation | The old manifests remain unrecorded. Recover original execution logs/archived hashes; do not rerun preparation into the old archive or attach today's hash as historical proof |

IDW comparison uses the same finite independent station-days for the model and
IDW, with all May 2022 dates excluded. Paired RMSE and MAE gains use whole-day
3- and 7-day blocks, resampling all stations on a day together within date-gap
segments. Nonlinear pooled RMSE is recomputed in every draw; 10,000 resamples
are used by default. Positive gains favour the model, including when actual
results favour IDW in some bins. Intensity bins are based on gauge rainfall;
the bin tables report point estimates and counts, not simultaneous intervals.
IDW remains deterministic; no invented IDW ensemble or temporal probabilistic
scores are introduced.

Checkpoint metadata uses restricted `torch.load(..., weights_only=True)` in the
existing Torch environment; it never instantiates a model or falls back to
unrestricted loading. `PAPER1_CHECKPOINT` can remap the actual file on PRISM.
Missing/unsupported metadata remains unresolved, not inferred from training
history. Measured training/background/analysis cost still needs the original
stage timings or an explicit benchmark, using the `--compute` schema in
[PAPER1_COMPLETION.md](PAPER1_COMPLETION.md). `sacct` allocation elapsed time
by itself cannot separate background and analysis cost for a multi-arm job.

## Other original records and data

1. **Native IMERG.** Supply the actual prepared V07B native 0.1-degree files,
   half-hourly-derived 24-hour windows ending at 03 UTC. Put one path per line
   in a list and set `PAPER1_NATIVE_IMERG_LIST`. No files are acquired by the
   post-processing scripts. The native comparison retains a separate finite
   intersection so it cannot change the primary five-product scores.
2. **Provider overlap.** Supply historical product/version station inventories
   with dated membership and documented local ID crosswalks. The schema and
   `PAPER1_UPSTREAM_INVENTORIES` are in
   [PAPER1_UPDATED_EVIDENCE.md](PAPER1_UPDATED_EVIDENCE.md). Proximity candidates
   and unlisted gauges remain unknown; these do not establish non-use.
5. **May 2022 selection.** Discover actual saved arrays at their real path:

   ```bash
   python scripts/97_audit_remaining_paper_evidence.py \
     --evidence-dir output/paper1_updated_evidence \
     --discover-selection-root data/processed/v2_dense_gauge_sweep/profiles \
     --out-dir output/paper1_remaining_evidence
   ```

   If at least two profiles actually contain `dense_s6_bwdb_r4` on May 2022
   dates, this writes `selection_candidates.json`. Set
   `PAPER1_SELECTION=output/paper1_remaining_evidence/selection_candidates.json`
   for the batch run. Script 92 still checks matching dates, original gauges,
   statistics, checkpoint, seed and sampler settings before scoring. Discovery
   alone is not a validated ranking. Profiles using a different arm are not
   substituted; a five-day sweep must be reported as five days, not a month.
7. **BMD and sparse holdouts.** First inventory all reference folds:

   ```bash
   python scripts/90_evaluate_cpcv2_paper1.py --profile bmd-reference --audit-only
   ```

   Only after all original folds are present, run the same command without
   `--audit-only`. The existing BMD reference has different guidance, network
   and selection exclusions; report it separately. Sparse-network experiments
   require actual original arrays and their own frozen split/settings contract.
   No sparse results are manufactured from the dense-network summary.
9. **Remaining calendar dates.** Script 97 writes `test_extension_plan.json`:
   1,306 dates outside the current 520-day archive contract, including
   July–September 2024 (92 days), all 2025 (365 days), and the dry-season gaps.
   This is a plan, not a statement that every remote checkout lacks those dates.
   Check remote inputs/outputs first, freeze the extension split and configuration,
   then prepare and sample missing dates into a separate archive. Keep the same
   selected model and DA profile, record statistics provenance, and generate
   original withheld-gauge arrays as well as production grids. Score the combined
   entire test archive, preserving the May 2022 exclusion. Do not replace the
   original frozen contract silently or retune on the new test dates.

## Audit returned exports without changing them

```bash
python scripts/97_audit_remaining_paper_evidence.py \
  --evidence-dir paper1_updated_evidence \
  --out-dir output/paper1_remaining_evidence
```

This writes a nine-item JSON/Markdown checklist and the extension plan in a
separate folder. It checks generated result hashes before calling them available;
input-store availability never satisfies the gridded-output requirement. On PRISM,
point `--evidence-dir` at the actual output folder. To recover just IDW comparisons
from a *new* daily export, use `--idw-samples PATH.csv`; old exports lacking model
means are rejected. Its original `completion_manifest.json` must accompany it
and identify the same country boundary and unchanged CSV hash.
`--checkpoint PATH.pt` reads just the evaluated identity/epoch.
Do not overwrite the original completed scoring manifest with a local raw-input
inventory run: copied summaries and locally absent raw files describe different
availability contexts.
