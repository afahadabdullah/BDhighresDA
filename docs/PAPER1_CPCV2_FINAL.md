# Final CPCv2 model and the first publication

Decision recorded 2026-10-01: the final learned model is **SURMA-Flow v1.0**,
the CPCv2 conditional rectified-flow **U-Net**, using
`configs/train_h100_cpc_v2.yaml` and `runs/prior_h100_cpc_v2/best.pt`.
GraphFlow and future GNN backbones are development experiments. They are not
part of this paper's selected model. This records the model choice; it does
not assert that evaluation or public release has finished.

## Assimilation winner and its provenance

The latest recorded production winner for the dense BMD+BWDB network is
**0.25° gauge super-observations + `dense_s6_bwdb_r4`**, as selected explicitly
by `V2_BMD_BWDB_SUPEROB_WINNER` in script 28 and the superob Slurm launchers.
It uses joint gauge/IMERG guidance, Huber delta 3, six-cell gauge-gradient
spreading, temperature 1, gauge gamma 0.01, IMERG gamma 0.001, exact 0.4°
IMERG footprints, footprint correlation length 0.75 cells, and 30 members.
The code applies a fourfold **variance** multiplier to assimilated records
whose ids start with `BWDB_`; this doubles their error SD. It does **not**
apply that source multiplier to merged `SOB_` clusters. Those use the common
superob representativeness budget. Preserve this distinction in the methods.
Gauge super-observations are passed to the existing point operator at the
cluster centroid; 0.25° here is the clustering mesh, not a new block operator.

The older BMD-only selected analysis is `v2_simul_s04_ig010`, with gamma 0.01
for both streams and five spatial station folds. The intermediate unmerged
BMD+BWDB winner was `v2_simul_s04_huber3`. These are separate observation
contracts using the same CPCv2 prior. The archived selection-score files are
needed to quote numerical winning margins; none are present locally.

The machine-readable evaluation contract is `configs/paper1_cpcv2_final.json`.
The default profile evaluates the final superob product. A separate
`bmd-reference` profile evaluates the original five-fold evidence. Scores from
the two layouts must not be pooled or interpreted as a matched method contest.

## Two publications

1. **Model paper:** model and observation operators; held-out test-year
   verification; assimilation ablations; ensemble calibration; spatial
   structure; heavy rain and computational cost. Use the saved test-year
   products. The configured test split is 2021–2025; the current archived
   protocol covers May–September 2021–2023 and May–June 2024 (520 days).
   State the actual seasonal coverage; 2025 and complete calendar years are
   not established by these archives.
2. **Historical-product paper:** planned 2000–2025 BRISHTI-05 archive,
   observing-network availability through time, long-record evaluation,
   climatology, extremes, uncertainty, data access and release. This is later
   work. Years 2000–2018 overlap prior training and belong to the product
   assessment rather than temporally independent model verification.

The first paper must distinguish stations withheld from the likelihood from
possible station overlap in the CHIRPS/CPC product families. CPC conditions
the model, CHIRPS supplies training targets, and IMERG enters assimilation;
none supplies independent gridded truth. All-station production gauges measure
assimilation fit. The final profile has one constrained 20% holdout per period
with a retained neighbour within 15 km and co-located gauges grouped at 5 km;
it is not exhaustive five-fold verification.

## Evaluate already generated data

Run from the repository root using the supported HPC Python environment:

```bash
# Inventory first; no scientific scores are created by an audit.
python scripts/90_evaluate_cpcv2_paper1.py --audit-only

# Final model: held-out scores, followed by existing maps and structure suite.
python scripts/90_evaluate_cpcv2_paper1.py --with-gridded

# Original BMD-only five-fold reference, kept in a separate output directory.
python scripts/90_evaluate_cpcv2_paper1.py --profile bmd-reference --with-gridded

# Equivalent CPU-only HPC submission for the final profile:
mkdir -p logs
sbatch slurm/cpcv2_paper1_evaluation.sbatch
```

`--root PATH` selects an archive in a different checkout. `--periods NAME ...`
evaluates an explicitly declared subset; missing files within that subset
still fail. `--comparators KEY ...` includes additional methods only if saved
in every requested fold. Withheld evaluation needs only the NPZ/JSON folds
and station manifests. `--with-gridded` additionally needs completed Zarrs and
the environment for the established script 55. It does not regenerate fields.

Outputs under `output/paper1_cpcv2/{superob-final,bmd-reference}/` include:

- `input_audit.json`: required inputs and missing-file inventory.
- `paper1_evaluation.{json,md}`: verified archived contract, provenance and scores.
- `daily_withheld_scores.csv`: pooled, period, station, source and intensity scores.
- `paired_crps_intervals.csv`: paired day-block intervals with 3- and 7-day blocks.
- `temporal_withheld_scores.csv`: calendar-month and complete-season means and
  daily temporal variability, with at least 80% observed coverage.
- `gridded/`: the existing reference-agreement, subgrid, spectra and plotting
  suite when requested. Interpret its probabilistic temporal aggregates with
  the temporal-coherence limitation below.

Both methods in a comparison use the same finite station-days. Daily
verification excludes the profile's selection dates: May 1–10, 2022 for the
BMD reference and all of May 2022 for the superob archive. Monthly verification
excludes affected months; seasonal verification excludes affected seasons and
requires complete May–September input coverage. June-only or May–June data
are never labelled a complete monsoon season. The bootstrap resamples paired
day blocks separately within contiguous segments; seasonal and excluded-date
gaps are preserved. It estimates temporal uncertainty for the available fixed
holdout, not uncertainty over alternative station networks.

Daily stochastic members have not been validated as temporally coherent
trajectories. The new paper tables therefore assess deterministic monthly and
seasonal means, without claiming ensemble CRPS or coverage for accumulated
rainfall. Posterior daily spread and temporal variability are distinct.

## Manuscript and completed result exports

`manuscript/BDhighresDA_arxiv.tex` was rewritten on 2026-10-01 as the CPCv2
model-paper preprint. It follows the broad organization of Manshausen et al.
(arXiv:2406.16947v3), with the project's own methods and completed test-period
results. `manuscript/BDhighresDA_arxiv_pre_CPCv2.tex` preserves the preceding
August draft, which centred on older OSSE and May 2018 process experiments.
`manuscript/REVIEW_AND_NEXT_STEPS.md` remains a historical review.

The completed HPC exports are copied to `paper1_cpcv2/superob-final/` in the
project root. They score 29,615 station-days on 489 non-selection dates;
fair CRPS is 8.775 for background and 6.263 for the selected analysis.
The new manuscript reports intensity-dependent failures and undercoverage
alongside the pooled gain. Its vector figures are reproduced with
`scripts/91_build_cpcv2_paper1_figures.py`; preparation notes and remaining
provenance/author decisions are in `manuscript/PAPER1_PREPARATION_NOTES.md`.

First-paper title: **SURMA-Flow: conditional rectified-flow rainfall
downscaling and observation assimilation over Bangladesh**.

The rewritten draft describes the final sampler and replaces old May 2018
headline numbers with audited test-period scores. It includes daily, source,
intensity and calibration diagnostics, deterministic temporal means, spatial
structure and a selected test-period case. Older OSSE numbers are not reused
as CPCv2 evidence. A network map, new ablations and measured compute can be
added when supporting records are available.

Before release or extending the evidence: recover selection reports, complete
any reference folds needed for claimed comparisons, retain checkpoint weights
and their SHA-256 identity,
verify the superob error-budget transform against checkpoint
stats, and verify actual observation coverage. The current superob preparation
launcher defaults to `data/processed/stats.json`, whereas CPCv2 names
`stats_cpc_v2.json`; archived superob manifests do not record that stats path.
Resolve this provenance gap before release or reuse of the transformed-error
budget. Numerical results and confidence intervals are now available for the
final profile; the copied BMD-reference folder still contains only an audit.
Compare practical non-generative baselines using the same withheld stations,
and record gauge overlap in upstream products as a limitation where unknown.

For the later 2000–2025 production, also audit the IMERG start date and changing
BMD/BWDB availability. A fixed learned model can accommodate missing observation
streams, but early years may not have the same observing-system contract.

## Complete draft and pending evidence workflow

The full manuscript now includes eight explicit missing-evidence slots and
written verification methods. `scripts/92_complete_cpcv2_paper1.py` generates
calibration, threshold scores, original-gauge IDW comparisons, network maps,
selection tables, validation curves, measured compute tables and separate
audited holdout summaries when their actual inputs are supplied. Missing
inputs remain pending; no synthetic result enters the manuscript.

See [PAPER1_COMPLETION.md](PAPER1_COMPLETION.md) for input schemas, the
CPU-only Slurm launcher and source-package command. The copied result bundle
contains summary exports; member-level arrays and original station tables
remain on the original compute archive.

## Bangladesh-only revision

Verification and map layers now use the snapshotted Bangladesh ADM0 boundary.
BWDB_CL130 and 214 scored pairs are excluded according to the original
catalogues; the current recoverable sample is 29,401 station-days at 132
withheld site IDs. CRPS is 8.768 / 6.258 and RMSE 20.795 / 17.070 mm/day.
See [PAPER1_BANGLADESH.md](PAPER1_BANGLADESH.md) for source identity, exact
sufficient-statistic reconstruction, pending raw-array diagnostics and the
full country-only Slurm rerun. The earlier wider-domain results are preserved.
