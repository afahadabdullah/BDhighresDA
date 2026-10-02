# Paper 1 evidence audit after the returned v2 evaluation

The latest local manuscript is `manuscript/BDhighresDA_arxiv.tex`. The verified
`paper1_updated_evidence_v2/` export covers **29,401 Bangladesh withheld
station-days, 132 gauges, and 489 independent dates over all four available
2021–2024 periods**. All of May 2022 is excluded from scored results.

## Already available: do not regenerate these just to fill the paper

- Daily, annual, intensity and temporal comparisons with CPC, CHIRPS and 0.4° IMERG.
- Paired IDW RMSE/MAE intervals and intensity gains in `additional/`.
- Sub-0.4° results in `evaluation/superob-final/gridded/`.
- Calibration, network geometry and training validation history.
- Hash-verified checkpoint identity: completed epoch 40, EMA weights.
- The configuration table already reports 57,493,729 parameters. This is the
  implemented architecture count, not a count extracted from checkpoint tensors.
  ERA5 is sourced at 0.25° and regridded to the 0.05° model grid.

The new collector retained 47 manifest-backed tables and metadata files locally.
The original gauge CSVs and ensemble NPZs are not in the copied export; they
must be read on PRISM. Local absence does not establish absence on PRISM.

## Still open in the manuscript

| Slot | Missing evidence | Recovery or new work |
|---|---|---|
| WG5 | Native 0.1° IMERG on the same withheld station-days | Supply prepared native files for every period to script 96. Check daily accumulation window/version and report common-sample attrition. |
| WG4 | Which withheld gauges enter CPC, CHIRPS or IMERG's gauge analysis | Dated provider inventories and station-ID crosswalks. All 396 product–gauge combinations remain unknown; proximity or nonmatches do not establish independence. |
| P5 | May 2022 profile-selection comparison | Recover original matched profile NPZ/JSON pairs; script 97/98 discovers candidates and script 92 validates and scores them. |
| P7 | Measured training, background and analysis cost | Executed logs plus Slurm records or a new measured benchmark. Scheduler allocation time alone does not establish stage timing. Epoch is already known. |
| P8 | BMD-only five-fold reference and sparse-network tests | Recover complete audited experiments or generate missing runs under frozen contracts. Existing distance strata do not replace thinning experiments. |
| KR | Probabilistic spatial interpolation baseline | Script 98 exports separate retained-only daily inputs and withheld verification ensembles. A specified, fitted and scored probabilistic baseline is still needed; deterministic IDW CRPS is not this result. |
| PR | Historical statistics used in super-observation preparation | Original executed preparation logs or archived content hashes. Current defaults and prospective manifests cannot prove historical identity. |
| FP | July–September 2024, 2025 and dry-season verification | Extend the contract and generate missing data after checking remote coverage. There are 1,306 dates outside the present contract: 92 wet-season dates in 2024, 153 in 2025, and 1,061 dry-season dates across 2021–2025. |
| META | Funding, compute acknowledgement, final author metadata and release/access statements | Author and allocation/award records, plus actual public release identifiers or accurate access statements. A DOI is not mandatory, but a local folder is not a public deposit. |

FP is a scope extension beyond the existing 489-date paper, not evidence that
some of those available dates were omitted. Claims about a full 2021–2025 or
all-season evaluation must wait for that extension. Native IMERG, the
probabilistic baseline, provenance and selection evidence are especially
important for interpreting the current model comparison.

## Claims that need correction before submission

These are wording/evidence issues, not numbers to invent:

1. Replace “never sees the verification gauges” with exclusion from the direct
   likelihood and super-observations. CPC/CHIRPS/IMERG can transmit upstream gauge
   information to SURMA-Flow. Network-level score similarity does not bound that
   effect or prove independent truth.
2. A roughly flat validation history does not establish test-score insensitivity
   to stopping epoch. Report epoch 40 and the validation history; a checkpoint
   sensitivity claim needs matched evaluation of additional checkpoints.
3. Lower CRPS than deterministic IDW establishes improvement over that point
   forecast. It does not establish superiority over probabilistic interpolation
   until KR is completed. Keep the under-dispersion caveat prominent.
4. Appendix F says only the last row needs new sampling. Missing P8 experiments
   can require new sampling too. Other gaps may need new post-processing or
   provider records.
5. The data-availability section calls `paper1_updated_evidence_v2` a repository
   folder, but it is currently untracked. Publish permitted derived outputs or
   state the actual access arrangement; do not publish restricted gauge data.

## Run the audit locally

```bash
python scripts/97_audit_remaining_paper_evidence.py
python scripts/98_collect_paper1_missing_information.py \
  --evidence-dir paper1_updated_evidence_v2 \
  --out-dir output/paper1_information_bundle
```

The audit prefers the completed v2 export, checks result hashes and tracks the
new KR and publication metadata gaps. Script 98 requires a **new** output
folder so stale results cannot survive a rerun. Use a different name if the
bundle already exists. It copies verified CSV/JSON/TeX/Markdown artifacts,
writes a manifest, a calendar extension plan, and unfilled compute/publication
metadata templates. These templates are not measurements.

## Collect the missing original inputs on PRISM

```bash
mkdir -p logs
sbatch slurm/cpcv2_collect_missing_information.sbatch
```

This reads the completed v2 export and original archive, validates all four
periods with script 90, and exports baseline inputs without training or
sampling. The job-specific destination is `output/paper1_information_JOBID/`.
Copy that folder back for the next paper update. Override `PAPER1_EVIDENCE_DIR`
or `PAPER1_ROOT` if the actual directories differ.

For original logs, create a text file with one executed-log path per line and
set `PAPER1_LOG_LIST`. Set `PAPER1_SELECTION_ROOT` to a directory of actual May
selection NPZ/JSON pairs. Set `PAPER1_JOB_ID_LIST` to a text file with one numeric
Slurm job ID per line to retrieve accounting. These are optional; the script
never infers a measurement from a filename or submits further jobs.

Direct invocation supports the same inputs:

```bash
python scripts/98_collect_paper1_missing_information.py \
  --evidence-dir output/paper1_updated_evidence_v2 \
  --root data/processed/v2_bmd_bwdb_superob_2021_2024 \
  --out-dir output/paper1_information_recovery \
  --prepare-baseline-inputs
```

Add `--logs PATH...`, `--selection-root DIRECTORY`, and/or
`--accounting-job-ids ID...` using actual records. Log matches retain line
numbers and source hashes; they remain candidates until attributed to the
specific checkpoint and stage. A failed collection retains a non-complete
manifest and must not be used as completed evidence.

The baseline export excludes every original withheld ID from retained inputs,
uses the Bangladesh polygon, checks archived truth and coordinates against the
raw CSV, preserves all 30 members, and excludes selection dates. Files ending
`verification_only.npz` must never be used to fit or tune covariance or variance.
Missing original files are listed explicitly. There are no synthetic scores.

Fill `compute.template.json` from measured records and pass it to script 92
`--compute` to generate the P7 table. Review discovered selection candidates
and pass their configuration to script 92 `--selection`; discovery alone does
not establish a matched experiment or a winner. The existing script-96 native
IMERG and upstream inventory inputs are documented in
[PAPER1_UPDATED_EVIDENCE.md](PAPER1_UPDATED_EVIDENCE.md).

**Keep the collection private:** it can contain original rainfall observations,
station coordinates and execution-log excerpts. Commit the collector and docs,
not this bundle. It does not create a kriging result, retrieve unavailable
provider inventories, recover lost historical provenance, or generate missing
rainfall ensembles.
