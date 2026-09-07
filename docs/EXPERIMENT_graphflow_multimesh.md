# SURMA-GraphFlow G0: regional multimesh backbone experiment

> SURMA-GraphFlow G0 is an experimental backbone and is not part of the frozen BRISHTI-05 production lineage unless separately validated and promoted.

Internal identifier: `graphflow_g0_multimesh`. Reference: CPCv2 at commit
`d072bd3e0418bd0b150bac530dc0a8dc62eca180`. Implementation and engineering
checks: 2026-09-07. No scientific training or real-data GraphFlow skill evaluation
has been run in this local checkout.

## Scientific contract

The question is whether explicit multiscale communication improves the learned
precipitation prior and its observation-response structure under the **same**
generative DA. Broader sensitivity alone is not evidence of better covariance;
unphysical distant responses would count against the architecture.

`configs/train_h100_cpc_graphflow_g0.yaml` is the CPCv2 configuration with only
architecture settings and the output directory changed. An automated test
compares the parsed configurations to enforce this. It retains:

- Training 1981–2018, validation 2019–2020, test 2021–2025.
- The same packed Zarr and `stats_cpc_v2.json`, CHIRPS target, square-root
  precipitation transforms, and standardized residual relative to CPC.
- Seven dynamic channels: CPC precipitation/validity and ERA5 TCWV, CAPE, U10,
  V10, MSL; seven static channels and two seasonal channels in the standard pack.
- CPCv2 multiscale conditioning, existing attention, dropout 0.2, wet sampling,
  coarse consistency, optimizer settings, EMA 0.9995, and validation selection.
- Batch 32, 150 epochs, 8 validation members and 30 validation sampling steps.

The normal grid tensor pipeline is unchanged. CPC and ERA5 state fields are
already interpolated to 0.05° by the existing packer; no graph-format data stores
are created. BMD and IMERG never become learned inputs or graph nodes. They
remain likelihood observations, permitting assimilation without retraining.

The rectified-flow equations remain `x_t = t*x1 + (1-t)*x0`, `x0 ~ N(0,I)`,
velocity target `x1-x0`, and `x1_hat = x_t + (1-t)*v`. Residual decoding precedes
the precipitation observation operators. The existing score/velocity conversion,
likelihood, errors, Heun integration, observation perturbations and Langevin
correctors are reused. No changes were made to `flow.py`, `guidance.py`,
`observation.py`, `sampler.py`, the transforms, datasets, or production DA config.

## Architecture and topology

`GraphFlowUNet` subclasses the existing U-Net. A parameter-free identity hook in
the baseline is called once after all ResBlocks/attention/conditioning injections
at each encoder level. G0 overrides it at **level 2**, immediately before the next
downsample and before saving that block's decoder skip. Earlier within-level
skips retain their normal representations.

For 128×128, the graph sees 32×32 nodes with 288 latent channels. Both dimensions
follow the actual input shape divided by four; the graph is not keyed to a
literal resolution. Any input satisfying the existing U-Net pyramid divisibility
rule remains supported, including rectangular canvases. Construction-time
`image_size` continues to control the inherited attention placement.

The graph performs:

1. Reshape `[B,C,Hg,Wg]` to `[B,N,C]`, project C→256.
2. Embed static geometry with an MLP and LayerNorm.
3. Apply 8 residual message-passing blocks with SiLU and LayerNorm.
4. Project 256→C, reshape to the grid and add to the incoming latent field.

The last projection is zero-initialized, so identical U-Net weights give exactly
the same initial output. The graph still executes at initialization; it is not
bypassed by a special-case branch. Its node features inherit time and all prior
conditioning through the CNN. There is no second time encoder, duplicate raw
meteorological feature stack or dynamic wind-dependent topology.

At stride `s` in `[1,2,4,8]`, both endpoints lie on the subset with row and column
multiples of `s`. Connect offsets `(0,s)`, `(s,0)`, `(s,s)`, `(s,-s)` and each
reverse, only when both endpoints are inside the domain. There is no periodic
wrap, no self edge, and no duplicate directed edge. Stride 1 connects every node.

At 32×32 there are **1,024 nodes and 10,176 directed edges**:

| Stride | Directed edges |
| --- | ---: |
| 1 | 7,812 |
| 2 | 1,860 |
| 4 | 420 |
| 8 | 84 |

Raw edge features are `[dx/s_max, dy/s_max, distance/s_max, level/max_level]`,
where displacement is destination minus source in latent-grid cells. These are
geometric scales, not precise physical support radii. Incoming degrees range
from 3 to 32 on the standard graph. A singleton latent grid has no edges and
uses a denominator of one for a safe zero aggregate.

Each block computes a message MLP from source node, destination node and static
encoded geometry. To avoid a `[B,E,3D]` concatenation, the first linear map is
factored into three projections before gathering/adding; this is algebraically
equivalent to the concatenated linear map. Messages are accumulated by shared
destination indices and divided by incoming degree. The node update is an MLP
of the normalized node and aggregate, followed by a residual addition. Edge
hidden states do not evolve across blocks.

The per-processor cache stores at most four shape/topology/device combinations.
It includes indices, raw features and degrees, is cleared on device/dtype moves,
and is absent from `state_dict`/EMA/checkpoints. Cache misses made under
`inference_mode` deliberately create normal tensors, allowing later guided
autograd calls to reuse them.

AMP follows the current trainer: BF16 on supported CUDA hardware, FP16 with
GradScaler otherwise. Graph aggregation alone uses FP32 (or preserves FP64 for
double-precision checks). Optional `graph.checkpoint_blocks: true` uses
non-reentrant activation checkpointing, including support for `autograd.grad` in
DA; the primary configuration leaves it **false**. The primary remains 256×8.

## Repository integration and isolation

The original single-stage workflow constructed `UNet` in training, preflight,
test plotting, extreme diagnostics, OSSE, BMD monthly examples, simultaneous
method sweeps, generic assimilation and the V7 stage-A comparison loader.
Validation receives a model object; EMA traverses the full state dictionary;
`VelocityOnly` preserves the existing velocity/hurdle split. Hierarchical
coarse/allocation branches have a separate scientific state model and remain
unchanged; their explicit U-Net constructors and historical smoke models are
intentionally retained.

Architecture selection is centralized in `models/factory.py`. Missing
`architecture` means `unet`; unknown or inconsistent settings fail. New
checkpoints store a full `model_config` (including input/output/conditioning
channels, original image size and all graph settings). Legacy `cfg`-based
checkpoints need no migration. Inference uses the existing `select_weights`
online/EMA choice and strict state loading. New-checkpoint resume checks metadata
to catch topology changes that weight shapes alone would miss.

Shared modifications are limited to the identity hook, model construction/loading,
checkpoint metadata and model summary. New experimental entry points are in
`scripts/graphflow_experiment.py` and `slurm/graphflow_g0.sbatch`. The latter
defaults to **benchmarking**, not full training. Experimental output roots must
be under `runs/<name-containing-graphflow>/`; existing checkpoints and completed
experimental folds are not silently overwritten. Production launchers,
`v2_simul_s04_ig010`, BRISHTI-05 stores and historical metadata are untouched.

### Added files

- `src/bdhires/models/{multimesh,graph_processor,graphflow_unet,factory}.py`
- `src/bdhires/eval/graphflow.py`
- `configs/train_h100_cpc_graphflow_g0.yaml`
- `scripts/{graphflow_experiment,benchmark_graphflow,diagnose_graphflow_sensitivity}.py`
- `slurm/graphflow_g0.sbatch`
- `tests/test_graphflow.py`
- This document.

### Modified files

- `src/bdhires/models/unet.py`, `src/bdhires/models/__init__.py`,
  `src/bdhires/utils/summary.py`.
- `scripts/train.py`, `scripts/preflight_training.py`.
- `scripts/08_plot_test_predictions.py`, `09_diagnose_extremes.py`, `10_osse.py`,
  `15_bmd_month_example.py`, `28_simultaneous_method_sweep.py`,
  `72_v7_two_stage_osse.py`, `assimilate.py`: architecture-aware loading only.

## Engineering verification

`tests/test_graphflow.py`: **36 passed, 1 CUDA test skipped** on CPU. Coverage:

- Deterministic, bidirectional, duplicate-free, non-wrapping topology; aligned
  coarse nodes; stride-1 reachability; degrees; multiple latent shapes; caching.
- Batch >1, 128×128 and rectangular inputs, with/without conditioning, one- and
  two-output hurdle models, CPU BF16 flow loss/backward.
- Baseline outputs equal the pre-hook forward implementation exactly.
- Graph zero-init outputs equal a baseline with **nonzero randomized weights
  throughout**, avoiding a vacuous all-zero U-Net-head comparison. Disabled graph
  models also strictly load baseline state dictionaries.
- The graph-modified representation reaches both the next downsample and the
  matching decoder skip. Graph projections/message blocks receive finite gradients.
- Real `guidance_grad` through physical point and S04 block operators, residual
  decoding, per-component spreading and optional activation checkpointing.
- New online/EMA checkpoint roundtrips, legacy checkpoints, metadata reconstruction,
  unguided/guided Heun sampling with a Langevin corrector, deterministic seeds,
  and changed outputs under per-member observation perturbations.
- Sensitivity norms, radial energy, anisotropy, and zero-innovation behavior.

The synthetic engineering run used an explicitly reduced network, two members,
32×32 fields and 20 fixed RF mini-batch updates. Loss fell from **2.043306 to
1.254835**, all gradients remained finite, EMA updated, a checkpoint was saved
and reloaded, and a three-step validation sampler remained finite. This is an
overfit/engineering check, **not** a scientific training result.

The initial broad test run with Zarr 3.3 encountered seven archive API failures.
The isolated test environment was adjusted to Zarr 2.18.7, matching those APIs;
no repository dependency definitions or archive code were changed. Three other
failures were reproduced on a separately extracted, untouched `d072bd3` snapshot:

1. `test_field_metrics_are_zero_for_identical_fields`: existing correlation is
   1.0666667 for identical fields rather than 1.
2. `test_flow_matching_clean_loss_is_added_and_reported_separately`: existing test
   unpacks three results but the current hurdle-capable loss returns four.
3. `test_dry_block_is_exact_zero_but_keeps_a_positive_occurrence_gradient`:
   existing hierarchical dry-branch occurrence gradient is zero.

These are recorded rather than changing CPCv2/other experiments to make the new
backbone's regression report green. The full-suite rerun with Zarr 2.18.7 was
**448 passed, 2 skipped, 3 pre-existing failures**, in 95.48 seconds. The local
`runs/graphflow_g0_multimesh/tests.xml` contains the detailed report.

## Measured compute

Full primary architectures, **128×128, batch 1, 16 conditioning channels**, FP32
on a macOS ARM CPU, PyTorch 2.14.0, four CPU threads. One warmup and three timed
repeats; table entries are medians. Initialized models with nonzero output
projections were used to exercise actual Jacobians. These desktop timings are
noisy and do not establish HPC throughput.

| Measure | CPCv2 U-Net | G0 256×8 |
| --- | ---: | ---: |
| Parameters | 57,493,729 | 61,399,809 |
| Forward + RF/coarse loss + backward | 1,682.9 ms | 2,290.0 ms |
| Training primitive throughput | 0.594 samples/s | 0.437 samples/s |
| One unguided forward | 558.8 ms | 624.6 ms |
| Three-step Heun RF trajectory | 2,674.1 ms | 3,067.3 ms |
| Composite guided gradient | 2,267.1 ms | 2,315.0 ms |
| Guided ratio | 1.000 | 1.021 |
| Peak CUDA memory | unavailable | unavailable |

G0 adds **3,906,080 parameters (6.8%)**. The measured CPU guidance ratio is below
the 1.5× engineering target, but CUDA performance/memory, batch 32, and 30-member
DA have **not** been verified. The benchmark includes the real coarse-consistency
loss and the existing physical gauge/block likelihood with component spreading.
Its synthetic error vector is a controlled cost benchmark, not a real IMERG error
calibration. It excludes optimizer/EMA state from the timed training primitive;
run the real-data preflight to determine full training memory fit.

The machine-readable local result is
`runs/graphflow_g0_multimesh/benchmark_cpu.json`. Primary batch size remains 32;
there is no demonstrated CUDA OOM justifying a batch reduction or accumulation
change. Optional 192×6 benchmarking is explicit and never replaces the primary.

## Commands (from the repository root)

Use the normal project Python on HPC. The local checks used an isolated temporary
environment at `/private/tmp/bdhires-graphflow-g0-env`; no packages were installed
into the user's existing project environment. No added graph-framework dependency
or network access is required by the model.

### A. Tests, synthetic training and short DA smoke

```bash
python -m pytest tests/test_graphflow.py -q
python scripts/graphflow_experiment.py smoke --steps 20
python -m pytest tests/test_graphflow.py -k checkpoint_ema_loss_guidance_and_sampler -q
python scripts/diagnose_graphflow_sensitivity.py --synthetic
```

The last command generates interior, terrain and mask-coast response maps,
profiles, spectra, `.npz` arrays and JSON metrics. These initial graph residuals
are zero, so the paired maps agree by construction; they show the diagnostic is
working, not that G0 improves influence.

### B. CPU / GPU benchmarks and full-memory preflight

```bash
python scripts/benchmark_graphflow.py --device cpu --batches 1
python scripts/benchmark_graphflow.py --device cuda --batches 1 8 16 30 32 \
  --repeats 5 --warmup 2 --out runs/graphflow_g0_multimesh/benchmark_cuda.json
python scripts/benchmark_graphflow.py --device cuda --batches 16 32 --alternative \
  --out runs/graphflow_g0_multimesh/benchmark_alternatives.json
python scripts/preflight_training.py --config configs/train_h100_cpc_graphflow_g0.yaml \
  --steps 3 --out runs/graphflow_g0_multimesh/real_data_preflight.json
```

The existing real-data preflight also requires the CPCv2 normalization diagnostic
report. Retain its data/stats checks. GPU OOM is recorded explicitly; no batch or
architecture is silently reduced. For batch-32 failure, assess activation
checkpointing and then implement/verify gradient accumulation before changing the
effective batch or optimizer frequency.

For allocated Grace/GH200 jobs:

```bash
mkdir -p logs
sbatch slurm/graphflow_g0.sbatch
sbatch --export=ALL,GRAPHFLOW_ACTION=preflight slurm/graphflow_g0.sbatch
```

### C. Matched short training screen, then full training

```bash
python scripts/graphflow_experiment.py train --stage screen --baseline
python scripts/graphflow_experiment.py train --stage screen
# After technical and scientific screening:
python scripts/graphflow_experiment.py train --stage full
```

The separate screen roots are `runs/cpcv2_graphflow_control_screen` and
`runs/prior_h100_cpc_graphflow_g0_screen`. Both use 30 epochs, the same effective
batch 32 and the same 30-epoch cosine schedule. This is shorter than the full
150-epoch schedule; do not compare a screened model to an archived fully trained
CPCv2 checkpoint as a matched optimization-budget experiment. Full G0 writes
`runs/prior_h100_cpc_graphflow_g0`. Nothing resumes automatically. Explicitly
resume with `--resume <that-experiment>/last.pt`.

The corresponding new Slurm actions are `control-screen`, `screen`, and `full`:

```bash
sbatch --export=ALL,GRAPHFLOW_ACTION=control-screen slurm/graphflow_g0.sbatch
sbatch --export=ALL,GRAPHFLOW_ACTION=screen slurm/graphflow_g0.sbatch
# Only after the screen warrants the full experiment:
sbatch --export=ALL,GRAPHFLOW_ACTION=full slurm/graphflow_g0.sbatch
```

### D. Prior validation and checkpoint sensitivity

```bash
python scripts/graphflow_experiment.py validate \
  --ckpt runs/prior_h100_cpc_graphflow_g0/best.pt \
  --out runs/graphflow_g0_multimesh/prior_validation
python scripts/diagnose_graphflow_sensitivity.py \
  --unet-ckpt runs/prior_h100_cpc_v2/best.pt \
  --graphflow-ckpt runs/prior_h100_cpc_graphflow_g0/best.pt \
  --date 2019-07-15 --out runs/graphflow_g0_multimesh/sensitivity_20190715
```

Validation reuses the checkpoint's exact `ValidationMonitor`: fixed held-out
cases, residual decoding, rainfall CRPS/bias/wet-frequency/coverage and existing
maps. Real sensitivity requires a date present in the validation split and
matching checkpoint data contracts. It uses the same RF-interpolated held-out
state for both models and a detached, equal transformed-space innovation relative
to each model's denoised point estimate. It reports raw and spread gradients,
one-step increments, total gradient norm, radial amplitude/energy, 90%-energy
radius, anisotropy, spectra and maximum distant response, with terrain and
validity-boundary overlays. Distances are grid cells. Scaled terrain is not metres;
the CHIRPS-validity boundary is a coast proxy. This is not a full analysis increment.

### E. Four-case frozen DA comparison

Supply the same canonical BMD CSV and already prepared S04 IMERG file used by
the CPCv2 development evaluation:

```bash
python scripts/graphflow_experiment.py da \
  --unet-ckpt runs/prior_h100_cpc_v2/best.pt \
  --graphflow-ckpt runs/prior_h100_cpc_graphflow_g0/best.pt \
  --stations data/stations/bmd_daily.csv \
  --imerg data/processed/v2_confirmatory_2021_2024/imerg_s04/2022_may_sep.nc \
  --start 2022-05-01 --end 2022-05-10 \
  --out runs/graphflow_g0_multimesh/frozen_da
```

Replace observation paths with the actual prepared files in the HPC workspace.
The wrapper verifies checkpoint architectures/contracts and calls the existing
simultaneous evaluation, selecting a copy of the frozen `v2_simul_s04_ig010`
settings under new names. It runs five identical spatial folds for both priors,
30 members, one-day background offset, factor-8/stride-1 S04, IMERG correlation
0.75 cells, gamma 0.01 for both streams, gauge spreading 6 cells and the unchanged
sampler/error parameters. Each run includes an unguided background. Outputs are
separate `cpcv2_control/` and `graphflow_g0_multimesh/` fold files; the G0 DA method
is `graphflow_g0_multimesh_frozen_da`, never the historical production name.

Use paired withheld-station scores to distinguish prior improvement from added
DA benefit. Prioritize CRPS, RMSE, MAE, bias, correlation, spread/RMSE, coverage,
wet-day/intensity-stratified and heavy-rain skill; reuse existing spatial FSS,
spectra, variograms and sub-footprint diagnostics where available. Training loss
alone cannot establish improvement. The ten default days are the existing
development window, not independent confirmation. Do not choose architecture on
the long test archive. A later confirmatory evaluation must preserve the existing
date/fold exclusions, including 2022-05-01..10 and May 2022 for monthly scores.

### F. May 1--10 matched HPC evaluation against completed CPCv2

After the full GraphFlow checkpoint is frozen, submit the five-fold development
comparison with:

```bash
git pull --ff-only origin main
bash slurm/submit_graphflow_g0_da_may2022.sh
```

This launcher runs only GraphFlow. It reuses the already completed CPCv2
`v2_simul_s04_ig010` folds under
`data/processed/v2_simultaneous_refinement/ing2022_s04`, requires the same BMD
daily files and prepared S04 IMERG product, and refuses a comparison if dates,
folds, observations, ensemble size, seed, or frozen DA parameters differ.
Override `GRAPHFLOW_CPC_ROOT`, `GRAPHFLOW_IMERG`, `GRAPHFLOW_CKPT`, or
`BMD_CKPT` only when the corresponding HPC paths differ.

The dependent summary writes:

```text
runs/graphflow_g0_multimesh/da_may2022/summary/
├── comparison.md
├── comparison.json
└── comparison.png
```

The summary reports four distinct cases (CPCv2 background/DA and GraphFlow
background/DA), paired CRPS confidence intervals, the difference in within-model
DA gain, withheld-gauge DA gain versus distance to the nearest assimilated
gauge, and grid-increment amplitude versus distance. The difference in
within-model gain is the clean test of whether GraphFlow extracts more useful
information from observations, rather than merely starting from a better prior.
May 1--10, 2022 was a development/selection window, so this is a controlled
diagnostic rather than an independent confirmatory claim.

## Remaining checks and limitations

- The full GraphFlow training run is complete. The matched real-observation DA
  folds and their pooled comparison remain pending on HPC.
- No learned GraphFlow checkpoint, rainfall CRPS/RMSE gain, or improved physical
  observation-response structure is claimed. Only synthetic/engineering outputs
  were generated locally.
- G0 augments the U-Net and retains its attention: it is not compute-matched.
  Multimesh phase is anchored to the latent crop origin and may matter for tiled
  or shifted inference. There is no learned coarsening or geographical metric.
- Autograd and topology construction are deterministic on CPU; CUDA scatter
  reductions can differ at floating-point roundoff level. Seed reproducibility
  is not a promise of cross-device bitwise identity.
- Future G1 can replace redundant attention at matched compute, G2 can investigate
  atmospheric edge features, and G3 can compare a different hierarchy. None are
  implemented in G0.
