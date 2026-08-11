# OSSE handover

Written for whoever (human or model) picks this up next. Repository:
`/panfs/ccds02/nobackup/people/afahad/project/BDDA/BDhighresDA` on the cluster.
Everything described here is committed and pushed.

---

## 1. Read this first

**The satellite path in the OSSE is broken and the cause is not yet known.**
Six hypotheses were proposed and each was wrong. Do not propose a seventh and
patch it. Read the code named in §5 before changing anything.

**Two CLI flags were silently inert on first use** (`--prior-temperature`, and
possibly `--satellite-sigma`). Both were "fixed"; only the first is confirmed
working. Any flag added here must be verified by checking that its value
changes the printed configuration line, not by assuming.

**`osse_report.json` and the `fit XXx sigma` line in the log are the primary
diagnostics.** They were ignored for six rounds of debugging while NPZ finite-
fractions were inspected instead. `fit` is the analysis misfit to the
observations in units of sigma; it should be O(1). Check it first, always.

---

## 2. What the OSSE is

CHIRPS at 0.05 degrees over 2021--2024 is the nature run. It supplies both
observation types, so they are mutually consistent by construction:

* **pseudo-gauges** -- CHIRPS sampled at the real BMD station coordinates
  (42 locations, `data/stations/data_2020_2025/Stations.csv`)
* **pseudo-satellite** -- CHIRPS averaged onto exact nested block means, or
  optionally the CPC conditioning field (`--satellite-source cpc`)

Scoring is at **withheld** pseudo-gauges, 8 of 42, matching the real-data fold
size so the OSSE and the real experiment can be read against each other.

The window is 2021--2024 and **not** 2020 onward: the prior splits train
`[1981, 2018]`, val `[2019, 2020]`, test `[2021, 2025]`, so scoring on 2020
would use years the checkpoint selection saw.

This is an optimistic upper bound -- CHIRPS supplies both the truth and the
observations -- and every result must be labelled as such.

---

## 3. Established results (gauges only; trust these)

### 3.1 The prior matters more than any assimilation setting

`runs/prior_h100_cpc_v2` beats `runs/prior_h100_cpc` (V1) decisively on
2023-03-19:

| field | corr | RMSE (mm/day) |
|---|--:|--:|
| V1 background | -- | 30.6 |
| V1 **analysis** | 0.389 | 28.8 |
| V2 background | **0.527** | 26.46 |
| V2 analysis (spread 6) | **0.561** | 24.97 |

**V2's background alone beats V1's analysis.** V2 adds
`multiscale_conditioning`, `coarse_consistency` (0.5 deg block-mean penalty)
and `wet_sampling`; all three target the "featureless background on heavy-rain
days" failure and evidently work.

Stats are checkpoint-bound (`scripts/10_osse.py` reads `checkpoint["cfg"]`), so
swapping `--ckpt` automatically swaps the zarr and stats. The
`[osse] checkpoint-bound data:` log line confirms which were used.

### 3.2 Unspread assimilation DEGRADES a good background

On V2, assimilating 34 exact gauges without spreading takes correlation from
0.527 (background) down to **0.476**, with the highest bullseye score measured.
Pinning individual pixels damages a field that was already structurally decent.

### 3.3 Any fix works, and they are interchangeable

| variant | RMSE | corr | subgrid r | bullseye |
|---|--:|--:|--:|--:|
| background | 26.46 | 0.527 | 0.008 | -- |
| base (no spreading) | 26.22 | 0.476 | 0.031 | 5.2 |
| s3 | 24.84 | 0.568 | 0.045 | 3.4 |
| s6 | 24.97 | 0.561 | 0.036 | 3.1 |
| s12 | 25.06 | 0.544 | 0.038 | 1.9 |
| s20 | 24.10 | 0.585 | 0.034 | 1.0 |
| g1e-2 | 25.12 | 0.585 | 0.043 | 5.6 |
| temp15 | 25.63 | 0.506 | 0.037 | 4.9 |

s3--s20 span 24.1--25.1 with no ordering (s20 best, s12 worst, s3 second) --
that is noise, not a curve. And `g1e-2` reaches the same 0.585 correlation by a
different route while *keeping* bullseye at 5.6. The mechanism is not
"spreading" specifically; it is "anything that stops the analysis over-fitting
individual pixels". **Do not tune this further on one day; it is fitting noise.**

### 3.4 Sub-footprint structure was never recovered

`subgrid r` is 0.031--0.045 in every one of eleven variants, against a
background of 0.008. Claim B (the analysis places rain correctly BELOW the
observation footprint) remains **unsupported**. No guidance setting moved it,
because it is not a guidance problem -- it is a prior/training question.

### 3.5 The holdout was adversarial by construction

`spread_holdout` seeds at the station farthest from the centroid and then
samples farthest-point, so it selected the most **isolated** and most
**peripheral** gauges. On BMD-like geometry it put all three planted isolates
into an 8-station holdout, leaving withheld stations up to **291 km** from the
nearest assimilated gauge with bearing gaps to **322 degrees**.

`neighbored_holdout` (in `src/bdhires/bmd.py`) excludes two distinct defects:

* **isolation** -- no neighbour within `radius_km` (default 75, inside the
  measured ~146 km variogram range)
* **edge** -- neighbours on one side only, caught by `max_bearing_gap_deg`
  (default 200; a 5x5 grid centre gives 90 degrees, its corner 270)

Support is checked against the **assimilated** stations, with a repair pass:
withholding a cluster otherwise strips the very neighbours that qualified its
members. On the real 42-station catalogue it gives 34.7/44.4/68.4 km and gaps
median 139, max 182.

**This raises apparent skill and must never be reported alone.** Arm 9
(`gauges_exact_spread_ref`) repeats the old isolated holdout unchanged; the
pair is the result.

---

## 4. The open bug: satellite arms produce identical degenerate output

Every satellite-enabled run gives **byte-identical** scores:

```
withheld CRPS 17.78 -> 35.39 (-99.1%)   assim -114.8%   fit 54.97x sigma
```

across:

| varied | values tried | result |
|---|---|---|
| footprint factor | 8 (0.4 deg), 10 (0.5 deg) | identical |
| source | CHIRPS block means, CPC | identical |
| mode | satellite-only, combined | identical |
| gradient spreading | 0, 6 cells | identical |
| satellite sigma | 0.05, 0.15, 0.35 | identical |

`cpconly` has **no gauges** and `cpc05` has 34, and they give the same number.
So the analysis is independent of **every** observation. This is not tuning,
not R, not the crop, not the operator's numerical values.

Supporting facts:

* gauges-only works: `fit` 0.22--0.46 sigma, CRPS +3 to +8%
* satellite observations are healthy: max|error| 7.6e-06 mm/day, 70.8% finite
* background is healthy: 82.5% finite (the land mask)
* the gridded `analysis` array is **0.000 finite**; station-space CRPS still
  computes, which is why it looked like a NaN bug rather than divergence

**Leading hypothesis, UNVERIFIED.** NaN observations (29.2% of footprints are
ocean blocks) enter `obs_log_likelihood(...).sum()`, making the gradient NaN;
`clip_norm` then turns NaN into a fixed-magnitude direction -- which the code's
own `perfect` branch docstring warns about -- giving the same degenerate
analysis regardless of input. Gauges-only survives because all 34 gauge
observations are finite.

This accounts for every observation above, including why the results are
identical. It is still a hypothesis.

### Hypotheses already ruled out

1. **Crop misalignment** -- factor 8 divides 128 evenly, needs no crop, fails.
2. **Land mask killing the observations** -- 70.8% of footprints are finite.
3. **Footprint factor** -- 8 and 10 fail identically.
4. **The gradient-spreading blur propagating NaN** -- spread 0 fails.
5. **CPC-specific code** -- `--satellite-source truth` fails.
6. **R/sigma too tight** -- 0.05, 0.15, 0.35 give identical results.

---

## 5. Where to look

Read these **before** changing anything:

* `src/bdhires/da/guidance.py::obs_log_likelihood` -- does it mask non-finite
  `y`, or does `.sum()` propagate NaN?
* `src/bdhires/da/guidance.py::guidance_grad` -- `clip_norm` behaviour on a
  non-finite gradient
* `src/bdhires/da/observation.py::build_R_multi`, `CompositeObsOperator` --
  how the gauge and satellite blocks are concatenated
* `scripts/10_osse.py` lines ~1150--1210 -- where `y_truth` / `y_assim` are
  built and `y_assim[~np.isfinite(y_truth)] = np.nan` is applied

Reproduce in ~2 minutes:

```bash
DAY_ROOT=/tmp/osse_debug OSSE_CKPT=runs/prior_h100_cpc_v2/best.pt \
  VARIANTS="sat04|0|||combined|truth|8" \
  sbatch slurm/osse_single_day.sbatch
grep "fit " logs/bdhires-osse-day-*.out | tail -1
```

A fix is confirmed when `fit` drops to O(1) and the gridded analysis is finite.

---

## 6. Scripts

### New in this work

| path | purpose |
|---|---|
| `slurm/osse_single_day.sbatch` | one day, N variants sequentially, minutes not hours. Variant spec `NAME\|SPREAD\|GAMMA\|TEMP\|MODE\|SAT_SOURCE\|SAT_FACTOR`. Skips variants whose `ensemble.npz` exists. |
| `scripts/48_single_day_compare.py` | side-by-side maps + metrics across variants. Defines the **bullseye** metric: increment amplitude within 1 cell of a gauge over its amplitude beyond 5 cells. Pure pinning scores high; a propagating field scores ~1. |
| `slurm/osse_exact_bmd.sbatch` | 13 arms: 0--2 exact gauges/satellite/both, 3--6 error sensitivities, 7--9 neighboured-holdout ingestion, 10--12 spreading sweep. |
| `slurm/submit_osse_exact_bmd.sh` | wrapper; a failed dependent submission does not read as a failed array. |
| `slurm/summarize_osse_bmd.sbatch` | CPU figure job. Partition is **`grace-cpuonly`** -- `grace-cpu` does not exist; `squeue` truncates to 9 chars and hides this. |
| `scripts/46_paper_figures.py` | Figs 1--3 (pipeline, measured observation error, fold design). |
| `scripts/47_osse_paper_figures.py` | Figs 4--7 + Table 1; imports `scripts/24`'s loaders so the arm schema has one definition. |
| `src/bdhires/paper/style.py` | `save_figure()` refuses a figure with no data; writes per-panel CSV plus a provenance manifest. |
| `slurm/make_paper_figures.sh` | one command for Figs 1--7. |

### Modified

* `scripts/10_osse.py` -- `--guidance-spread-cells`, `--guidance-gamma`,
  `--prior-temperature`, `--satellite-source {truth,cpc}`, `--satellite-sigma`,
  `--holdout-layout neighbored`, `--holdout-neighbor-km`,
  `--holdout-max-gap-deg`. Overrides are applied **before** the configs are
  constructed (they were not, which made `--prior-temperature` inert).
* `src/bdhires/bmd.py` -- `nearest_neighbour_km`, `max_bearing_gap_deg`,
  `neighbored_holdout`, `_pairwise_km`.
* `src/bdhires/da/guidance.py` -- `GuidanceConfig.spread_cells` and
  `spread_gradient` (mask-aware Gaussian blur of the guidance gradient;
  default 0.0 leaves every earlier result bit-identical).
* `configs/da.yaml` -- `guidance.spread_cells: 0.0`.

### Documents

* `docs/ablation_pseudo_satellite.tex` -- what adding observation error does,
  and the R-inflation double-count (the inflation applied to R is also used to
  MANUFACTURE the observation, giving the synthetic satellite sd 0.91
  correlated error against the gauges' 0.269 white).
* `docs/ingestion_results.tex` -- the real-data eleven-arm screen.
* `docs/PAPER_OUTLINE.md` -- OSSE-led paper structure, every figure marked
  have / needs-a-run / needs-code.

---

## 7. Cluster notes

* GH200 aarch64 nodes. `srun` inside an allocation fails with "CPU binding
  outside of job step allocation"; use the `run_step` launcher pattern.
  **Fourteen sbatch files still use raw `srun` and will fail**, including
  `osse_paper.sbatch`, `osse.sbatch`, `osse_final.sbatch`, `train_h100.sbatch`.
* `python3` on PATH is an x86_64 module build and dies with "cannot execute
  binary file". Use `$ENV_PREFIX/bin/python` or `awk`.
* CPU partition is `grace-cpuonly`.
* `scripts/10_osse.py` defaults `--bmd-stations` to `data/bmd/Stations.csv`,
  which does not exist; pass `data/stations/data_2020_2025/Stations.csv`.
* The Mac working copy is on OneDrive and leaves stale `.git/*.lock` files
  constantly. `rm -f .git/index.lock .git/HEAD.lock` before committing.

---

## 8. Suggested order of work

1. **Fix the satellite path** (§4, §5). Nothing satellite-related is
   interpretable until `fit` is O(1). One focused session.
2. **Re-run the three exact arms** (`ARRAY=0-2` of `osse_exact_bmd.sbatch`)
   and confirm `satellite_exact_bmd` is strongly positive. If it is still
   negative with the truth's own block means as the observation, the defect is
   in the operator or likelihood, not the observations.
3. **Then** the CPC question the user actually asked: does forcing at 0.5 deg
   improve the pattern? Run `cpc05` against `sat05` (perfect coarse) --
   `sat05` is the ceiling, and the gap between them is CPC's disagreement with
   CHIRPS. Note CPC is also the prior's input channel, so a gain means the
   prior was not honouring its own conditioning, not that CPC adds information.
4. **Paper figures** (`slurm/make_paper_figures.sh`).
5. **Sub-footprint structure** (§3.4) is a training question -- the
   CHIRPS-versus-IMERG target argument in `docs/PAPER_OUTLINE.md` §6.3 --
   not something more DA tuning will reach.

---

## 9. Process notes

Recorded because the failure mode repeated and cost most of a day.

* Six wrong diagnoses of the satellite bug, each **patched before being
  verified**. The pattern was: observe a symptom, propose a plausible cause,
  change code, re-run, fail, repeat. Reading the relevant function once would
  have been faster than all six.
* `osse_report.json` carries `fit XXx sigma`, which distinguishes divergence
  from a NaN bug immediately. It was not consulted until round six.
* Two added flags were inert on first use because overrides were applied after
  the config object was constructed. Symptom: two variants producing
  **bit-identical** results. Treat identical rows across configurations that
  should differ as a bug in the harness, not a null result.
* Changing three things at once (satellite source, crop, factor) meant the
  first failure had three candidate causes. One variable per run.
