# V5-HR — 0.05° Bangladesh precipitation reanalysis: workflow specification

Supersedes V3-SG/V4. One change drives everything: the conservation support
moves from CPC's 0.5° to IMERG's 0.1°, so the amount field is **observed**
rather than interpolated and the downscaling problem becomes 2× instead of 10×.

---

## 1. Inputs

Everything below is already on disk. No new acquisition except ERA5 `tp`.

| Input | Native | Role | New? |
|---|---|---|---|
| **IMERG V07B Final** | 0.1°, daily | **amount backbone** — conditioning | moved from observation → conditioning |
| CPC unified gauge | 0.5°, daily | second amount channel; gauge-informed bias | unchanged |
| ERA5 `tp` | 0.25°, daily | daily timing, dynamically constrained | **new channel** |
| ERA5 `tcwv, cape, u10, v10, msl` | 0.25° | thermodynamics / dynamics | unchanged |
| Static: elevation, slope, aspect, distance-to-coast | 0.05° | persistent subgrid structure | unchanged |
| Derived: `w_oro = U10·∇h`, `−∇·U10`, onshore component | 0.05° | day-varying orographic forcing | **new, computed in-pipeline** |
| **CHIRPS v2.0** | 0.05°, daily | training target — *structure teacher* | role narrowed |
| **BMD gauges** | point, daily | assimilated + withheld verification | unchanged |

**Period.** 2000-01-01 → present, set by IMERG Final. Train 2000–2018 (~6 900
days), validate 2019–2020, confirm 2021–2024. May 1–10 2022 excluded from
confirmatory claims.

---

## 2. Grids

The original project grids nest exactly at 0.1°, so `bd_cpc` / `wide_cpc`, the
halo canvas and the mod-10 crop lattice are all retired.

```
fine    0.05°   WIDE 256×256  (training)      BD 128×128  (production)
coarse  0.1°    WIDE 128×128                  BD  64×64          factor = 2
crop    128×128, origins ≡ 0 (mod 2), 128 % 2^3 = 0 for a 3-level U-Net
```

Verified: BD (87.6–94.0 E, 20.3–26.7 N) and WIDE (84.0–96.8 E, 16.0–28.8 N)
both close on 0.1° cell edges, and IMERG V07's native edges are multiples of
0.1°. Legacy V1/V2 comparisons work directly again — no crop translation.

---

## 3. State and decoder

```
m   coarse hurdle latents  (2 ch @ 0.1°)   wetness logit, transformed amount
z   fine allocation latents (2 ch @ 0.05°) wetness logit, centred log weight
x = R(m, z)                                 0.05° mm/day, conserving to m
```

`R` is unchanged from v5: conservative smooth base (2 iterations), hard
per-block normalisation, straight-through occurrence. At factor 2 each block is
4 cells, so the seam the smooth base removes is small to begin with.

---

## 4. Models trained — 4 runs, 3 shipped

| # | Model | Grid | State | Conditioned on | Purpose |
|---|---|---|---|---|---|
| 1 | **Model A** — coarse amount flow | 0.1°, 128² | 2 ch | IMERG, CPC, ERA5, season, static (block means) | how much rain per 0.1° cell |
| 1b | Model A-det — deterministic hurdle | 0.1° | — | same | baseline: is stochastic coarse uncertainty worth it? |
| 2 | **Model B** — allocation flow | 0.05°, 128² | 2 ch | `m` + fine predictors + corruption level | how that splits 4 ways |
| 3 | **Coupled flow** — joint multiscale | both | 4 ch | same, both branches | **operational prior and DA prior** |

Only #3 generates the product. #1 and #2 exist to be trained cheaply,
diagnosed separately, and initialise #3's branches. #1b answers one ablation.

**Training order:** A → B (conditioning augmentation from the start) → couple
and fine-tune. Same as V3-SG; only the support and the conditioning change.

---

## 5. Outputs

```
data/processed/v5_hr/
├── targets/wide_v5.zarr            fine_mm, coarse_mm, coarse_state,
│                                   allocation_state, coarse_cond, fine_cond,
│                                   fine_valid, coarse_valid, cell_area, lat, lon
├── checkpoints/{coarse,allocation,joint}.pt
├── background/<period>.zarr        30-member unassimilated ensemble
├── analysis/<period>/
│   ├── gauges_only.zarr            PRIMARY PRODUCT
│   ├── imerg_only.zarr             ablation
│   └── simultaneous.zarr           ablation
└── evaluation/<period>/            JSON, CSV, figures
```

**The deliverable** is `analysis/*/gauges_only.zarr`: a 30-member 0.05° daily
precipitation ensemble for Bangladesh, 2000–present, conserving to observed
0.1° IMERG amounts and fitted to BMD gauges.

---

## 6. Observation handling

| Stream | Role in V5-HR | Why |
|---|---|---|
| IMERG | **conditioning** | it is the daily observation; using it here puts the best daily constraint where it does the most work |
| BMD gauges | **assimilated**, and withheld folds verify | the only genuinely independent local truth |
| CPC | conditioning | gauge-informed, carries bias information IMERG lacks |
| CHIRPS | training target only | never assimilated, never verification |

No double counting: IMERG conditions and is never assimilated in the primary
product. The `imerg_only` and `simultaneous` arms remain as declared ablations
so the observational-authority contrast is still measurable — but the headline
product assimilates gauges only.

---

## 7. Experiment arms

| Arm | Amount | Allocation | Obs | Question |
|---|---|---|---|---|
| `block_null_oracle` | exact CHIRPS 0.1° | repeated block | — | zero-subgrid null |
| `smooth_null_oracle` | exact CHIRPS 0.1° | smooth base only | — | benefit beyond removing edges |
| `clim_ratio_null_oracle` | exact CHIRPS 0.1° | monthly climatology | — | **daily information beyond climatology** |
| `oracle_deterministic` | exact CHIRPS 0.1° | deterministic U-Net | — | predictable fine structure |
| `oracle_flow` | exact CHIRPS 0.1° | Model B | — | learnable distribution |
| `coupled_oracle` | clamped to truth | coupled fine branch | — | architecture/coupling cost |
| `*_null_operational` | Model A members | smooth / climatology | — | matched operational nulls |
| `operational_background` | coupled flow | coupled | — | prior downscaling |
| **`gauges_only`** | coupled flow | coupled | BMD | **primary product** |
| `imerg_only` | coupled flow | coupled | IMERG | ablation |
| `simultaneous` | coupled flow | coupled | IMERG+BMD | ablation |

---

## 8. Evaluation

**Primary endpoint** — pooled five-fold withheld-gauge anomaly ΔCRPS of
`gauges_only` against `clim_ratio_null_operational`, member-wise, day-block
bootstrap.

**CHIRPS is scored distributionally, not as daily placement**: radial and
axis-separated spectra, variograms, wet-area fraction, orographic ratio by
terrain class, extreme quantiles, seam index. Daily point agreement with CHIRPS
is reported as context with its measured ceiling (CHIRPS↔CPC daily r = 0.33),
never as the headline.

**Reference lines on every pattern panel**: CHIRPS↔CPC, CHIRPS↔IMERG, and the
model at the same support. No score is reported without its achievable ceiling.

---

## 9. What is retired

`BD_CPC`, `WIDE_CPC`, the 160-cell halo canvas, the mod-10 crop lattice, the
factor-40 crop-size rule, and the 0.5° conservation support. The V4 archive and
checkpoint stay readable through `resolve_archive_encoding` for reproduction.
