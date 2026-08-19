# V7 — CPCv2 at 0.1° plus a factor-2 downscaler

Supersedes V3-SG / V5. One diagnosis drives the whole design.

## Why V5 failed

Hard conservation to CPC's 0.5° support means the analysis increment can only do
two things: change a block's total, or redistribute inside it. The first
dominates — `amount_share` is ~0.91 in every gauge arm — so **a single point
gauge produces a 55 km-wide uniform rescale**.

That single fact explains every result:

| observation | outcome | why |
|---|---|---|
| BMD gauges | degrade withheld gauges | a 55 km correction reaches a station 75 km away at nearly full strength, with the sign that station never asked for |
| IMERG | neutral to helpful | its observations are already at block scale, so a block-quantised response is the right one |
| perfect gauges (OSSE) | still degrade | it is the *response* that is quantised, not the observation |
| block-mean pseudo-gauges | identical to point | same reason |
| CPCv2 | gauges helped **more** than IMERG | soft consistency, so the increment was shaped by the prior covariance and stayed local |

**Conservation is a property the prior should have. It is not a property the
analysis can afford.** An analysis that cannot depart from CPC cannot use an
observation that says CPC was wrong — and on heavy days CPC is wrong by a lot.

V5 also compressed amplitude: regression slope **0.473**, over-predicting light
days and under-predicting heavy ones by ~40%. CPCv2 already carries the answer to
that as well — see `wet_sampling` below.

## Structure

```
CPC 0.5°  ──▶  [A] CPCv2 at 0.1°  ──▶  0.1° ensemble
   ERA5, static, season                     │
                                            ▼
                              DA: IMERG + BMD, simultaneous
                              soft consistency only, no hard constraint
                                            │
                                            ▼
                                   0.1° analysis ensemble
                                            │
                              [B] allocation flow, factor 2
                                            │
                                            ▼
                                    0.05° ensemble
                                            │
                              DA: BMD gauges only
                                            │
                                            ▼
                                    0.05° PRODUCT
```

IMERG is assimilated at 0.1° and nowhere else — below that support it carries no
information, so re-using it at stage B would be double counting for nothing.
Gauges act at both stages, on orthogonal components of the same measurement: the
cell amount at 0.1°, the within-cell structure at 0.05°. Inflate σ modestly at
stage B so the posterior is not over-tightened.

## Stage A is CPCv2, verbatim

Not a re-derivation. `04_regrid_and_pack.py` already places CHIRPS
(nearest-selected), CPC (**bilinear, with an explicit coverage field and missing
values renormalised over the available weights**) and ERA5 (bilinear) on one
common grid, and every choice in it has been exercised. `71_v7_coarsen_pack_archive.py`
takes that output and reduces it by an exact area-weighted factor of 2 —
0.05 → 0.1 is a whole factor on the same lattice, so nothing is interpolated and
nothing is invented. Array names, channel order and the attribute block survive,
so **`06_compute_stats.py` and `scripts/train.py` run unchanged**.

`configs/train_v7_meso.yaml` differs from `train_h100_cpc_v2.yaml` in exactly
three places, each forced by the resolution:

| | v2 | V7 stage A | why |
|---|---|---|---|
| `data.zarr` / `stats` | 0.05° archive | coarsened archive + its own stats | statistics belong to the archive they were measured on |
| `data.crop` | 128 | 64 | **the same 6.4° of ground**, so `attn_resolutions: [16, 32]` still land on real U-Net levels (64→32→16→8) |
| `coarse_consistency.factor` | 10 | 5 | 0.1° → 0.5° is five cells |

Everything else is byte-identical, and a test asserts it. Three of those
inherited settings are the reason for the whole redesign:

- **`coarse_consistency` is a penalty** (`target_weight: 0.10`, `cpc_weight: 0.02`),
  not a decoder constraint. This is the property that made v2's gauge DA work.
- **`wet_sampling`** (`day_quantile: 0.90`, `crop_quantile: 0.95`) oversamples wet
  days and wet crops — the direct answer to V5's amplitude compression.
- **No hurdle.** A plain flow on sqrt-transformed precipitation. V5's dry bias
  and its all-dry members lived in the occurrence gate.

## Stage B is V3-SG's allocation branch at factor 2

The part that demonstrably works: it beat the smooth-base null on 10 of 12 days,
conserves to 3e-7, and closed the blockiness artifact (seam 1.28 against CHIRPS's
1.03). At factor 2 a block is four cells, so a gauge acting here redistributes
over three neighbours instead of ninety-nine, and the increment is quantised to
11 km rather than 55 km.

`conditioning_augmentation` matters more than usual: stage B trains on clean 0.1°
targets but runs on **analysed** 0.1° fields, whose error looks nothing like
clean CHIRPS. Calibrate `max_coarse_noise` against the stage-A analysis spread
once that exists, rather than leaving it at a guess.

## Two archives, and how they meet

| stage | archive | built by | grid |
|---|---|---|---|
| A | `bd_wide_cpc_0p1.zarr` | `71_v7_coarsen_pack_archive.py` from the v2 pack | `WIDE` 256² → **128² @ 0.1°** |
| B | `v7/wide_v7.zarr` | `56_build_chirps_subgrid_targets.py --factor 2 --coarse-res 0.1` | `WIDE_CPC` 240² @ 0.05°, coarse **120² @ 0.1°** |

**These nest exactly.** Both grids share the origin (84.0 E, 16.0 N) and the
0.1° lattice, and `WIDE_CPC` is the first 240 cells of `WIDE`. So stage B's 0.1°
grid is the first 120×120 of stage A's 128×128 — the interface is a crop, not a
regrid. Nothing is interpolated between the stages.

`56` needed one change to support this: its coarse support used to have to *be*
CPC's native 0.5° grid, and it asserted the coordinates matched. It now
**block-replicates** CPC when the support is finer (which preserves the CPC block
mean exactly, because a constant equals its own mean), still selects exactly when
the support is native so V5 reproduction is untouched, and refuses a driver finer
than the coarse grid rather than silently subsampling it.

## Inputs and period

| input | role | new? |
|---|---|---|
| CHIRPS 0.05° | target, both stages | unchanged |
| CPC 0.5° | conditioning, stage A | unchanged |
| ERA5 `tp, tcwv, cape, u10, v10, msl` | conditioning | unchanged |
| static elevation, slope, aspect, coast | conditioning, both stages | unchanged |
| IMERG 0.1° | **observation only, never conditioning** | not needed for training |
| BMD gauges | observation, and withheld folds verify | unchanged |

**Training is 1981–2024 with no IMERG dependency** — the full 44-year record, all
samples retained. Train 1981–2018, validate 2019–2020, confirm 2021–2024.

## Outputs

```
runs/v7/{meso,allocation}/best.pt
data/processed/
├── bd_wide_cpc_0p1.zarr        stage A inputs (coarsened v2 pack)
├── stats_v7_meso.json          stage A normalisation
├── v7/wide_v7.zarr             stage B targets
└── v7/{background,analysis}/   ensembles — analysis/fine is THE PRODUCT
```

## Two regimes, stated plainly

The emulator spans 1981–2024. The assimilated analysis exists only where
observations do. Report them as two products with different claims rather than
one product with a silently varying observation base.

## The risk this design carries

Stage B conserves to the analysed 0.1° field, so gauges there can only
redistribute within a 4-cell block — the same mechanism that failed at 0.5°, now
quantised to 11 km instead of 55 km. Daily precipitation error correlates over
roughly 10–50 km, so an 11 km increment is plausible where a 55 km one was not.
**Plausible, not proven.** The first stage-B pilot must measure it: withheld-gauge
CRPS with and without stage-B gauge assimilation. If it degrades again, drop
stage-B DA and take the product from the stage-A analysis downscaled by the
emulator alone — still a complete, defensible product.

## Retired from V3-SG / V5

The coupled joint flow, the 0.5° coarse hurdle flow, hard conservation to CPC in
the analysis, factor-10 single-step downscaling, and the `S_m` / `C_m`-`C_z`
authority framing in its current form — replaced by an explicit scale assignment
enforced by construction rather than measured after the fact.

## Kept

From CPCv2: the packer, `06_compute_stats.py`, `scripts/train.py`, the model and
schedule, soft coarse consistency, wet sampling. From V3-SG: the allocation flow,
the conservative smooth base, the straight-through occurrence estimator, the
frozen-encoding contract and schema gate, the DA guidance and observation
operators, the five-fold neighboured holdout, the day-block bootstrap, the claim
ladder, and scripts 56/57/58/60/61/64.
