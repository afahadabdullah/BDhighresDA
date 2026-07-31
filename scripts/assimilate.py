#!/usr/bin/env python
"""Produce the analysis: ERA5 + IMERG downscaled to 5 km and corrected by BMD gauges.

    python scripts/assimilate.py --config configs/da.yaml \
        --ckpt runs/prior_h100/best.pt --start 2021-01-01 --end 2021-12-31 \
        --out data/processed/bdhires_2021.nc

Modes (``--mode``):
    background   ERA5-conditioned generation, no observations -- the first guess
    analysis     background + IMERG + gauge guidance          -- the product
    prior        unconditional prior + observations           -- Manshausen-style
                 pure SDA, the ablation that isolates what ERA5 contributes

Ensemble spread comes from three deliberately separate sources (see
docs/METHODOLOGY.md Section 6):
    1. downscaling ambiguity  -> the x0 draw, widened by sampler.prior_temperature
    2. background uncertainty -> optionally a different ERA5-EDA member per member
    3. observation error      -> per-member perturbed observations
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.da import (  # noqa: E402
    BilinearObsOperator, BlockAverageObsOperator, CompositeObsOperator,
    GuidanceConfig, SamplerConfig, build_R_multi, perturb_observations,
)
from bdhires.da.sampler import assimilate as run_assim  # noqa: E402
from bdhires.data import PrecipDataset, DatasetConfig, load_stations  # noqa: E402
from bdhires.grids import WIDE, crop_offsets, get_grid  # noqa: E402
from bdhires.models import RectifiedFlow, UNet, select_weights  # noqa: E402
from bdhires.transforms import (  # noqa: E402
    load_climatology,
    CondTransform,
    PrecipTransform,
    ResidualSpec,
)


def load_model(ckpt_path: str, cond_channels: int, crop: int, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ck["cfg"]
    model = UNet(in_channels=1, cond_channels=cond_channels, out_channels=1,
                 image_size=crop, **cfg["model"])
    model.load_state_dict(select_weights(ck), strict=True)
    return model.to(device).eval(), cfg


def build_observations(cfg, ds, grid, tf, times, device, assim_stations=None):
    """Assemble the observation vector, operator and error covariance.

    Both observing systems go through the SAME likelihood -- there is no
    conditioning path for either of them.  The network sees only ERA5 and the
    static fields, which keeps a clean separation between the dynamical
    background (prior) and everything that actually measured rainfall
    (likelihood).  See docs/METHODOLOGY.md Section 4.

    * ``gauges``  -- BMD daily rain gauges, bilinear point operator, ~35 points.
    * ``imerg``   -- IMERG 0.1 deg footprints, exact 2x2 block-average operator,
                     ~3.5k valid footprints over the Bangladesh grid.

    Returns ``(H, y_all, R, corr_blocks)``.  ``y_all`` has shape (T, S_total) in
    TRANSFORMED units, NaN wherever an observation is missing.  ``corr_blocks``
    describes which slices of the vector should receive spatially correlated
    perturbations when building per-member observation draws.
    """
    obs_cfg = cfg["observations"]
    ops, ys, specs, corr_blocks = [], [], [], []
    offset = 0

    if obs_cfg["gauges"]["enabled"]:
        ss, values = load_stations(obs_cfg["gauges"]["csv"], times, grid=grid,
                                   min_coverage=obs_cfg["gauges"]["min_coverage"])
        if assim_stations:
            keep = np.load(assim_stations)
            ss, values = ss.subset(keep), values[:, keep]
        y = tf.forward(values)
        y[~np.isfinite(values)] = np.nan
        ops.append(BilinearObsOperator(grid, ss.lat, ss.lon).to(device))
        ys.append(y)
        specs.append((len(ss), obs_cfg["gauges"]["sigma_obs"],
                      obs_cfg["gauges"]["representativeness"]))
        offset += len(ss)
        print(f"[obs] gauges: {len(ss)} stations")

    if obs_cfg["imerg"]["enabled"]:
        f = obs_cfg["imerg"]["factor"]
        op = BlockAverageObsOperator(f, valid=ds.fixed_valid).to(device)
        keep = op.valid_mask()
        # IMERG lives on the model grid but was regridded conservatively, so its
        # 2x2 block mean IS the native 0.1 deg footprint value.
        spatial_slices = ds.fixed_spatial_slices()
        raw = np.stack(
            [
                np.asarray(ds.z["imerg"][int(j)][spatial_slices])
                for j in range(len(times))
            ]
        )
        if obs_cfg["imerg"].get("bias_correction"):
            raw = apply_qm(raw, obs_cfg["imerg"]["bias_correction"], times)
        nlat, nlon = grid.nlat // f, grid.nlon // f
        coarse = raw.reshape(len(times), nlat, f, nlon, f).mean(axis=(2, 4))
        flat = coarse.reshape(len(times), -1)
        y = tf.forward(flat)
        y[~np.isfinite(flat)] = np.nan
        if keep is not None:
            y[:, ~keep.cpu().numpy()] = np.nan
        ops.append(op)
        ys.append(y)
        specs.append((y.shape[1], obs_cfg["imerg"]["sigma_obs"],
                      obs_cfg["imerg"]["representativeness"]))
        corr_blocks.append((offset, nlat, nlon, obs_cfg["imerg"]["error_corr_cells"]))
        offset += y.shape[1]
        n_ok = int(keep.sum()) if keep is not None else y.shape[1]
        print(f"[obs] imerg: {n_ok} footprints (sigma={obs_cfg['imerg']['sigma_obs']}, "
              f"error corr length {obs_cfg['imerg']['error_corr_cells']} cells)")

    if not ops:
        return None, None, None, []
    H = CompositeObsOperator(ops).to(device) if len(ops) > 1 else ops[0]
    return H, np.concatenate(ys, axis=1), build_R_multi(specs, device=device), corr_blocks


def apply_qm(field: np.ndarray, qm_path: str, times) -> np.ndarray:
    """Apply the per-cell, per-season quantile map fitted by 07_bias_correct_imerg.py."""
    z = np.load(qm_path)
    q_src, q_dst, seasons = z["q_src"], z["q_dst"], z["seasons"]  # (S,Q,H,W),(S,Q,H,W),(S,)
    months = times.astype("datetime64[M]").astype(int) % 12 + 1
    out = np.empty_like(field)
    for si, sm in enumerate(seasons):
        sel = np.isin(months, sm)
        if not sel.any():
            continue
        src, dst = q_src[si], q_dst[si]
        f = field[sel]
        res = np.empty_like(f)
        for i in range(f.shape[1]):
            for j in range(f.shape[2]):
                res[:, i, j] = np.interp(f[:, i, j], src[:, i, j], dst[:, i, j])
        out[sel] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="analysis", choices=["background", "analysis", "prior"])
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--assim-stations", default=None,
                    help="npy file of station indices to assimilate (rest are withheld)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = get_grid(cfg["data"]["grid"])
    stats = json.loads(Path(cfg["data"]["stats"]).read_text())
    tf = PrecipTransform.from_dict(stats["precip_transform"])

    ds = PrecipDataset(
        DatasetConfig(
            root=cfg["data"]["zarr"],
            crop=grid.nlon,
            random_crop=False,
            crop_origin=crop_offsets(WIDE, grid),
        ),
        tf,
        cond_mean=np.asarray(stats["cond_mean"], np.float32),
        cond_std=np.asarray(stats["cond_std"], np.float32),
        cond_transform=CondTransform.from_stats(stats),
        residual=ResidualSpec.from_stats(stats),
        climatology=load_climatology(config["data"]["stats"], stats),
    )
    times = ds.time
    sel = np.where((times >= np.datetime64(args.start)) & (times <= np.datetime64(args.end)))[0]
    print(f"assimilating {len(sel)} days over grid {grid.name} {grid.shape}")

    model, _ = load_model(args.ckpt, ds.total_cond_channels, grid.nlon, device)
    flow = RectifiedFlow()

    # `background` runs unguided, so it takes the uninflated sampler block:
    # without observations to pull members back, prior tempering is pure error.
    sampler_key = (
        "background_sampler"
        if args.mode == "background" and "background_sampler" in cfg
        else "sampler"
    )
    scfg = SamplerConfig(**cfg[sampler_key])
    scfg = dataclasses.replace(scfg, mask_fill=ds.mask_fill)
    print(f"sampler block: {sampler_key} (mode={args.mode})")
    gcfg = GuidanceConfig(**cfg["guidance"])
    n_members = args.members or cfg["ensemble"]["members"]

    # ---- observation streams --------------------------------------------
    H = y_all = R = None
    corr_blocks: list = []
    if args.mode != "background":
        H, y_all, R, corr_blocks = build_observations(
            cfg, ds, grid, tf, times, device, assim_stations=args.assim_stations)
    perturb = bool(cfg["ensemble"].get("perturb_observations", True))

    valid = ds.fixed_valid
    mask = torch.from_numpy(valid[None, None]).to(device)

    out = np.full((len(sel), n_members, grid.nlat, grid.nlon), np.nan, np.float32)
    for k, j in enumerate(sel):
        item = ds[int(j)]
        cond = item["cond"][None].to(device)
        base = item["base"][None].to(device)
        y = None
        if args.mode != "background":
            if perturb:
                # one observation draw per member -- the single cheapest fix for
                # under-dispersion (see da/observation.perturb_observations)
                yj = perturb_observations(y_all[j], R, n_members, seed=int(j),
                                          corr_blocks=corr_blocks)
                yj[:, ~np.isfinite(y_all[j])] = np.nan
            else:
                yj = np.repeat(y_all[j][None], n_members, axis=0)
            y = torch.from_numpy(yj[:, None].astype(np.float32)).to(device)  # (M,1,S)
        out_raw = run_assim(
            model,
            None if args.mode == "prior" else cond,
            (n_members, 1, grid.nlat, grid.nlon),
            device, H=H, y=y, R=R, cfg=scfg, gcfg=gcfg, flow=flow, mask=mask,
            to_precip=lambda x, b=base: ds.residual.decode(x, b),
        )
        # Decode to transformed-precipitation space before storing.
        out[k] = ds.residual.decode(out_raw, base).squeeze(1).cpu().numpy()
        if k % 20 == 0:
            print(f"  {k}/{len(sel)}  {str(times[j])[:10]}", flush=True)

    precip = tf.inverse(out)                     # back to mm/day
    precip = np.where(valid[None, None] > 0, precip, np.nan)

    da = xr.DataArray(
        precip,
        dims=("time", "member", "lat", "lon"),
        coords=dict(time=times[sel], member=np.arange(n_members), lat=grid.lat, lon=grid.lon),
        name="precip",
        attrs=dict(units="mm/day", long_name="daily precipitation", mode=args.mode),
    )
    out_ds = da.to_dataset()
    out_ds["precip_mean"] = da.mean("member")
    out_ds.attrs.update(
        title="BDhighresDA generative reanalysis of daily precipitation",
        source=f"flow-matching downscaling of ERA5+IMERG with BMD gauge assimilation ({args.mode})",
        checkpoint=str(args.ckpt),
        grid=grid.name,
        resolution_deg=grid.res,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in out_ds.data_vars}
    out_ds.to_netcdf(args.out, encoding=enc)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
