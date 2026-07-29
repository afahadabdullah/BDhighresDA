#!/usr/bin/env python
"""Produce the analysis: ERA5 + IMERG downscaled to 5 km and corrected by BMD gauges.

    python scripts/assimilate.py --config configs/da.yaml \
        --ckpt runs/stageB/final.pt --start 2021-01-01 --end 2021-12-31 \
        --out data/processed/bdhires_2021.nc

Modes (``--mode``):
    background   conditional generation only (no stations)  -- the "first guess"
    analysis     conditional generation + station guidance   -- the product
    prior        unconditional prior + station guidance      -- Manshausen-style
                 pure SDA, useful as an ablation to show what ERA5/IMERG add
"""
from __future__ import annotations

import argparse
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
    GuidanceConfig, SamplerConfig, build_R_multi,
)
from bdhires.da.sampler import assimilate as run_assim  # noqa: E402
from bdhires.data import PrecipDataset, DatasetConfig, load_stations  # noqa: E402
from bdhires.grids import get_grid  # noqa: E402
from bdhires.models import RectifiedFlow, UNet  # noqa: E402
from bdhires.transforms import PrecipTransform  # noqa: E402


def load_model(ckpt_path: str, cond_channels: int, crop: int, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ck["cfg"]
    model = UNet(in_channels=1, cond_channels=cond_channels, out_channels=1,
                 image_size=crop, **cfg["model"])
    state = ck.get("ema") or ck["model"]
    model.load_state_dict({k: v for k, v in state.items()}, strict=True)
    return model.to(device).eval(), cfg


def build_observations(cfg, ds, grid, tf, times, device, assim_stations=None):
    """Assemble the observation vector, operator and error covariance.

    Two streams, each independently switchable in ``configs/da.yaml``:

    * ``gauges``  -- BMD daily rain gauges, bilinear point operator.
    * ``imerg``   -- IMERG 0.1 deg footprints, exact 2x2 block-average operator.
      Only active when ``observations.imerg.mode == "assimilate"``; when it is
      ``"condition"`` IMERG is instead a conditioning channel of the network
      and never enters the likelihood.  See docs/METHODOLOGY.md Section 3.6.

    Returns ``(H, y_all, R)`` with ``y_all`` of shape (T, S_total) in
    TRANSFORMED units and NaN wherever an observation is missing.
    """
    obs_cfg = cfg["observations"]
    ops, ys, specs = [], [], []

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
        print(f"[obs] gauges: {len(ss)} stations")

    if obs_cfg["imerg"]["mode"] == "assimilate":
        ic = ds.z.attrs["imerg_cond_index"] if hasattr(ds.z, "attrs") else cfg["data"]["imerg_cond_index"]
        f = obs_cfg["imerg"]["factor"]
        op = BlockAverageObsOperator(f, valid=ds.valid).to(device)
        keep = op.valid_mask()
        # IMERG is stored on the model grid; the 2x2 mean of that field IS the
        # native 0.1 deg value because the packing step regridded conservatively.
        raw = np.stack([np.asarray(ds.z["cond"][int(j)][ic]) for j in range(len(times))])
        if obs_cfg["imerg"].get("bias_correction"):
            raw = apply_qm(raw, obs_cfg["imerg"]["bias_correction"], times)
        coarse = raw.reshape(len(times), grid.nlat // f, f, grid.nlon // f, f).mean(axis=(2, 4))
        y = tf.forward(coarse).reshape(len(times), -1)
        y[~np.isfinite(coarse.reshape(len(times), -1))] = np.nan
        if keep is not None:
            y[:, ~keep.cpu().numpy()] = np.nan
        ops.append(op)
        ys.append(y)
        specs.append((y.shape[1], obs_cfg["imerg"]["sigma_obs"],
                      obs_cfg["imerg"]["representativeness"]))
        print(f"[obs] imerg: {int((keep.sum() if keep is not None else y.shape[1]))} footprints "
              f"(sigma={obs_cfg['imerg']['sigma_obs']})")

    if not ops:
        return None, None, None
    H = CompositeObsOperator(ops).to(device) if len(ops) > 1 else ops[0]
    return H, np.concatenate(ys, axis=1), build_R_multi(specs, device=device)


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
        DatasetConfig(root=cfg["data"]["zarr"], crop=grid.nlon, random_crop=False),
        tf,
        cond_mean=np.asarray(stats["cond_mean"], np.float32),
        cond_std=np.asarray(stats["cond_std"], np.float32),
    )
    times = ds.time
    sel = np.where((times >= np.datetime64(args.start)) & (times <= np.datetime64(args.end)))[0]
    print(f"assimilating {len(sel)} days over grid {grid.name} {grid.shape}")

    model, _ = load_model(args.ckpt, ds.total_cond_channels, grid.nlon, device)
    flow = RectifiedFlow()

    scfg = SamplerConfig(**cfg["sampler"])
    gcfg = GuidanceConfig(**cfg["guidance"])
    n_members = args.members or cfg["ensemble"]["members"]

    # ---- observation streams --------------------------------------------
    H = y_all = R = None
    if args.mode != "background":
        H, y_all, R = build_observations(cfg, ds, grid, tf, times, device,
                                         assim_stations=args.assim_stations)

    mask = torch.from_numpy(ds.valid[None, None]).to(device)

    out = np.full((len(sel), n_members, grid.nlat, grid.nlon), np.nan, np.float32)
    for k, j in enumerate(sel):
        item = ds[int(j)]
        cond = item["cond"][None].to(device)
        y = None
        if args.mode != "background":
            yj = torch.from_numpy(y_all[j][None, None]).to(device)  # (1,1,S)
            y = yj.expand(n_members, -1, -1)
        out[k] = run_assim(
            model,
            None if args.mode == "prior" else cond,
            (n_members, 1, grid.nlat, grid.nlon),
            device, H=H, y=y, R=R, cfg=scfg, gcfg=gcfg, flow=flow, mask=mask,
        ).squeeze(1).cpu().numpy()
        if k % 20 == 0:
            print(f"  {k}/{len(sel)}  {str(times[j])[:10]}", flush=True)

    precip = tf.inverse(out)                     # back to mm/day
    precip = np.where(ds.valid[None, None] > 0, precip, np.nan)

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
