#!/usr/bin/env python
"""Verification against withheld stations and against CHIRPS, plus baselines.

    # cross-validated station scores (3-fold rotation of the gauge network)
    python scripts/evaluate.py --config configs/da.yaml --ckpt runs/stageB/final.pt \
        --start 2021-01-01 --end 2023-12-31 --cv-folds 3 --out results/cv2021_2023.json

    # tune Gamma / sigma_obs on pseudo-observations before touching real gauges
    python scripts/evaluate.py --config configs/da.yaml --ckpt runs/stageB/final.pt \
        --start 2019-01-01 --end 2020-12-31 --tune --out results/tuning.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.da import BilinearObsOperator, GuidanceConfig, SamplerConfig, build_R, split_stations  # noqa: E402
from bdhires.da.sampler import assimilate as run_assim  # noqa: E402
from bdhires.data import DatasetConfig, PrecipDataset, load_stations  # noqa: E402
from bdhires.eval import crps_ensemble, fss_series, rank_histogram, spread_skill, summarize  # noqa: E402
from bdhires.grids import get_grid  # noqa: E402
from bdhires.models import RectifiedFlow, UNet  # noqa: E402
from bdhires.transforms import PrecipTransform  # noqa: E402

from assimilate import load_model  # noqa: E402


def station_values_from_field(field_mm, grid, lat, lon):
    """Sample gridded ensemble (N, H, W) at station points -> (N, S)."""
    Hop = BilinearObsOperator(grid, lat, lon)
    x = torch.from_numpy(np.nan_to_num(field_mm, nan=0.0)).float()[:, None]
    return Hop(x)[:, 0].numpy()


def run_case(model, ds, grid, tf, times, sel, Hop, y_all, R, scfg, gcfg, members, device, mask,
             cond_on=True):
    out = np.empty((len(sel), members, grid.nlat, grid.nlon), np.float32)
    for k, j in enumerate(sel):
        item = ds[int(j)]
        cond = item["cond"][None].to(device) if cond_on else None
        y = None
        if Hop is not None:
            y = torch.from_numpy(y_all[j][None, None]).to(device).expand(members, -1, -1)
        out[k] = run_assim(model, cond, (members, 1, grid.nlat, grid.nlon), device,
                           H=Hop, y=y, R=R, cfg=scfg, gcfg=gcfg, flow=RectifiedFlow(),
                           mask=mask).squeeze(1).cpu().numpy()
    return tf.inverse(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--cv-folds", type=int, default=3)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = get_grid(cfg["data"]["grid"])
    stats = json.loads(Path(cfg["data"]["stats"]).read_text())
    tf = PrecipTransform.from_dict(stats["precip_transform"])

    ds = PrecipDataset(
        DatasetConfig(root=cfg["data"]["zarr"], crop=grid.nlon, random_crop=False), tf,
        cond_mean=np.asarray(stats["cond_mean"], np.float32),
        cond_std=np.asarray(stats["cond_std"], np.float32),
    )
    times = ds.time
    sel = np.where((times >= np.datetime64(args.start)) & (times <= np.datetime64(args.end)))[0]
    if args.max_days:
        sel = sel[:: max(1, len(sel) // args.max_days)][: args.max_days]

    model, _ = load_model(args.ckpt, ds.total_cond_channels, grid.nlon, device)
    mask = torch.from_numpy(ds.valid[None, None]).to(device)

    ss, values = load_stations(cfg["stations"]["csv"], times, grid=grid,
                               min_coverage=cfg["stations"]["min_coverage"])
    y_all = tf.forward(values)
    y_all[~np.isfinite(values)] = np.nan

    scfg = SamplerConfig(**cfg["sampler"])
    results: dict = {"period": [args.start, args.end], "n_days": int(len(sel)),
                     "n_stations": int(len(ss))}

    # ------------------------------------------------------------ tuning
    if args.tune:
        grid_search = list(itertools.product(cfg["tuning"]["gamma"], cfg["tuning"]["sigma_obs"]))
        assim_idx, eval_idx = split_stations(ss, n_folds=3, seed=0)[0]
        Hop = BilinearObsOperator(grid, ss.lat[assim_idx], ss.lon[assim_idx]).to(device)
        best = None
        for gamma, sigma in grid_search:
            gcfg = GuidanceConfig(**{**cfg["guidance"], "gamma": gamma})
            R = build_R(len(assim_idx), sigma, device=device,
                        representativeness=cfg["stations"]["representativeness"])
            pred = run_case(model, ds, grid, tf, times, sel, Hop, y_all[:, assim_idx], R,
                            scfg, gcfg, args.members, device, mask)
            obs = values[sel][:, eval_idx]
            est = np.stack([station_values_from_field(pred[k], grid, ss.lat[eval_idx],
                                                      ss.lon[eval_idx])
                            for k in range(len(sel))])          # (T, N, S)
            sc = summarize(est.mean(axis=1), obs)
            sc["crps"] = crps_ensemble(np.moveaxis(est, 1, 0), obs)
            results.setdefault("tuning", []).append(
                dict(gamma=gamma, sigma_obs=sigma, **sc))
            print(f"gamma={gamma:g} sigma={sigma:g}  rmse={sc['rmse']:.3f} crps={sc['crps']:.3f}")
            if best is None or sc["crps"] < best[0]:
                best = (sc["crps"], gamma, sigma)
        results["best"] = dict(crps=best[0], gamma=best[1], sigma_obs=best[2])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2, default=float))
        print(json.dumps(results["best"], indent=2))
        return

    # -------------------------------------------- cross-validated evaluation
    gcfg = GuidanceConfig(**cfg["guidance"])
    per_fold = []
    for f, (assim_idx, eval_idx) in enumerate(split_stations(ss, n_folds=args.cv_folds, seed=0)):
        Hop = BilinearObsOperator(grid, ss.lat[assim_idx], ss.lon[assim_idx]).to(device)
        R = build_R(len(assim_idx), cfg["stations"]["sigma_obs"], device=device,
                    representativeness=cfg["stations"]["representativeness"])
        obs = values[sel][:, eval_idx]
        fold: dict = {"fold": f, "n_assim": int(len(assim_idx)), "n_eval": int(len(eval_idx))}

        for name, use_obs, use_cond in [("background", False, True),
                                        ("analysis", True, True),
                                        ("prior_sda", True, False)]:
            pred = run_case(model, ds, grid, tf, times, sel,
                            Hop if use_obs else None, y_all[:, assim_idx] if use_obs else None,
                            R if use_obs else None, scfg, gcfg, args.members, device, mask,
                            cond_on=use_cond)
            est = np.stack([station_values_from_field(pred[k], grid, ss.lat[eval_idx],
                                                      ss.lon[eval_idx]) for k in range(len(sel))])
            sc = summarize(est.mean(axis=1), obs)
            sc["crps"] = crps_ensemble(np.moveaxis(est, 1, 0), obs)
            sk, sp = spread_skill(np.moveaxis(est, 1, 0), obs)
            sc.update(spread=sp, skill=sk, spread_skill_ratio=sp / sk if sk else np.nan)
            sc["rank_hist"] = rank_histogram(np.moveaxis(est, 1, 0), obs).tolist()
            # gridded verification against CHIRPS (the training target)
            truth = np.stack([np.asarray(ds.z["target"][int(j)]) for j in sel])
            sc["fss"] = {f"{t}mm_{w}px": v for (t, w), v in
                         fss_series(pred.mean(axis=1), truth,
                                    cfg["verification"]["thresholds"],
                                    cfg["verification"]["windows"]).items()}
            fold[name] = sc
            print(f"fold {f} {name:11s} rmse={sc['rmse']:.3f} crps={sc['crps']:.3f} "
                  f"spread/skill={sc['spread_skill_ratio']:.2f}", flush=True)

        # ---- baselines evaluated at the same withheld stations -------------
        truth = np.stack([np.asarray(ds.z["target"][int(j)]) for j in sel])
        fold["baseline_chirps"] = summarize(
            station_values_from_field(truth, grid, ss.lat[eval_idx], ss.lon[eval_idx]), obs)
        icond = cfg["data"].get("imerg_cond_index")
        if icond is not None:
            imerg = np.stack([np.asarray(ds.z["cond"][int(j)][icond]) for j in sel])
            fold["baseline_imerg"] = summarize(
                station_values_from_field(imerg, grid, ss.lat[eval_idx], ss.lon[eval_idx]), obs)
        ecosnd = cfg["data"].get("era5_tp_cond_index")
        if ecosnd is not None:
            era = np.stack([np.asarray(ds.z["cond"][int(j)][ecosnd]) for j in sel])
            fold["baseline_era5"] = summarize(
                station_values_from_field(era, grid, ss.lat[eval_idx], ss.lon[eval_idx]), obs)
        per_fold.append(fold)

    results["folds"] = per_fold
    for name in ("background", "analysis", "prior_sda"):
        results.setdefault("mean", {})[name] = {
            k: float(np.mean([f[name][k] for f in per_fold]))
            for k in ("rmse", "mae", "bias", "crps", "spread_skill_ratio")
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results["mean"], indent=2))


if __name__ == "__main__":
    main()
