#!/usr/bin/env python
"""Observing-system simulation experiment: how much does assimilation buy?

CHIRPS is treated as the nature run.  Pseudo-gauges sample it at station
locations, observation error is added, a subset is assimilated, and the analysis
is scored against the truth.  Because the true full field is known, this answers
questions that real gauges never can: how much skill the DA adds, how that scales
with network density, and whether the guidance hyperparameters are set sensibly.

THREE SCOPES, WHICH ANSWER DIFFERENT QUESTIONS
    assimilated  Does the analysis actually move to the observations it was
                 given?  This is a FIT diagnostic, not a skill one -- it is
                 circular as a measure of accuracy, but it is the first thing to
                 look at when a DA system underperforms.  Reported alongside the
                 assumed observation error: if the analysis sits much further
                 from the observations than sigma, the guidance is too weak
                 (gamma too large, or R too loose); much closer, and it is
                 over-fitting noise it should be smoothing.

    withheld     Stations excluded from the assimilation.  This is the honest
                 skill number and the headline.

    full field   Every land cell.  Secondary, but informative: only a few hundred
                 of ~13.5k cells are ever constrained, so most of the domain is
                 genuinely out of sample even though the observations came from
                 CHIRPS.

WHAT IS MEASURED
    background   ERA5-conditioned generation, no observations   (the control)
    analysis     the same, guided by the assimilated pseudo-gauges
    improvement  the reduction in error from one to the other

    Reporting both, on identical days with identical seeds, is the point.  An
    analysis RMSE in isolation says nothing; the drop from the background is the
    quantity that measures the assimilation.

OBSERVATION ERROR
    Added in TRANSFORMED space with standard deviation sqrt(sigma_obs^2 +
    representativeness^2), which is exactly what ``build_R`` tells the likelihood
    to expect.  Perturbing in mm and then transforming would make the assumed R
    wrong and would quietly mis-tune everything downstream.

    ``--obs-error perfect`` sets that to ~0.  It is not realistic; it bounds what
    the DA could achieve with flawless gauges, which is what makes a
    disappointing realistic result interpretable.

    python scripts/10_osse.py --ckpt runs/prior_h100_v5/best.pt \
        --networks 10,25,50,100,200,bmd --days 20 --members 16
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.da import (  # noqa: E402
    BilinearObsOperator,
    GuidanceConfig,
    SamplerConfig,
    StationSet,
    build_R,
    perturb_observations,
)
from bdhires.da.sampler import assimilate as run_assim  # noqa: E402
from bdhires.data import DatasetConfig, PrecipDataset, load_stations  # noqa: E402
from bdhires.eval import crps_ensemble  # noqa: E402
from bdhires.grids import WIDE, crop_offsets, get_grid  # noqa: E402
from bdhires.models import RectifiedFlow, UNet, select_weights  # noqa: E402
from bdhires.transforms import (  # noqa: E402
    CondTransform,
    PrecipTransform,
    ResidualSpec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/da.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument(
        "--networks",
        default="10,25,50,100,200,bmd",
        help="comma-separated station counts, plus 'bmd' for the real geometry",
    )
    parser.add_argument(
        "--obs-error",
        default="realistic,perfect",
        help="comma-separated: 'realistic' uses da.yaml, 'perfect' uses ~0",
    )
    parser.add_argument(
        "--withhold",
        type=float,
        default=0.4,
        help="fraction of stations held out of the assimilation for scoring",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="sweep gamma / sigma_obs / prior_temperature instead of network size",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-figure", default="data/processed/osse.png")
    parser.add_argument("--out-report", default="data/processed/osse.json")
    parser.add_argument(
        "--dump",
        default=None,
        help="write the raw ensembles for ONE configuration to this .npz so "
             "scripts/11_da_diagnostics.py can plot the verification suite "
             "without re-running the sampling",
    )
    parser.add_argument(
        "--dump-network",
        default=None,
        help="which network to dump (default: the largest synthetic one)",
    )
    parser.add_argument("--dump-obs-error", default="realistic")
    return parser.parse_args()


def synthetic_network(
    n: int, grid, valid: np.ndarray, rng: np.random.Generator
) -> StationSet:
    """Random station locations over land, at least a cell from the boundary."""
    land = np.argwhere(valid)
    margin = 2
    inside = land[
        (land[:, 0] >= margin)
        & (land[:, 0] < valid.shape[0] - margin)
        & (land[:, 1] >= margin)
        & (land[:, 1] < valid.shape[1] - margin)
    ]
    if n > len(inside):
        raise ValueError(f"cannot place {n} stations in {len(inside)} land cells")
    picked = inside[rng.choice(len(inside), size=n, replace=False)]
    # jitter within the cell so stations are not all exactly on grid centres
    lat = grid.lat[picked[:, 0]] + (rng.random(n) - 0.5) * grid.res
    lon = grid.lon[picked[:, 1]] + (rng.random(n) - 0.5) * grid.res
    return StationSet(lat=lat, lon=lon, ids=np.arange(n))


def sample_at_stations(field_mm: np.ndarray, grid, stations: StationSet) -> np.ndarray:
    """Bilinear sample of (N, H, W) mm fields at station points -> (N, S)."""
    operator = BilinearObsOperator(grid, stations.lat, stations.lon)
    tensor = torch.from_numpy(np.nan_to_num(field_mm, nan=0.0)).float()
    if tensor.ndim == 2:
        tensor = tensor[None]
    return operator(tensor[:, None])[:, 0].numpy()


def score(members_mm: np.ndarray, truth: np.ndarray) -> dict:
    """Ensemble scores for (M, ...) members against (...) truth."""
    finite = np.isfinite(truth)
    if not finite.any():
        return {}
    ensemble = members_mm[:, finite]
    observed = truth[finite]
    mean = ensemble.mean(axis=0)
    spread = ensemble.std(axis=0, ddof=1)
    difference = mean - observed
    correlation = (
        float(np.corrcoef(mean, observed)[0, 1])
        if mean.std() > 0 and observed.std() > 0
        else float("nan")
    )
    low, high = np.quantile(ensemble, [0.05, 0.95], axis=0)
    return {
        "rmse_mm": float(np.sqrt(np.mean(difference**2))),
        "mae_mm": float(np.mean(np.abs(difference))),
        "bias_mm": float(np.mean(difference)),
        "crps_mm": float(crps_ensemble(ensemble, observed)),
        "correlation": correlation,
        "spread_mm": float(spread.mean()),
        "coverage_90": float(np.mean((observed >= low) & (observed <= high))),
        "n": int(finite.sum()),
    }


def aggregate(per_day: list[dict]) -> dict:
    """Mean over days of each metric, ignoring days that produced nothing."""
    keys = {k for d in per_day for k in d}
    out = {}
    for key in sorted(keys):
        values = [d[key] for d in per_day if key in d and np.isfinite(d[key])]
        out[key] = float(np.mean(values)) if values else float("nan")
    return out


def improvement(background: dict, analysis: dict, key: str) -> float:
    """Percent reduction in an error metric, positive = analysis is better."""
    b, a = background.get(key), analysis.get(key)
    if b is None or a is None or not np.isfinite(b) or b == 0:
        return float("nan")
    return float(100.0 * (b - a) / b)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    stats = json.loads(Path(config["data"]["stats"]).read_text())
    transform = PrecipTransform.from_dict(stats["precip_transform"])
    residual = ResidualSpec.from_stats(stats)
    grid = get_grid(config["data"]["grid"])

    dataset = PrecipDataset(
        DatasetConfig(
            root=config["data"]["zarr"],
            crop=grid.nlon,
            random_crop=False,
            crop_origin=crop_offsets(WIDE, grid),
        ),
        transform,
        cond_mean=np.asarray(stats["cond_mean"], np.float32),
        cond_std=np.asarray(stats["cond_std"], np.float32),
        cond_transform=CondTransform.from_stats(stats),
        residual=residual,
    )
    valid = dataset.fixed_valid > 0
    slices = dataset.fixed_spatial_slices()

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    training_config = checkpoint["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        in_channels=1,
        cond_channels=dataset.total_cond_channels,
        out_channels=1,
        image_size=grid.nlon,
        **training_config["model"],
    )
    model.load_state_dict(select_weights(checkpoint), strict=True)
    model = model.to(device).eval()
    flow = RectifiedFlow()
    mask = torch.from_numpy(valid.astype(np.float32)[None, None]).to(device)

    # -- days -----------------------------------------------------------------
    times = dataset.time
    test_years = training_config["data"]["years"]["test"]
    start = np.datetime64(args.start or f"{test_years[0]}-01-01")
    end = np.datetime64(args.end or f"{test_years[1]}-12-31")
    eligible = np.where((times >= start) & (times <= end))[0]
    if args.month:
        months = times[eligible].astype("datetime64[M]").astype(int) % 12 + 1
        eligible = eligible[months == args.month]
    if not len(eligible):
        raise ValueError("no days match the requested window")
    days = eligible[
        np.linspace(0, len(eligible) - 1, min(args.days, len(eligible))).astype(int)
    ]

    # -- station networks -----------------------------------------------------
    rng = np.random.default_rng(args.seed)
    networks: dict[str, StationSet] = {}
    for token in args.networks.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "bmd":
            csv = config["observations"]["gauges"]["csv"]
            if not Path(csv).is_file():
                print(f"[osse] skipping 'bmd': {csv} not found")
                continue
            bmd, _ = load_stations(csv, times, grid=grid, min_coverage=0.0)
            networks[f"bmd ({len(bmd)})"] = bmd
        else:
            n = int(token)
            networks[f"{n}"] = synthetic_network(n, grid, valid, rng)
    if not networks:
        raise ValueError("no station networks were built")

    gauge_cfg = config["observations"]["gauges"]
    error_levels = {}
    for token in args.obs_error.split(","):
        token = token.strip().lower()
        if token == "realistic":
            error_levels["realistic"] = (
                float(gauge_cfg["sigma_obs"]),
                float(gauge_cfg["representativeness"]),
            )
        elif token == "perfect":
            error_levels["perfect"] = (1e-3, 0.0)
    if not error_levels:
        raise ValueError("no observation-error levels selected")

    base_sampler = replace(
        SamplerConfig(**config["sampler"]), mask_fill=dataset.mask_fill
    )
    base_guidance = GuidanceConfig(**config["guidance"])

    if args.tune:
        combinations = [
            {"gamma": g, "sigma_obs": s, "prior_temperature": t}
            for g, s, t in itertools.product(
                config["tuning"]["gamma"],
                config["tuning"]["sigma_obs"],
                config["tuning"]["prior_temperature"],
            )
        ]
        networks = {name: networks[name] for name in list(networks)[:1]}
        error_levels = {"realistic": error_levels.get("realistic", (0.1, 0.25))}
        print(f"[osse] tuning mode: {len(combinations)} combinations on "
              f"network '{list(networks)[0]}'")
    else:
        combinations = [{}]

    print(
        f"[osse] {len(days)} days x {args.members} members x "
        f"{len(networks)} networks x {len(error_levels)} error levels x "
        f"{len(combinations)} settings on {device}",
        flush=True,
    )

    # -- the background is identical for every configuration on a given day ----
    background_cache: dict[int, np.ndarray] = {}
    truth_cache: dict[int, np.ndarray] = {}
    for index in days:
        position = int(np.where(dataset.index == index)[0][0])
        item = dataset[position]
        base = item["base"][None].to(device)
        with torch.inference_mode():
            generated = run_assim(
                model,
                item["cond"][None].to(device),
                (args.members, 1, grid.nlat, grid.nlon),
                device,
                cfg=replace(base_sampler, seed=args.seed + int(index),
                            prior_temperature=1.0, n_corrections=0),
                flow=flow,
                mask=mask,
                to_precip=lambda x, b=base: residual.decode(x, b),
            )
        background_cache[int(index)] = transform.inverse(
            residual.decode(generated, base)[:, 0].float().cpu().numpy()
        )
        target = np.asarray(dataset.z["target"][int(index)][slices], dtype=np.float32)
        truth_cache[int(index)] = np.where(valid, target, np.nan)
    print(f"[osse] cached {len(background_cache)} background ensembles", flush=True)

    dump_target = args.dump_network
    if args.dump and dump_target is None:
        synthetic = [n for n in networks if n.replace(" ", "").isdigit()]
        dump_target = max(synthetic, key=int) if synthetic else list(networks)[0]
    dumped = None

    results = []
    for name, stations in networks.items():
        n_stations = len(stations)
        n_withhold = max(1, int(round(args.withhold * n_stations)))
        split_rng = np.random.default_rng(args.seed + n_stations)
        order = split_rng.permutation(n_stations)
        eval_idx = np.sort(order[:n_withhold])
        assim_idx = np.sort(order[n_withhold:])

        for error_name, (sigma, representativeness) in error_levels.items():
            for setting in combinations:
                sigma_used = setting.get("sigma_obs", sigma)
                guidance = replace(
                    base_guidance, gamma=setting.get("gamma", base_guidance.gamma)
                )
                sampler_cfg = replace(
                    base_sampler,
                    prior_temperature=setting.get(
                        "prior_temperature", base_sampler.prior_temperature
                    ),
                )
                R = build_R(
                    len(assim_idx), sigma_used, device=device,
                    representativeness=representativeness,
                )
                noise_sd = float(np.sqrt(sigma**2 + representativeness**2))
                operator = BilinearObsOperator(
                    grid, stations.lat[assim_idx], stations.lon[assim_idx]
                ).to(device)

                background_days, analysis_days = [], []
                field_background, field_analysis = [], []
                assim_background, assim_analysis = [], []
                fit_transformed = []
                collect = (
                    args.dump
                    and name == dump_target
                    and error_name == args.dump_obs_error
                    and not setting
                )
                store: dict[str, list] = {
                    k: [] for k in (
                        "background", "analysis", "truth",
                        "obs_transformed", "truth_at_stations",
                    )
                }
                for index in days:
                    index = int(index)
                    position = int(np.where(dataset.index == index)[0][0])
                    item = dataset[position]
                    base = item["base"][None].to(device)
                    truth = truth_cache[index]

                    # Observations: truth sampled at stations, error added in
                    # TRANSFORMED space so it matches what R claims.
                    truth_at_stations = sample_at_stations(truth, grid, stations)[0]
                    observation_rng = np.random.default_rng(args.seed + index)
                    y_transformed = transform.forward(
                        np.clip(truth_at_stations, 0.0, None)
                    ) + observation_rng.normal(0.0, noise_sd, n_stations)
                    y_assim = y_transformed[assim_idx].astype(np.float32)

                    perturbed = perturb_observations(
                        y_assim, R, args.members, seed=index
                    )
                    y_tensor = torch.from_numpy(
                        perturbed[:, None].astype(np.float32)
                    ).to(device)

                    # NOT under torch.inference_mode(): guidance differentiates
                    # the likelihood back through the network, and inference mode
                    # permanently marks its tensors as unusable by autograd --
                    # torch.enable_grad() inside it does not help.  The sampler
                    # already applies no_grad() where it is safe to.
                    generated = run_assim(
                        model,
                        item["cond"][None].to(device),
                        (args.members, 1, grid.nlat, grid.nlon),
                        device,
                        H=operator,
                        y=y_tensor,
                        R=R,
                        cfg=replace(sampler_cfg, seed=args.seed + index),
                        gcfg=guidance,
                        flow=flow,
                        mask=mask,
                        to_precip=lambda x, b=base: residual.decode(x, b),
                    ).detach()
                    analysis = transform.inverse(
                        residual.decode(generated, base)[:, 0].float().cpu().numpy()
                    )
                    analysis = np.where(valid[None], analysis, np.nan)
                    background = background_cache[index]

                    # HEADLINE: withheld stations only.
                    truth_eval = truth_at_stations[eval_idx]
                    background_days.append(
                        score(
                            sample_at_stations(background, grid, stations)[:, eval_idx],
                            truth_eval,
                        )
                    )
                    analysis_days.append(
                        score(
                            sample_at_stations(analysis, grid, stations)[:, eval_idx],
                            truth_eval,
                        )
                    )
                    # FIT: at the assimilated stations, against the observations
                    # the DA was actually handed (not against the truth).
                    analysis_at_assim = sample_at_stations(
                        analysis, grid, stations
                    )[:, assim_idx]
                    background_at_assim = sample_at_stations(
                        background, grid, stations
                    )[:, assim_idx]
                    assim_analysis.append(
                        score(analysis_at_assim, truth_at_stations[assim_idx])
                    )
                    assim_background.append(
                        score(background_at_assim, truth_at_stations[assim_idx])
                    )
                    # Distance from the ensemble mean to the observation, in the
                    # transformed units the likelihood works in, so it is directly
                    # comparable with the assumed observation-error sd.
                    fit_transformed.append(
                        float(
                            np.sqrt(
                                np.mean(
                                    (
                                        transform.forward(
                                            np.clip(analysis_at_assim.mean(axis=0), 0, None)
                                        )
                                        - y_assim
                                    )
                                    ** 2
                                )
                            )
                        )
                    )

                    # Secondary: the whole field.
                    field_background.append(score(background, truth))
                    field_analysis.append(score(analysis, truth))

                    if collect:
                        store["background"].append(background.astype(np.float32))
                        store["analysis"].append(analysis.astype(np.float32))
                        store["truth"].append(truth.astype(np.float32))
                        store["obs_transformed"].append(
                            y_transformed.astype(np.float32)
                        )
                        store["truth_at_stations"].append(
                            truth_at_stations.astype(np.float32)
                        )

                entry = {
                    "network": name,
                    "n_stations": n_stations,
                    "n_assimilated": int(len(assim_idx)),
                    "n_withheld": int(len(eval_idx)),
                    "obs_error": error_name,
                    "obs_noise_sd_transformed": noise_sd,
                    **{f"setting_{k}": v for k, v in setting.items()},
                    "withheld_background": aggregate(background_days),
                    "withheld_analysis": aggregate(analysis_days),
                    "assimilated_background": aggregate(assim_background),
                    "assimilated_analysis": aggregate(assim_analysis),
                    "field_background": aggregate(field_background),
                    "field_analysis": aggregate(field_analysis),
                    # Consistency check: analysis-minus-observation distance in
                    # transformed units against the sd the likelihood assumed.
                    "fit_rms_transformed": float(np.mean(fit_transformed)),
                    "assumed_obs_sd_transformed": noise_sd,
                    "fit_ratio": float(np.mean(fit_transformed) / noise_sd)
                    if noise_sd > 0
                    else float("nan"),
                }
                for metric in ("rmse_mm", "crps_mm", "mae_mm"):
                    for scope in ("withheld", "assimilated", "field"):
                        entry[f"{scope}_improvement_{metric}"] = improvement(
                            entry[f"{scope}_background"],
                            entry[f"{scope}_analysis"],
                            metric,
                        )
                if collect and store["truth"]:
                    dumped = dict(
                        background=np.stack(store["background"]),
                        analysis=np.stack(store["analysis"]),
                        truth=np.stack(store["truth"]),
                        obs_transformed=np.stack(store["obs_transformed"]),
                        truth_at_stations=np.stack(store["truth_at_stations"]),
                        station_lat=stations.lat,
                        station_lon=stations.lon,
                        assim_idx=assim_idx,
                        eval_idx=eval_idx,
                        valid=valid,
                        obs_noise_sd=np.float32(noise_sd),
                        grid_name=np.str_(grid.name),
                        network=np.str_(name),
                        obs_error=np.str_(error_name),
                        days=np.array(
                            [str(times[int(i)].astype("datetime64[D]")) for i in days]
                        ),
                        precip_transform=np.str_(
                            json.dumps(stats["precip_transform"])
                        ),
                    )

                results.append(entry)
                print(
                    f"  {name:>12s}  {error_name:<9s} "
                    + (f"{setting} " if setting else "")
                    + f"withheld CRPS {entry['withheld_background']['crps_mm']:6.2f} -> "
                    f"{entry['withheld_analysis']['crps_mm']:6.2f} "
                    f"({entry['withheld_improvement_crps_mm']:+5.1f}%)   "
                    f"assim {entry['assimilated_improvement_crps_mm']:+5.1f}%   "
                    f"field {entry['field_improvement_crps_mm']:+5.1f}%   "
                    f"fit {entry['fit_ratio']:.2f}x sigma",
                    flush=True,
                )

    report = {
        "checkpoint": str(args.ckpt),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "days": [str(times[int(i)].astype("datetime64[D]")) for i in days],
        "members": args.members,
        "withhold_fraction": args.withhold,
        "mode": "tuning" if args.tune else "network sweep",
        "note": (
            "Headline numbers are from WITHHELD stations. Observations are drawn "
            "from CHIRPS, so scores at assimilated locations are circular. "
            "Full-field scores are secondary but informative because only a few "
            "hundred of ~13.5k land cells are ever constrained."
        ),
        "results": results,
    }
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_report).write_text(json.dumps(report, indent=2) + "\n")

    if args.dump:
        if dumped is None:
            print(f"[osse] nothing dumped: no configuration matched "
                  f"network={dump_target!r} obs_error={args.dump_obs_error!r}")
        else:
            Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.dump, **dumped)
            size = Path(args.dump).stat().st_size / 2**20
            print(f"wrote {args.dump} ({size:.0f} MiB) "
                  f"-- network '{dumped['network']}', {dumped['obs_error']} obs")

    plot_results(results, args, Path(args.out_figure))
    print(f"\nwrote {args.out_figure}")
    print(f"wrote {args.out_report}")


def plot_results(results: list[dict], args, output: Path) -> None:
    if not results:
        return
    figure, axes = plt.subplots(2, 3, figsize=(18, 9.5), constrained_layout=True)
    colours = plt.get_cmap("tab10").colors
    levels = sorted({r["obs_error"] for r in results})

    panels = [
        ("crps_mm", "CRPS (mm day$^{-1}$)", "A.  Withheld-station CRPS", "withheld"),
        ("rmse_mm", "RMSE (mm day$^{-1}$)", "B.  Withheld-station RMSE", "withheld"),
        ("coverage_90", "Fraction", "C.  Withheld 90% coverage", "withheld"),
        ("crps_mm", "CRPS (mm day$^{-1}$)", "D.  Full-field CRPS (secondary)", "field"),
        ("spread_mm", "mm day$^{-1}$", "E.  Full-field ensemble spread", "field"),
    ]
    for axis, (metric, ylabel, title, scope) in zip(axes.ravel(), panels):
        for number, level in enumerate(levels):
            rows = [r for r in results if r["obs_error"] == level]
            rows.sort(key=lambda r: r["n_stations"])
            x = [r["n_stations"] for r in rows]
            axis.plot(
                x, [r[f"{scope}_background"].get(metric, np.nan) for r in rows],
                marker="s", ms=4, ls="--", color=colours[number],
                label=f"background ({level})", alpha=0.6,
            )
            axis.plot(
                x, [r[f"{scope}_analysis"].get(metric, np.nan) for r in rows],
                marker="o", ms=5, color=colours[number], label=f"analysis ({level})",
            )
        if metric == "coverage_90":
            axis.axhline(0.90, color="black", ls="--", lw=1.0, label="nominal 0.90")
        axis.set_xscale("log")
        axis.set_xlabel("Stations assimilated")
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=10.5)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, frameon=False)

    axis = axes.ravel()[5]
    for number, level in enumerate(levels):
        rows = [r for r in results if r["obs_error"] == level]
        rows.sort(key=lambda r: r["n_stations"])
        axis.plot(
            [r["n_stations"] for r in rows],
            [r["withheld_improvement_crps_mm"] for r in rows],
            marker="o", ms=5, color=colours[number], label=f"withheld ({level})",
        )
        axis.plot(
            [r["n_stations"] for r in rows],
            [r["field_improvement_crps_mm"] for r in rows],
            marker="s", ms=4, ls="--", color=colours[number],
            label=f"full field ({level})", alpha=0.6,
        )
    axis.axhline(0.0, color="black", lw=0.9)
    axis.set_xscale("log")
    axis.set_xlabel("Stations assimilated")
    axis.set_ylabel("CRPS reduction (%)")
    axis.set_title("F.  What the assimilation buys", fontsize=10.5)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, frameon=False)

    figure.suptitle(
        "BDhighresDA OSSE - pseudo-gauges drawn from CHIRPS, scored on WITHHELD stations\n"
        f"{args.ckpt}   |   {args.days} days   |   {args.members} members   |   "
        f"{args.withhold:.0%} of each network withheld from assimilation\n"
        "Dashed = background (no observations), solid = analysis. "
        "The gap between them IS the assimilation.",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=115)
    plt.close(figure)


if __name__ == "__main__":
    main()
