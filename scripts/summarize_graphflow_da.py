#!/usr/bin/env python3
"""Compare GraphFlow G0 with CPCv2 on matched May 1--10, 2022 DA folds.

The comparison deliberately separates prior skill, end-to-end analysis skill,
and the *additional* CRPS improvement supplied by the same frozen DA method.
All uncertainty is paired in circular day blocks, keeping stations from the
same weather day together. Positive reported differences favour GraphFlow.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_graphflow_da_summary", ROOT / "scripts/49_summarize_v2_gauge_sweep.py"
)
_summary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_summary)

CPC_BACKGROUND = "background"
GRAPH_BACKGROUND = "background"
DEFAULT_CPC_DA = "v2_simul_s04_ig010"
DEFAULT_GRAPH_DA = "graphflow_g0_multimesh_frozen_da"
DISTANCE_EDGES_KM = np.asarray([0, 25, 50, 100, 150, 250, np.inf], float)
# These fields were added after the original BMD-only CPCv2 May-2022 archive
# was written. Their defaults preserve the original single-network behaviour,
# so an absent key in that immutable archive is semantically identical to the
# explicit GraphFlow value. Do not add DA-tuning fields here.
LEGACY_SPEC_DEFAULTS = {
    "secondary_source_prefix": "BWDB_",
    "secondary_r_multiplier": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpc-dumps", nargs="+", required=True)
    parser.add_argument("--cpc-reports", nargs="+", required=True)
    parser.add_argument("--graph-dumps", nargs="+", required=True)
    parser.add_argument("--graph-reports", nargs="+", required=True)
    parser.add_argument("--cpc-method", default=DEFAULT_CPC_DA)
    parser.add_argument("--graph-method", default=DEFAULT_GRAPH_DA)
    parser.add_argument("--block-days", type=int, default=3)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=202209)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-markdown", required=True)
    parser.add_argument("--out-plot", required=True)
    return parser.parse_args()


def _same(first: np.ndarray, second: np.ndarray, label: str, atol: float = 0.0) -> None:
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape:
        raise ValueError(f"{label}: shapes differ: {first.shape} versus {second.shape}")
    if np.issubdtype(first.dtype, np.number) and np.issubdtype(second.dtype, np.number):
        equal = np.allclose(first, second, rtol=0.0, atol=atol, equal_nan=True)
    else:
        equal = np.array_equal(first.astype(str), second.astype(str))
    if not equal:
        raise ValueError(f"{label}: paired CPCv2 and GraphFlow inputs differ")


def _method_spec(report: dict, method: str) -> dict:
    spec = dict(report["variants"][method]["spec"])
    spec.pop("name", None)
    spec.pop("note", None)
    for key, value in LEGACY_SPEC_DEFAULTS.items():
        spec.setdefault(key, value)
    return spec


def validate_pairing(cpc: list[dict], graph: list[dict], cpc_method: str, graph_method: str) -> None:
    if len(cpc) != 5 or len(graph) != 5:
        raise ValueError("the comparison requires five CPCv2 and five GraphFlow folds")
    immutable_scope = (
        "start", "end", "n_days", "members", "background_day_offset", "seed",
        "holdout_folds", "holdout_fold", "precip_transform", "config_overrides",
    )
    identical_arrays = (
        "times", "station_ids", "eval_idx", "assim_idx", "station_lat",
        "station_lon", "grid_lat", "grid_lon", "gauge_mm", "raw_imerg_mm",
        "condition", "chirps", "valid",
    )
    for cpc_fold, graph_fold in zip(cpc, graph):
        if cpc_fold["fold"] != graph_fold["fold"]:
            raise ValueError("CPCv2 and GraphFlow fold numbers do not align")
        cpc_scope = cpc_fold["report"]["scope"]
        graph_scope = graph_fold["report"]["scope"]
        for key in immutable_scope:
            if cpc_scope.get(key) != graph_scope.get(key):
                raise ValueError(
                    f"fold {cpc_fold['fold']} differs on {key}: "
                    f"{cpc_scope.get(key)!r} versus {graph_scope.get(key)!r}"
                )
        for key in identical_arrays:
            if key not in cpc_fold["dump"] or key not in graph_fold["dump"]:
                raise ValueError(f"paired dump is missing {key!r}")
            _same(
                cpc_fold["dump"][key], graph_fold["dump"][key],
                f"fold {cpc_fold['fold']} {key}", atol=1.0e-6,
            )
        cpc_spec = _method_spec(cpc_fold["report"], cpc_method)
        graph_spec = _method_spec(graph_fold["report"], graph_method)
        differences = {
            key: {"cpcv2": cpc_spec.get(key), "graphflow": graph_spec.get(key)}
            for key in sorted(set(cpc_spec) | set(graph_spec))
            if cpc_spec.get(key) != graph_spec.get(key)
        }
        if differences:
            raise ValueError(
                f"fold {cpc_fold['fold']}: CPCv2 and GraphFlow DA settings differ: "
                f"{json.dumps(differences, sort_keys=True)}"
            )


def haversine_km(lat: np.ndarray, lon: np.ndarray, other_lat: np.ndarray, other_lon: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(np.asarray(lat, float))[:, None]
    lon = np.deg2rad(np.asarray(lon, float))[:, None]
    other_lat = np.deg2rad(np.asarray(other_lat, float))[None]
    other_lon = np.deg2rad(np.asarray(other_lon, float))[None]
    dlat, dlon = other_lat - lat, other_lon - lon
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(other_lat) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def withheld_distances(folds: list[dict]) -> np.ndarray:
    blocks = []
    for item in folds:
        dump = item["dump"]
        eval_idx = np.asarray(dump["eval_idx"], int)
        assim_idx = np.asarray(dump["assim_idx"], int)
        distances = haversine_km(
            dump["station_lat"][eval_idx], dump["station_lon"][eval_idx],
            dump["station_lat"][assim_idx], dump["station_lon"][assim_idx],
        )
        blocks.append(np.min(distances, axis=1))
    return np.concatenate(blocks)


def distance_skill(distance: np.ndarray, cpc_gain: np.ndarray, graph_gain: np.ndarray) -> list[dict]:
    output = []
    for lower, upper in zip(DISTANCE_EDGES_KM[:-1], DISTANCE_EDGES_KM[1:]):
        stations = (distance >= lower) & (distance < upper)
        cpc_values = cpc_gain[:, stations]
        graph_values = graph_gain[:, stations]
        finite = np.isfinite(cpc_values) & np.isfinite(graph_values)
        output.append({
            "lower_km": float(lower),
            "upper_km": None if np.isinf(upper) else float(upper),
            "stations": int(stations.sum()),
            "station_days": int(finite.sum()),
            "cpcv2_da_gain_crps": float(np.nanmean(cpc_values)) if finite.any() else None,
            "graphflow_da_gain_crps": float(np.nanmean(graph_values)) if finite.any() else None,
            "graphflow_extra_da_gain_crps": (
                float(np.nanmean(graph_values - cpc_values)) if finite.any() else None
            ),
        })
    return output


def locality_curve(folds: list[dict], method: str) -> dict:
    curves, edges = [], None
    for item in folds:
        locality = item["report"]["variants"][method]["increment_locality"]
        current_edges = np.asarray(locality["edges_km"], float)
        if edges is None:
            edges = current_edges
        elif not np.array_equal(edges, current_edges):
            raise ValueError("increment-locality bins differ across folds")
        curves.append([
            np.nan if value is None else float(value)
            for value in locality["mean_abs_increment_mm"]
        ])
    values = np.nanmean(np.asarray(curves, float), axis=0)
    return {"edges_km": edges.tolist(), "mean_abs_increment_mm": values.tolist()}


def paired(difference: np.ndarray, args: argparse.Namespace, seed_offset: int) -> dict:
    return _summary.circular_block_bootstrap(
        difference, args.block_days, args.n_resamples, args.seed + seed_offset
    )


def verdict(result: dict) -> str:
    if result["ci_low"] > 0:
        return "graphflow_better"
    if result["ci_high"] < 0:
        return "graphflow_worse"
    return "unresolved"


def fmt(result: dict) -> str:
    return f"{result['difference']:+.3f} [{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]"


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    cpc = _summary.validate_and_load(
        [Path(path) for path in args.cpc_dumps],
        [Path(path) for path in args.cpc_reports],
    )
    graph = _summary.validate_and_load(
        [Path(path) for path in args.graph_dumps],
        [Path(path) for path in args.graph_reports],
    )
    validate_pairing(cpc, graph, args.cpc_method, args.graph_method)

    cpc_bg_metrics, cpc_bg = _summary.pooled_variant(cpc, CPC_BACKGROUND)
    cpc_da_metrics, cpc_da = _summary.pooled_variant(cpc, args.cpc_method)
    graph_bg_metrics, graph_bg = _summary.pooled_variant(graph, GRAPH_BACKGROUND)
    graph_da_metrics, graph_da = _summary.pooled_variant(graph, args.graph_method)
    _same(cpc_bg["truth"], graph_bg["truth"], "pooled withheld observations", atol=1.0e-6)

    metrics = {
        "cpcv2_background": cpc_bg_metrics,
        "cpcv2_da": cpc_da_metrics,
        "graphflow_background": graph_bg_metrics,
        "graphflow_da": graph_da_metrics,
    }
    comparisons = {
        "graphflow_background_vs_cpcv2_background": paired(
            cpc_bg["crps"] - graph_bg["crps"], args, 0
        ),
        "graphflow_da_vs_cpcv2_da": paired(
            cpc_da["crps"] - graph_da["crps"], args, 10_000
        ),
        "graphflow_da_vs_graphflow_background": paired(
            graph_bg["crps"] - graph_da["crps"], args, 20_000
        ),
        "cpcv2_da_vs_cpcv2_background": paired(
            cpc_bg["crps"] - cpc_da["crps"], args, 30_000
        ),
        "graphflow_extra_da_gain_vs_cpcv2": paired(
            (graph_bg["crps"] - graph_da["crps"])
            - (cpc_bg["crps"] - cpc_da["crps"]),
            args, 40_000,
        ),
    }
    for result in comparisons.values():
        result["verdict"] = verdict(result)

    distance = withheld_distances(graph)
    cpc_gain = cpc_bg["crps"] - cpc_da["crps"]
    graph_gain = graph_bg["crps"] - graph_da["crps"]
    distance_results = distance_skill(distance, cpc_gain, graph_gain)
    locality = {
        "cpcv2": locality_curve(cpc, args.cpc_method),
        "graphflow": locality_curve(graph, args.graph_method),
    }

    scope = {
        "start": cpc[0]["report"]["scope"]["start"],
        "end": cpc[0]["report"]["scope"]["end"],
        "members": cpc[0]["report"]["scope"]["members"],
        "folds": 5,
        "stations": len(cpc[0]["dump"]["station_ids"]),
        "block_days": args.block_days,
        "n_resamples": args.n_resamples,
        "cpc_method": args.cpc_method,
        "graph_method": args.graph_method,
    }
    prior = comparisons["graphflow_background_vs_cpcv2_background"]
    final = comparisons["graphflow_da_vs_cpcv2_da"]
    extra = comparisons["graphflow_extra_da_gain_vs_cpcv2"]
    labels = {
        "cpcv2_background": "CPCv2 background",
        "cpcv2_da": "CPCv2 DA",
        "graphflow_background": "GraphFlow background",
        "graphflow_da": "GraphFlow DA",
    }
    order = list(labels)
    lines = [
        "# GraphFlow G0 versus CPCv2: matched DA test",
        "",
        f"- Period: **{scope['start']} through {scope['end']}**",
        f"- {scope['stations']} BMD stations, each withheld once in five spatial folds",
        f"- {scope['members']} ensemble members; paired {scope['block_days']}-day circular bootstrap",
        "- Positive paired CRPS differences favour GraphFlow",
        "",
        "| Case | CRPS | RMSE | MAE | Bias | Corr | Spread/RMSE | Cov90 |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for name in order:
        value = metrics[name]
        lines.append(
            f"| {labels[name]} | {value['crps']:.3f} | {value['rmse']:.3f} | "
            f"{value['mae']:.3f} | {value['bias']:+.3f} | {value['correlation']:.3f} | "
            f"{value['spread_skill']:.3f} | {value['coverage_90']:.3f} |"
        )
    lines += [
        "",
        "## Paired answers",
        "",
        f"- Better background: **{fmt(prior)} mm/day — {verdict(prior)}**.",
        f"- Better final analysis: **{fmt(final)} mm/day — {verdict(final)}**.",
        f"- Extra value extracted from the same observations: **{fmt(extra)} mm/day — {verdict(extra)}**.",
        "",
        "The third line is the clean DA attribution: it subtracts each model's own background "
        "before comparing models. A broader increment is not automatically better; useful "
        "propagation requires positive withheld-gauge DA gain away from assimilated stations.",
        "",
        "## DA gain by distance to the nearest assimilated gauge",
        "",
        "| Distance (km) | Stations | CPCv2 gain | GraphFlow gain | GraphFlow extra gain |",
        "|:--|--:|--:|--:|--:|",
    ]
    for item in distance_results:
        upper = "∞" if item["upper_km"] is None else f"{item['upper_km']:.0f}"
        values = [item["cpcv2_da_gain_crps"], item["graphflow_da_gain_crps"], item["graphflow_extra_da_gain_crps"]]
        rendered = ["—" if value is None else f"{value:+.3f}" for value in values]
        lines.append(
            f"| {item['lower_km']:.0f}–{upper} | {item['stations']} | "
            f"{rendered[0]} | {rendered[1]} | {rendered[2]} |"
        )

    output = _clean({
        "scope": scope,
        "sources": {
            "cpcv2": [item["dump_path"] for item in cpc],
            "graphflow": [item["dump_path"] for item in graph],
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "distance_to_assimilated_gauge": distance_results,
        "grid_increment_locality": locality,
        "interpretation": {
            "background": verdict(prior),
            "end_to_end_analysis": verdict(final),
            "additional_observation_value": verdict(extra),
        },
    })
    out_json, out_markdown, out_plot = map(
        Path, (args.out_json, args.out_markdown, args.out_plot)
    )
    for path in (out_json, out_markdown, out_plot):
        path.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    out_markdown.write_text("\n".join(lines) + "\n")

    colors = ["#7A5195", "#EF5675", "#2F4B7C", "#00A6A6"]
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    positions = np.arange(len(order))
    axes[0, 0].barh(positions, [metrics[name]["crps"] for name in order], color=colors)
    axes[0, 0].set_yticks(positions, [labels[name] for name in order]); axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("Fair CRPS (mm/day)"); axes[0, 0].set_title("A. Withheld-gauge skill")

    key_order = [
        "graphflow_background_vs_cpcv2_background",
        "graphflow_da_vs_cpcv2_da",
        "graphflow_extra_da_gain_vs_cpcv2",
    ]
    centres = np.asarray([comparisons[key]["difference"] for key in key_order])
    lows = np.asarray([comparisons[key]["ci_low"] for key in key_order])
    highs = np.asarray([comparisons[key]["ci_high"] for key in key_order])
    axes[0, 1].errorbar(centres, np.arange(3), xerr=np.vstack([centres - lows, highs - centres]), fmt="o", capsize=4)
    axes[0, 1].axvline(0, color="black", ls="--")
    axes[0, 1].set_yticks(np.arange(3), ["background", "final DA", "extra DA value"])
    axes[0, 1].invert_yaxis(); axes[0, 1].set_xlabel("CRPS gain favouring GraphFlow")
    axes[0, 1].set_title("B. Paired 95% intervals")

    distance_x, cpc_y, graph_y = [], [], []
    for item in distance_results:
        if item["stations"] and item["upper_km"] is not None:
            distance_x.append((item["lower_km"] + item["upper_km"]) / 2)
            cpc_y.append(item["cpcv2_da_gain_crps"])
            graph_y.append(item["graphflow_da_gain_crps"])
    axes[0, 2].plot(distance_x, cpc_y, "o-", label="CPCv2")
    axes[0, 2].plot(distance_x, graph_y, "o-", label="GraphFlow")
    axes[0, 2].axhline(0, color="black", ls="--"); axes[0, 2].legend()
    axes[0, 2].set_xlabel("Nearest assimilated gauge (km)"); axes[0, 2].set_ylabel("DA CRPS gain")
    axes[0, 2].set_title("C. Useful observation reach")

    for name, style in (("cpcv2", "o-"), ("graphflow", "o-")):
        curve = locality[name]
        edges = np.asarray(curve["edges_km"])
        x = (edges[:-2] + edges[1:-1]) / 2
        y = np.asarray(curve["mean_abs_increment_mm"])[:-1]
        axes[1, 0].plot(x, y, style, label=name)
    axes[1, 0].legend(); axes[1, 0].set_xlabel("Nearest assimilated gauge (km)")
    axes[1, 0].set_ylabel("Mean |analysis − background| (mm/day)")
    axes[1, 0].set_title("D. Grid increment propagation")

    axes[1, 1].barh(positions, [metrics[name]["bias"] for name in order], color=colors)
    axes[1, 1].axvline(0, color="black"); axes[1, 1].set_yticks(positions, [labels[name] for name in order])
    axes[1, 1].invert_yaxis(); axes[1, 1].set_xlabel("Bias (mm/day)"); axes[1, 1].set_title("E. Bias")

    width = 0.38
    axes[1, 2].barh(positions - width / 2, [metrics[name]["spread_skill"] for name in order], height=width, label="spread/RMSE")
    axes[1, 2].barh(positions + width / 2, [metrics[name]["coverage_90"] for name in order], height=width, label="90% coverage")
    axes[1, 2].axvline(0.9, color="black", ls=":"); axes[1, 2].set_yticks(positions, [labels[name] for name in order])
    axes[1, 2].invert_yaxis(); axes[1, 2].legend(); axes[1, 2].set_title("F. Ensemble reliability")
    figure.suptitle("GraphFlow G0 vs CPCv2 — May 1–10, 2022 matched DA")
    figure.savefig(out_plot, dpi=160)
    plt.close(figure)

    print("\n".join(lines))
    print(f"\n[done] wrote {out_json}, {out_markdown}, and {out_plot}")


if __name__ == "__main__":
    main()
