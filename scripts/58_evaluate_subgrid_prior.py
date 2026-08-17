#!/usr/bin/env python3
"""Evaluate V3-SG archives with subgrid-aware, member-wise diagnostics.

The sample store must contain one array per method with shape ``(time, member,
lat, lon)`` (a deterministic ``(time, lat, lon)`` array is also accepted).
The target store is the archive produced by script 56.  If latent arrays named
``<method>_coarse_state`` and ``<method>_allocation_state`` are present, the
script additionally reports the physical amount/allocation authority split
relative to ``background``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.da import authority_decomposition  # noqa: E402
from bdhires.data import (  # noqa: E402
    resolve_archive_encoding,
    SubgridEncoding,
    area_weighted_block_mean,
    encoding_metadata,
)
from bdhires.models import HierarchicalState  # noqa: E402


def _open(path: str):
    import zarr

    return zarr.open_group(path, mode="r")


def _take_time(array, indices: np.ndarray) -> np.ndarray:
    """Zarr-v2/v3-compatible orthogonal selection on the leading dimension."""
    try:
        return np.asarray(array.oindex[indices.tolist()])
    except (AttributeError, TypeError):
        return np.stack([np.asarray(array[int(index)]) for index in indices])


def _parse_crop(text: str | None, attrs) -> tuple[slice, slice]:
    value = text
    if value is None:
        stored = attrs.get("target_crop")
        if stored is None:
            return slice(None), slice(None)
        if len(stored) != 4:
            raise ValueError("sample target_crop attribute must be [r0,r1,c0,c1]")
        return slice(int(stored[0]), int(stored[1])), slice(int(stored[2]), int(stored[3]))
    try:
        row, column = value.split(",")
        r0, r1 = map(int, row.split(":"))
        c0, c1 = map(int, column.split(":"))
    except Exception as exc:
        raise ValueError("--target-crop must be r0:r1,c0:c1") from exc
    return slice(r0, r1), slice(c0, c1)


def _ensemble(values: np.ndarray) -> np.ndarray:
    if values.ndim == 3:
        values = values[:, None]
    if values.ndim != 4:
        raise ValueError("sample arrays must have shape (T,M,H,W) or (T,H,W)")
    return values.astype(np.float32)


def _block_anomaly(
    values: torch.Tensor, area: torch.Tensor, valid: torch.Tensor, factor: int
) -> torch.Tensor:
    # Treat time*member as one batch and remove each member's own block mean.
    time, members, height, width = values.shape
    flat = values.reshape(time * members, 1, height, width)
    mean, _, _ = area_weighted_block_mean(flat, area, valid, factor, 0.0)
    expanded = mean.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
    return (flat - expanded).reshape(time, members, height, width)


def _crps_ensemble(ensemble: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Gridpoint CRPS without constructing an MxM member-pair tensor."""
    members = ensemble.shape[1]
    first = (ensemble - truth[:, None]).abs().mean(dim=1)
    if members == 1:
        return first
    ordered, _ = ensemble.sort(dim=1)
    rank = torch.arange(1, members + 1, device=ensemble.device, dtype=ensemble.dtype)
    coefficient = (2.0 * rank - members - 1.0).view(1, members, 1, 1)
    pair_half = (ordered * coefficient).sum(dim=1) / (members * members)
    return first - pair_half


def _seam_index(values: np.ndarray, valid: np.ndarray, factor: int) -> float:
    mean = np.nanmean(values, axis=(0, 1))
    vertical = np.abs(np.diff(mean, axis=1))
    horizontal = np.abs(np.diff(mean, axis=0))
    vvalid = valid[:, 1:] & valid[:, :-1]
    hvalid = valid[1:] & valid[:-1]
    vseam = np.zeros(vertical.shape, bool)
    hseam = np.zeros(horizontal.shape, bool)
    vseam[:, factor - 1 :: factor] = True
    hseam[factor - 1 :: factor, :] = True
    seam = np.concatenate([vertical[vvalid & vseam], horizontal[hvalid & hseam]])
    interior = np.concatenate([vertical[vvalid & ~vseam], horizontal[hvalid & ~hseam]])
    return float(np.mean(seam) / max(float(np.mean(interior)), 1.0e-12))


class Accumulator:
    def __init__(self):
        self.count = 0
        self.grid_count = 0
        self.sum_x = self.sum_y = 0.0
        self.sum_x2 = self.sum_y2 = self.sum_xy = 0.0
        self.abs_error = self.square_error = self.bias = self.crps = 0.0
        self.anomaly_crps = self.anomaly_square = 0.0
        self.wet_x = self.wet_y = 0.0
        self.max_conservation_error = 0.0

    def update(self, ensemble, truth, anomaly, truth_anomaly, valid, coarse=None, area=None, factor=10):
        finite = valid[None, None] & torch.isfinite(ensemble) & torch.isfinite(truth[:, None])
        target = truth[:, None].expand_as(ensemble)
        x = ensemble[finite].double()
        y = target[finite].double()
        self.count += x.numel()
        self.grid_count += int(valid.sum()) * truth.shape[0]
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_x2 += float((x * x).sum())
        self.sum_y2 += float((y * y).sum())
        self.sum_xy += float((x * y).sum())
        difference = x - y
        self.abs_error += float(difference.abs().sum())
        self.square_error += float((difference * difference).sum())
        self.bias += float(difference.sum())
        crps = _crps_ensemble(ensemble, truth)
        self.crps += float(crps[:, valid].sum())
        anomaly_crps = _crps_ensemble(anomaly, truth_anomaly)
        self.anomaly_crps += float(anomaly_crps[:, valid].sum())
        anomaly_target = truth_anomaly[:, None].expand_as(anomaly)
        self.anomaly_square += float(
            ((anomaly - anomaly_target).square() * valid[None, None]).sum()
        )
        self.wet_x += float(((ensemble >= 0.1) & valid[None, None]).sum())
        self.wet_y += float(((truth >= 0.1) & valid[None]).sum()) * ensemble.shape[1]
        if coarse is not None:
            recovered, _, _ = area_weighted_block_mean(
                ensemble.reshape(-1, 1, *ensemble.shape[-2:]), area, valid, factor, 0.0
            )
            expected = coarse.reshape(-1, 1, *coarse.shape[-2:])
            self.max_conservation_error = max(
                self.max_conservation_error,
                float((recovered - expected).abs().max()),
            )

    def result(self):
        count = max(self.count, 1)
        covariance = self.sum_xy - self.sum_x * self.sum_y / count
        variance_x = self.sum_x2 - self.sum_x * self.sum_x / count
        variance_y = self.sum_y2 - self.sum_y * self.sum_y / count
        correlation = covariance / max(np.sqrt(max(variance_x, 0.0) * max(variance_y, 0.0)), 1.0e-12)
        return {
            "crps_mm_day": self.crps / max(self.grid_count, 1),
            "mae_mm_day": self.abs_error / count,
            "rmse_mm_day": np.sqrt(self.square_error / count),
            "bias_mm_day": self.bias / count,
            "pooled_member_correlation": correlation,
            "subgrid_anomaly_crps_mm_day": self.anomaly_crps / max(self.grid_count, 1),
            "subgrid_anomaly_rmse_mm_day": np.sqrt(self.anomaly_square / count),
            "wet_fraction": self.wet_x / count,
            "target_wet_fraction": self.wet_y / count,
            "max_conservation_error_mm_day": self.max_conservation_error,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-store", required=True)
    parser.add_argument("--sample-store", required=True)
    parser.add_argument("--methods", required=True, help="comma-separated Zarr array names")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chunk-days", type=int, default=8)
    parser.add_argument(
        "--target-crop",
        help="fine-grid slice r0:r1,c0:c1; defaults to sample-store target_crop attr",
    )
    args = parser.parse_args()
    target = _open(args.target_store)
    samples = _open(args.sample_store)
    if not target.attrs.get("complete", False):
        raise ValueError("target store is not marked complete")
    target_encoding, target_schema = resolve_archive_encoding(target.attrs)
    if not samples.attrs.get("complete", False):
        raise ValueError("sample store is not marked complete")
    if samples.attrs.get("schema") != "cpc_v3_hierarchical_samples_v3":
        raise ValueError("sample store was not produced by the audited V3 writer")
    if samples.attrs.get("archive_uses_likelihood_hard_decoder") is not True:
        raise ValueError("sample archive lacks a passing hard-decoder round-trip audit")
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    if not methods:
        raise ValueError("--methods must name at least one sample array")
    missing = [method for method in methods if method not in samples]
    if missing:
        raise ValueError(f"sample store is missing methods {missing}")
    target_time = np.asarray(target["time"][:], np.int64)
    sample_time = np.asarray(samples["time"][:], np.int64)
    if len(np.unique(sample_time)) != len(sample_time):
        raise ValueError("sample time axis contains duplicate dates")
    # The sample store's ``time`` is the OBSERVATION LABEL.  BMD accumulates to
    # the following morning and IMERG is aligned to that window, so a file
    # labelled D+1 constrains state date D.  Every sampled field -- and the
    # CHIRPS it must be scored against -- lives on the state date, so the target
    # lookup uses that, not the label.  Pairing on the label fetched the wrong
    # day's CHIRPS and drove every pattern score toward zero.
    if "state_date" in samples:
        state_time = np.asarray(samples["state_date"][:], np.int64)
    else:
        day_ns = np.timedelta64(1, "D").astype("timedelta64[ns]").astype(np.int64)
        state_time = sample_time + int(
            samples.attrs.get("condition_day_offset", 0)
        ) * day_ns
    if state_time.shape != sample_time.shape:
        raise ValueError("state_date and time axes have different lengths")
    target_lookup = {int(value): index for index, value in enumerate(target_time)}
    missing_times = [int(value) for value in state_time if int(value) not in target_lookup]
    if missing_times:
        missing_dates = np.asarray(missing_times, dtype="datetime64[ns]")
        raise ValueError(
            f"sample STATE dates are absent from target store: {missing_dates[:5]}"
        )
    target_index = np.asarray([target_lookup[int(value)] for value in state_time], np.int64)
    fine_slice = _parse_crop(args.target_crop, samples.attrs)
    valid_np = np.asarray(target["fine_valid"][:], bool)[fine_slice]
    area_np = np.asarray(target["cell_area"][:], np.float32)[fine_slice]
    target_lat = np.asarray(target["lat"][:], np.float32)[fine_slice[0]]
    target_lon = np.asarray(target["lon"][:], np.float32)[fine_slice[1]]
    valid = torch.from_numpy(valid_np)
    area = torch.from_numpy(area_np)
    if "subgrid_encoding" not in target.attrs:
        raise ValueError("target store lacks frozen subgrid_encoding metadata")
    if "subgrid_encoding" not in samples.attrs:
        raise ValueError("sample store lacks frozen subgrid_encoding metadata")
    # Resolved once against the target's schema.  A bare from_mapping here
    # would hand a legacy v4 archive the current smooth-base default and decode
    # it differently from the sampler that wrote it.
    encoding = target_encoding
    sample_encoding, _ = resolve_archive_encoding(
        {"schema": target_schema, "subgrid_encoding": samples.attrs["subgrid_encoding"]}
    )
    if encoding_metadata(sample_encoding) != encoding_metadata(encoding):
        raise ValueError("sample and target stores use different subgrid encodings")
    if not np.array_equal(np.asarray(samples["valid"][:], bool), valid_np):
        raise ValueError("sample and target stores use different fine validity masks")
    if not np.allclose(
        np.asarray(samples["cell_area"][:], np.float32), area_np, rtol=0.0, atol=0.0
    ):
        raise ValueError("sample and target stores use different fine-cell areas")
    if not np.array_equal(np.asarray(samples["lat"][:], np.float32), target_lat):
        raise ValueError("sample and target stores use different latitude coordinates")
    if not np.array_equal(np.asarray(samples["lon"][:], np.float32), target_lon):
        raise ValueError("sample and target stores use different longitude coordinates")
    accumulators = {method: Accumulator() for method in methods}
    seams = {method: [] for method in methods}

    for start in range(0, len(sample_time), args.chunk_days):
        stop = min(start + args.chunk_days, len(sample_time))
        truth = torch.from_numpy(
            _take_time(target["fine_mm"], target_index[start:stop]).astype(np.float32)
        )[(slice(None), *fine_slice)]
        truth_anomaly = _block_anomaly(truth[:, None], area, valid, encoding.factor)[:, 0]
        for method in methods:
            ensemble_np = _ensemble(np.asarray(samples[method][start:stop]))
            if ensemble_np.shape[-2:] != valid_np.shape:
                raise ValueError(
                    f"{method} has spatial shape {ensemble_np.shape[-2:]}, but the "
                    f"selected target crop has shape {valid_np.shape}"
                )
            ensemble = torch.from_numpy(ensemble_np)
            anomaly = _block_anomaly(ensemble, area, valid, encoding.factor)
            coarse_name = f"{method}_coarse_mm"
            coarse = (
                torch.from_numpy(np.asarray(samples[coarse_name][start:stop], np.float32))
                if coarse_name in samples else None
            )
            if coarse is not None:
                if coarse.ndim == 3:
                    coarse = coarse[:, None]
                if coarse.ndim != 4:
                    raise ValueError(
                        f"{coarse_name} must have shape (T,M,Hc,Wc) or (T,Hc,Wc)"
                    )
                if coarse.shape[1] == 1 and ensemble.shape[1] > 1:
                    coarse = coarse.expand(-1, ensemble.shape[1], -1, -1)
                if coarse.shape[1] != ensemble.shape[1]:
                    raise ValueError(f"{coarse_name} member count differs from {method}")
                expected_coarse = (
                    valid_np.shape[0] // encoding.factor,
                    valid_np.shape[1] // encoding.factor,
                )
                if coarse.shape[-2:] != expected_coarse:
                    raise ValueError(
                        f"{coarse_name} has spatial shape {coarse.shape[-2:]}, "
                        f"expected {expected_coarse}"
                    )
            accumulators[method].update(
                ensemble, truth, anomaly, truth_anomaly, valid,
                coarse=coarse, area=area, factor=encoding.factor,
            )
            seams[method].append(_seam_index(ensemble_np, valid_np, encoding.factor))
        print(f"evaluated {stop}/{len(sample_time)}", flush=True)

    results = {}
    for method in methods:
        results[method] = accumulators[method].result()
        results[method]["seam_index"] = float(np.mean(seams[method]))

    # Optional physical authority decomposition from archived latent states.
    if "background_coarse_state" in samples and "background_allocation_state" in samples:
        factor = encoding.factor
        if (
            (fine_slice[0].start or 0) % factor
            or (fine_slice[1].start or 0) % factor
            or (fine_slice[0].stop is not None and fine_slice[0].stop % factor)
            or (fine_slice[1].stop is not None and fine_slice[1].stop % factor)
        ):
            raise ValueError("authority evaluation target crop must close on CPC blocks")
        coarse_slice = (
            slice(
                (fine_slice[0].start or 0) // factor,
                None if fine_slice[0].stop is None else fine_slice[0].stop // factor,
            ),
            slice(
                (fine_slice[1].start or 0) // factor,
                None if fine_slice[1].stop is None else fine_slice[1].stop // factor,
            ),
        )
        coarse_valid = torch.from_numpy(
            np.asarray(target["coarse_valid"][:], bool)[coarse_slice]
        )[None, None]
        if not np.array_equal(
            np.asarray(samples["coarse_valid"][:], bool), coarse_valid[0, 0].numpy()
        ):
            raise ValueError("sample and target stores use different coarse validity masks")
        authority = {}
        for method in methods:
            if method == "background":
                continue
            coarse_name = f"{method}_coarse_state"
            allocation_name = f"{method}_allocation_state"
            if coarse_name not in samples or allocation_name not in samples:
                continue
            amount_total = allocation_total = residual_max = 0.0

            def latent(array, start, stop):
                value = torch.from_numpy(np.asarray(array[start:stop], np.float32))
                if value.ndim == 5:
                    value = value.reshape(-1, *value.shape[-3:])
                if value.ndim != 4:
                    raise ValueError("latent state arrays must be (T,M,C,H,W) or (T,C,H,W)")
                return value
            for start in range(0, len(sample_time), args.chunk_days):
                stop = min(start + args.chunk_days, len(sample_time))
                background = HierarchicalState(
                    latent(samples["background_coarse_state"], start, stop),
                    latent(samples["background_allocation_state"], start, stop),
                )
                analysis = HierarchicalState(
                    latent(samples[coarse_name], start, stop),
                    latent(samples[allocation_name], start, stop),
                )
                amount, allocation, residual = authority_decomposition(
                    background, analysis, coarse_valid, valid[None, None], area,
                    encoding,
                )
                amount_total += float(amount.abs().sum())
                allocation_total += float(allocation.abs().sum())
                residual_max = max(residual_max, float(residual.abs().max()))
            authority[method] = {
                "amount_share": amount_total / max(amount_total + allocation_total, 1.0e-12),
                "amount_abs_integral": amount_total,
                "allocation_abs_integral": allocation_total,
                "closure_max_abs_mm_day": residual_max,
            }
        if authority:
            results["authority"] = authority

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "cpc_v3_subgrid_evaluation_v1",
        "target_store": args.target_store,
        "sample_store": args.sample_store,
        "dates": [
            str(state_time[0].view("datetime64[ns]").astype("datetime64[D]")),
            str(state_time[-1].view("datetime64[ns]").astype("datetime64[D]")),
        ],
        "memberwise_subgrid_anomalies": True,
        "results": results,
    }
    (output / "v3_subgrid_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# V3-SG subgrid evaluation", "",
        "Subgrid anomalies remove each ensemble member's own area-weighted 0.5-degree mean.", "",
        "| Method | CRPS | anomaly CRPS | RMSE | anomaly RMSE | r | wet frac | seam | conservation max |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for method in methods:
        value = results[method]
        lines.append(
            f"| `{method}` | {value['crps_mm_day']:.3f} | "
            f"{value['subgrid_anomaly_crps_mm_day']:.3f} | {value['rmse_mm_day']:.3f} | "
            f"{value['subgrid_anomaly_rmse_mm_day']:.3f} | "
            f"{value['pooled_member_correlation']:.3f} | {value['wet_fraction']:.3f} | "
            f"{value['seam_index']:.3f} | {value['max_conservation_error_mm_day']:.2e} |"
        )
    (output / "v3_subgrid_metrics.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {output / 'v3_subgrid_metrics.json'}", flush=True)
    print(f"wrote {output / 'v3_subgrid_metrics.md'}", flush=True)


if __name__ == "__main__":
    main()
