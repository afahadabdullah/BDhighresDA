"""Decide whether a low CHIRPS reference correlation is physics or plumbing.

Panel D of the v4 pilot compares every analysis against CHIRPS.  That number is
only interpretable next to a reference: two independent same-day products of the
same storm must agree substantially once both are reduced to a common support.
When CHIRPS agrees with CPC at r=0.26 and with IMERG at r=0.15, either

  * early-May convection over this domain really is that poorly constrained by
    a sparse gauge analysis and an IR/PMW retrieval, or
  * one of the three fields is misregistered -- shifted in time, flipped in
    latitude, or cropped from the wrong window.

Guessing between those from maps wastes days.  This script scores CHIRPS
against CPC and IMERG under every cheap misregistration hypothesis and prints
the ranked table.  If ``identity`` wins, the disagreement is real and the model
is being asked to reproduce a target that its own conditioning barely predicts.
If some transform wins by a wide margin, that transform names the bug.

Read from the sample store, because that is exactly what the evaluator reads:
context_chirps_mm, context_cpc_mm and context_imerg_mm were all written on the
identical canvas by script 60, so any disagreement here is not a regridding
artefact introduced by this diagnostic.

    python scripts/62_diagnose_target_alignment.py \\
      --sample-store data/processed/v4_da_test/<run>/v4_da_samples.zarr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.da import AreaWeightedBlockObsOperator  # noqa: E402
from bdhires.data import area_weighted_block_mean, resolve_archive_encoding  # noqa: E402


def mean_daily_correlation(left: np.ndarray, right: np.ndarray, keep: np.ndarray) -> float:
    """Mean over days of the spatial correlation on the common valid support."""
    daily = []
    for a, b in zip(left, right):
        a = a.reshape(-1)
        b = b.reshape(-1)
        selected = keep.reshape(-1) & np.isfinite(a) & np.isfinite(b)
        if selected.sum() < 3:
            continue
        if a[selected].std() <= 0.0 or b[selected].std() <= 0.0:
            continue
        daily.append(float(np.corrcoef(a[selected], b[selected])[0, 1]))
    return float(np.mean(daily)) if daily else float("nan")


def spatial_variants(field: np.ndarray) -> dict[str, np.ndarray]:
    """Cheap misregistration hypotheses that a coordinate check would miss."""
    variants = {
        "identity": field,
        "lat_flip": field[:, ::-1, :],
        "lon_flip": field[:, :, ::-1],
        "lat_lon_flip": field[:, ::-1, ::-1],
    }
    if field.shape[-2] == field.shape[-1]:
        variants["transpose"] = np.swapaxes(field, -1, -2)
    return {name: np.ascontiguousarray(value) for name, value in variants.items()}


def day_shifted(left: np.ndarray, right: np.ndarray, shift: int):
    """Align ``left`` at day t against ``right`` at day t+shift."""
    if shift == 0:
        return left, right
    if shift > 0:
        return left[:-shift], right[shift:]
    return left[-shift:], right[:shift]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-store", required=True)
    parser.add_argument("--max-day-shift", type=int, default=2)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    store = zarr.open_group(args.sample_store, mode="r")
    for name in ("context_chirps_mm", "context_cpc_mm", "valid", "cell_area"):
        if name not in store:
            raise ValueError(f"sample store lacks {name}; rerun script 60")
    encoding, schema = resolve_archive_encoding(
        {
            "schema": store.attrs.get("source_target_schema", store.attrs.get("schema")),
            "subgrid_encoding": store.attrs["subgrid_encoding"],
        }
    ) if "subgrid_encoding" in store.attrs else (None, None)
    factor = int(encoding.factor) if encoding is not None else 10

    chirps = np.asarray(store["context_chirps_mm"][:], np.float32)
    cpc = np.asarray(store["context_cpc_mm"][:], np.float32)
    valid = np.asarray(store["valid"][:], bool)
    area = np.asarray(store["cell_area"][:], np.float32)
    offset = int(store.attrs.get("condition_day_offset", 0))

    print(f"sample store  : {args.sample_store}")
    print(f"days          : {chirps.shape[0]}   fine grid {chirps.shape[1:]}")
    print(f"conditioning  : {offset:+d} day(s) relative to the scored date")
    print(f"target schema : {schema!r}")
    print()

    # CHIRPS on the coarse CPC support, so the two live on identical arrays.
    coarse_chirps, retained, _ = area_weighted_block_mean(
        torch.from_numpy(chirps)[:, None], torch.from_numpy(area),
        torch.from_numpy(valid), factor=factor, valid_area_threshold=0.0,
    )
    coarse_chirps = coarse_chirps[:, 0].numpy()
    coarse_keep = retained[0, 0].numpy().astype(bool)
    if coarse_chirps.shape[1:] != cpc.shape[1:]:
        raise ValueError(
            f"block-mean CHIRPS {coarse_chirps.shape[1:]} does not match stored CPC "
            f"{cpc.shape[1:]}; the canvas and the coarse grid disagree"
        )

    results = {"reference": "CPC 0.5 degree", "scores": {}}
    rows = []
    for shift in range(-args.max_day_shift, args.max_day_shift + 1):
        left, right = day_shifted(coarse_chirps, cpc, shift)
        if left.shape[0] < 2:
            continue
        for name, variant in spatial_variants(left).items():
            score = mean_daily_correlation(variant, right, coarse_keep)
            rows.append((score, name, shift))
            results["scores"][f"{name}@{shift:+d}"] = score

    rows.sort(key=lambda row: (-row[0] if np.isfinite(row[0]) else 1.0))
    print("CHIRPS vs CPC on the common 0.5-degree support")
    print(f"  {'transform':<14} {'day shift':>9}   r")
    for score, name, shift in rows[:10]:
        marker = "  <-- as evaluated" if (name == "identity" and shift == 0) else ""
        print(f"  {name:<14} {shift:>+9d}   {score:6.3f}{marker}")

    imerg_note = None
    if "context_imerg_mm" in store and "imerg_factor" in store.attrs:
        imerg = np.asarray(store["context_imerg_mm"][:], np.float32)
        crop = tuple(int(value) for value in store.attrs["imerg_canvas_crop"])
        operator = AreaWeightedBlockObsOperator(
            int(store.attrs["imerg_factor"]), area, valid=valid,
            min_valid_frac=0.5, crop=crop,
        )
        keep = operator.valid_mask().numpy().astype(bool)
        imerg_rows = []
        for shift in range(-args.max_day_shift, args.max_day_shift + 1):
            left, right = day_shifted(chirps, imerg, shift)
            if left.shape[0] < 2:
                continue
            for name, variant in spatial_variants(left).items():
                with torch.no_grad():
                    reduced = operator(
                        torch.from_numpy(np.ascontiguousarray(variant))[:, None]
                    )[:, 0].numpy()
                score = mean_daily_correlation(
                    reduced, right.reshape(right.shape[0], -1), keep
                )
                imerg_rows.append((score, name, shift))
                results["scores"][f"imerg:{name}@{shift:+d}"] = score
        imerg_rows.sort(key=lambda row: (-row[0] if np.isfinite(row[0]) else 1.0))
        print()
        print("CHIRPS vs IMERG on the 0.4-degree footprint support")
        print(f"  {'transform':<14} {'day shift':>9}   r")
        for score, name, shift in imerg_rows[:10]:
            marker = "  <-- as evaluated" if (name == "identity" and shift == 0) else ""
            print(f"  {name:<14} {shift:>+9d}   {score:6.3f}{marker}")
        imerg_note = imerg_rows[0]

    best = rows[0]
    baseline = results["scores"].get("identity@+0", float("nan"))
    print()
    if best[1] == "identity" and best[2] == 0:
        print(
            "VERDICT: identity wins.  The fields are registered correctly and the\n"
            "         low reference is a real property of this period: a sparse\n"
            "         gauge analysis and an IR product genuinely disagree about\n"
            "         where early-May convection fell.  Treat r as the achievable\n"
            "         ceiling, not as 1.0 -- a model at r=0.02 against a ceiling of\n"
            f"         {baseline:.2f} still has essentially no pattern skill, but the\n"
            "         gap to close is that ceiling, not perfect agreement."
        )
    else:
        print(
            f"VERDICT: '{best[1]}' at day shift {best[2]:+d} scores {best[0]:.3f} versus\n"
            f"         {baseline:.3f} as currently evaluated.  That is a registration\n"
            "         bug, not a model result.  Fix it before reading any panel."
        )
    if imerg_note is not None and not (imerg_note[1] == "identity" and imerg_note[2] == 0):
        print(
            f"         IMERG agrees: '{imerg_note[1]}' at {imerg_note[2]:+d} wins there too."
        )

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
