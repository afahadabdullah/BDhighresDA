"""Training dataset backed by the packed Zarr store built by ``04_regrid_and_pack.py``.

Store layout (all on the 0.05 deg target grid, latitude ascending)::

    <root>.zarr
      time      (T,)              datetime64[ns]
      lat       (H,)  lon (W,)
      target    (T, H, W)         CHIRPS daily precip, mm/day, NaN over ocean
      cond      (T, Ccond, H, W)  ERA5 predictors only, already regridded
      static    (Cstat, H, W)     orography, land-sea mask, lat/lon encodings
      valid     (H, W)            1 where CHIRPS has data (land)

IMERG and gauges are deliberately absent. They are observations read by the
assimilation workflow, not part of the prior-training dataset.

Random-crop augmentation: because we only have ~16k daily fields, we train on
random 128x128 crops of the 256x256 ``wide`` domain.  Crops preserve geography
(no flips -- orography would be destroyed) and the absolute position is fed to
the network through the static positional-encoding channels, so the model can
still learn location-specific climatology.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..transforms import CondTransform, PrecipTransform, ResidualSpec


@dataclass
class DatasetConfig:
    root: str
    crop: int = 128
    random_crop: bool = True
    crop_origin: tuple[int, int] | None = None  # (row, column) for fixed crops
    years: tuple[int, int] | None = None      # inclusive
    seasonal_encoding: bool = True
    era5_member: int | None = None   # ERA5-EDA member index, or None for the
                                     # deterministic HRES analysis
    cond_channels: tuple[str, ...] | None = None   # subset to keep, in order
    min_valid_fraction: float = 0.3  # reject random crops with less land than
                                     # this (see _crop_box)
    max_crop_tries: int = 32


class PrecipDataset(Dataset):
    def __init__(
        self,
        cfg: DatasetConfig,
        transform: PrecipTransform,
        cond_mean: np.ndarray | None = None,
        cond_std: np.ndarray | None = None,
        split_index: np.ndarray | None = None,
        store=None,
        cond_transform: CondTransform | None = None,
        residual: ResidualSpec | None = None,
        climatology: np.ndarray | None = None,
    ):
        self.cfg = cfg
        if store is not None:
            # any mapping of name -> array-like; used by the smoke test and by
            # callers that already hold an open store
            self.z = store
        else:
            import zarr

            self.z = zarr.open(str(Path(cfg.root)), mode="r")
        self.time = np.asarray(self.z["time"][:], dtype="datetime64[ns]")
        self.transform = transform

        idx = np.arange(len(self.time)) if split_index is None else np.asarray(split_index)
        if cfg.years is not None:
            yrs = self.time.astype("datetime64[Y]").astype(int) + 1970
            idx = idx[(yrs[idx] >= cfg.years[0]) & (yrs[idx] <= cfg.years[1])]
        self.index = idx

        self.cond_mean = cond_mean
        self.cond_std = cond_std
        self.valid = np.asarray(self.z["valid"][:], dtype=np.float32)
        self.static = np.asarray(self.z["static"][:], dtype=np.float32)
        self.H, self.W = self.valid.shape
        self.n_cond = self.z["cond"].shape[1]
        # Optional channel subset.  The prior is meant to see the synoptic
        # situation, not a precipitation forecast, so era5_tp is dropped from the
        # conditioning stack while the dynamical channels stay.
        self.all_cond_channels = [
            str(name) for name in self.z.attrs.get("cond_channels", [])
        ]
        if cfg.cond_channels:
            missing = set(cfg.cond_channels) - set(self.all_cond_channels)
            if missing:
                raise ValueError(
                    f"requested conditioning channels not in the store: "
                    f"{sorted(missing)}; available: {self.all_cond_channels}"
                )
            self.cond_index = np.array(
                [self.all_cond_channels.index(n) for n in cfg.cond_channels]
            )
            self.cond_channels = list(cfg.cond_channels)
        else:
            self.cond_index = np.arange(self.n_cond)
            self.cond_channels = list(self.all_cond_channels)
        self.n_cond = len(self.cond_index)

        self.cond_transform = cond_transform or CondTransform.identity(self.n_cond)
        self.residual = residual or ResidualSpec()
        # (366, H, W) per-pixel day-of-year CHIRPS climatology, mm/day.
        self.climatology = climatology
        if self.residual.enabled and self.residual.base == "climatology":
            if climatology is None:
                raise ValueError(
                    "residual.base is 'climatology' but no climatology array was "
                    "supplied; run 06_compute_stats.py --residual-base climatology"
                )

        # Zero is not "no rain" in transformed space: forward(0 mm) is -mu/sd.
        # Filling masked cells with a literal 0.0 would encode a moderate rain
        # rate over the Bay of Bengal, which the global attention blocks then mix
        # into the land field (docs/DIAGNOSIS_epoch119.md item 5).
        self.mask_fill = float(np.asarray(self.transform.forward(np.float32(0.0))))
        if self.residual.enabled:
            # In residual mode the masked value means "no correction to ERA5",
            # which is both finite and the least informative thing to say.
            self.mask_fill = float(self.residual.fill)

    def __len__(self) -> int:
        return len(self.index)

    def _crop_box(self, rng: np.random.Generator):
        c = self.cfg.crop
        if c >= self.H:
            return 0, 0, self.H, self.W
        if self.cfg.random_crop:
            if self.cfg.crop_origin is not None:
                raise ValueError("crop_origin cannot be combined with random_crop=True")
            # The wide domain reaches down to 16 N, so its southern third is open
            # Bay of Bengal where CHIRPS is entirely absent.  A crop landing there
            # has a near-empty loss mask and contributes essentially no gradient
            # while still costing a full forward/backward.  Resample until the
            # crop carries enough land (docs/DIAGNOSIS_epoch119.md item 6).
            best = None
            for _ in range(max(1, self.cfg.max_crop_tries)):
                r0 = int(rng.integers(0, self.H - c + 1))
                c0 = int(rng.integers(0, self.W - c + 1))
                fraction = float(
                    self.valid[r0 : r0 + c, c0 : c0 + c].mean()
                )
                if best is None or fraction > best[0]:
                    best = (fraction, r0, c0)
                if fraction >= self.cfg.min_valid_fraction:
                    break
            _, r0, c0 = best
        elif self.cfg.crop_origin is not None:
            r0, c0 = self.cfg.crop_origin
            if r0 < 0 or c0 < 0 or r0 + c > self.H or c0 + c > self.W:
                raise ValueError(
                    f"fixed crop {(r0, c0, c, c)} is outside "
                    f"the dataset grid {(self.H, self.W)}"
                )
        else:
            r0 = (self.H - c) // 2
            c0 = (self.W - c) // 2
        return r0, c0, c, c

    def fixed_spatial_slices(self) -> tuple[slice, slice]:
        """Return the spatial slices used by a deterministic crop."""
        if self.cfg.random_crop:
            raise ValueError("random-crop datasets do not have fixed spatial slices")
        r0, c0, ch, cw = self._crop_box(np.random.default_rng(0))
        return slice(r0, r0 + ch), slice(c0, c0 + cw)

    @property
    def fixed_valid(self) -> np.ndarray:
        """Land-validity mask on the deterministic output crop."""
        return self.valid[self.fixed_spatial_slices()]

    def _seasonal(self, t: np.datetime64, h: int, w: int) -> np.ndarray:
        doy = (t.astype("datetime64[D]") - t.astype("datetime64[Y]")).astype(int)
        ang = 2 * np.pi * doy / 365.25
        return np.stack(
            [np.full((h, w), np.sin(ang), np.float32), np.full((h, w), np.cos(ang), np.float32)]
        )

    def _base_field(self, j: int, cond_raw: np.ndarray, sl) -> np.ndarray:
        """Raw mm/day field the residual is taken against."""
        if self.residual.base == "climatology":
            doy = (
                self.time[j].astype("datetime64[D]")
                - self.time[j].astype("datetime64[Y]")
            ).astype(int)
            return self.climatology[min(doy, self.climatology.shape[0] - 1)][sl]
        if self.residual.base == "era5_tp":
            # NOTE: indexes the FULL cond stack, so it still works when era5_tp
            # has been dropped from the conditioning channels the network sees.
            return cond_raw[self.residual.base_channel]
        return np.zeros(cond_raw.shape[1:], np.float32)

    def __getitem__(self, i: int):
        j = int(self.index[i])
        rng = np.random.default_rng(torch.initial_seed() % (2**31) + i)
        r0, c0, ch, cw = self._crop_box(rng)
        sl = (slice(r0, r0 + ch), slice(c0, c0 + cw))

        target = np.asarray(self.z["target"][j][sl], dtype=np.float32)
        cond_all = np.asarray(
            self.z["cond"][j][(slice(None), *sl)], dtype=np.float32
        )
        static = self.static[(slice(None), *sl)]
        valid = self.valid[sl]

        # NaNs (ocean / CHIRPS fill) are excluded by the mask, and filled with
        # the transform of 0 mm rather than a literal 0.0 -- see self.mask_fill.
        finite = np.isfinite(target)
        target = np.where(finite, target, 0.0)
        mask = (valid * finite).astype(np.float32)

        # The residual base, in the SAME transformed space as the target.
        base = self.transform.forward(
            np.clip(self._base_field(j, cond_all, sl), 0.0, None)
        ).astype(np.float32)

        x1 = np.where(
            mask > 0,
            self.residual.encode(self.transform.forward(target), base),
            self.mask_fill,
        )
        x1 = x1.astype(np.float32)[None]

        cond = cond_all[self.cond_index]
        cond = self.cond_transform.forward(cond, channel_axis=0)
        if self.cond_mean is not None:
            cond = (cond - self.cond_mean[:, None, None]) / self.cond_std[:, None, None]
        cond = np.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0)

        parts = [cond, static]
        if self.cfg.seasonal_encoding:
            parts.append(self._seasonal(self.time[j], ch, cw))
        cond_full = np.concatenate(parts, axis=0).astype(np.float32)

        return {
            "x1": torch.from_numpy(x1),
            "cond": torch.from_numpy(cond_full),
            "mask": torch.from_numpy(mask[None]),
            "time": torch.tensor(self.time[j].astype("datetime64[s]").astype(np.int64)),
            "target_mm": torch.from_numpy(target[None]),
            "crop": torch.tensor([r0, c0]),
            # Transformed ERA5 tp.  Needed to turn a residual prediction back
            # into precipitation; harmless (and still useful as a baseline) when
            # the residual parameterisation is off.
            "base": torch.from_numpy(base[None]),
        }

    @property
    def total_cond_channels(self) -> int:
        return self.n_cond + self.static.shape[0] + (2 if self.cfg.seasonal_encoding else 0)


def year_split(time: np.ndarray, train: tuple[int, int], val: tuple[int, int], test: tuple[int, int]):
    yrs = time.astype("datetime64[Y]").astype(int) + 1970
    m = lambda a, b: np.where((yrs >= a) & (yrs <= b))[0]  # noqa: E731
    return m(*train), m(*val), m(*test)
