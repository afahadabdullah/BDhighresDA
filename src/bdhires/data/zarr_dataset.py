"""Training dataset backed by the packed Zarr store built by ``04_regrid_and_pack.py``.

Store layout (all on the 0.05 deg target grid, latitude ascending)::

    <root>.zarr
      time      (T,)              datetime64[ns]
      lat       (H,)  lon (W,)
      target    (T, H, W)         CHIRPS daily precip, mm/day, NaN over ocean
      cond      (T, Ccond, H, W)  ERA5 (+ IMERG) predictors, already regridded
      static    (Cstat, H, W)     orography, land-sea mask, lat/lon encodings
      valid     (H, W)            1 where CHIRPS has data (land)

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

from ..transforms import PrecipTransform


@dataclass
class DatasetConfig:
    root: str
    crop: int = 128
    random_crop: bool = True
    years: tuple[int, int] | None = None      # inclusive
    seasonal_encoding: bool = True
    # Channels of ``cond`` to zero out.  Set this to the IMERG index to train an
    # ERA5-only prior -- required when IMERG is assimilated as an OBSERVATION
    # rather than used as conditioning, otherwise the same information enters
    # both the prior and the likelihood and gets double-counted.
    zero_cond_channels: tuple[int, ...] = ()


class PrecipDataset(Dataset):
    def __init__(
        self,
        cfg: DatasetConfig,
        transform: PrecipTransform,
        cond_mean: np.ndarray | None = None,
        cond_std: np.ndarray | None = None,
        split_index: np.ndarray | None = None,
        store=None,
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

    def __len__(self) -> int:
        return len(self.index)

    def _crop_box(self, rng: np.random.Generator):
        c = self.cfg.crop
        if c >= self.H:
            return 0, 0, self.H, self.W
        if self.cfg.random_crop:
            r0 = int(rng.integers(0, self.H - c + 1))
            c0 = int(rng.integers(0, self.W - c + 1))
        else:
            r0 = (self.H - c) // 2
            c0 = (self.W - c) // 2
        return r0, c0, c, c

    def _seasonal(self, t: np.datetime64, h: int, w: int) -> np.ndarray:
        doy = (t.astype("datetime64[D]") - t.astype("datetime64[Y]")).astype(int)
        ang = 2 * np.pi * doy / 365.25
        return np.stack(
            [np.full((h, w), np.sin(ang), np.float32), np.full((h, w), np.cos(ang), np.float32)]
        )

    def __getitem__(self, i: int):
        j = int(self.index[i])
        rng = np.random.default_rng(torch.initial_seed() % (2**31) + i)
        r0, c0, ch, cw = self._crop_box(rng)
        sl = (slice(r0, r0 + ch), slice(c0, c0 + cw))

        target = np.asarray(self.z["target"][j][sl], dtype=np.float32)
        cond = np.asarray(self.z["cond"][j][(slice(None), *sl)], dtype=np.float32)
        static = self.static[(slice(None), *sl)]
        valid = self.valid[sl]

        # NaNs (ocean / CHIRPS fill) -> 0 in transformed space, excluded by mask
        finite = np.isfinite(target)
        target = np.where(finite, target, 0.0)
        mask = (valid * finite).astype(np.float32)

        x1 = self.transform.forward(target)[None] * mask[None]

        if self.cond_mean is not None:
            cond = (cond - self.cond_mean[:, None, None]) / self.cond_std[:, None, None]
        cond = np.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0)
        for c in self.cfg.zero_cond_channels:
            cond[c] = 0.0

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
        }

    @property
    def total_cond_channels(self) -> int:
        return self.n_cond + self.static.shape[0] + (2 if self.cfg.seasonal_encoding else 0)


def year_split(time: np.ndarray, train: tuple[int, int], val: tuple[int, int], test: tuple[int, int]):
    yrs = time.astype("datetime64[Y]").astype(int) + 1970
    m = lambda a, b: np.where((yrs >= a) & (yrs <= b))[0]  # noqa: E731
    return m(*train), m(*val), m(*test)
