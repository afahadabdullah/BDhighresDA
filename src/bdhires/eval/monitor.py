"""In-training validation on real sampled fields.

The flow-matching validation loss is a poor selection signal: it is a masked MSE
against a random-``t`` velocity target, so it is dominated by the irreducible
noise of the ``t`` and ``x0`` draws.  Between epoch 60 and epoch 119 of the first
production run it moved less than the metric's own sampling noise, which made
"best.pt" close to arbitrary (docs/DIAGNOSIS_epoch119.md item 4).

This module fixes that by periodically running the *actual sampler* on a small,
fixed set of held-out days and scoring the resulting ensemble the way the model
will really be judged: CRPS, bias, spatial correlation, spread and interval
coverage in mm/day.

Because Bangladesh precipitation is overwhelmingly a monsoon phenomenon, the
default case selection is July: a wet extreme and a typical day, both chosen once
at construction so the same dates are re-sampled at every epoch and the resulting
curves are directly comparable.

Outputs, all under ``<out_dir>``:

* ``history.jsonl``      one row per evaluation, append-only
* ``epoch_XXXX.png``     map panel for that epoch
* ``progress.png``       metric-vs-epoch curves, rewritten each time

Design rule: monitoring must never kill a 250k-step run.  :meth:`run` catches its
own exceptions and returns ``None`` on failure.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class MonitorConfig:
    """When and how to run the sampled validation."""

    enabled: bool = True
    start_epoch: int = 10        # skip the early epochs; samples are noise there
    every: int = 5               # evaluate every N epochs after start_epoch
    members: int = 8             # ensemble size (cost scales linearly)
    n_steps: int = 30            # sampler steps; 30 is plenty for monitoring
    cfg_scale: float = 2.0       # match background_sampler in configs/da.yaml
    month: int = 7               # July: the heart of the BD monsoon
    quantiles: tuple[float, ...] = (0.5, 0.99)   # typical + wet extreme
    max_cases: int = 2
    save_maps: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "MonitorConfig":
        if not d:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in known}
        if "quantiles" in kwargs:
            kwargs["quantiles"] = tuple(kwargs["quantiles"])
        return cls(**kwargs)


@dataclass
class _Case:
    index: int
    date: str
    quantile: float
    domain_mean_mm: float
    target: np.ndarray = field(repr=False)
    era5: np.ndarray | None = field(default=None, repr=False)
    baseline: dict | None = None      # ERA5-vs-CHIRPS scores for this day

    @property
    def label(self) -> str:
        return f"{self.date}  (q{int(round(self.quantile * 100)):02d})"


class ValidationMonitor:
    """Sample a fixed set of held-out monsoon days and score them.

    ``dataset`` must be a fixed-crop :class:`~bdhires.data.PrecipDataset` on the
    production grid, restricted to the validation years.
    """

    def __init__(
        self,
        dataset,
        transform,
        device,
        out_dir: str | Path,
        cfg: MonitorConfig | None = None,
        era5_tp_index: int = 0,
        cond_transform=None,
        cond_mean: np.ndarray | None = None,
        cond_std: np.ndarray | None = None,
        extent: "tuple[float, float, float, float] | None" = None,
    ):
        self.extent = extent      # (lon_min, lon_max, lat_min, lat_max) for axes
        self.cfg = cfg or MonitorConfig()
        self.ds = dataset
        self.transform = transform
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.out_dir / "history.jsonl"
        self.era5_tp_index = era5_tp_index
        self.cond_transform = cond_transform
        self.cond_mean = cond_mean
        self.cond_std = cond_std

        self.valid = self.ds.fixed_valid > 0
        self.slices = self.ds.fixed_spatial_slices()
        self.cases = self._select_cases()
        self._attach_baselines()

    # -- case selection ----------------------------------------------------

    def _select_cases(self) -> list[_Case]:
        """Pick the same monsoon days every epoch, by domain-mean CHIRPS quantile."""
        time = self.ds.time[self.ds.index]
        months = time.astype("datetime64[M]").astype(int) % 12 + 1
        pool = np.where(months == self.cfg.month)[0]
        if not len(pool):     # no such month in the validation window
            pool = np.arange(len(self.ds.index))

        means = np.empty(len(pool), dtype=np.float64)
        for k, position in enumerate(pool):
            target = np.asarray(
                self.ds.z["target"][int(self.ds.index[position])][self.slices],
                dtype=np.float32,
            )
            means[k] = np.nanmean(np.where(self.valid, target, np.nan))

        cases: list[_Case] = []
        used: set[int] = set()
        for quantile in self.cfg.quantiles[: self.cfg.max_cases]:
            order = np.argsort(np.abs(means - float(np.quantile(means, quantile))))
            pick = next((int(p) for p in order if int(pool[p]) not in used), None)
            if pick is None:
                continue
            used.add(int(pool[pick]))
            index = int(self.ds.index[pool[pick]])
            target = np.asarray(
                self.ds.z["target"][index][self.slices], dtype=np.float32
            )
            cases.append(
                _Case(
                    index=index,
                    date=str(self.ds.time[index].astype("datetime64[D]")),
                    quantile=float(quantile),
                    domain_mean_mm=float(means[pick]),
                    target=np.where(self.valid, target, np.nan),
                )
            )
        return cases

    def _attach_baselines(self) -> None:
        """Score raw ERA5 against CHIRPS for each case, once.

        This is the bar the model has to clear.  The epoch-119 failure was a
        model whose ensemble mean correlated *worse* with CHIRPS than the ERA5
        field it was conditioned on, and that was invisible until someone
        compared two numbers by hand.  Drawing it as a reference line on every
        progress figure makes it impossible to miss.
        """
        for case in self.cases:
            try:
                item = self.ds[self._position_of(case.index)]
                case.era5 = self._era5_mm(item)
                keep = self.valid & np.isfinite(case.target) & np.isfinite(case.era5)
                predicted = case.era5[keep].astype(np.float64)
                observed = case.target[keep].astype(np.float64)
                difference = predicted - observed
                case.baseline = {
                    "rmse_mm": float(np.sqrt(np.mean(difference**2))),
                    # a deterministic forecast's CRPS is its MAE
                    "crps_mm": float(np.mean(np.abs(difference))),
                    "mae_mm": float(np.mean(np.abs(difference))),
                    "bias_mm": float(np.mean(difference)),
                    "spatial_correlation": (
                        float(np.corrcoef(predicted, observed)[0, 1])
                        if predicted.std() > 0 and observed.std() > 0
                        else float("nan")
                    ),
                }
            except Exception as exc:  # pragma: no cover - diagnostics only
                print(
                    f"[validation monitor] no ERA5 baseline for {case.date}: {exc!r}",
                    flush=True,
                )

    def describe(self) -> str:
        cases = ", ".join(
            f"{c.date} (q{int(round(c.quantile * 100)):02d}, "
            f"{c.domain_mean_mm:.1f} mm)"
            for c in self.cases
        )
        return (
            f"sampled validation every {self.cfg.every} epochs from epoch "
            f"{self.cfg.start_epoch}, {self.cfg.members} members: {cases or 'none'}"
        )

    def should_run(self, epoch: int) -> bool:
        """``epoch`` is the 0-based loop index.

        Counting is done on ``epoch + 1`` -- the number of completed epochs -- to
        match the ``train.ckpt_every`` convention.  The monitor is called from
        inside the checkpoint block, so ``train.ckpt_every`` must divide
        ``every`` or this never fires; :func:`validate_cadence` checks that.
        """
        done = epoch + 1
        return (
            self.cfg.enabled
            and bool(self.cases)
            and done >= self.cfg.start_epoch
            and done % max(1, self.cfg.every) == 0
        )

    def validate_cadence(self, ckpt_every: int) -> None:
        """Fail loudly at startup rather than silently never evaluating."""
        if not self.cfg.enabled:
            return
        if self.cfg.every % max(1, ckpt_every) != 0:
            raise ValueError(
                f"validation.every ({self.cfg.every}) must be a multiple of "
                f"train.ckpt_every ({ckpt_every}); the monitor only gets a "
                f"chance to run on checkpoint epochs, so otherwise it would "
                f"never fire"
            )

    # -- evaluation --------------------------------------------------------

    def run(self, model, ema, epoch: int, step: int) -> dict | None:
        """Sample every case with the EMA weights and score it.

        Returns the summary dict, or ``None`` if anything went wrong -- a broken
        diagnostic must not take the training run down with it.
        """
        try:
            return self._run(model, ema, epoch, step)
        except Exception as exc:  # pragma: no cover - defensive by design
            print(f"[validation monitor] skipped at epoch {epoch}: {exc!r}", flush=True)
            return None

    def _run(self, model, ema, epoch: int, step: int) -> dict:
        import torch

        from ..da.sampler import SamplerConfig, sample
        from ..models.flow import RectifiedFlow

        started = time.time()
        online = {k: v.detach().clone() for k, v in model.state_dict().items()}
        try:
            ema.copy_to(model)
            model.eval()

            scfg = SamplerConfig(
                n_steps=self.cfg.n_steps,
                heun=True,
                schedule_power=1.0,
                noise_scale=0.0,
                cfg_scale=self.cfg.cfg_scale,
                prior_temperature=1.0,     # unguided background: never inflate
                n_corrections=0,
                mask_fill=self.ds.mask_fill,
                seed=1234,                 # fixed: epoch-to-epoch changes are the model
            )
            flow = RectifiedFlow()
            mask = torch.from_numpy(
                self.valid.astype(np.float32)[None, None]
            ).to(self.device)
            shape = (self.cfg.members, 1, *self.valid.shape)

            records = []
            for case in self.cases:
                item = self.ds[self._position_of(case.index)]
                with torch.no_grad():
                    generated = sample(
                        model,
                        item["cond"][None].to(self.device),
                        shape,
                        self.device,
                        cfg=scfg,
                        flow=flow,
                        mask=mask,
                    )
                members = self.transform.inverse(
                    generated[:, 0].float().cpu().numpy()
                )
                members = np.where(self.valid[None], members, np.nan)
                records.append(
                    {
                        "case": case,
                        "members": members,
                        "era5": self._era5_mm(item),
                        "metrics": self._metrics(members, case.target),
                    }
                )
        finally:
            model.load_state_dict(online)
            model.train()

        summary = {
            "epoch": int(epoch),
            "step": int(step),
            "seconds": round(time.time() - started, 1),
            "members": self.cfg.members,
            "cfg_scale": self.cfg.cfg_scale,
            "cases": [
                {
                    "date": r["case"].date,
                    "quantile": r["case"].quantile,
                    "domain_mean_target_mm": r["case"].domain_mean_mm,
                    **r["metrics"],
                }
                for r in records
            ],
        }
        summary["mean_crps_mm"] = float(
            np.mean([c["crps_mm"] for c in summary["cases"]])
        )
        with self.history_path.open("a") as handle:
            handle.write(json.dumps(summary) + "\n")

        if self.cfg.save_maps:
            self._plot_maps(records, epoch)
        self._plot_progress()
        return summary

    def _position_of(self, index: int) -> int:
        """Map a store index back to a dataset position."""
        return int(np.where(self.ds.index == index)[0][0])

    def _era5_mm(self, item) -> np.ndarray:
        """Recover the ERA5 tp channel in mm/day for the comparison panel."""
        values = item["cond"][self.era5_tp_index].numpy()
        if self.cond_mean is not None and self.cond_std is not None:
            values = (
                values * self.cond_std[self.era5_tp_index]
                + self.cond_mean[self.era5_tp_index]
            )
        if self.cond_transform is not None:
            values = self.cond_transform.inverse_channel(values, self.era5_tp_index)
        return np.where(self.valid, values, np.nan)

    def _metrics(self, members: np.ndarray, target: np.ndarray) -> dict:
        from .metrics import crps_ensemble

        keep = self.valid & np.isfinite(target)
        mean = np.mean(members, axis=0)
        spread = np.std(members, axis=0, ddof=1)
        predicted = mean[keep].astype(np.float64)
        observed = target[keep].astype(np.float64)
        difference = predicted - observed
        correlation = (
            float(np.corrcoef(predicted, observed)[0, 1])
            if predicted.std() > 0 and observed.std() > 0
            else float("nan")
        )
        low = np.quantile(members[:, keep], 0.05, axis=0)
        high = np.quantile(members[:, keep], 0.95, axis=0)
        return {
            "crps_mm": float(crps_ensemble(members[:, keep], observed)),
            "rmse_mm": float(np.sqrt(np.mean(difference**2))),
            "mae_mm": float(np.mean(np.abs(difference))),
            "bias_mm": float(np.mean(difference)),
            "spatial_correlation": correlation,
            "mean_spread_mm": float(np.mean(spread[keep])),
            "interval_90_coverage": float(
                np.mean((observed >= low) & (observed <= high))
            ),
            "prediction_max_mm": float(predicted.max()),
            "target_max_mm": float(observed.max()),
        }

    # -- figures -----------------------------------------------------------

    #: Column headers for the map panel.  Lettered so they can be referred to
    #: unambiguously in notes and in the paper.
    MAP_COLUMNS = [
        ("A", "ERA5 input", "total precipitation, regridded"),
        ("B", "CHIRPS target", "observed truth"),
        ("C", "Model ensemble mean", "{members} members"),
        ("D", "Single member", "realism / texture check"),
        ("E", "Model error", "ensemble mean - CHIRPS"),
        ("F", "Ensemble spread", "standard deviation"),
    ]

    def _decorate(self, axis, *, left: bool, bottom: bool) -> None:
        """Degree ticks when the geographic extent is known, else bare axes."""
        if self.extent is None:
            axis.set_xticks([])
            axis.set_yticks([])
            return
        lon0, lon1, lat0, lat1 = self.extent
        axis.set_xticks(np.arange(np.ceil(lon0), np.floor(lon1) + 1, 2))
        axis.set_yticks(np.arange(np.ceil(lat0), np.floor(lat1) + 1, 2))
        axis.tick_params(labelsize=7, length=2)
        if bottom:
            axis.set_xlabel("Longitude (deg E)", fontsize=8)
        else:
            axis.set_xticklabels([])
        if left:
            axis.set_ylabel("Latitude (deg N)", fontsize=8)
        else:
            axis.set_yticklabels([])
        axis.grid(alpha=0.15, linewidth=0.4, linestyle=":")

    def _plot_maps(self, records: list[dict], epoch: int) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = len(records)
        figure, axes = plt.subplots(
            rows,
            6,
            figsize=(23, 4.6 * rows),
            constrained_layout=True,
            squeeze=False,
        )
        rain = plt.get_cmap("viridis").copy()
        rain.set_bad("white")
        error = plt.get_cmap("RdBu_r").copy()
        error.set_bad("white")
        spread_cmap = plt.get_cmap("magma").copy()
        spread_cmap.set_bad("white")
        imshow_kwargs = dict(origin="lower", interpolation="nearest")
        if self.extent is not None:
            imshow_kwargs["extent"] = self.extent

        for row, record in enumerate(records):
            case, members = record["case"], record["members"]
            mean = np.mean(members, axis=0)
            spread = np.std(members, axis=0, ddof=1)
            metrics = record["metrics"]

            # One rainfall scale across A-D so the four are directly comparable.
            pooled = np.concatenate(
                [
                    record["era5"][self.valid],
                    case.target[self.valid],
                    mean[self.valid],
                    members[0][self.valid],
                ]
            )
            vmax = max(5.0, float(np.nanpercentile(pooled, 99.0)))
            limit = max(
                2.0,
                float(np.nanpercentile(np.abs((mean - case.target)[self.valid]), 99.0)),
            )

            baseline = case.baseline or {}
            panels = [
                (record["era5"], rain, 0.0, vmax,
                 f"RMSE {baseline.get('rmse_mm', float('nan')):.2f}   "
                 f"r {baseline.get('spatial_correlation', float('nan')):.2f}"),
                (case.target, rain, 0.0, vmax,
                 f"domain mean {case.domain_mean_mm:.2f}   "
                 f"max {metrics['target_max_mm']:.1f}"),
                (mean, rain, 0.0, vmax,
                 f"RMSE {metrics['rmse_mm']:.2f}   "
                 f"r {metrics['spatial_correlation']:.2f}   "
                 f"CRPS {metrics['crps_mm']:.2f}"),
                (members[0], rain, 0.0, vmax,
                 f"max {float(np.nanmax(members[0])):.1f}"),
                (mean - case.target, error, -limit, limit,
                 f"bias {metrics['bias_mm']:+.2f}   MAE {metrics['mae_mm']:.2f}"),
                (spread, spread_cmap, 0.0,
                 max(1.0, float(np.nanpercentile(spread[self.valid], 99.0))),
                 f"mean {metrics['mean_spread_mm']:.2f}   "
                 f"cov90 {metrics['interval_90_coverage'] * 100:.0f}%"),
            ]

            images = []
            for column, (values, cmap, vmin, vhigh, note) in enumerate(panels):
                axis = axes[row, column]
                images.append(
                    axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vhigh,
                                **imshow_kwargs)
                )
                self._decorate(axis, left=column == 0, bottom=row == rows - 1)
                if row == 0:
                    letter, title, subtitle = self.MAP_COLUMNS[column]
                    axis.set_title(
                        f"{letter}.  {title}\n"
                        f"{subtitle.format(members=self.cfg.members)}",
                        fontsize=10.5,
                        pad=8,
                    )
                axis.text(
                    0.02, 0.02, note, transform=axis.transAxes,
                    ha="left", va="bottom", fontsize=7.5, zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.82, pad=2.0),
                )

            # Row identifier in the left margin, clear of the latitude label.
            axes[row, 0].annotate(
                f"{case.date}\nq{int(round(case.quantile * 100)):02d} case\n"
                f"{case.domain_mean_mm:.1f} mm day$^{{-1}}$",
                xy=(0, 0.5), xycoords="axes fraction",
                xytext=(-96, 0), textcoords="offset points",
                ha="center", va="center", fontsize=10.5, fontweight="bold",
            )

            bar = figure.colorbar(
                images[0], ax=axes[row, 0:4].tolist(),
                orientation="horizontal", shrink=0.6, aspect=45, pad=0.02,
            )
            bar.set_label("Daily precipitation (mm day$^{-1}$)", fontsize=9)
            bar.ax.tick_params(labelsize=7)
            bar = figure.colorbar(
                images[4], ax=axes[row, 4], orientation="horizontal",
                shrink=0.85, aspect=16, pad=0.02,
            )
            bar.set_label("Error (mm day$^{-1}$)", fontsize=9)
            bar.ax.tick_params(labelsize=7)
            bar = figure.colorbar(
                images[5], ax=axes[row, 5], orientation="horizontal",
                shrink=0.85, aspect=16, pad=0.02,
            )
            bar.set_label("Spread (mm day$^{-1}$)", fontsize=9)
            bar.ax.tick_params(labelsize=7)

        figure.suptitle(
            "BDhighresDA sampled validation - held-out ERA5-conditioned background\n"
            f"Epoch {epoch + 1}   |   EMA weights   |   {self.cfg.members}-member "
            f"ensemble   |   {self.cfg.n_steps} sampler steps, CFG w="
            f"{self.cfg.cfg_scale:g}, prior temperature 1.0\n"
            "Panels A-D share a common rainfall scale within each row; "
            "rows are independent held-out days",
            fontsize=13.5,
        )
        figure.savefig(self.out_dir / f"epoch_{epoch + 1:04d}.png", dpi=110)
        plt.close(figure)

    #: (key, axis label, subtitle, lower-is-better) for the progress figure.
    PROGRESS_PANELS = [
        ("crps_mm", "CRPS (mm day$^{-1}$)",
         "Probabilistic error - the selection metric", True),
        ("rmse_mm", "RMSE (mm day$^{-1}$)",
         "Ensemble-mean deterministic error", True),
        ("spatial_correlation", "Pearson correlation",
         "Spatial pattern agreement with CHIRPS", False),
        ("bias_mm", "Mean error (mm day$^{-1}$)",
         "Domain-mean bias - zero is the target", None),
        ("mean_spread_mm", "Ensemble standard deviation (mm day$^{-1}$)",
         "Predictive spread", None),
        ("interval_90_coverage", "Fraction of valid grid cells",
         "90% interval coverage - nominal 0.90", None),
    ]

    def _plot_progress(self) -> None:
        """Metric-vs-epoch curves with the ERA5 baseline drawn in.

        The reference lines are the point of this figure: a conditional model
        that cannot beat the field it is conditioned on has not learned to use
        its conditioning.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        rows = [
            json.loads(line)
            for line in self.history_path.read_text().splitlines()
            if line.strip()
        ]
        if len(rows) < 2:
            return
        epochs = [r["epoch"] + 1 for r in rows]
        colours = plt.get_cmap("tab10").colors

        figure, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
        for axis, (key, ylabel, subtitle, lower_better) in zip(
            axes.ravel(), self.PROGRESS_PANELS
        ):
            has_baseline = False
            for number, case in enumerate(self.cases):
                colour = colours[number % len(colours)]
                series = [
                    r["cases"][number][key]
                    for r in rows
                    if number < len(r["cases"])
                ]
                axis.plot(
                    epochs[: len(series)], series, marker="o", ms=3.2,
                    lw=1.4, color=colour, label=case.label,
                )
                reference = (case.baseline or {}).get(key)
                if reference is not None and reference == reference:
                    # Unlabelled: one shared legend entry is added below instead
                    # of repeating "ERA5 baseline" once per case per panel.
                    axis.axhline(reference, color=colour, ls=":", lw=1.4)
                    has_baseline = True
            if key == "bias_mm":
                axis.axhline(0.0, color="black", lw=0.9)
            if key == "interval_90_coverage":
                axis.axhline(0.90, color="black", ls="--", lw=1.0)
                axis.set_ylim(0.0, 1.0)

            handles, labels = axis.get_legend_handles_labels()
            if has_baseline:
                handles.append(Line2D([], [], color="grey", ls=":", lw=1.4))
                labels.append("ERA5 input baseline")
            if lower_better is not None:
                pooled = [np.mean([c[key] for c in r["cases"]]) for r in rows]
                best = int(np.argmin(pooled) if lower_better else np.argmax(pooled))
                axis.axvline(epochs[best], color="grey", ls="--", lw=1.0)
                handles.append(Line2D([], [], color="grey", ls="--", lw=1.0))
                labels.append(f"best: epoch {epochs[best]}")

            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.set_title(subtitle, fontsize=10.5)
            axis.grid(alpha=0.25)
            axis.legend(
                handles, labels, fontsize=7.2, frameon=True, framealpha=0.85,
                edgecolor="none", loc="best",
            )

        month = (
            "July" if self.cfg.month == 7 else f"month {self.cfg.month}"
        )
        figure.suptitle(
            "BDhighresDA training progress - sampled validation on held-out "
            f"{month} days\n"
            f"EMA weights   |   {self.cfg.members}-member ensemble   |   "
            f"{self.cfg.n_steps} sampler steps, CFG w={self.cfg.cfg_scale:g}   |   "
            f"evaluated every {self.cfg.every} epochs\n"
            "Dotted lines are the raw ERA5 input scored against CHIRPS: the "
            "model must beat its own conditioning",
            fontsize=13.5,
        )
        figure.savefig(self.out_dir / "progress.png", dpi=115)
        plt.close(figure)
