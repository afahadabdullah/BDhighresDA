"""In-training sampled validation for the V3-SG / V7 subgrid branches.

``scripts/57_train_subgrid_oracle.py`` selects ``best.pt`` from a masked MSE
against a random-``t`` velocity target.  That number is dominated by the noise of
the ``t`` and ``x0`` draws, so it moves very little once a run is past its early
epochs and says almost nothing about the field the sampler actually produces.
``bdhires.eval.monitor`` fixed exactly this for the CPCv2 trainer; this module is
its counterpart for the branches trained by script 57, which use a different
dataset, a different decoder contract and a different model signature.

What it measures, at a cadence tied to the kept checkpoints:

* **allocation** -- sample the branch given the TRUE coarse amounts, reconstruct
  through the hard mass-conserving decoder, and score the part the branch
  actually controls.  The headline number is the within-block anomaly
  correlation against the conservative smooth base, because a full-field
  correlation is dominated by coarse amounts the branch was handed for free.
  Conservation error and the seam index come along, since those are the two ways
  this stage has failed before.
* **coarse** -- sample the hurdle branch, decode to mm/day, and score CRPS,
  ensemble-mean pattern correlation and per-member wet fraction.  Per-member
  statistics are kept separate from ensemble-mean ones on purpose: the mean of N
  members is wetter in area and weaker in amount than any member, so comparing
  it against an observed wet fraction manufactures an alarm out of nothing.

Cases are chosen once, at construction, by RAINFALL quantile rather than by the
calendar -- evenly spaced positions in a date-ordered validation split land on
1 January and 31 December, dry-season days where a pattern correlation is close
to pure noise.  The same days are then re-sampled at every evaluation, so the
curves in ``history.jsonl`` are directly comparable across the run.

Outputs, all under ``<out_dir>/diagnostics``:

* ``history.jsonl``   one append-only row per evaluation
* ``epoch_XXXX.png``  map panel for that epoch
* ``progress.png``    metric-vs-epoch curves, rewritten each time

Design rule, inherited from ``bdhires.eval.monitor``: a diagnostic must never
take down a 48-hour training run.  :meth:`run` catches its own exceptions and
returns ``None``.  It does NOT, however, allow itself to be silently disabled --
:meth:`validate_cadence` raises at startup if the configured cadence could never
fire, because a monitor that never runs looks exactly like a healthy one.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data import (
    area_weighted_block_mean,
    conservative_smooth_upsample,
    decode_coarse_amount,
    reconstruct_from_amount,
)

SUPPORTED_STAGES = ("coarse", "allocation")


@dataclass
class SubgridMonitorConfig:
    """When and how to run the sampled diagnostic.

    Defaults are the cheap end on purpose: this runs inside a training loop, so
    ``cases * members * n_steps * 2`` forward passes are paid out of the epoch
    budget every time it fires.
    """

    enabled: bool = True
    start_epoch: int = 10      # early samples are noise; do not spend GPU on them
    every: int = 10            # must divide into the checkpoint cadence
    cases: int = 2             # days re-sampled at every evaluation
    members: int = 4           # ensemble size (cost scales linearly)
    n_steps: int = 50          # Heun steps; 50 is converged for these branches
    seed: int = 20220503
    save_maps: bool = True
    # Rainfall quantiles of the validation split.  Two cases give a typical day
    # and a wet one, which is the pair that matters: V5 looked acceptable in the
    # middle of the distribution and lost 40 percent of the amplitude in the tail.
    quantiles: tuple[float, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict | None) -> "SubgridMonitorConfig":
        """Build from config, REFUSING unknown keys.

        Silently dropping an unrecognised key turns a typo into a diagnostic
        that quietly runs with different settings than the config claims.
        """
        if not raw:
            return cls()
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(
                f"unknown sampled_validation keys {unknown}; expected any of "
                f"{sorted(known)}"
            )
        kwargs = dict(raw)
        if "quantiles" in kwargs and kwargs["quantiles"] is not None:
            kwargs["quantiles"] = tuple(float(q) for q in kwargs["quantiles"])
        return cls(**kwargs)

    def resolved_quantiles(self) -> tuple[float, ...]:
        if self.quantiles:
            return self.quantiles
        if self.cases <= 1:
            return (0.90,)
        return tuple(np.linspace(0.50, 0.95, self.cases))


# --------------------------------------------------------------------------
# sampling and scoring primitives
# --------------------------------------------------------------------------


def heun_sample(velocity_fn, shape, n_steps: int, device, generator) -> torch.Tensor:
    """Integrate the rectified-flow ODE from noise at t=0 to the sample at t=1.

    Deliberately the same Heun scheme as ``bdhires.da.hierarchical_sampler`` and
    ``scripts/64``: a monitor that integrates differently from the sampler used
    at inference is measuring a model nobody will ever run.  The final step is
    Euler because the second evaluation would land past t=1.
    """
    state = torch.randn(shape, device=device, generator=generator)
    times = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t0, t1 = times[i], times[i + 1]
        dt = t1 - t0
        batch_t = t0.expand(shape[0])
        v0 = velocity_fn(state, batch_t)
        if i == n_steps - 1:
            state = state + dt * v0
        else:
            probe = state + dt * v0
            v1 = velocity_fn(probe, t1.expand(shape[0]))
            state = state + 0.5 * dt * (v0 + v1)
    return state


def pattern_correlation(a: np.ndarray, b: np.ndarray, keep: np.ndarray) -> float:
    x = a.reshape(-1)[keep.reshape(-1)]
    y = b.reshape(-1)[keep.reshape(-1)]
    if x.std() <= 0.0 or y.std() <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def crps(ensemble: np.ndarray, truth: np.ndarray, keep: np.ndarray) -> float:
    """Fair CRPS of an ensemble against a deterministic truth, in mm/day.

    The second term carries the (m-1) correction, so this is comparable across
    ensemble sizes -- without it a 4-member CRPS and a 16-member CRPS are not
    the same quantity and a change of ``members`` looks like a change of skill.
    """
    members = ensemble.reshape(ensemble.shape[0], -1)[:, keep.reshape(-1)]
    observed = truth.reshape(-1)[keep.reshape(-1)]
    m = members.shape[0]
    skill = np.abs(members - observed[None]).mean()
    if m < 2:
        return float(skill)
    spread = np.abs(members[:, None] - members[None, :]).mean() * m / (2.0 * (m - 1))
    return float(skill - spread)


def seam_index(field: np.ndarray, valid: np.ndarray, factor: int) -> float:
    """Mean gradient across block edges over the mean gradient inside blocks.

    A block-constant base is flat inside a block, so every gradient it has is a
    seam and the index diverges.  A field with genuine subgrid structure sits
    near one.  At factor 2 every interior gradient is also adjacent to a seam,
    so read this as a trend across epochs, not as an absolute.
    """
    if factor < 2:
        return float("nan")
    vertical = np.abs(np.diff(field, axis=1))
    horizontal = np.abs(np.diff(field, axis=0))
    vvalid = valid[:, 1:] & valid[:, :-1]
    hvalid = valid[1:] & valid[:-1]
    vseam = np.zeros(vertical.shape, bool)
    hseam = np.zeros(horizontal.shape, bool)
    vseam[:, factor - 1::factor] = True
    hseam[factor - 1::factor, :] = True
    edge = np.concatenate([vertical[vvalid & vseam], horizontal[hvalid & hseam]])
    interior = np.concatenate([vertical[vvalid & ~vseam], horizontal[hvalid & ~hseam]])
    if edge.size == 0 or interior.size == 0 or float(interior.mean()) <= 0.0:
        return float("nan")
    return float(edge.mean() / interior.mean())


def within_block_anomaly(
    field: torch.Tensor, area: torch.Tensor, valid: torch.Tensor, factor: int
) -> np.ndarray:
    """Remove each block's own mean, leaving only what allocation controls."""
    mean, _, _ = area_weighted_block_mean(field, area, valid, factor, 0.0)
    expanded = mean.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
    return (field - expanded)[:, 0].numpy()


# --------------------------------------------------------------------------


@dataclass
class _Case:
    position: int
    date: str
    quantile: float
    domain_mean_mm: float

    @property
    def label(self) -> str:
        return f"{self.date} (q{int(round(self.quantile * 100)):02d})"


class SubgridMonitor:
    """Sample a fixed pair of held-out days and score them, mid-training.

    ``dataset`` must be a fixed-crop :class:`~bdhires.data.SubgridDataset`
    restricted to the validation years -- the same object script 57 already
    builds for its loss-based validation.
    """

    def __init__(
        self,
        dataset,
        device,
        out_dir: str | Path,
        stage: str,
        cfg: SubgridMonitorConfig | None = None,
    ):
        if stage not in SUPPORTED_STAGES:
            raise ValueError(
                f"sampled validation supports {SUPPORTED_STAGES}, not {stage!r}"
            )
        self.stage = stage
        self.cfg = cfg or SubgridMonitorConfig()
        self.ds = dataset
        self.device = device
        self.out_dir = Path(out_dir)
        self.encoding = dataset.encoding
        self.factor = int(dataset.encoding.factor)
        self.wet_threshold = float(dataset.encoding.wet_threshold_mm)
        if self.cfg.enabled:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.out_dir / "history.jsonl"
        self.cases = self._select_cases() if self.cfg.enabled else []

    # -- case selection ----------------------------------------------------

    def _domain_means(self) -> np.ndarray:
        """Domain-mean rainfall per validation day, read in chunks.

        One pass over the validation split at construction.  Reading day by day
        would issue one zarr request per day; chunking keeps it to a handful.
        """
        valid = self.ds.valid.astype(bool)
        index = np.asarray(self.ds.index)
        means = np.empty(index.size, np.float64)
        step = 64
        for start in range(0, index.size, step):
            stop = min(start + step, index.size)
            block = np.asarray(self.ds.z["fine_mm"][index[start:stop]], np.float32)
            if block.ndim == 4:      # (T,1,H,W)
                block = block[:, 0]
            means[start:stop] = block[:, valid].mean(axis=1)
        return means

    def _select_cases(self) -> list[_Case]:
        quantiles = self.cfg.resolved_quantiles()
        means = self._domain_means()
        order = np.argsort(means)
        cases: list[_Case] = []
        seen: set[int] = set()
        for q in quantiles:
            position = int(order[int(round(float(q) * (len(order) - 1)))])
            # Two quantiles can land on the same day in a short split; nudge
            # rather than silently evaluating the same field twice.
            while position in seen and len(seen) < len(order):
                position = int(order[(int(np.flatnonzero(order == position)[0]) + 1)
                                     % len(order)])
            seen.add(position)
            date = str(
                self.ds.time[self.ds.index[position]].astype("datetime64[D]")
            )
            cases.append(_Case(position, date, float(q), float(means[position])))
        return cases

    # -- cadence -----------------------------------------------------------

    def should_run(self, epoch: int) -> bool:
        """``epoch`` is zero-based, as script 57 counts it."""
        if not self.cfg.enabled or not self.cases:
            return False
        completed = epoch + 1
        return (
            completed >= int(self.cfg.start_epoch)
            and completed % max(1, int(self.cfg.every)) == 0
        )

    def validate_cadence(self, keep_every: int) -> None:
        """Fail loudly at startup rather than silently never evaluating."""
        if not self.cfg.enabled:
            return
        keep_every = max(1, int(keep_every))
        if int(self.cfg.every) % keep_every != 0:
            raise ValueError(
                f"sampled_validation.every ({self.cfg.every}) must be a multiple "
                f"of train.keep_every ({keep_every}) so that every diagnostic "
                f"lands on a kept checkpoint; otherwise the picture and the "
                f"weights that produced it do not both survive"
            )

    def describe(self) -> str:
        days = ", ".join(case.label for case in self.cases)
        return (
            f"{self.stage}: {len(self.cases)} cases x {self.cfg.members} members "
            f"x {self.cfg.n_steps} steps every {self.cfg.every} epochs "
            f"from epoch {self.cfg.start_epoch}  [{days}]"
        )

    # -- evaluation --------------------------------------------------------

    def run(self, model, epoch: int, weights: str = "ema") -> dict | None:
        """Sample every case with the CURRENT weights of ``model`` and score it.

        Script 57 swaps EMA weights into ``model`` before validating and swaps
        them back afterwards; call this inside that window so the diagnostic
        measures the weights that will actually be checkpointed.  Returns None
        on any failure -- a broken picture must not end the run.
        """
        try:
            return self._run(model, epoch, weights)
        except Exception as exc:  # pragma: no cover - defensive by design
            import traceback

            print(
                f"[sampled validation] SKIPPED at epoch {epoch + 1}: {exc!r}",
                flush=True,
            )
            traceback.print_exc()
            return None

    def _run(self, model, epoch: int, weights: str) -> dict:
        started = time.time()
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                entries = [self._evaluate_case(model, case) for case in self.cases]
        finally:
            model.train(was_training)

        summary = {
            "epoch": int(epoch + 1),
            "stage": self.stage,
            "weights": weights,
            "members": int(self.cfg.members),
            "n_steps": int(self.cfg.n_steps),
            "seconds": round(time.time() - started, 1),
            "cases": [
                {key: value for key, value in entry.items() if key != "_panel"}
                for entry in entries
            ],
        }
        summary["mean"] = self._aggregate(summary["cases"])
        with self.history_path.open("a") as handle:
            handle.write(json.dumps(summary) + "\n")
        print(f"[sampled validation] {self._headline(summary)}", flush=True)
        if self.cfg.save_maps:
            self._draw(entries, epoch)
            self._draw_progress()
        return summary

    def _aggregate(self, cases: list[dict]) -> dict:
        keys = [
            key
            for key in cases[0]
            if isinstance(cases[0][key], (int, float)) and key != "quantile"
        ]
        return {
            key: float(np.nanmean([case[key] for case in cases])) for key in keys
        }

    def _headline(self, summary: dict) -> str:
        mean = summary["mean"]
        parts = [f"epoch {summary['epoch']:4d}", f"{summary['seconds']:.0f}s"]
        if self.stage == "allocation":
            parts += [
                f"anomaly_r={mean.get('anomaly_r', float('nan')):.3f}",
                f"(smooth null {mean.get('smooth_anomaly_r', float('nan')):.3f})",
                f"crps={mean.get('crps_mm', float('nan')):.3f}mm",
                f"seam={mean.get('seam_index', float('nan')):.2f}",
                f"conservation={mean.get('conservation_abs_mm', float('nan')):.2e}mm",
            ]
        else:
            parts += [
                f"crps={mean.get('crps_mm', float('nan')):.3f}mm",
                f"pattern_r={mean.get('ensemble_pattern_r', float('nan')):.3f}",
                f"member_wet={mean.get('member_wet_fraction', float('nan')):.3f}",
                f"truth_wet={mean.get('truth_wet_fraction', float('nan')):.3f}",
            ]
        return "  ".join(parts)

    # -- one case ----------------------------------------------------------

    def _evaluate_case(self, model, case: _Case) -> dict:
        members = int(self.cfg.members)
        item = self.ds[case.position]
        device = self.device
        # A per-case seed derived from the configured one, so the same day gets
        # the same noise at every epoch: an apparent change is then the model
        # moving, not the draw.
        generator = torch.Generator(device=device).manual_seed(
            int(self.cfg.seed) + case.position
        )

        fine_valid = item["fine_valid"][None].to(device)
        coarse_valid = item["coarse_valid"][None].to(device)
        area = item["cell_area"][None].to(device)
        coarse_truth = item["coarse_mm"][None].to(device)
        fine_truth = item["fine_mm"][None].to(device)
        entry = {
            "date": case.date,
            "quantile": case.quantile,
            "domain_mean_mm": case.domain_mean_mm,
        }

        if self.stage == "coarse":
            condition = item["coarse_cond"][None].to(device).expand(members, -1, -1, -1)
            state = heun_sample(
                lambda s, t: model(s, t, condition),
                (members, 2, *coarse_truth.shape[-2:]),
                int(self.cfg.n_steps), device, generator,
            )
            sampled = decode_coarse_amount(
                state, coarse_valid.expand(members, -1, -1, -1),
                self.encoding, hard=True,
            )[:, 0].cpu().numpy()
            truth = coarse_truth[0, 0].cpu().numpy()
            keep = item["coarse_valid"][0].numpy().astype(bool)
            flat_keep = keep.reshape(-1)
            entry.update({
                "crps_mm": crps(sampled, truth, keep),
                "ensemble_pattern_r": pattern_correlation(
                    sampled.mean(axis=0), truth, keep
                ),
                "member_pattern_r": float(np.mean([
                    pattern_correlation(member, truth, keep) for member in sampled
                ])),
                "ensemble_mean_bias_mm": float(
                    sampled.mean(axis=0).reshape(-1)[flat_keep].mean()
                    - truth.reshape(-1)[flat_keep].mean()
                ),
                # Per member, never on the ensemble mean: the mean of N members
                # is wetter in area and weaker in amount than any of them.
                "member_wet_fraction": float(np.mean([
                    (member.reshape(-1)[flat_keep] >= self.wet_threshold).mean()
                    for member in sampled
                ])),
                "truth_wet_fraction": float(
                    (truth.reshape(-1)[flat_keep] >= self.wet_threshold).mean()
                ),
                "dry_member_count": int(sum(
                    1 for member in sampled
                    if member.reshape(-1)[flat_keep].std() <= 0.0
                )),
            })
            entry["_panel"] = {
                "truth": truth, "members": sampled, "keep": keep,
                "label": case.label,
            }
            return entry

        # -- allocation, given the TRUE coarse amounts -----------------------
        coarse_context = item["coarse_state"][None].to(device).expand(
            members, -1, -1, -1
        )
        condition = item["fine_cond"][None].to(device).expand(members, -1, -1, -1)
        state = heun_sample(
            lambda s, t: model(s, t, condition, coarse_context, 0.0),
            (members, 2, *fine_truth.shape[-2:]),
            int(self.cfg.n_steps), device, generator,
        )
        field = reconstruct_from_amount(
            coarse_truth.expand(members, -1, -1, -1),
            state,
            fine_valid.expand(members, -1, -1, -1),
            area.expand(members, -1, -1, -1),
            self.encoding, hard=True,
        ).cpu()

        truth_cpu = fine_truth.cpu()
        area_cpu = area.cpu()
        valid_cpu = fine_valid.cpu()
        coarse_cpu = coarse_truth.cpu()
        keep = item["fine_valid"][0].numpy().astype(bool)
        flat_keep = keep.reshape(-1)

        smooth_null = conservative_smooth_upsample(
            coarse_cpu, area_cpu, valid_cpu, self.factor,
            int(self.encoding.smooth_base_iterations),
        )
        anomaly_truth = within_block_anomaly(
            truth_cpu, area_cpu, valid_cpu, self.factor
        )[0]
        anomaly_model = within_block_anomaly(
            field, area_cpu, valid_cpu, self.factor
        )
        anomaly_smooth = within_block_anomaly(
            smooth_null, area_cpu, valid_cpu, self.factor
        )[0]

        # Conservation, reported absolutely.  A ratio is meaningless in a dry
        # block -- rounding dust over zero looks catastrophic.
        block_mean, _, _ = area_weighted_block_mean(
            field, area_cpu, valid_cpu, self.factor, 0.0
        )
        coarse_keep = item["coarse_valid"][0].numpy().astype(bool)
        conservation = np.abs(
            block_mean[:, 0].numpy() - coarse_cpu[0, 0].numpy()[None]
        )[:, coarse_keep]

        sampled = field[:, 0].numpy()
        truth = truth_cpu[0, 0].numpy()
        entry.update({
            # The headline: what the branch controls, against the null it must beat.
            "anomaly_r": float(np.mean([
                pattern_correlation(member, anomaly_truth, keep)
                for member in anomaly_model
            ])),
            "smooth_anomaly_r": pattern_correlation(anomaly_smooth, anomaly_truth, keep),
            "crps_mm": crps(sampled, truth, keep),
            "ensemble_mean_mae_mm": float(
                np.abs(sampled.mean(axis=0) - truth).reshape(-1)[flat_keep].mean()
            ),
            "ensemble_pattern_r": pattern_correlation(
                sampled.mean(axis=0), truth, keep
            ),
            "seam_index": float(np.nanmean([
                seam_index(member, keep, self.factor) for member in sampled
            ])),
            "truth_seam_index": seam_index(truth, keep, self.factor),
            "conservation_abs_mm": float(conservation.mean()),
            "conservation_max_mm": float(conservation.max()),
            "member_wet_fraction": float(np.mean([
                (member.reshape(-1)[flat_keep] >= self.wet_threshold).mean()
                for member in sampled
            ])),
            "truth_wet_fraction": float(
                (truth.reshape(-1)[flat_keep] >= self.wet_threshold).mean()
            ),
        })
        entry["_panel"] = {
            "truth": truth,
            "members": sampled,
            "smooth": smooth_null[0, 0].numpy(),
            "keep": keep,
            "label": case.label,
        }
        return entry

    # -- figures -----------------------------------------------------------

    def _draw(self, entries: list[dict], epoch: int) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        panels = [entry["_panel"] for entry in entries]
        has_null = "smooth" in panels[0]
        columns = 3 + (1 if has_null else 0)
        figure, axes = plt.subplots(
            len(panels), columns,
            figsize=(3.4 * columns, 3.6 * len(panels)),
            squeeze=False,
        )

        def draw(axis, field, title, vmax, mask):
            shown = np.where(mask, field, np.nan)
            image = axis.imshow(
                shown, origin="lower", cmap="turbo", vmin=0.0, vmax=vmax
            )
            axis.set_title(title, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            return image

        for row, (panel, entry) in enumerate(zip(panels, entries)):
            mask = panel["keep"]
            truth = panel["truth"]
            members = panel["members"]
            vmax = float(np.nanpercentile(truth[mask], 99.0)) or 1.0
            column = 0
            image = draw(axes[row][column], truth, f"CHIRPS  {panel['label']}", vmax, mask)
            column += 1
            if has_null:
                draw(
                    axes[row][column], panel["smooth"],
                    f"smooth null  r={entry['smooth_anomaly_r']:.2f}", vmax, mask,
                )
                column += 1
            draw(
                axes[row][column], members[0],
                f"member 1  seam={seam_index(members[0], mask, self.factor):.2f}"
                if self.stage == "allocation" else "member 1",
                vmax, mask,
            )
            column += 1
            headline = (
                f"anomaly r={entry['anomaly_r']:.2f}"
                if self.stage == "allocation"
                else f"pattern r={entry['ensemble_pattern_r']:.2f}"
            )
            draw(
                axes[row][column], members.mean(axis=0),
                f"ensemble mean ({len(members)})  {headline}", vmax, mask,
            )
            figure.colorbar(
                image, ax=axes[row].tolist(), fraction=0.025, pad=0.01, label="mm/day"
            )

        figure.suptitle(
            f"{self.stage} branch, epoch {epoch + 1}  "
            f"({self.cfg.members} members, {self.cfg.n_steps} steps)",
            fontsize=11,
        )
        figure.savefig(
            self.out_dir / f"epoch_{epoch + 1:04d}.png", dpi=110,
            bbox_inches="tight",
        )
        plt.close(figure)

    def _draw_progress(self) -> None:
        """Metric-vs-epoch curves, rewritten from history each time."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = [
            json.loads(line)
            for line in self.history_path.read_text().splitlines()
            if line.strip()
        ]
        if len(rows) < 2:
            return
        epochs = [row["epoch"] for row in rows]
        if self.stage == "allocation":
            tracked = [
                ("anomaly_r", "within-block anomaly r", "smooth_anomaly_r"),
                ("crps_mm", "CRPS (mm/day)", None),
                ("seam_index", "seam index", "truth_seam_index"),
            ]
        else:
            tracked = [
                ("crps_mm", "CRPS (mm/day)", None),
                ("ensemble_pattern_r", "ensemble pattern r", None),
                ("member_wet_fraction", "wet fraction", "truth_wet_fraction"),
            ]
        figure, axes = plt.subplots(
            1, len(tracked), figsize=(4.2 * len(tracked), 3.4), squeeze=False
        )
        for axis, (key, label, reference) in zip(axes[0], tracked):
            axis.plot(epochs, [row["mean"].get(key) for row in rows], marker="o", ms=3)
            if reference is not None:
                values = [row["mean"].get(reference) for row in rows]
                if any(value is not None and np.isfinite(value) for value in values):
                    axis.plot(
                        epochs, values, linestyle="--", color="0.5",
                        label=reference.replace("_", " "),
                    )
                    axis.legend(fontsize=7)
            axis.set_xlabel("epoch")
            axis.set_title(label, fontsize=9)
            axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(self.out_dir / "progress.png", dpi=110)
        plt.close(figure)
