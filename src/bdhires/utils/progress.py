"""Training progress reporting.

Two very different consumers, so two rendering modes:

* **Interactive (TTY)** -- a real in-place progress bar, redrawn with ``\\r``.
* **SLURM log (not a TTY)** -- periodic one-line records.  A ``\\r`` bar written
  to a ``.out`` file is either an unreadable single mega-line or thousands of
  redraws, so the bar is suppressed automatically rather than by a flag.

Both modes report the same numbers: running and smoothed loss, learning rate,
throughput, epoch ETA and whole-run ETA.  The run ETA is the one that matters at
580 epochs -- it answers "will this finish inside the 24 h allocation".
"""

from __future__ import annotations

import shutil
import sys
import time


def format_duration(seconds: float) -> str:
    """``3725`` -> ``'1h02m05s'``; ``95`` -> ``'1m35s'``."""
    if seconds is None or seconds != seconds or seconds < 0:      # NaN-safe
        return "--"
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d{hours:02d}h{minutes:02d}m"
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def format_count(n: int) -> str:
    """Thousands separators, because 251140 is hard to read at a glance."""
    return f"{n:,}"


class ProgressReporter:
    """Per-step progress within an epoch, plus whole-run projection.

    ``steps_per_epoch`` is ``len(dataloader)``; ``total_epochs`` is the configured
    epoch count, so the run ETA accounts for a resumed run correctly.
    """

    def __init__(
        self,
        total_epochs: int,
        steps_per_epoch: int,
        log_every: int = 50,
        start_epoch: int = 0,
        stream=None,
        smoothing: float = 0.98,
        bar_width: int = 24,
        force_bar: bool | None = None,
    ):
        self.total_epochs = max(1, int(total_epochs))
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.log_every = max(1, int(log_every))
        self.start_epoch = int(start_epoch)
        self.stream = stream or sys.stdout
        self.smoothing = smoothing
        self.bar_width = bar_width
        self.use_bar = (
            force_bar
            if force_bar is not None
            else bool(getattr(self.stream, "isatty", lambda: False)())
        )

        self.run_started = time.time()
        self.epoch = start_epoch
        self.epoch_started = self.run_started
        self.steps_this_epoch = 0
        self.steps_seen = 0            # steps timed in THIS process
        self.running_sum = 0.0
        self.smoothed: float | None = None
        self._bar_dirty = False

    # -- lifecycle ---------------------------------------------------------

    def begin_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.epoch_started = time.time()
        self.steps_this_epoch = 0
        self.running_sum = 0.0

    def update(self, loss: float, lr: float) -> None:
        """Call once per optimiser step."""
        self.steps_this_epoch += 1
        self.steps_seen += 1
        self.running_sum += loss
        self.smoothed = (
            loss
            if self.smoothed is None
            else self.smoothing * self.smoothed + (1 - self.smoothing) * loss
        )
        if self.use_bar:
            self._draw_bar(lr)
        elif self.steps_this_epoch % self.log_every == 0:
            self._write_line(lr)

    def end_epoch(self, extra: str = "") -> str:
        """Clear the bar and return a one-line epoch summary."""
        self._clear_bar()
        elapsed = time.time() - self.epoch_started
        mean = self.running_sum / max(1, self.steps_this_epoch)
        parts = [
            f"[epoch {self.epoch + 1}/{self.total_epochs}]",
            f"loss {mean:.4f}",
            f"took {format_duration(elapsed)}",
            f"({self._rate():.2f} it/s)",
            f"run eta {format_duration(self._run_eta())}",
        ]
        if extra:
            parts.append(extra)
        return "  ".join(parts)

    # -- internals ---------------------------------------------------------

    def _rate(self) -> float:
        elapsed = time.time() - self.run_started
        return self.steps_seen / elapsed if elapsed > 0 else 0.0

    def _epoch_eta(self) -> float:
        rate = self._rate()
        remaining = self.steps_per_epoch - self.steps_this_epoch
        return remaining / rate if rate > 0 else float("nan")

    def _run_eta(self) -> float:
        rate = self._rate()
        if rate <= 0:
            return float("nan")
        done = (self.epoch - self.start_epoch) * self.steps_per_epoch + self.steps_this_epoch
        total = (self.total_epochs - self.start_epoch) * self.steps_per_epoch
        return max(0, total - done) / rate

    def _stats(self, lr: float) -> str:
        mean = self.running_sum / max(1, self.steps_this_epoch)
        smoothed = self.smoothed if self.smoothed is not None else mean
        # Deliberately NOT called "ema": in this project EMA means the weight
        # average controlled by train.use_ema, and seeing "(ema ...)" in the step
        # line while use_ema was false was understandably alarming.  This is a
        # smoothed view of the loss values only.
        return (
            f"loss {mean:.4f} (smoothed {smoothed:.4f})  "
            f"lr {lr:.2e}  {self._rate():.2f} it/s"
        )

    def _write_line(self, lr: float) -> None:
        print(
            f"  ep {self.epoch + 1}/{self.total_epochs}  "
            f"step {self.steps_this_epoch}/{self.steps_per_epoch}  "
            f"{self._stats(lr)}  "
            f"epoch eta {format_duration(self._epoch_eta())}  "
            f"run eta {format_duration(self._run_eta())}",
            file=self.stream,
            flush=True,
        )

    def _draw_bar(self, lr: float) -> None:
        fraction = self.steps_this_epoch / self.steps_per_epoch
        filled = int(self.bar_width * fraction)
        bar = "#" * filled + "." * (self.bar_width - filled)
        text = (
            f"\rep {self.epoch + 1}/{self.total_epochs} "
            f"|{bar}| {fraction * 100:5.1f}% "
            f"{self.steps_this_epoch}/{self.steps_per_epoch}  "
            f"{self._stats(lr)}  "
            f"eta {format_duration(self._epoch_eta())}"
        )
        width = shutil.get_terminal_size((120, 24)).columns
        self.stream.write(text[: width - 1].ljust(width - 1))
        self.stream.flush()
        self._bar_dirty = True

    def _clear_bar(self) -> None:
        if self.use_bar and self._bar_dirty:
            width = shutil.get_terminal_size((120, 24)).columns
            self.stream.write("\r" + " " * (width - 1) + "\r")
            self.stream.flush()
            self._bar_dirty = False
