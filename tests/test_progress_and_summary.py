"""Tests for the training progress reporter and the startup summary.

Both are pure formatting, so these run without torch.
"""
from __future__ import annotations

import io
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# bdhires.utils.__init__ imports dist, which needs torch; load the leaf modules
# directly so these stay runnable on a machine without it.
import importlib.util  # noqa: E402


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


progress = _load("_bd_progress", "src/bdhires/utils/progress.py")
summary = _load("_bd_summary", "src/bdhires/utils/summary.py")


# --------------------------------------------------------------------------
# format_duration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "0s"), (45, "45s"), (95, "1m35s"), (3725, "1h02m05s"), (90061, "1d01h01m")],
)
def test_format_duration(seconds, expected):
    assert progress.format_duration(seconds) == expected


def test_format_duration_handles_nan_and_negative():
    assert progress.format_duration(float("nan")) == "--"
    assert progress.format_duration(-5) == "--"
    assert progress.format_duration(None) == "--"


# --------------------------------------------------------------------------
# ProgressReporter
# --------------------------------------------------------------------------


def _reporter(bar: bool, **kwargs):
    stream = io.StringIO()
    stream.isatty = lambda: bar          # noqa: E731
    defaults = dict(
        total_epochs=580, steps_per_epoch=433, log_every=100, stream=stream
    )
    defaults.update(kwargs)
    return progress.ProgressReporter(**defaults), stream


def test_bar_is_suppressed_when_not_a_tty():
    """A \\r bar in a SLURM .out file is unreadable; it must auto-disable."""
    quiet, _ = _reporter(bar=False)
    loud, _ = _reporter(bar=True)
    assert quiet.use_bar is False
    assert loud.use_bar is True


def test_non_tty_logs_only_every_log_every_steps():
    reporter, stream = _reporter(bar=False, log_every=100)
    reporter.begin_epoch(0)
    for _ in range(433):
        reporter.update(0.15, 1e-4)
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    assert len(lines) == 4                      # 100, 200, 300, 400
    assert "step 100/433" in lines[0]
    assert "run eta" in lines[0]
    assert "\r" not in stream.getvalue()


def test_tty_writes_a_bar_and_clears_it():
    reporter, stream = _reporter(bar=True)
    reporter.begin_epoch(0)
    for _ in range(10):
        reporter.update(0.15, 1e-4)
    assert "\r" in stream.getvalue()
    assert "|" in stream.getvalue()
    reporter.end_epoch()
    assert stream.getvalue().rstrip(" ").endswith("\r")   # cleared


def test_epoch_summary_reports_mean_loss_and_epoch_number():
    reporter, _ = _reporter(bar=False)
    reporter.begin_epoch(11)
    for value in (0.10, 0.20, 0.30):
        reporter.update(value, 1e-4)
    line = reporter.end_epoch("peak 41.2 GiB")
    assert "[epoch 12/580]" in line          # 0-based index -> 1-based display
    assert "loss 0.2000" in line
    assert "peak 41.2 GiB" in line
    assert "run eta" in line


def test_smoothed_loss_tracks_recent_steps_more_than_the_epoch_mean():
    """The whole point of showing both: when loss falls mid-epoch, the running
    mean is dragged up by early steps while the smoothed value follows the
    current level."""
    reporter, _ = _reporter(bar=False)
    reporter.begin_epoch(0)
    for _ in range(50):
        reporter.update(1.0, 1e-4)
    for _ in range(50):
        reporter.update(0.0, 1e-4)
    running = reporter.running_sum / reporter.steps_this_epoch
    assert running == pytest.approx(0.5)
    assert reporter.smoothed < running
    # ...and it keeps converging toward the new level
    before = reporter.smoothed
    for _ in range(100):
        reporter.update(0.0, 1e-4)
    assert reporter.smoothed < before


def test_smoothed_loss_starts_at_the_first_observation():
    reporter, _ = _reporter(bar=False)
    reporter.begin_epoch(0)
    reporter.update(0.42, 1e-4)
    assert reporter.smoothed == pytest.approx(0.42)


def test_run_eta_accounts_for_a_resumed_start_epoch():
    """A run resumed at epoch 500 has 80 epochs left, not 580."""
    fresh, _ = _reporter(bar=False, start_epoch=0)
    resumed, _ = _reporter(bar=False, start_epoch=500)
    for reporter in (fresh, resumed):
        reporter.begin_epoch(reporter.start_epoch)
        reporter.run_started -= 10.0          # 10 s of history
        for _ in range(10):
            reporter.update(0.1, 1e-4)
    assert resumed._run_eta() < fresh._run_eta() / 5


def test_progress_never_divides_by_zero_before_any_step():
    reporter, _ = _reporter(bar=False)
    reporter.begin_epoch(0)
    assert reporter._rate() >= 0.0
    line = reporter.end_epoch()
    assert "loss" in line


# --------------------------------------------------------------------------
# training_summary
# --------------------------------------------------------------------------


class _FakeUNet:
    base_channels_arg = 96
    channel_mult = (1, 2, 3, 4)
    num_res_blocks = 2
    num_heads = 4
    dropout = 0.1
    image_size = 128
    attn_resolutions = (16, 32)
    num_parameters = 38_412_193
    num_trainable_parameters = 38_412_193

    def levels(self):
        out, resolution = [], self.image_size
        for level, mult in enumerate(self.channel_mult):
            out.append(
                {
                    "level": level,
                    "resolution": resolution,
                    "channels": self.base_channels_arg * mult,
                    "attention": resolution in self.attn_resolutions,
                    "res_blocks": 2,
                }
            )
            if level != len(self.channel_mult) - 1:
                resolution //= 2
        out.append(
            {
                "level": "mid",
                "resolution": resolution,
                "channels": self.base_channels_arg * self.channel_mult[-1],
                "attention": True,
                "res_blocks": 2,
            }
        )
        return out

    def parameters_by_module(self):
        return {"in_conv": 13_920, "down": 12_402_048, "up": 22_290_144}


class _FakeDataset:
    H = W = 256
    n_cond = 6
    total_cond_channels = 15

    def __init__(self, n):
        self._n = n
        self.static = types.SimpleNamespace(shape=(7, 4, 4))

    def __len__(self):
        return self._n


_STATS = {
    "precip_transform": {"kind": "log1p", "eps": 0.1, "mu": 1.2031, "sd": 0.9442},
    "cond_transform": {"kinds": ["log1p", "none", "sqrt", "none", "none", "none"]},
    "cond_channels": [
        "era5_tp", "era5_tcwv", "era5_cape", "era5_u10", "era5_v10", "era5_msl",
    ],
    "static_channels": ["elev", "slope", "lsm", "sin_lon", "cos_lon", "sin_lat", "cos_lat"],
}


def _summary(stats=None, monitor=None, **kwargs):
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "train_h100.yaml").read_text())
    params = dict(
        cfg=cfg,
        config_path="configs/train_h100.yaml",
        model=_FakeUNet(),
        train_ds=_FakeDataset(13880),
        val_ds=_FakeDataset(731),
        device="cuda:0",
        world_size=1,
        amp_dtype="torch.bfloat16",
        steps_per_epoch=433,
        total_steps=251140,
        stats=_STATS if stats is None else stats,
        monitor=monitor,
    )
    params.update(kwargs)
    return summary.training_summary(**params)


def test_summary_reports_the_facts_that_matter():
    text = _summary()
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "train_h100.yaml").read_text())
    for expected in [
        cfg["train"]["out_dir"],          # never the old run directory
        Path(cfg["data"]["stats"]).name,
        "era5_tp:log1p",                  # the v2 conditioning
        "era5_cape:sqrt",
        "13,880 days",
        "251,140 steps",
        "38.41 M",
        "bfloat16 autocast",
        "training from scratch",
    ]:
        assert expected in text, expected
    assert "runs/prior_h100\n" not in text


def test_summary_shows_attention_at_the_right_levels():
    text = _summary()
    rows = [line for line in text.splitlines() if "x128" in line or "x16 " in line]
    architecture = "\n".join(text.splitlines())
    # attn_resolutions is (16, 32): 128 and 64 must NOT have attention
    assert "128x128" in architecture
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("0 ") and "128x128" in stripped:
            assert stripped.endswith("-")
        if stripped.startswith("3 ") and "16x16" in stripped:
            assert stripped.endswith("yes")
    assert rows


def test_summary_states_the_target_parameterisation():
    """A reader must be able to tell absolute from residual at a glance."""
    absolute = _summary()
    assert "ABSOLUTE" in absolute

    residual_stats = dict(_STATS)
    residual_stats["residual"] = {
        "enabled": True, "mean": 0.31, "std": 1.42, "base_channel": 0
    }
    residual_stats["residual_summary"] = {
        "base_channel_name": "era5_tp", "encoded_mean": 0.0,
        "encoded_std": 1.0, "encoded_abs_max": 6.2, "base_correlation": 0.71,
    }
    text = _summary(stats=residual_stats)
    assert "RESIDUAL" in text
    assert "era5_tp" in text
    assert "skill floor" in text


def test_summary_flags_a_pre_v2_stats_file():
    stats = {k: v for k, v in _STATS.items() if k != "cond_transform"}
    assert "identity (pre-v2 stats file)" in _summary(stats=stats)


def test_summary_notes_a_resume():
    text = _summary(resumed_from="runs/prior_h100_v2/last.pt", start_epoch=99)
    assert "last.pt" in text
    assert "starting at epoch 100" in text


def test_summary_describes_the_validation_monitor():
    monitor = types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            start_epoch=10, every=5, members=8, n_steps=30, cfg_scale=2.0
        ),
        cases=[
            types.SimpleNamespace(date="2019-07-18", quantile=0.5, domain_mean_mm=7.4),
            types.SimpleNamespace(date="2020-07-27", quantile=0.99, domain_mean_mm=41.2),
        ],
    )
    text = _summary(monitor=monitor)
    assert "2019-07-18" in text
    assert "q99" in text
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "train_h100.yaml").read_text())
    expected = sum(
        1
        for e in range(cfg["train"]["epochs"])
        if (e + 1) >= 10 and (e + 1) % 5 == 0
    )
    assert f"{expected} evaluations" in text
    assert "sampled CRPS" in text
    assert "1,920 NFE" in text             # 8 x 30 x 2 (Heun) x 2 (CFG) x 2 cases


def test_summary_warns_when_sampled_validation_is_off():
    text = _summary(monitor=None)
    assert "disabled" in text
    assert "sampling noise" in text


def test_summary_lines_are_not_absurdly_wide():
    for line in _summary().splitlines():
        assert len(line) < 120, line


# --------------------------------------------------------------------------
# Figure labelling
# --------------------------------------------------------------------------

monitor = _load("_bd_monitor", "src/bdhires/eval/monitor.py")


def test_map_columns_are_lettered_and_complete():
    columns = monitor.ValidationMonitor.MAP_COLUMNS
    assert [letter for letter, _, _ in columns] == list("ABCDEF")
    titles = [title for _, title, _ in columns]
    assert "ERA5 input" in titles
    assert "CHIRPS target" in titles
    assert "Model ensemble mean" in titles
    # every column carries an explanatory subtitle, not just a bare name
    assert all(subtitle.strip() for _, _, subtitle in columns)


def test_member_count_is_substituted_into_the_column_subtitle():
    _, _, subtitle = monitor.ValidationMonitor.MAP_COLUMNS[2]
    assert subtitle.format(members=8) == "8 members"


def test_progress_panels_cover_the_diagnostic_metrics():
    keys = [key for key, _, _, _ in monitor.ValidationMonitor.PROGRESS_PANELS]
    assert keys == [
        "crps_mm", "rmse_mm", "spatial_correlation",
        "bias_mm", "mean_spread_mm", "interval_90_coverage",
    ]
    for key, ylabel, subtitle, _ in monitor.ValidationMonitor.PROGRESS_PANELS:
        assert subtitle.strip(), key
        # every panel states its units or what the number is
        assert ylabel.strip(), key
    units = dict((k, y) for k, y, _, _ in monitor.ValidationMonitor.PROGRESS_PANELS)
    assert "mm day" in units["crps_mm"]
    assert "mm day" in units["rmse_mm"]
    assert "mm day" in units["bias_mm"]


def test_lower_is_better_flags_match_the_metric_direction():
    direction = {
        key: lower for key, _, _, lower in monitor.ValidationMonitor.PROGRESS_PANELS
    }
    assert direction["crps_mm"] is True         # minimise
    assert direction["rmse_mm"] is True         # minimise
    assert direction["spatial_correlation"] is False   # maximise
    assert direction["bias_mm"] is None         # neither: zero is the target
    assert direction["interval_90_coverage"] is None   # target is 0.90


def test_test_prediction_column_titles_are_lettered_with_units():
    spec = importlib.util.spec_from_file_location(
        "_bd_plot", ROOT / "scripts" / "08_plot_test_predictions.py"
    )
    try:
        plot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plot)
    except ImportError as exc:                  # cartopy/torch not installed here
        pytest.skip(f"08_plot_test_predictions needs {exc.name}")
    titles = plot.map_column_titles(16)
    assert len(titles) == 6
    for letter, title in zip("ABCDEF", titles):
        assert title.startswith(f"{letter}."), title
        assert "mm day" in title, title       # every column states its units
    assert "16-member" in titles[2]


# --------------------------------------------------------------------------
# v4: optional EMA, retained checkpoints, early stopping
# --------------------------------------------------------------------------


def _select_weights():
    """Load select_weights without importing torch-dependent siblings."""
    spec = importlib.util.spec_from_file_location(
        "_bd_flow_sel", ROOT / "src" / "bdhires" / "models" / "flow.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        pytest.skip(f"needs {exc.name}")
    return module.select_weights


def test_select_weights_prefers_ema_when_the_run_used_it():
    select = _select_weights()
    ckpt = {"model": {"a": 1}, "ema": {"a": 2}, "weights": "ema"}
    assert select(ckpt) == {"a": 2}


def test_select_weights_uses_online_weights_when_ema_is_off():
    select = _select_weights()
    ckpt = {"model": {"a": 1}, "ema": None, "weights": "model"}
    assert select(ckpt) == {"a": 1}


def test_select_weights_handles_checkpoints_written_before_the_flag():
    """Old checkpoints have no 'weights' key but always carry usable EMA."""
    select = _select_weights()
    assert select({"model": {"a": 1}, "ema": {"a": 2}}) == {"a": 2}
    assert select({"model": {"a": 1}}) == {"a": 1}


def test_select_weights_rejects_an_inconsistent_checkpoint():
    select = _select_weights()
    with pytest.raises(ValueError, match="carries none"):
        select({"model": {"a": 1}, "ema": None, "weights": "ema"})
    with pytest.raises(ValueError, match="neither"):
        select({"cfg": {}})


def _early_stop(crps_series, patience, best=float("inf")):
    """Mirror of the train.py early-stopping bookkeeping."""
    stale, stopped_at = 0, None
    for i, crps in enumerate(crps_series):
        if crps < best:
            best, stale = crps, 0
        else:
            stale += 1
            if patience and stale >= patience:
                stopped_at = i
                break
    return stopped_at, best


def test_early_stop_fires_after_patience_evaluations_without_improvement():
    improving = [10.0, 9.0, 8.0, 7.0]
    assert _early_stop(improving, patience=3)[0] is None
    # improves, then plateaus for exactly `patience` evaluations
    series = [10.0, 9.0, 9.5, 9.6, 9.7]
    stopped, best = _early_stop(series, patience=3)
    assert stopped == 4
    assert best == 9.0


def test_early_stop_would_have_caught_the_v3_degradation():
    """v3 CRPS bottomed near epoch 125 and rose for the rest of the run."""
    # mean CRPS, evaluations every 5 epochs from epoch 90 onward
    series = [13.1, 13.0, 12.9, 13.0, 13.2, 13.3, 13.4, 13.5, 13.6]
    stopped, _ = _early_stop(series, patience=6)
    assert stopped is not None and stopped < len(series)


def test_early_stop_disabled_never_fires():
    assert _early_stop([5.0, 6.0, 7.0, 8.0, 9.0], patience=0)[0] is None


def test_summary_reports_early_stopping_either_way():
    import copy

    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "train_h100.yaml").read_text())
    on, off = copy.deepcopy(cfg), copy.deepcopy(cfg)
    on["train"]["early_stop_patience"] = 6
    off["train"]["early_stop_patience"] = 0
    assert "without a CRPS improvement" in _summary(cfg=on)
    assert "disabled - runs the full schedule" in _summary(cfg=off)


def test_preflight_gate_is_opt_in():
    """The commit pin blocked three submissions and caught nothing."""
    sbatch = (ROOT / "slurm" / "train_h100.sbatch").read_text()
    assert '"${REQUIRE_TRAINING_PREFLIGHT:-0}"' in sbatch, (
        "the preflight gate must default to off"
    )
    assert "PREFLIGHT_PIN_COMMIT" in sbatch, (
        "there must still be a way to re-enable strict checking"
    )
    # the statistics checksum stays fatal: it is a data check, not a code one
    preflight = (ROOT / "scripts" / "preflight_training.py").read_text()
    assert "--strict-commit" in preflight
    assert "rerun them so the diagnostic describes the statistics in use" in preflight


def test_config_keeps_cfg_off_and_retains_snapshots():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "train_h100.yaml").read_text())
    train, validation = cfg["train"], cfg["validation"]
    assert train["cond_dropout"] == 0.0, "unused at cfg_scale 1.0"
    assert validation["cfg_scale"] == 1.0
    assert train["keep_every"] > 0, "v3 lost its peak to overwriting"
    assert train["early_stop_patience"] >= 0, "0 disables early stopping"
    assert train["keep_every"] % train["ckpt_every"] == 0, (
        "snapshots are written from inside the checkpoint block"
    )


def test_background_sampler_stays_uninflated():
    """Config-only mirror of the SamplerConfig test in test_conditioning_fixes.

    That one is @needs_torch and so never runs in an environment without torch,
    which is exactly how a stale assertion (cfg_scale > 1.0) survived until it
    hit the cluster. These are plain YAML checks, so they run everywhere.
    """
    import yaml

    da = yaml.safe_load((ROOT / "configs" / "da.yaml").read_text())
    background = da["background_sampler"]
    assert background["prior_temperature"] == 1.0, "no inflation without obs"
    assert background["n_corrections"] == 0
    assert background["schedule_power"] <= 1.0
    assert background["cfg_scale"] >= 1.0
    assert da["sampler"]["prior_temperature"] >= 1.0


def test_cond_dropout_and_cfg_scale_stay_consistent():
    """Paying for an unconditional branch only makes sense if sampling uses it."""
    import yaml

    train_cfg = yaml.safe_load((ROOT / "configs" / "train_h100.yaml").read_text())
    da_cfg = yaml.safe_load((ROOT / "configs" / "da.yaml").read_text())
    dropout = train_cfg["train"]["cond_dropout"]
    for name, block in (
        ("validation", train_cfg["validation"]),
        ("background_sampler", da_cfg["background_sampler"]),
    ):
        if block["cfg_scale"] != 1.0:
            assert dropout > 0, (
                f"{name} uses CFG w={block['cfg_scale']} but cond_dropout is 0, "
                f"so there is no unconditional branch to blend with"
            )
    if dropout == 0.0:
        assert train_cfg["validation"]["cfg_scale"] == 1.0
        assert da_cfg["background_sampler"]["cfg_scale"] == 1.0


def test_summary_reports_the_ema_setting_either_way():
    import copy

    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "train_h100.yaml").read_text())
    text = _summary(cfg=cfg)
    assert "early stop" in text
    assert "retained snapshot" in text

    on, off = copy.deepcopy(cfg), copy.deepcopy(cfg)
    on["train"]["use_ema"] = True
    off["train"]["use_ema"] = False
    assert "DISABLED" in _summary(cfg=off)
    assert "EMA shadow" not in _summary(cfg=off)
    assert "DISABLED" not in _summary(cfg=on)
    assert "EMA shadow" in _summary(cfg=on)


def test_step_line_does_not_say_ema():
    """"EMA" in this project means the weight average (train.use_ema).

    Using it for the smoothed loss made a run with use_ema=false look like EMA
    was still active.
    """
    reporter, stream = _reporter(bar=False, log_every=10)
    reporter.begin_epoch(0)
    for _ in range(10):
        reporter.update(0.15, 1e-4)
    line = stream.getvalue()
    assert "smoothed" in line
    assert "ema" not in line.lower()


# --------------------------------------------------------------------------
# guided sampling must not be wrapped in torch.inference_mode()
# --------------------------------------------------------------------------


def test_no_guided_sampler_call_runs_under_inference_mode():
    """inference_mode() permanently bars its tensors from autograd.

    Guidance differentiates the observation likelihood back through the network,
    so a guided call inside inference_mode fails deep in the backward engine with
    "element 0 of tensors does not require grad" -- a message that points
    nowhere. This scan catches it at the call site instead.
    """
    import ast

    offenders = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            if not any(
                "inference_mode" in ast.dump(item.context_expr)
                for item in node.items
            ):
                continue
            body = ast.dump(node)
            if "'H'" in body or "'gcfg'" in body:      # a guided sampler call
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        f"guided sampling inside torch.inference_mode(): {offenders}"
    )


def test_guidance_guards_against_inference_mode():
    """The guard must exist and name the fix, not just fail."""
    source = (ROOT / "src" / "bdhires" / "da" / "guidance.py").read_text()
    assert "is_inference_mode_enabled" in source
    assert "cannot run" in source and "no_grad" in source
