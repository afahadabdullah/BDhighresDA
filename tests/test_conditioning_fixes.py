"""Regression tests for the epoch-119 diagnosis fixes.

Covers the four changes in docs/DIAGNOSIS_epoch119.md that can be checked
without a GPU or the packed Zarr store:

1. conditioning-channel variance stabilisation (log1p tp, sqrt cape)
2. classifier-free guidance in the sampler
3. masked cells filled with transform(0 mm), not a literal 0.0
4. random crops rejected when they carry too little land

The numpy-only tests run anywhere; the sampler tests need torch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bdhires.transforms import (  # noqa: E402
    CondTransform,
    PrecipTransform,
    ResidualSpec,
)

# The transform tests are pure numpy and must run anywhere.  Only the sampler and
# dataset tests need torch, so guard those individually rather than skipping the
# whole module on a machine without it.
try:  # pragma: no cover
    import torch

    from bdhires.data import DatasetConfig, PrecipDataset
    from bdhires.da.sampler import SamplerConfig, apply_mask, sample
except ImportError:  # pragma: no cover
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="needs torch")

CHANNELS = ["era5_tp", "era5_tcwv", "era5_cape", "era5_u10", "era5_v10", "era5_msl"]


def _skew(a: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    return float((((a - a.mean()) / a.std()) ** 3).mean())


def _fake_cond(rng, n=8, h=16, w=16) -> np.ndarray:
    """ERA5-like predictors: heavy-tailed tp and cape, near-Gaussian rest."""
    return np.stack(
        [
            rng.gamma(0.2, 40.0, (n, h, w)),        # tp, mm/day
            rng.normal(50.0, 10.0, (n, h, w)),      # tcwv
            rng.gamma(1.5, 600.0, (n, h, w)),       # cape
            rng.normal(0.0, 4.0, (n, h, w)),        # u10
            rng.normal(0.0, 4.0, (n, h, w)),        # v10
            rng.normal(101000.0, 400.0, (n, h, w)),  # msl
        ],
        axis=1,
    ).astype(np.float32)


# --------------------------------------------------------------------------
# 1. conditioning transforms
# --------------------------------------------------------------------------


def test_default_spec_targets_only_the_skewed_channels():
    ctf = CondTransform.for_channels(CHANNELS)
    assert ctf.kinds == ("log1p", "none", "sqrt", "none", "none", "none")


def test_transform_stabilises_variance():
    """The point of the change: tp must stop being a +20 sigma outlier field."""
    cond = _fake_cond(np.random.default_rng(0))
    out = CondTransform.for_channels(CHANNELS).forward(cond, channel_axis=1)
    assert _skew(cond[:, 0]) > 3.0          # raw tp is badly skewed
    assert abs(_skew(out[:, 0])) < 1.0      # transformed tp is not
    assert abs(_skew(out[:, 2])) < 1.0      # ditto cape


def test_transform_leaves_other_channels_untouched_and_does_not_mutate():
    cond = _fake_cond(np.random.default_rng(1))
    original = cond.copy()
    out = CondTransform.for_channels(CHANNELS).forward(cond, channel_axis=1)
    for i in (1, 3, 4, 5):
        assert np.array_equal(out[:, i], original[:, i])
    assert np.array_equal(cond, original), "forward() mutated its input"


def test_chw_and_nchw_paths_agree():
    """PrecipDataset passes (C,H,W); 06_compute_stats.py passes (N,C,H,W)."""
    cond = _fake_cond(np.random.default_rng(2))
    ctf = CondTransform.for_channels(CHANNELS)
    assert np.allclose(
        ctf.forward(cond[0], channel_axis=0),
        ctf.forward(cond, channel_axis=1)[0],
    )


def test_forward_channel_matches_the_stacked_forward():
    cond = _fake_cond(np.random.default_rng(7))
    ctf = CondTransform.for_channels(CHANNELS)
    stacked = ctf.forward(cond, channel_axis=1)
    for i in range(len(CHANNELS)):
        assert np.allclose(ctf.forward_channel(cond[:, i], i), stacked[:, i]), i


def test_standardising_raw_values_with_transformed_stats_is_detectably_broken():
    """Regression for the normalization-diagnostics failure.

    cond_mean/cond_std are computed after the transform.  Applying them to raw
    values leaves tp and cape wildly off while the untransformed channels look
    fine -- which is exactly the signature the QA figure reported (sampled CAPE
    mean 34.8 sigma, std 54.6).
    """
    cond = _fake_cond(np.random.default_rng(8))
    ctf = CondTransform.for_channels(CHANNELS)
    transformed = ctf.forward(cond, channel_axis=1)
    mean = transformed.mean(axis=(0, 2, 3))
    std = transformed.std(axis=(0, 2, 3))

    def standard(values):
        return abs(values.mean()) <= 0.5 and 0.5 <= values.std() <= 1.5

    for i, name in enumerate(CHANNELS):
        wrong = (cond[:, i] - mean[i]) / std[i]
        right = (ctf.forward_channel(cond[:, i], i) - mean[i]) / std[i]
        assert standard(right), f"{name} should be standard after the transform"
        if ctf.kinds[i] == "none":
            assert standard(wrong), f"{name} is untransformed and should be fine"
        else:
            assert not standard(wrong), (
                f"{name} is transformed; standardising raw values must NOT look "
                f"standard, or the diagnostic cannot catch this bug"
            )


def test_every_consumer_of_cond_stats_also_applies_the_cond_transform():
    """Lint: cond_mean/cond_std and CondTransform must travel together.

    Two call sites were missed when the transform was introduced
    (06_plot_normalization.py and evaluate.py) and both produced silently wrong
    normalisation.  This scan is cheap insurance against a third.
    """
    offenders = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        text = path.read_text()
        uses_stats = 'stats["cond_mean"]' in text or '"cond_mean"' in text
        if not uses_stats:
            continue
        if "CondTransform" not in text and "cond_transform" not in text:
            offenders.append(path.name)
    assert not offenders, (
        f"these read cond_mean/cond_std but never mention the conditioning "
        f"transform: {offenders}"
    )


@pytest.mark.parametrize("channel", [0, 2])
def test_inverse_round_trip(channel):
    cond = _fake_cond(np.random.default_rng(3))
    ctf = CondTransform.for_channels(CHANNELS)
    out = ctf.forward(cond, channel_axis=1)
    assert np.allclose(
        ctf.inverse_channel(out[:, channel], channel),
        cond[:, channel],
        rtol=1e-4,
        atol=1e-4,
    )


def test_channel_count_mismatch_raises():
    ctf = CondTransform.for_channels(CHANNELS)
    with pytest.raises(ValueError, match="channels"):
        ctf.forward(_fake_cond(np.random.default_rng(4))[:, :3], channel_axis=1)


def test_from_stats_defaults_to_identity_for_old_files():
    """Statistics written before this change must still load and be a no-op."""
    old = {"cond_mean": [0.0] * 6, "cond_std": [1.0] * 6}
    assert CondTransform.from_stats(old).kinds == ("none",) * 6
    cond = _fake_cond(np.random.default_rng(5))
    assert np.array_equal(
        CondTransform.from_stats(old).forward(cond, channel_axis=1), cond
    )


def test_from_stats_round_trips_through_json():
    import json

    ctf = CondTransform.for_channels(CHANNELS)
    stats = json.loads(
        json.dumps({"cond_transform": ctf.to_dict(), "cond_mean": [0.0] * 6})
    )
    assert CondTransform.from_stats(stats) == ctf


@needs_torch
def test_torch_and_numpy_backends_agree():
    cond = _fake_cond(np.random.default_rng(6))
    ctf = CondTransform.for_channels(CHANNELS)
    np_out = ctf.forward(cond.astype(np.float64), channel_axis=1)
    pt_out = ctf.forward(torch.from_numpy(cond).double(), channel_axis=1)
    assert np.allclose(np_out, pt_out.numpy(), rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------
# 3. mask fill
# --------------------------------------------------------------------------


def test_literal_zero_fill_would_encode_rain():
    """Documents the bug: 0.0 in transformed space is not 0 mm."""
    tf = PrecipTransform(kind="log1p", eps=0.1, mu=1.2, sd=0.9)
    assert float(tf.inverse(np.float32(0.0))) > 0.1     # the leak
    fill = float(np.asarray(tf.forward(np.float32(0.0))))
    assert abs(float(tf.inverse(np.float32(fill)))) < 1e-6


@needs_torch
def test_apply_mask_holds_masked_cells_at_fill():
    x = torch.randn(2, 1, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., :4, :] = 1.0
    out = apply_mask(x, mask, fill=-1.3333)
    assert torch.allclose(out[..., :4, :], x[..., :4, :])
    assert torch.allclose(out[..., 4:, :], torch.full_like(out[..., 4:, :], -1.3333))
    assert apply_mask(x, None, fill=-1.0) is x


# --------------------------------------------------------------------------
# 2. classifier-free guidance
# --------------------------------------------------------------------------


class _LinearVelocity(torch.nn.Module if torch is not None else object):
    """u = a*x + b*mean(cond). Conditional and unconditional branches differ."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.b = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)

    def forward(self, x, t, cond=None):
        out = self.a * x
        if cond is not None:
            out = out + self.b * cond.mean(dim=1, keepdim=True)
        return out


def _run(cfg_scale, cond, seed=0):
    model = _LinearVelocity().eval()
    cfg = SamplerConfig(
        n_steps=8, heun=True, schedule_power=1.0, n_corrections=0,
        prior_temperature=1.0, cfg_scale=cfg_scale, seed=seed,
    )
    with torch.no_grad():
        return sample(model, cond, (3, 1, 8, 8), torch.device("cpu"), cfg=cfg)


@needs_torch
def test_cfg_scale_one_is_plain_conditional_sampling():
    cond = torch.randn(1, 4, 8, 8)
    assert torch.allclose(_run(1.0, cond), _run(1.0, cond))


@needs_torch
def test_cfg_amplifies_the_conditional_signal():
    """w>1 must push the sample further from the unconditional result."""
    cond = torch.randn(1, 4, 8, 8)
    uncond = _run(1.0, torch.zeros_like(cond))
    near = (_run(1.0, cond) - uncond).abs().mean()
    far = (_run(3.0, cond) - uncond).abs().mean()
    assert far > near * 1.5, (float(near), float(far))


@needs_torch
def test_cfg_is_a_noop_without_conditioning():
    assert torch.allclose(_run(1.0, None), _run(3.0, None))


@needs_torch
def test_sampler_defaults_are_the_neutral_background():
    """Guard against the epoch-119 settings creeping back in as defaults."""
    cfg = SamplerConfig()
    assert cfg.prior_temperature == 1.0
    assert cfg.n_corrections == 0
    assert cfg.schedule_power <= 1.0
    assert cfg.cfg_scale == 1.0


@needs_torch
def test_config_blocks_construct_and_background_is_uninflated():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "da.yaml").read_text())
    background = SamplerConfig(**cfg["background_sampler"])
    analysis = SamplerConfig(**cfg["sampler"])
    assert background.prior_temperature == 1.0, "unguided background must not inflate"
    assert background.n_corrections == 0
    assert analysis.prior_temperature >= 1.0

    # This originally asserted cfg_scale > 1.0, on the assumption that CFG was
    # worth having because cond_dropout had already paid for it.  Measured on the
    # v3 checkpoint that assumption is false -- w=2.0 cost 0.22 of spatial
    # correlation on the 2024-08-17 case while leaving bias and spread untouched.
    # The invariant that actually holds is consistency: CFG above 1 requires an
    # unconditional branch to blend with, and that branch only exists if
    # cond_dropout > 0.
    train_cfg = yaml.safe_load((ROOT / "configs" / "train_h100.yaml").read_text())
    cond_dropout = train_cfg["train"]["cond_dropout"]
    assert background.cfg_scale >= 1.0
    if background.cfg_scale > 1.0:
        assert cond_dropout > 0, (
            f"background_sampler.cfg_scale={background.cfg_scale} needs an "
            f"unconditional branch, but train.cond_dropout is {cond_dropout}"
        )


# --------------------------------------------------------------------------
# 4. crop rejection
# --------------------------------------------------------------------------


class _FakeStore(dict):
    """Minimal mapping standing in for the packed Zarr store."""


def _fake_store(n_days=4, size=64, land_rows=slice(32, 64)):
    valid = np.zeros((size, size), np.float32)
    valid[land_rows] = 1.0                       # northern half is land
    target = np.random.default_rng(0).gamma(
        0.3, 8.0, (n_days, size, size)
    ).astype(np.float32)
    target[:, valid == 0] = np.nan               # CHIRPS is land-only
    return _FakeStore(
        time=np.array(
            ["2000-01-01", "2000-04-01", "2000-07-01", "2000-10-01"][:n_days],
            dtype="datetime64[ns]",
        ),
        valid=valid,
        static=np.zeros((7, size, size), np.float32),
        target=target,
        cond=_fake_cond(np.random.default_rng(1), n=n_days, h=size, w=size),
    )


def _dataset(**kwargs):
    store = _fake_store()
    cfg = DatasetConfig(root="unused", crop=32, random_crop=True, **kwargs)
    return PrecipDataset(
        cfg,
        PrecipTransform(kind="log1p", eps=0.1, mu=1.2, sd=0.9),
        store=store,
        cond_transform=CondTransform.for_channels(CHANNELS),
    )


@needs_torch
def test_crop_rejection_raises_the_land_fraction():
    """Half the fake domain is ocean, as the real wide grid's south third is."""
    strict = _dataset(min_valid_fraction=0.9)
    loose = _dataset(min_valid_fraction=0.0, max_crop_tries=1)
    strict_land = np.mean([strict[i % 4]["mask"].mean().item() for i in range(64)])
    loose_land = np.mean([loose[i % 4]["mask"].mean().item() for i in range(64)])
    assert strict_land > loose_land + 0.15, (strict_land, loose_land)


@needs_torch
def test_masked_cells_carry_the_transform_of_zero_not_zero():
    ds = _dataset(min_valid_fraction=0.0)
    item = ds[0]
    x1, mask = item["x1"].numpy(), item["mask"].numpy()
    assert (mask == 0).any(), "test needs at least one masked cell"
    assert np.allclose(x1[mask == 0], ds.mask_fill)
    assert not np.allclose(ds.mask_fill, 0.0)


@needs_torch
def test_dataset_applies_the_conditioning_transform():
    ds = _dataset(min_valid_fraction=0.0)
    raw_tp = np.asarray(ds.z["cond"][0][0])
    seen_tp = ds[0]["cond"].numpy()[0]
    assert _skew(raw_tp) > 2.0
    assert abs(_skew(seen_tp)) < abs(_skew(raw_tp))


# --------------------------------------------------------------------------
# 5. sampled-validation monitor cadence
# --------------------------------------------------------------------------


def _monitor_cfg(**kwargs):
    from bdhires.eval.monitor import MonitorConfig

    return MonitorConfig(**kwargs)


class _CadenceOnly:
    """Exercise should_run/validate_cadence without building a dataset."""

    def __init__(self, cfg):
        from bdhires.eval.monitor import ValidationMonitor

        self.cfg = cfg
        self.cases = [object()]
        self.should_run = ValidationMonitor.should_run.__get__(self)
        self.validate_cadence = ValidationMonitor.validate_cadence.__get__(self)


def test_monitor_fires_at_epoch_10_then_every_fifth():
    """The requested schedule: first at 10 completed epochs, then every 5."""
    m = _CadenceOnly(_monitor_cfg(start_epoch=10, every=5))
    fired = [e for e in range(60) if m.should_run(e)]
    # epoch index is 0-based, so "10 completed epochs" is index 9
    assert fired[:5] == [9, 14, 19, 24, 29]
    assert all((e + 1) % 5 == 0 for e in fired)
    assert not any(m.should_run(e) for e in range(9))


def test_monitor_cadence_must_divide_ckpt_every():
    """A silent never-fires was the trap here; it must raise instead."""
    m = _CadenceOnly(_monitor_cfg(start_epoch=10, every=5))
    m.validate_cadence(5)          # 5 % 5 == 0, fine
    m.validate_cadence(1)
    with pytest.raises(ValueError, match="never fire"):
        m.validate_cadence(10)     # monitor every 5 but checkpoints every 10


def test_monitor_disabled_never_runs():
    m = _CadenceOnly(_monitor_cfg(enabled=False))
    assert not any(m.should_run(e) for e in range(100))
    m.validate_cadence(999)        # disabled: no cadence complaint


def test_monitor_config_from_dict_ignores_unknown_keys():
    from bdhires.eval.monitor import MonitorConfig

    cfg = MonitorConfig.from_dict(
        {"every": 20, "quantiles": [0.5, 0.9], "not_a_real_key": 1}
    )
    assert cfg.every == 20
    assert cfg.quantiles == (0.5, 0.9)
    assert MonitorConfig.from_dict(None).enabled is True


def test_training_configs_have_a_consistent_monitor_cadence():
    """Guards the real configs, not just the code."""
    import yaml

    from bdhires.eval.monitor import MonitorConfig

    for name in ("train_h100.yaml", "train_v100.yaml"):
        cfg = yaml.safe_load((ROOT / "configs" / name).read_text())
        monitor = MonitorConfig.from_dict(cfg.get("validation"))
        if not monitor.enabled:
            continue
        ckpt_every = cfg["train"]["ckpt_every"]
        assert monitor.every % ckpt_every == 0, name
        assert cfg["data"]["monitor_grid"] == "bd"
        # the monitor builds a UNet-sized fixed crop, so these must agree
        assert cfg["data"]["crop"] == 128, name


def test_h100_config_pairs_its_stats_file_with_a_matching_run_dir():
    """Statistics files and run directories must move together.

    Each parameterisation produces checkpoints that are not interchangeable:
      v1  absolute target, raw z-scored conditioning
      v2  absolute target, transformed conditioning
      v3  residual target, transformed conditioning
    A retrain must never land on top of a previous run's directory, and the two
    settings must not drift apart.
    """
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "train_h100.yaml").read_text())
    stats, out_dir = cfg["data"]["stats"], cfg["train"]["out_dir"]
    assert stats != "data/processed/stats.json", "v1 stats are absolute-target"
    assert out_dir != "runs/prior_h100", "would overwrite the v1 baseline"

    # The statistics version need NOT equal the run version -- v4 legitimately
    # reuses stats_v3 because only training settings changed, not the data
    # parameterisation.  What must hold is that the run directory is new, so a
    # previous run's checkpoints are never overwritten.
    assert stats.rsplit("_", 1)[-1].removesuffix(".json").startswith("v"), stats
    previous = {
        "runs/prior_h100", "runs/prior_h100_v2",
        "runs/prior_h100_v3", "runs/prior_h100_v4",
    }
    assert out_dir not in previous, (
        f"out_dir {out_dir} would overwrite an earlier run; bump the suffix"
    )


# --------------------------------------------------------------------------
# 6. residual target parameterisation
# --------------------------------------------------------------------------

TF = PrecipTransform(kind="log1p", eps=0.1, mu=1.2, sd=0.9)


def _spec(**kwargs):
    return ResidualSpec(enabled=True, mean=0.3, std=1.4, **kwargs)


def test_residual_encode_decode_round_trips():
    """The only property that really matters: decode(encode(x)) == x."""
    rng = np.random.default_rng(0)
    chirps = rng.gamma(0.4, 12.0, (32, 32))
    era5 = rng.gamma(0.5, 9.0, (32, 32))
    spec = _spec()
    target_t, base_t = TF.forward(chirps), TF.forward(era5)
    encoded = spec.encode(target_t, base_t)
    assert np.allclose(spec.decode(encoded, base_t), target_t, atol=1e-6)
    # and all the way back to mm
    assert np.allclose(TF.inverse(spec.decode(encoded, base_t)), chirps, rtol=1e-5)


def test_disabled_spec_is_the_identity():
    """Absolute-target checkpoints must be unaffected by the new code path."""
    spec = ResidualSpec()
    values = np.linspace(-3, 3, 25)
    base = np.full_like(values, 7.7)
    assert np.array_equal(spec.encode(values, base), values)
    assert np.array_equal(spec.decode(values, base), values)
    assert spec.fill == 0.0


def test_zero_residual_reproduces_era5_exactly():
    """The skill floor: the model can always fall back to its own conditioning."""
    rng = np.random.default_rng(1)
    era5 = rng.gamma(0.5, 9.0, (16, 16))
    base_t = TF.forward(era5)
    spec = _spec()
    # a network output of `fill` means "no correction"
    prediction = np.full_like(base_t, spec.fill)
    assert np.allclose(TF.inverse(spec.decode(prediction, base_t)), era5, rtol=1e-5)


def test_residual_output_can_never_be_negative():
    """Non-negativity is automatic because reconstruction goes through inverse()."""
    rng = np.random.default_rng(2)
    era5 = rng.gamma(0.5, 9.0, (64, 64))
    base_t = TF.forward(era5)
    spec = _spec()
    # even a wildly negative residual cannot produce negative rainfall
    extreme = np.full_like(base_t, -50.0)
    recovered = TF.inverse(spec.decode(extreme, base_t))
    assert np.isfinite(recovered).all()
    assert (recovered >= 0).all()


def test_residual_is_better_conditioned_than_an_mm_space_residual():
    """Why log space: an mm-space residual is heteroscedastic, log space is not.

    Split the domain into light and heavy rain and compare the spread of the
    residual in each. In mm space the heavy half is far noisier; in transformed
    space the two are comparable, which is what an MSE loss needs.
    """
    rng = np.random.default_rng(3)
    chirps = rng.gamma(0.4, 12.0, 20000)
    era5 = chirps * rng.lognormal(0.0, 0.5, 20000)      # multiplicative error
    light, heavy = chirps < np.median(chirps), chirps >= np.median(chirps)

    mm_ratio = (chirps - era5)[heavy].std() / ((chirps - era5)[light].std() + 1e-9)
    log_residual = TF.forward(chirps) - TF.forward(era5)
    log_ratio = log_residual[heavy].std() / (log_residual[light].std() + 1e-9)

    assert mm_ratio > 5.0, mm_ratio          # mm space: wildly heteroscedastic
    assert log_ratio < 2.0, log_ratio        # log space: roughly homoscedastic


def test_residual_spec_round_trips_through_json():
    import json

    spec = _spec(base_channel=0)
    restored = ResidualSpec.from_stats(
        json.loads(json.dumps({"residual": spec.to_dict()}))
    )
    assert restored == spec


def test_from_stats_defaults_to_absolute_for_older_files():
    assert ResidualSpec.from_stats({"cond_mean": [0.0]}).enabled is False


@needs_torch
def test_encode_decode_agree_between_numpy_and_torch():
    spec = _spec()
    values = np.linspace(-2.0, 2.0, 40).astype(np.float32)
    base = np.linspace(0.0, 3.0, 40).astype(np.float32)
    np_out = spec.decode(values, base)
    pt_out = spec.decode(torch.from_numpy(values), torch.from_numpy(base))
    assert np.allclose(np_out, pt_out.numpy(), atol=1e-6)


@needs_torch
def test_dataset_in_residual_mode_masks_to_no_correction():
    store = _fake_store()
    cfg = DatasetConfig(root="unused", crop=32, random_crop=True, min_valid_fraction=0.0)
    spec = _spec()
    ds = PrecipDataset(
        cfg, TF, store=store,
        cond_transform=CondTransform.for_channels(CHANNELS),
        residual=spec,
    )
    assert ds.mask_fill == pytest.approx(spec.fill)
    item = ds[0]
    x1, mask, base = (
        item["x1"].numpy(), item["mask"].numpy(), item["base"].numpy()
    )
    assert base.shape == x1.shape
    assert np.allclose(x1[mask == 0], spec.fill)
    # decoding the masked cells reproduces the ERA5 base there
    decoded = spec.decode(x1, base)
    assert np.allclose(decoded[mask == 0], base[mask == 0], atol=1e-5)


@needs_torch
def test_dataset_residual_decodes_back_to_the_stored_target():
    """End-to-end: what the network is asked to learn must reconstruct CHIRPS."""
    store = _fake_store()
    cfg = DatasetConfig(root="unused", crop=32, random_crop=False, crop_origin=(32, 0))
    ds = PrecipDataset(
        cfg, TF, store=store,
        cond_transform=CondTransform.for_channels(CHANNELS),
        residual=_spec(),
    )
    item = ds[0]
    x1, base, mask = (
        item["x1"].numpy(), item["base"].numpy(), item["mask"].numpy()
    )
    recovered = TF.inverse(ds.residual.decode(x1, base))
    expected = item["target_mm"].numpy()
    assert np.allclose(recovered[mask > 0], expected[mask > 0], rtol=1e-4, atol=1e-4)


@needs_torch
def test_absolute_mode_dataset_is_unchanged_by_the_residual_code():
    store = _fake_store()
    cfg = DatasetConfig(root="unused", crop=32, random_crop=False, crop_origin=(32, 0))
    kwargs = dict(
        store=store, cond_transform=CondTransform.for_channels(CHANNELS)
    )
    absolute = PrecipDataset(cfg, TF, **kwargs)[0]
    explicit = PrecipDataset(cfg, TF, residual=ResidualSpec(), **kwargs)[0]
    assert np.allclose(absolute["x1"].numpy(), explicit["x1"].numpy())
    # x1 is still the plain transformed target
    mask = absolute["mask"].numpy()
    direct = TF.forward(absolute["target_mm"].numpy())
    assert np.allclose(absolute["x1"].numpy()[mask > 0], direct[mask > 0], atol=1e-5)


# --------------------------------------------------------------------------
# 7. v6: climatology residual base and conditioning-channel selection
# --------------------------------------------------------------------------


def test_climatology_base_round_trips_and_floors_at_climatology():
    """Zero residual must reproduce CLIMATOLOGY, not ERA5."""
    rng = np.random.default_rng(0)
    chirps = rng.gamma(0.4, 12.0, (48, 48))
    climatology = rng.gamma(1.5, 3.0, (48, 48))
    spec = ResidualSpec(enabled=True, mean=0.31, std=1.42, base="climatology")
    base_t = TF.forward(climatology)

    encoded = spec.encode(TF.forward(chirps), base_t)
    assert np.allclose(TF.inverse(spec.decode(encoded, base_t)), chirps,
                       rtol=1e-4, atol=1e-4)
    zero = np.full_like(base_t, spec.fill)
    assert np.allclose(TF.inverse(spec.decode(zero, base_t)), climatology,
                       rtol=1e-5, atol=1e-5)
    extreme = np.full_like(base_t, -50.0)
    assert (TF.inverse(spec.decode(extreme, base_t)) >= 0).all()


def test_residual_base_survives_serialisation():
    import json

    for base in ("era5_tp", "climatology"):
        spec = ResidualSpec(enabled=True, mean=0.1, std=1.0, base=base)
        restored = ResidualSpec.from_stats(
            json.loads(json.dumps({"residual": spec.to_dict()}))
        )
        assert restored.base == base
    # files written before the field existed default to era5_tp
    assert ResidualSpec.from_stats(
        {"residual": {"enabled": True, "mean": 0.0, "std": 1.0}}
    ).base == "era5_tp"


@needs_torch
def test_conditioning_channel_subset_drops_era5_tp():
    store = _fake_store()
    store["cond"] = store["cond"]           # (T, 6, H, W)
    keep = ("era5_tcwv", "era5_cape", "era5_u10", "era5_v10", "era5_msl")
    cfg = DatasetConfig(
        root="unused", crop=32, random_crop=False, crop_origin=(32, 0),
        cond_channels=keep,
    )
    ds = PrecipDataset(
        cfg, TF, store=store,
        cond_transform=CondTransform.for_channels(list(keep)),
    )
    assert ds.cond_channels == list(keep)
    assert "era5_tp" not in ds.cond_channels
    assert ds.n_cond == 5
    item = ds[0]
    # 5 ERA5 + 7 static + 2 seasonal
    assert item["cond"].shape[0] == ds.total_cond_channels == 14


@needs_torch
def test_dropping_a_channel_changes_what_the_network_sees():
    """Guard against the subset silently being ignored."""
    store = _fake_store()
    cfg_all = DatasetConfig(root="unused", crop=32, random_crop=False,
                            crop_origin=(32, 0))
    keep = ("era5_tcwv", "era5_cape", "era5_u10", "era5_v10", "era5_msl")
    cfg_sub = DatasetConfig(root="unused", crop=32, random_crop=False,
                            crop_origin=(32, 0), cond_channels=keep)
    full = PrecipDataset(cfg_all, TF, store=store,
                         cond_transform=CondTransform.for_channels(CHANNELS))[0]
    sub = PrecipDataset(cfg_sub, TF, store=store,
                        cond_transform=CondTransform.for_channels(list(keep)))[0]
    assert full["cond"].shape[0] == sub["cond"].shape[0] + 1
    # the retained ERA5 channels must be identical, just re-indexed
    assert np.allclose(full["cond"].numpy()[1:6], sub["cond"].numpy()[0:5])


@needs_torch
def test_unknown_conditioning_channel_raises():
    store = _fake_store()
    cfg = DatasetConfig(root="unused", crop=32, random_crop=False,
                        crop_origin=(32, 0), cond_channels=("era5_nonsense",))
    with pytest.raises(ValueError, match="not in the store"):
        PrecipDataset(cfg, TF, store=store)


@needs_torch
def test_climatology_base_without_a_climatology_array_raises():
    store = _fake_store()
    cfg = DatasetConfig(root="unused", crop=32, random_crop=False,
                        crop_origin=(32, 0))
    with pytest.raises(ValueError, match="climatology"):
        PrecipDataset(
            cfg, TF, store=store,
            residual=ResidualSpec(enabled=True, mean=0.0, std=1.0,
                                  base="climatology"),
        )


def test_v6_config_removes_precipitation_from_the_conditioning():
    """The claim 'the analysis beats every input' needs no precip in the prior."""
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "train_h100.yaml").read_text())
    channels = cfg["data"].get("cond_channels")
    assert channels, "v6 must name its conditioning channels explicitly"
    assert not any("tp" in name for name in channels), (
        f"a precipitation channel is still in the prior's conditioning: {channels}"
    )
    assert "era5_tcwv" in channels and "era5_cape" in channels, (
        "the dynamical channels should stay -- they describe the situation, "
        "not the rainfall"
    )
