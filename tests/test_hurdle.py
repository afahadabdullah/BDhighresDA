"""Tests for the hurdle (dry/wet) head that gives the prior an atom at zero.

The defect being fixed: rectified flow is a continuous density and CHIRPS has
~54% of its mass at exactly 0 mm.  The v1 prior therefore rained on 64.9% of
cells against a target of 45.9%, worst in dry regions -- which is where the BMD
gauges sit, so it surfaced as +5.88 mm/day at stations.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bdhires.models.flow import (  # noqa: E402
    RectifiedFlow,
    VelocityOnly,
    apply_dry_mask,
    flow_matching_loss,
    predict_dry_logit,
    split_prediction,
)
from bdhires.transforms import PrecipTransform  # noqa: E402


class TwoChannelStub(torch.nn.Module):
    """Minimal stand-in: channel 0 is velocity, channel 1 is the dry logit."""

    def __init__(self, out_channels: int = 2, dry_bias: float = 0.0):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, out_channels, 3, padding=1)
        self.out_channels = out_channels
        with torch.no_grad():
            if out_channels > 1:
                self.conv.bias[1] = dry_bias

    def forward(self, x, t, cond=None):
        return self.conv(x)


# ------------------------------------------------------------------ masking


def test_dry_mask_writes_the_transformed_value_of_zero_not_zero():
    """In transformed space "no rain" is -mu/sd, so writing 0.0 would be wrong."""
    transform = PrecipTransform(kind="log1p", eps=0.1, mu=1.194, sd=2.179)
    dry_value = float(np.asarray(transform.forward(np.float32(0.0))))
    assert dry_value == pytest.approx(-0.548, abs=1e-2)

    field = torch.full((1, 1, 4, 4), 0.9)
    logit = torch.full((1, 1, 4, 4), 10.0)  # confidently dry
    out = apply_dry_mask(field, logit, dry_value=dry_value)
    assert torch.allclose(out, torch.full_like(out, dry_value))
    # and it really decodes to zero rainfall
    back = transform.inverse(out.numpy())
    assert np.allclose(back, 0.0, atol=1e-6)


def test_dry_mask_leaves_confident_wet_cells_untouched():
    field = torch.randn(2, 1, 8, 8)
    logit = torch.full((2, 1, 8, 8), -10.0)  # confidently wet
    out = apply_dry_mask(field, logit, dry_value=-0.548)
    assert torch.allclose(out, field)


def test_sample_mode_reproduces_the_target_wet_fraction():
    """Bernoulli masking must give the right marginal, which thresholding need not."""
    torch.manual_seed(0)
    p_dry = 0.55
    logit = torch.full((1, 1, 400, 400), float(np.log(p_dry / (1 - p_dry))))
    field = torch.zeros_like(logit)
    out = apply_dry_mask(field, logit, dry_value=-1.0, mode="sample")
    observed_dry = (out == -1.0).float().mean().item()
    assert observed_dry == pytest.approx(p_dry, abs=0.01)


def test_unknown_mask_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown dry mask mode"):
        apply_dry_mask(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), 0.0, mode="magic")


# ------------------------------------------------------------------ plumbing


def test_split_prediction_separates_the_two_heads():
    pred = torch.stack(
        [torch.full((1, 4, 4), 1.0), torch.full((1, 4, 4), 2.0)], dim=1
    ).squeeze(2)
    velocity, dry = split_prediction(pred, hurdle=True)
    assert velocity.shape[1] == 1 and dry.shape[1] == 1
    assert torch.allclose(velocity, torch.ones_like(velocity))
    assert torch.allclose(dry, torch.full_like(dry, 2.0))


def test_split_prediction_rejects_a_single_channel_model():
    with pytest.raises(ValueError, match="out_channels >= 2"):
        split_prediction(torch.zeros(1, 1, 4, 4), hurdle=True)


def test_velocity_only_hides_the_hurdle_channel_from_the_sampler():
    """The sampler and guidance code must keep seeing a 1-channel velocity."""
    model = TwoChannelStub(out_channels=2)
    x = torch.randn(2, 1, 8, 8)
    t = torch.full((2,), 0.5)
    assert model(x, t).shape[1] == 2
    assert VelocityOnly(model)(x, t).shape[1] == 1


def test_velocity_only_is_a_no_op_for_single_output_models():
    model = TwoChannelStub(out_channels=1)
    x = torch.randn(1, 1, 8, 8)
    t = torch.full((1,), 0.5)
    assert torch.allclose(VelocityOnly(model)(x, t), model(x, t))


def test_predict_dry_logit_refuses_a_model_without_the_head():
    with pytest.raises(ValueError, match="no hurdle head"):
        predict_dry_logit(TwoChannelStub(out_channels=1), torch.randn(1, 1, 8, 8), None)


# ------------------------------------------------------------------ the loss


def test_hurdle_loss_is_zero_when_the_head_is_already_right():
    """A confidently-correct classifier should contribute no BCE."""
    torch.manual_seed(0)
    flow = RectifiedFlow()
    x1 = torch.randn(2, 1, 8, 8)
    dry = torch.zeros(2, 1, 8, 8)

    class Perfect(torch.nn.Module):
        def forward(self, x, t, cond=None):
            velocity = torch.zeros_like(x)
            logit = torch.full_like(x, -30.0)  # confidently wet, matching dry=0
            return torch.cat([velocity, logit], dim=1)

    _, _, _, hurdle = flow_matching_loss(
        Perfect(), x1, None, flow, dry_target=dry,
        cond_dropout=0.0, return_components=True,
    )
    assert hurdle.item() == pytest.approx(0.0, abs=1e-6)


def test_hurdle_loss_penalises_a_wrong_classifier():
    torch.manual_seed(0)
    flow = RectifiedFlow()
    x1 = torch.randn(2, 1, 8, 8)
    dry = torch.ones(2, 1, 8, 8)  # everything dry

    class Wrong(torch.nn.Module):
        def forward(self, x, t, cond=None):
            return torch.cat([torch.zeros_like(x), torch.full_like(x, -5.0)], dim=1)

    _, _, _, hurdle = flow_matching_loss(
        Wrong(), x1, None, flow, dry_target=dry,
        cond_dropout=0.0, return_components=True,
    )
    assert hurdle.item() > 4.0


def test_dry_weight_shifts_the_flow_loss_toward_dry_cells():
    """dry_weight>1 must actually change the objective, not just be accepted."""
    torch.manual_seed(0)
    flow = RectifiedFlow()
    x1 = torch.randn(4, 1, 16, 16)
    dry = (torch.rand(4, 1, 16, 16) > 0.5).float()
    model = TwoChannelStub()

    def flow_loss_at(weight):
        torch.manual_seed(123)  # identical t and x0 draws
        _, fm, _, _ = flow_matching_loss(
            model, x1, None, flow, dry_target=dry, dry_weight=weight,
            cond_dropout=0.0, return_components=True,
        )
        return fm.item()

    assert flow_loss_at(1.0) != pytest.approx(flow_loss_at(4.0), rel=1e-6)


def test_hurdle_off_preserves_the_original_three_component_behaviour():
    """Without dry_target the loss must behave exactly as it did before."""
    torch.manual_seed(0)
    flow = RectifiedFlow()
    x1 = torch.randn(2, 1, 8, 8)
    model = TwoChannelStub(out_channels=1)
    total, fm, coarse = flow_matching_loss(
        model, x1, None, flow, cond_dropout=0.0, return_components=True
    )[:3]
    assert torch.isfinite(total) and coarse.item() == 0.0
    assert total.item() == pytest.approx(fm.item(), rel=1e-6)


def test_mask_excludes_ocean_from_both_losses():
    """Ocean cells must not contribute to the hurdle BCE either."""
    torch.manual_seed(0)
    flow = RectifiedFlow()
    x1 = torch.randn(2, 1, 8, 8)
    dry = torch.ones(2, 1, 8, 8)
    mask = torch.zeros(2, 1, 8, 8)
    mask[:, :, :4, :] = 1.0

    class Wrong(torch.nn.Module):
        def forward(self, x, t, cond=None):
            logit = torch.full_like(x, -5.0)
            logit[:, :, :4, :] = 5.0  # right on land, wrong over "ocean"
            return torch.cat([torch.zeros_like(x), logit], dim=1)

    _, _, _, hurdle = flow_matching_loss(
        Wrong(), x1, None, flow, mask=mask, dry_target=dry,
        cond_dropout=0.0, return_components=True,
    )
    # only the land half counts, and there the classifier is right
    assert hurdle.item() < 0.02


# ------------------------------------------------------------- the whole point


def test_hurdle_head_can_hit_a_wet_fraction_a_continuous_model_cannot():
    """The v1 defect, reproduced and then fixed, in the smallest possible form.

    Truth has an ATOM: dry cells sit exactly at T(0). A continuous model cannot
    place mass on a point, so its dry cells land NEAR the atom with spread (and,
    in residual mode, a little above it from under-subtracting the wet CPC
    base). Enough of that spread crosses the 1 mm threshold to inflate the wet
    fraction -- the v1 prior read 0.649 against 0.459.

    The mask must use the CONDITIONAL P(dry) the head predicts. Applying the
    marginal independently of the field would zero wet cells at the same rate as
    dry ones and overshoot into a dry bias, which is a real way to get this
    wrong.
    """
    transform = PrecipTransform(kind="log1p", eps=0.1, mu=1.194, sd=2.179)
    dry_value = float(np.asarray(transform.forward(np.float32(0.0))))
    generator = torch.Generator().manual_seed(0)
    n = 200_000

    true_dry = torch.rand(1, 1, n, 1, generator=generator) < 0.541
    truth = torch.where(
        true_dry, torch.tensor(dry_value),
        torch.randn(1, 1, n, 1, generator=generator) * 0.7 + 1.2,
    )
    wet_truth = float((transform.inverse(truth.numpy()) >= 1.0).mean())

    continuous = torch.where(
        true_dry,
        torch.randn(1, 1, n, 1, generator=generator) * 0.95 + (dry_value + 0.45),
        torch.randn(1, 1, n, 1, generator=generator) * 0.7 + 1.2,
    )
    wet_before = float((transform.inverse(continuous.numpy()) >= 1.0).mean())

    # informative but imperfect classifier, i.e. P(dry | field)
    logit = torch.where(true_dry, 2.5, -2.5) + torch.randn(
        1, 1, n, 1, generator=generator
    )
    masked = apply_dry_mask(
        continuous, logit, dry_value=dry_value, mode="sample", generator=generator
    )
    wet_after = float((transform.inverse(masked.numpy()) >= 1.0).mean())

    assert wet_before > wet_truth + 0.08, (wet_before, wet_truth)
    assert abs(wet_after - wet_truth) < 0.45 * abs(wet_before - wet_truth)


def test_validation_loss_ignores_the_hurdle_channel():
    """Regression: val_ema read 31.8 on hurdle arms against 0.16 without one.

    ``flow_matching_loss`` used to decide whether to split the model output by
    asking whether a ``dry_target`` had been supplied.  Validation passes none,
    so a 2-channel model went unsplit and its dry LOGIT was broadcast against
    the velocity target and scored as flow error.  best.pt would have been
    selected on that number.
    """
    torch = pytest.importorskip("torch")
    from bdhires.models.flow import RectifiedFlow, flow_matching_loss

    class TwoChannel(torch.nn.Module):
        """Velocity in channel 0, a large constant logit in channel 1."""

        def forward(self, x, t, cond=None):
            velocity = torch.zeros_like(x)
            logit = torch.full_like(x, 8.0)
            return torch.cat([velocity, logit], dim=1)

    class OneChannel(torch.nn.Module):
        def forward(self, x, t, cond=None):
            return torch.zeros_like(x)

    torch.manual_seed(0)
    x1 = torch.randn(2, 1, 8, 8)
    flow = RectifiedFlow()

    torch.manual_seed(1)
    two = flow_matching_loss(TwoChannel(), x1, None, flow, cond_dropout=0.0)
    torch.manual_seed(1)
    one = flow_matching_loss(OneChannel(), x1, None, flow, cond_dropout=0.0)

    # The hurdle channel must not contribute: both models predict zero velocity.
    assert float(two) == pytest.approx(float(one), rel=1e-5)
    assert float(two) < 10.0


def test_predict_dry_logit_broadcasts_a_single_case_over_an_ensemble():
    """Regression: the validation monitor skipped every epoch on hurdle arms.

    It samples an ENSEMBLE against a single case's conditioning, then reads the
    dry logit.  Without expanding the conditioning the UNet concatenates a
    batch-8 state onto a batch-1 condition and raises, which the monitor caught
    and reported as a skip -- silently discarding the wet-fraction metric that
    the whole v3 decision gate rests on.
    """
    torch = pytest.importorskip("torch")
    from bdhires.models.flow import predict_dry_logit

    class NeedsMatchingBatch(torch.nn.Module):
        def forward(self, x, t, cond=None):
            if cond is not None and cond.shape[0] != x.shape[0]:
                raise RuntimeError(
                    "Sizes of tensors must match except in dimension 1"
                )
            return torch.cat([torch.zeros_like(x), torch.ones_like(x)], dim=1)

    ensemble = torch.zeros(8, 1, 6, 6)
    single_case = torch.zeros(1, 7, 6, 6)

    logit = predict_dry_logit(NeedsMatchingBatch(), ensemble, single_case)

    assert logit.shape == (8, 1, 6, 6)


def test_predict_dry_logit_refuses_an_ambiguous_batch():
    torch = pytest.importorskip("torch")
    from bdhires.models.flow import predict_dry_logit

    class Any(torch.nn.Module):
        def forward(self, x, t, cond=None):
            return torch.cat([x, x], dim=1)

    with pytest.raises(ValueError, match="cannot broadcast"):
        predict_dry_logit(Any(), torch.zeros(8, 1, 4, 4), torch.zeros(3, 7, 4, 4))
