"""Scientific invariants for the V3-SG hierarchical implementation."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bdhires.da import (  # noqa: E402
    AreaWeightedBlockObsOperator,
    GuidanceConfig,
    HierarchicalObservations,
    HierarchicalSamplerConfig,
    authority_decomposition,
    hierarchical_guidance_grad,
    sample_hierarchical,
)
from bdhires.data import (  # noqa: E402
    SubgridEncoding,
    SubgridDataset,
    SubgridDatasetConfig,
    aligned_production_canvas,
    allocation_log_weight_target,
    area_weighted_block_mean,
    decode_and_reconstruct,
    encode_subgrid_targets,
    reconstruct_from_amount,
    validate_aligned_crop,
    validate_cpc_alignment,
)
from bdhires.grids import BD, BD_CPC, WIDE, WIDE_CPC, crop_offsets  # noqa: E402
from bdhires.models import (  # noqa: E402
    AllocationFlow,
    CoarseHurdleFlow,
    CoupledSubgridFlow,
    HierarchicalRectifiedFlow,
    HierarchicalState,
)
from bdhires.models.hierarchical_subgrid import (  # noqa: E402
    _hurdle_velocity_mse,
    _occurrence_loss,
    allocation_flow_matching_loss,
)
from bdhires.models.unet import UNet  # noqa: E402
from bdhires.zarr_output import write_hierarchical_sample_zarr  # noqa: E402


def _encoding(**kwargs):
    values = {
        "factor": 10,
        "amount_sqrt_mean": 0.0,
        "amount_sqrt_std": 1.0,
        "dequant_noise": 0.0,
    }
    values.update(kwargs)
    return SubgridEncoding(**values)


def test_v3_domains_close_on_cpc_edges_and_preserve_legacy_bd_crop():
    validate_cpc_alignment(BD_CPC)
    validate_cpc_alignment(WIDE_CPC)
    assert WIDE_CPC.shape == (240, 240)
    assert np.array_equal(WIDE_CPC.lat, WIDE.lat[:240])
    assert np.array_equal(WIDE_CPC.lon, WIDE.lon[:240])
    assert crop_offsets(BD_CPC, BD) == (6, 2)


def test_crop_lattice_rejects_legacy_128_and_off_phase_origins():
    validate_aligned_crop((20, 30), 120, factor=10, downsamplings=3)
    with pytest.raises(ValueError, match="lcm"):
        validate_aligned_crop((0, 0), 128, factor=10, downsamplings=3)
    with pytest.raises(ValueError, match="modulo"):
        validate_aligned_crop((1, 0), 120, factor=10, downsamplings=3)


def test_production_canvas_contains_complete_bd_cpc_core_on_block_phase():
    outer, core = aligned_production_canvas(WIDE_CPC, BD_CPC, canvas=160)
    assert outer == (slice(70, 230), slice(60, 220))
    assert core == (slice(10, 150), slice(10, 140))


def test_full_and_coastal_blocks_conserve_area_weighted_amount_exactly():
    torch.manual_seed(3)
    encoding = _encoding(valid_area_threshold=0.50)
    coarse = torch.tensor([[[[7.0, 2.5]]]])
    allocation = torch.randn(1, 2, 10, 20)
    allocation[:, 1] = 4.0
    valid = torch.ones(1, 1, 10, 20, dtype=torch.bool)
    valid[..., :4, 10:15] = False  # a retained partial/coastal block
    area = torch.linspace(1.0, 1.2, 10)[:, None].expand(10, 20)
    field = reconstruct_from_amount(coarse, allocation, valid, area, encoding)
    recovered, retained, _ = area_weighted_block_mean(
        field, area, valid, factor=10, valid_area_threshold=0.50
    )
    assert retained.all()
    assert torch.allclose(recovered, coarse, atol=2.0e-5, rtol=2.0e-6)
    assert (field >= 0.0).all()
    assert (field[~valid] == 0.0).all()


def test_dry_block_is_exact_zero_but_keeps_a_positive_occurrence_gradient():
    encoding = _encoding()
    coarse_state = torch.tensor([[[[2.0]], [[-4.0]]]], requires_grad=True)
    allocation = torch.zeros(1, 2, 10, 10, requires_grad=True)
    allocation.data[:, 1] = 4.0
    valid = torch.ones(1, 1, 10, 10, dtype=torch.bool)
    area = torch.ones(10, 10)
    from bdhires.data import decode_and_reconstruct

    field = decode_and_reconstruct(
        coarse_state, allocation, torch.ones(1, 1, 1, 1, dtype=torch.bool),
        valid, area, encoding,
    )
    assert torch.count_nonzero(field) == 0
    field.sum().backward()
    assert torch.isfinite(coarse_state.grad).all()
    assert coarse_state.grad[0, 1, 0, 0].abs() > 0.0


def test_empty_wet_mask_fallback_activates_one_valid_cell_and_conserves():
    encoding = _encoding()
    coarse = torch.tensor([[[[5.0]]]])
    allocation = torch.zeros(1, 2, 10, 10)
    allocation[:, 1] = -20.0
    allocation[:, 1, 4, 7] = -10.0  # still hard-dry, but the largest probability
    field, diagnostics = reconstruct_from_amount(
        coarse, allocation, torch.ones(10, 10), torch.ones(10, 10), encoding,
        return_diagnostics=True,
    )
    assert diagnostics.empty_wet_fallbacks == 1
    assert torch.count_nonzero(field) == 1
    # One wet 0.05-degree cell must carry 100 times the 0.5-degree mean.
    assert field[0, 0, 4, 7] == pytest.approx(500.0)


def test_target_dequantisation_is_invariant_to_preparation_chunking():
    torch.manual_seed(8)
    encoding = SubgridEncoding(factor=10, dequant_seed=44, dequant_noise=0.1)
    fine = torch.rand(4, 1, 10, 10)
    valid = torch.ones(10, 10)
    area = torch.ones(10, 10)
    together = encode_subgrid_targets(fine, valid, area, encoding, sample_offset=12)
    pieces = [
        encode_subgrid_targets(
            fine[index : index + 1], valid, area, encoding, sample_offset=12 + index
        ).allocation_state
        for index in range(4)
    ]
    assert torch.equal(together.allocation_state, torch.cat(pieces))


def test_allocation_target_is_centered_standardised_and_exactly_reconstructable():
    encoding = _encoding(intensity_log_mean=0.4, intensity_log_std=1.7)
    fine = torch.linspace(0.2, 8.0, 200).reshape(1, 1, 10, 20)
    valid = torch.ones(10, 20, dtype=torch.bool)
    area = torch.linspace(1.0, 1.2, 10)[:, None].expand(10, 20)
    log_weight, wet = allocation_log_weight_target(fine, valid, area, encoding)
    block_mean, _, _ = area_weighted_block_mean(
        log_weight, area, wet, factor=10, valid_area_threshold=0.0
    )
    assert block_mean.abs().max() < 2.0e-6

    targets = encode_subgrid_targets(fine, valid, area, encoding)
    restored_log_weight = (
        targets.allocation_state[:, :1] * encoding.intensity_log_std
        + encoding.intensity_log_mean
    )
    assert torch.allclose(restored_log_weight, log_weight, atol=2.0e-6)
    reconstructed = decode_and_reconstruct(
        targets.coarse_state,
        targets.allocation_state,
        targets.coarse_valid,
        targets.fine_valid,
        area,
        encoding,
    )
    assert torch.allclose(reconstructed, fine, atol=3.0e-5, rtol=3.0e-6)


def test_dry_allocation_intensity_is_finite_neutral_not_inverse_softplus_tail():
    encoding = _encoding(intensity_log_mean=0.6, intensity_log_std=1.5)
    fine = torch.zeros(1, 1, 10, 10)
    fine[..., 2, 3] = 2.0
    fine[..., 5, 7] = 5.0
    targets = encode_subgrid_targets(
        fine, torch.ones(10, 10), torch.ones(10, 10), encoding
    )
    dry = fine < encoding.wet_threshold_mm
    expected = -encoding.intensity_log_mean / encoding.intensity_log_std
    assert torch.isfinite(targets.allocation_state).all()
    assert torch.allclose(
        targets.allocation_state[:, :1][dry],
        torch.full_like(targets.allocation_state[:, :1][dry], expected),
    )
    assert targets.allocation_state[:, :1][dry].abs().max() < 1.0

    dry_targets = encode_subgrid_targets(
        torch.zeros(1, 1, 10, 10),
        torch.ones(10, 10),
        torch.ones(10, 10),
        _encoding(amount_sqrt_mean=2.0, amount_sqrt_std=0.5),
    )
    assert torch.count_nonzero(dry_targets.coarse_state[:, :1]) == 0


def test_occurrence_bce_is_minimised_at_the_finite_dequantised_target():
    mask = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    target = torch.tensor([[[[0.0]], [[3.8918204]]]])
    matched = target.clone()
    saturated = target.clone()
    saturated[:, 1] = 20.0
    assert _occurrence_loss(matched, target, mask) < _occurrence_loss(
        saturated, target, mask
    )


def test_hurdle_velocity_loss_ignores_decoder_inactive_positive_channel():
    mask = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    clean = torch.tensor([[[[0.0, 0.0]], [[-4.0, 4.0]]]])
    prediction = torch.zeros_like(clean)
    baseline = torch.zeros_like(clean)
    changed = baseline.clone()
    changed[:, 0, 0, 0] = 1000.0  # dry: physically inactive
    assert torch.equal(
        _hurdle_velocity_mse(prediction, baseline, clean, mask),
        _hurdle_velocity_mse(prediction, changed, clean, mask),
    )
    changed[:, 0, 0, 1] = 2.0  # wet: must be learned
    assert _hurdle_velocity_mse(prediction, changed, clean, mask) > 0.0


class _CaptureAllocation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.level = None
        self.context = None

    def forward(self, state, time, fine_cond, context, level):
        self.level = level.detach().clone()
        self.context = context.detach().clone()
        return torch.zeros_like(state)


class _FixedFlow:
    def sample_t(self, batch, device):
        return torch.full((batch,), 0.25, device=device)

    def interpolate(self, target, time):
        zeros = torch.zeros_like(target)
        return zeros, zeros, zeros

    def x1_hat(self, state, time, velocity):
        return velocity


def test_phase2_coarse_corruption_matches_joint_one_minus_t(monkeypatch):
    model = _CaptureAllocation()
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.zeros_like(value))
    allocation_flow_matching_loss(
        model,
        torch.zeros(2, 2, 10, 10),
        None,
        torch.ones(2, 2, 1, 1),
        torch.ones(2, 1, 10, 10, dtype=torch.bool),
        flow=_FixedFlow(),
        max_coarse_noise=1.0,
        clean_probability=0.0,
    )
    assert torch.allclose(model.level, torch.full((2,), 0.75))
    assert torch.allclose(model.context, torch.full((2, 2, 1, 1), 0.25))


def test_allocation_coarse_context_has_exact_block_boundaries():
    model = AllocationFlow(
        0, image_size=40, base_channels=8, channel_mult=(1, 2),
        num_res_blocks=1, attn_resolutions=(20,), num_heads=1,
    )

    class CaptureNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.condition = None

        def forward(self, state, time, condition):
            self.condition = condition.detach().clone()
            return torch.zeros_like(state)

    capture = CaptureNet()
    model.net = capture
    coarse = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    model(torch.zeros(1, 2, 40, 40), torch.tensor([0.5]), None, coarse, 0.5)
    expected = coarse.repeat_interleave(10, -2).repeat_interleave(10, -1)
    assert torch.equal(capture.condition[:, :2], expected)


class _MemoryStore(dict):
    def __init__(self, *args, attrs, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs = attrs


def test_validation_dataset_tiles_the_complete_240_canvas():
    store = _MemoryStore(
        {
            "time": np.asarray(
                ["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"
            ).astype(np.int64),
            "fine_valid": np.ones((240, 240), bool),
            "cell_area": np.ones((240, 240), np.float32),
            "coarse_valid": np.ones((24, 24), bool),
        },
        attrs={
            "schema": "cpc_v3_subgrid_v2",
            "subgrid_encoding": {"factor": 10},
        },
    )
    dataset = SubgridDataset(
        SubgridDatasetConfig(
            root="unused", crop=120, random_crop=False, tile_domain=True
        ),
        store=store,
    )
    assert dataset.validation_origins == ((0, 0), (0, 120), (120, 0), (120, 120))
    assert len(dataset) == 8


def test_unet_rejects_raw_odd_cpc_core_with_actionable_error():
    model = UNet(
        in_channels=2, cond_channels=0, out_channels=2, image_size=14,
        base_channels=8, channel_mult=(1, 2, 3), num_res_blocks=1,
        num_heads=1,
    )
    with pytest.raises(ValueError, match="aligned production canvas"):
        model(torch.zeros(1, 2, 14, 13), torch.tensor([0.5]))


def test_branch_transfer_has_identical_initial_velocities():
    torch.manual_seed(9)
    coarse = CoarseHurdleFlow(
        0, image_size=4, base_channels=8, channel_mult=(1, 2),
        num_res_blocks=1, num_heads=1,
    )
    allocation = AllocationFlow(
        0, image_size=40, base_channels=8, channel_mult=(1, 2),
        num_res_blocks=1, attn_resolutions=(20,), num_heads=1,
    )
    joint = CoupledSubgridFlow(coarse, allocation)
    # Literal transfer is an inference-time invariant; disable dropout before
    # comparing two otherwise separate forward calls.
    joint.eval()
    coarse_state = torch.randn(2, 2, 4, 4)
    fine_state = torch.randn(2, 2, 40, 40)
    time = torch.full((2,), 0.35)
    expected_coarse = coarse(coarse_state, time)
    expected_fine = allocation(fine_state, time, None, coarse_state, 1.0 - time)
    actual = joint(coarse_state, fine_state, time)
    assert torch.equal(actual.coarse, expected_coarse)
    assert torch.equal(actual.allocation, expected_fine)


class _ToyJoint(torch.nn.Module):
    def forward(
        self, coarse_state, allocation_state, t, coarse_cond=None, fine_cond=None,
        **kwargs,
    ):
        # Connected zeros retain the exact denoised-state graph.
        return HierarchicalState(coarse_state * 0.0, allocation_state * 0.0)


class _Point(torch.nn.Module):
    def forward(self, field):
        return field[:, :, 3, 4].unsqueeze(-1)


def test_point_likelihood_gradient_reaches_both_coarse_and_allocation_states():
    encoding = _encoding()
    coarse = torch.tensor([[[[2.0]], [[4.0]]]])
    allocation = torch.zeros(1, 2, 10, 10)
    allocation[:, 1] = 4.0
    state = HierarchicalState(coarse, allocation)
    observations = HierarchicalObservations(
        _Point(), torch.tensor([[[10.0]]]), torch.tensor([1.0]),
        GuidanceConfig(gamma=1.0e-3, clip_norm=None),
    )
    _, gradient, _ = hierarchical_guidance_grad(
        _ToyJoint(), state, torch.tensor([0.7]), None, None, observations,
        HierarchicalRectifiedFlow(), encoding,
        torch.ones(1, 1, 1, 1, dtype=torch.bool),
        torch.ones(1, 1, 10, 10, dtype=torch.bool),
        torch.ones(10, 10), 1.0,
    )
    assert gradient.coarse.abs().sum() > 0.0
    assert gradient.allocation.abs().sum() > 0.0
    assert torch.isfinite(gradient.coarse).all()
    assert torch.isfinite(gradient.allocation).all()


def test_sampler_evaluates_terminal_hard_observation_departures():
    coarse = CoarseHurdleFlow(
        0, image_size=1, base_channels=8, channel_mult=(1,),
        num_res_blocks=1, num_heads=1,
    )
    allocation = AllocationFlow(
        0, image_size=10, base_channels=8, channel_mult=(1,),
        num_res_blocks=1, attn_resolutions=(), num_heads=1,
    )
    model = CoupledSubgridFlow(coarse, allocation)
    sample = sample_hierarchical(
        model,
        None,
        None,
        (2, 2, 1, 1),
        (2, 2, 10, 10),
        torch.ones(1, 1, 1, 1, dtype=torch.bool),
        torch.ones(1, 1, 10, 10, dtype=torch.bool),
        torch.ones(10, 10),
        _encoding(),
        observations=HierarchicalObservations(
            _Point(),
            torch.zeros(2, 1, 1),
            torch.ones(1),
            GuidanceConfig(clip_norm=None),
        ),
        config=HierarchicalSamplerConfig(n_steps=1, heun=False, seed=12),
    )
    assert sample.diagnostics["terminal_decoder_consistent"] is True
    assert sample.diagnostics["terminal_hard_decode_max_abs_mm_day"] == 0.0
    assert sample.diagnostics["terminal_valid_observation_count"] == 2
    assert np.isfinite(sample.diagnostics["terminal_log_likelihood_mean"])


def test_area_weighted_04_and_aligned_05_operators_preserve_uniform_fields():
    field = torch.full((2, 1, 40, 40), 7.25)
    area = np.linspace(1.0, 1.3, 40)[:, None] * np.ones((1, 40))
    imerg = AreaWeightedBlockObsOperator(8, area)
    aligned_control = AreaWeightedBlockObsOperator(10, area)
    assert torch.allclose(imerg(field), torch.full_like(imerg(field), 7.25))
    assert torch.allclose(
        aligned_control(field), torch.full_like(aligned_control(field), 7.25)
    )


def test_physical_authority_decomposition_closes_exactly():
    encoding = _encoding()
    background = HierarchicalState(
        torch.tensor([[[[2.0]], [[4.0]]]]),
        torch.cat([torch.zeros(1, 1, 10, 10), torch.full((1, 1, 10, 10), 4.0)], 1),
    )
    analysis = HierarchicalState(
        torch.tensor([[[[2.3]], [[4.0]]]]),
        background.allocation + 0.1 * torch.randn_like(background.allocation),
    )
    amount, allocation, residual = authority_decomposition(
        background, analysis,
        torch.ones(1, 1, 1, 1, dtype=torch.bool),
        torch.ones(1, 1, 10, 10, dtype=torch.bool),
        torch.ones(10, 10), encoding,
    )
    assert amount.shape == allocation.shape == residual.shape
    assert residual.abs().max() < 1.0e-5


def test_authority_decomposition_assigns_pure_amount_and_allocation_changes():
    encoding = _encoding()
    coarse = torch.tensor([[[[2.0]], [[4.0]]]])
    allocation_state = torch.cat(
        [torch.zeros(1, 1, 10, 10), torch.full((1, 1, 10, 10), 4.0)], 1
    )
    background = HierarchicalState(coarse, allocation_state)
    masks = (
        torch.ones(1, 1, 1, 1, dtype=torch.bool),
        torch.ones(1, 1, 10, 10, dtype=torch.bool),
        torch.ones(10, 10),
    )

    amount_analysis = HierarchicalState(
        coarse + torch.tensor([[[[0.5]], [[0.0]]]]), allocation_state
    )
    amount, allocation, residual = authority_decomposition(
        background, amount_analysis, *masks, encoding
    )
    assert amount.abs().sum() > 0.0
    assert allocation.abs().max() < 1.0e-7
    assert residual.abs().max() < 1.0e-6

    changed_allocation = allocation_state.clone()
    changed_allocation[:, 0, :5] = 1.0
    allocation_analysis = HierarchicalState(coarse, changed_allocation)
    amount, allocation, residual = authority_decomposition(
        background, allocation_analysis, *masks, encoding
    )
    assert amount.abs().max() < 1.0e-7
    assert allocation.abs().sum() > 0.0
    assert residual.abs().max() < 1.0e-6


def test_hierarchical_archive_requires_and_roundtrips_terminal_hard_field(tmp_path):
    fields = {"background": np.ones((2, 2, 10, 10), np.float32)}
    coarse_states = {"background": np.ones((2, 2, 2, 1, 1), np.float32)}
    allocation_states = {"background": np.ones((2, 2, 2, 10, 10), np.float32)}
    diagnostics = {
        "background": {
            "terminal_decoder_consistent": True,
            "terminal_hard_decode_max_abs_mm_day": 0.0,
        }
    }
    with pytest.raises(ValueError, match="terminal hard-decoder diagnostic"):
        write_hierarchical_sample_zarr(
            tmp_path / "rejected.zarr",
            fields=fields,
            coarse_states=coarse_states,
            allocation_states=allocation_states,
            selected_times=np.asarray(["2021-05-01", "2021-05-02"]),
            lat=np.arange(10, dtype=np.float32),
            lon=np.arange(10, dtype=np.float32),
            valid=np.ones((10, 10), bool),
            diagnostics={"background": {"terminal_decoder_consistent": False}},
        )
    output = tmp_path / "samples.zarr"
    write_hierarchical_sample_zarr(
        output,
        fields=fields,
        coarse_states=coarse_states,
        allocation_states=allocation_states,
        selected_times=np.asarray(["2021-05-01", "2021-05-02"], dtype="datetime64[D]"),
        lat=np.arange(10, dtype=np.float32),
        lon=np.arange(10, dtype=np.float32),
        valid=np.ones((10, 10), bool),
        diagnostics=diagnostics,
        coarse_mm={"background": np.ones((2, 2, 1, 1), np.float32)},
    )
    import zarr

    archive = zarr.open_group(str(output), mode="r")
    assert archive.attrs["complete"] is True
    assert archive.attrs["archive_uses_likelihood_hard_decoder"] is True
    assert archive.attrs["serialization_max_abs_mm_day"] == 0.0
    assert np.array_equal(archive["background"][:], fields["background"])
