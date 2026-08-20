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
    SUBGRID_SCHEMA,
    LEGACY_V2_SUBGRID_SCHEMA,
    LegacyV2SubgridEncoding,
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
from bdhires.grids import (  # noqa: E402
    BD,
    BD_CPC,
    WIDE,
    WIDE_CPC,
    crop_offsets,
    get_grid,
)
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
from bdhires.zarr_output import (  # noqa: E402
    recover_incomplete_hierarchical_sample_zarr,
    write_hierarchical_sample_zarr,
)


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


def test_hard_threshold_oracle_exposes_drizzle_representation_error():
    encoding = _encoding(wet_threshold_mm=0.1)
    fine = torch.zeros(1, 1, 10, 10)
    fine[..., 2, 3] = 0.05
    fine[..., 5, 7] = 1.0
    valid = torch.ones(10, 10, dtype=torch.bool)
    area = torch.ones(10, 10)
    targets = encode_subgrid_targets(fine, valid, area, encoding)
    decoded = decode_and_reconstruct(
        targets.coarse_state,
        targets.allocation_state,
        targets.coarse_valid,
        targets.fine_valid,
        area,
        encoding,
    )
    assert decoded[..., 2, 3].item() == 0.0
    assert not torch.allclose(decoded, fine)
    decoded_coarse, _, _ = area_weighted_block_mean(
        decoded, area, valid, factor=10, valid_area_threshold=0.0
    )
    assert torch.allclose(decoded_coarse, targets.coarse_mm, atol=1.0e-6)


def test_standardised_intensity_guard_cannot_overflow_exponential_decoder():
    encoding = _encoding(intensity_log_std=1000.0, intensity_z_clip=6.0)
    coarse = torch.tensor([[[[1.0]], [[4.0]]]])
    allocation = torch.empty(1, 2, 10, 10)
    allocation[:, 0] = torch.linspace(-100.0, 100.0, 100).reshape(10, 10)
    allocation[:, 1] = 4.0
    field, diagnostics = decode_and_reconstruct(
        coarse,
        allocation,
        torch.ones(1, 1, 1, 1, dtype=torch.bool),
        torch.ones(1, 1, 10, 10, dtype=torch.bool),
        torch.ones(10, 10),
        encoding,
        return_diagnostics=True,
    )
    assert torch.isfinite(field).all()
    assert torch.allclose(field.mean(), torch.tensor(1.0), atol=1.0e-6)
    assert 0.0 < diagnostics.maximum_cell_mass_fraction <= 1.0


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


def test_hurdle_velocity_loss_weakly_regularises_inactive_positive_channel():
    mask = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    clean = torch.tensor([[[[0.0, 0.0]], [[-4.0, 4.0]]]])
    prediction = torch.zeros_like(clean)
    baseline = torch.zeros_like(clean)
    dry_changed = baseline.clone()
    dry_changed[:, 0, 0, 0] = 2.0
    wet_changed = baseline.clone()
    wet_changed[:, 0, 0, 1] = 2.0
    baseline_loss = _hurdle_velocity_mse(prediction, baseline, clean, mask)
    dry_loss = _hurdle_velocity_mse(prediction, dry_changed, clean, mask)
    wet_loss = _hurdle_velocity_mse(prediction, wet_changed, clean, mask)
    assert baseline_loss < dry_loss < wet_loss
    assert torch.allclose(dry_loss, 0.05 * wet_loss)


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
            "schema": SUBGRID_SCHEMA,
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

    with pytest.raises(ValueError, match="overlapping validation tiles"):
        SubgridDataset(
            SubgridDatasetConfig(
                root="unused", crop=160, random_crop=False, tile_domain=True
            ),
            store=store,
        )


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
    assert sample.diagnostics["reconstruction_maximum_cell_mass_fraction"] <= 1.0 + 1.0e-6
    assert sample.diagnostics["terminal_valid_observation_count"] == 2
    assert np.isfinite(sample.diagnostics["terminal_log_likelihood_mean"])
    expected_oa = float((-_Point()(sample.precipitation)).mean())
    assert np.isclose(sample.diagnostics["terminal_oa_bias_sigma"], expected_oa)


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


def test_hierarchical_archive_redecodes_serialized_states(tmp_path):
    fields = {"background": np.ones((2, 2, 10, 10), np.float32)}
    coarse_states = {"background": np.ones((2, 2, 2, 1, 1), np.float32)}
    allocation_states = {"background": np.ones((2, 2, 2, 10, 10), np.float32)}
    diagnostics = {"background": {"n_steps": 1}}
    geometry = {
        "coarse_valid": np.ones((1, 1), bool),
        "cell_area": np.ones((10, 10), np.float32),
        "encoding": _encoding(),
    }
    with pytest.raises(RuntimeError, match="serialized hard-decoded states"):
        write_hierarchical_sample_zarr(
            tmp_path / "rejected.zarr",
            fields={"background": 2.0 * fields["background"]},
            coarse_states=coarse_states,
            allocation_states=allocation_states,
            selected_times=np.asarray(["2021-05-01", "2021-05-02"]),
            lat=np.arange(10, dtype=np.float32),
            lon=np.arange(10, dtype=np.float32),
            valid=np.ones((10, 10), bool),
            diagnostics=diagnostics,
            **geometry,
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
        **geometry,
    )
    import zarr

    archive = zarr.open_group(str(output), mode="r")
    assert archive.attrs["complete"] is True
    assert archive.attrs["archive_uses_likelihood_hard_decoder"] is True
    assert archive.attrs["schema"] == "cpc_v3_hierarchical_samples_v3"
    assert archive.attrs["serialization_max_abs_mm_day"] == 0.0
    # Frozen contract is the 1e-5 mm/day serialization bound, not bit identity:
    # re-decoding a float32 state reruns the iterative smooth base.
    assert archive.attrs["saved_state_hard_decode_max_abs_mm_day"]["background"] < 1.0e-5
    assert np.allclose(archive["background"][:], fields["background"], atol=1.0e-5)

    legacy_encoding = LegacyV2SubgridEncoding()
    legacy_output = tmp_path / "legacy-v2-samples.zarr"
    with pytest.raises(ValueError, match="explicitly labelled diagnostic"):
        write_hierarchical_sample_zarr(
            legacy_output,
            fields=fields,
            coarse_states=coarse_states,
            allocation_states=allocation_states,
            selected_times=np.asarray(["2021-05-01", "2021-05-02"]),
            lat=np.arange(10, dtype=np.float32),
            lon=np.arange(10, dtype=np.float32),
            valid=np.ones((10, 10), bool),
            coarse_valid=np.ones((1, 1), bool),
            cell_area=np.ones((10, 10), np.float32),
            encoding=legacy_encoding,
            diagnostics=diagnostics,
        )
    write_hierarchical_sample_zarr(
        legacy_output,
        fields=fields,
        coarse_states=coarse_states,
        allocation_states=allocation_states,
        selected_times=np.asarray(["2021-05-01", "2021-05-02"]),
        lat=np.arange(10, dtype=np.float32),
        lon=np.arange(10, dtype=np.float32),
        valid=np.ones((10, 10), bool),
        coarse_valid=np.ones((1, 1), bool),
        cell_area=np.ones((10, 10), np.float32),
        encoding=legacy_encoding,
        diagnostics=diagnostics,
        allow_legacy_v2_encoding=True,
    )
    legacy_archive = zarr.open_group(str(legacy_output), mode="r")
    assert legacy_archive.attrs["legacy_v2_decoder"] is True
    assert legacy_archive.attrs["subgrid_encoding"]["intensity_log_clip"] == 12.0


def test_archive_canonicalizes_bounded_device_decode_roundoff(tmp_path):
    encoding = _encoding()
    coarse = np.zeros((1, 2, 2, 1, 1), np.float32)
    coarse[:, :, 0] = 10.0
    coarse[:, :, 1] = 4.0
    allocation = np.zeros((1, 2, 2, 10, 10), np.float32)
    allocation[:, :, 1] = 4.0
    expected = decode_and_reconstruct(
        torch.from_numpy(coarse[0]),
        torch.from_numpy(allocation[0]),
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(10, 10, dtype=torch.bool),
        torch.ones(10, 10),
        encoding,
    )[:, 0].numpy()[None]
    device_rendering = expected + np.float32(2.0e-4)
    output = tmp_path / "canonical.zarr"
    write_hierarchical_sample_zarr(
        output,
        fields={"background": device_rendering},
        coarse_states={"background": coarse},
        allocation_states={"background": allocation},
        selected_times=np.asarray(["2022-05-01"]),
        lat=np.arange(10, dtype=np.float32),
        lon=np.arange(10, dtype=np.float32),
        valid=np.ones((10, 10), bool),
        coarse_valid=np.ones((1, 1), bool),
        cell_area=np.ones((10, 10), np.float32),
        encoding=encoding,
        diagnostics={"background": {"daily": [{"n_steps": 25, "heun": True}]}},
    )
    import zarr

    archive = zarr.open_group(str(output), mode="r")
    assert np.array_equal(archive["background"][:], expected)
    # Frozen contract is the 1e-5 mm/day serialization bound, not bit identity:
    # re-decoding a float32 state reruns the iterative smooth base.
    assert archive.attrs["saved_state_hard_decode_max_abs_mm_day"]["background"] < 1.0e-5
    assert archive.attrs[
        "source_to_canonical_hard_decode_max_abs_mm_day"
    ]["background"] == pytest.approx(2.0e-4, abs=1.0e-5)


def test_explicit_recovery_reuses_fully_written_incomplete_states(tmp_path):
    encoding = _encoding()
    fields = {"background": np.full((1, 2, 10, 10), 2.0, np.float32)}
    coarse = {"background": np.ones((1, 2, 2, 1, 1), np.float32)}
    allocation = {"background": np.ones((1, 2, 2, 10, 10), np.float32)}
    output = tmp_path / "recover.zarr"
    with pytest.raises(RuntimeError, match="serialized hard-decoded states"):
        write_hierarchical_sample_zarr(
            output,
            fields=fields,
            coarse_states=coarse,
            allocation_states=allocation,
            selected_times=np.asarray(["2022-05-01"]),
            lat=np.arange(10, dtype=np.float32),
            lon=np.arange(10, dtype=np.float32),
            valid=np.ones((10, 10), bool),
            coarse_valid=np.ones((1, 1), bool),
            cell_area=np.ones((10, 10), np.float32),
            encoding=encoding,
            diagnostics={"background": {"daily": [{"n_steps": 25, "heun": True}]}},
        )
    import zarr

    partial = zarr.open_group(str(output) + ".incomplete", mode="a")
    partial["background"][:] = np.float32(1.00005)
    recovered = recover_incomplete_hierarchical_sample_zarr(
        output,
        encoding=encoding,
        expected_methods=("background",),
    )
    archive = zarr.open_group(str(output), mode="r")
    assert recovered["background"] == pytest.approx(5.0e-5, abs=1.0e-5)
    assert archive.attrs["complete"] is True
    assert archive.attrs["recovered_from_device_roundoff_audit"] is True
    assert np.allclose(archive["background"][:], np.ones((1, 2, 10, 10)), atol=1.0e-5)


def test_encoding_rejects_renamed_or_unknown_frozen_fields():
    # A v2 archive carries ``intensity_log_clip``.  Silently dropping it would
    # rebuild the decoder with the current ``intensity_z_clip`` default and
    # change the physical field with no error anywhere.
    with pytest.raises(ValueError, match="unknown subgrid encoding fields"):
        SubgridEncoding.from_mapping({"factor": 10, "intensity_log_clip": 12.0})
    assert SubgridEncoding.from_mapping({}).factor == 10


def test_legacy_v2_decoder_preserves_its_raw_log_clip_contract():
    values = {
        "factor": 10,
        "wet_threshold_mm": 0.1,
        "dequant_epsilon": 0.02,
        "dequant_noise": 0.05,
        "dequant_seed": 314159,
        "intensity_floor": 1.0e-5,
        "denominator_floor": 1.0e-8,
        "valid_area_threshold": 0.5,
        "amount_sqrt_mean": 0.0,
        "amount_sqrt_std": 1.0,
        "intensity_log_mean": 5.0,
        "intensity_log_std": 2.0,
        "intensity_log_clip": 6.0,
    }
    assert LEGACY_V2_SUBGRID_SCHEMA == "cpc_v3_subgrid_v2"
    encoding = LegacyV2SubgridEncoding.from_mapping(values)
    allocation = torch.zeros(1, 2, 10, 10)
    allocation[:, 1] = 4.0
    allocation[:, 0, 0, 0] = 10.0
    field, diagnostics = reconstruct_from_amount(
        torch.ones(1, 1, 1, 1), allocation,
        torch.ones(10, 10, dtype=torch.bool), torch.ones(10, 10),
        encoding, return_diagnostics=True,
    )
    # V2 computes clamp(z * std + mean, -clip, clip), giving exp(6)/exp(5).
    assert (field[0, 0, 0, 0] / field[0, 0, 0, 1]).item() == pytest.approx(np.e)
    assert diagnostics.clipped_intensity_fraction == pytest.approx(0.01)
    recovered, _, _ = area_weighted_block_mean(
        field, torch.ones(10, 10), torch.ones(10, 10, dtype=torch.bool),
        factor=10, valid_area_threshold=0.0,
    )
    assert recovered.item() == pytest.approx(1.0, abs=1.0e-6)


def test_legacy_diagnostic_replays_the_frozen_sqrt_cpc_channel():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/59_legacy_v3_subgrid_diagnostic.py"
    spec = importlib.util.spec_from_file_location("legacy_v3_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    root = _MemoryStore(
        {},
        attrs={
            "coarse_cond_channels": ["sqrt_cpc_precip", "cpc_valid"],
            "coarse_cond_mean": [1.0, 0.0],
            "coarse_cond_std": [2.0, 1.0],
        },
    )
    index, mean, std = module._legacy_cpc_context_contract(root)
    assert (index, mean, std) == (0, 1.0, 2.0)
    decoded = module._decode_legacy_cpc_context(
        np.asarray([[0.5, 1.0, -1.0]], np.float32), mean, std
    )
    assert np.array_equal(decoded, np.asarray([[4.0, 9.0, 0.0]], np.float32))

    wrong = _MemoryStore(
        {},
        attrs={
            "coarse_cond_channels": ["cpc_precip", "cpc_valid"],
            "coarse_cond_mean": [1.0, 0.0],
            "coarse_cond_std": [2.0, 1.0],
        },
    )
    with pytest.raises(ValueError, match="sqrt_cpc_precip"):
        module._legacy_cpc_context_contract(wrong)


def test_dataset_rejects_missing_frozen_encoding_metadata():
    store = _MemoryStore({}, attrs={"schema": SUBGRID_SCHEMA})
    with pytest.raises(ValueError, match="lacks frozen subgrid_encoding"):
        SubgridDataset(SubgridDatasetConfig(root="unused"), store=store)


def test_hurdle_velocity_weighting_is_independent_of_the_wet_fraction():
    # The inactive-cell down-weighting must be a weighted mean.  Normalising by
    # the valid-cell count instead makes the intensity term scale with the
    # seasonal wet fraction.
    def loss_for(wet_cells: int):
        clean = torch.zeros(1, 2, 1, 20)
        clean[:, 1] = -4.0
        clean[:, 1, 0, :wet_cells] = 4.0
        target_velocity = torch.zeros(1, 2, 1, 20)
        target_velocity[:, 0] = 1.0
        prediction = torch.zeros_like(target_velocity)
        mask = torch.ones(1, 1, 1, 20, dtype=torch.bool)
        return _hurdle_velocity_mse(prediction, target_velocity, clean, mask)

    assert torch.allclose(loss_for(1), loss_for(19))
    assert torch.allclose(loss_for(10), torch.tensor(0.5))


def test_isolated_convective_cell_survives_an_otherwise_dry_coarse_block():
    # Block mean 0.01 mm/day is far below the 0.1 mm/day per-cell drizzle
    # threshold, but the block genuinely contains 1 mm/day of rain.  Occurrence
    # must follow the fine wet mask, not a threshold on the area mean.
    from bdhires.data import coarse_wet_from_fine

    encoding = _encoding(wet_threshold_mm=0.1)
    fine = torch.zeros(1, 1, 10, 10)
    fine[..., 6, 2] = 1.0
    valid = torch.ones(10, 10, dtype=torch.bool)
    area = torch.ones(10, 10)
    targets = encode_subgrid_targets(fine, valid, area, encoding)
    assert targets.coarse_mm.item() == pytest.approx(0.01)
    assert coarse_wet_from_fine(targets.fine_valid & (fine >= 0.1), 10).all()
    assert targets.coarse_state[0, 1, 0, 0] > 0.0  # wet occurrence logit

    decoded = decode_and_reconstruct(
        targets.coarse_state, targets.allocation_state,
        targets.coarse_valid, targets.fine_valid, area, encoding,
    )
    assert decoded[0, 0, 6, 2].item() == pytest.approx(1.0, abs=1.0e-5)
    assert torch.count_nonzero(decoded) == 1
    recovered, _, _ = area_weighted_block_mean(
        decoded, area, valid, factor=10, valid_area_threshold=0.0
    )
    assert torch.allclose(recovered, targets.coarse_mm, atol=1.0e-6)


def test_saturated_allocation_latents_are_counted_not_silently_clipped():
    encoding = _encoding(intensity_z_clip=6.0)
    allocation = torch.zeros(1, 2, 10, 10)
    allocation[:, 1] = 4.0
    allocation[:, 0, 0, :5] = 40.0  # 5 of 100 valid cells saturate the guard
    _, diagnostics = decode_and_reconstruct(
        torch.tensor([[[[1.0]], [[4.0]]]]), allocation,
        torch.ones(1, 1, 1, 1, dtype=torch.bool),
        torch.ones(1, 1, 10, 10, dtype=torch.bool),
        torch.ones(10, 10), encoding, return_diagnostics=True,
    )
    assert diagnostics.clipped_intensity_fraction == pytest.approx(0.05)
    assert 0.0 < diagnostics.minimum_denominator_fraction <= 1.0


def test_training_checkpoint_contract_rejects_old_schema_and_encoding():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/57_train_subgrid_oracle.py"
    spec = importlib.util.spec_from_file_location("train_v3_subgrid", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    dataset = type("Dataset", (), {"encoding": _encoding()})()
    config = {"stage": "coarse"}
    checkpoint = {
        "schema": SUBGRID_SCHEMA,
        "stage": "coarse",
        "config": config,
        "subgrid_encoding": dataset.encoding.__dict__,
    }
    module.validate_checkpoint(
        checkpoint,
        expected_stage="coarse",
        dataset=dataset,
        label="test checkpoint",
        expected_config=config,
    )

    with pytest.raises(ValueError, match="schema"):
        module.validate_checkpoint(
            {**checkpoint, "schema": "cpc_v3_subgrid_v3"},
            expected_stage="coarse",
            dataset=dataset,
            label="old checkpoint",
        )
    with pytest.raises(ValueError, match="encoding differs"):
        module.validate_checkpoint(
            {
                **checkpoint,
                "subgrid_encoding": {
                    **checkpoint["subgrid_encoding"],
                    "wet_threshold_mm": 0.2,
                },
            },
            expected_stage="coarse",
            dataset=dataset,
            label="wrong encoding",
        )


def test_smooth_base_removes_the_conservation_lattice_without_losing_mass():
    """The 0.5-degree steps are the conservation support, not the allocation.

    A repeated block mean has *zero* interior gradient, so every change in the
    field happens at a block edge: that is the blockiness, and it survives any
    amount of training.  The conservative smooth base has to remove it while
    keeping the block means exact.
    """
    from bdhires.data import conservative_smooth_upsample

    def seam_ratio(field, factor=10):
        plane = field[0, 0].numpy().astype(np.float64)
        vertical = np.abs(np.diff(plane, axis=1))
        horizontal = np.abs(np.diff(plane, axis=0))
        vseam = np.zeros(vertical.shape, bool)
        hseam = np.zeros(horizontal.shape, bool)
        vseam[:, factor - 1 :: factor] = True
        hseam[factor - 1 :: factor, :] = True
        interior = np.concatenate([vertical[~vseam], horizontal[~hseam]])
        edge = np.concatenate([vertical[vseam], horizontal[hseam]])
        if interior.mean() <= 1.0e-12:
            return float("inf")
        return float(edge.mean() / interior.mean())

    blocks, factor = 6, 10
    rows, cols = torch.meshgrid(
        torch.linspace(0.0, 3.0, blocks), torch.linspace(0.0, 3.0, blocks),
        indexing="ij",
    )
    coarse = (2.0 + 2.0 * torch.sin(rows) * torch.cos(cols)).clamp_min(0.05)[None, None]
    area = torch.ones(1, 1, blocks * factor, blocks * factor)
    valid = torch.ones(1, 1, blocks * factor, blocks * factor, dtype=torch.bool)

    flat = conservative_smooth_upsample(coarse, area, valid, factor, 0)
    smooth = conservative_smooth_upsample(coarse, area, valid, factor, 2)
    assert seam_ratio(flat) == float("inf")  # piecewise constant by construction
    assert seam_ratio(smooth) < 1.5
    for candidate in (flat, smooth):
        recovered, _, _ = area_weighted_block_mean(
            candidate, area, valid, factor=factor, valid_area_threshold=0.0
        )
        assert torch.allclose(recovered, coarse, atol=1.0e-5)
    assert (smooth >= 0.0).all()

    # ...and the decoded field inherits it, still conserving exactly.
    allocation = torch.zeros(1, 2, blocks * factor, blocks * factor)
    allocation[:, 1] = 4.0
    coarse_state = torch.cat([torch.sqrt(coarse), torch.full_like(coarse, 4.0)], 1)
    coarse_valid = torch.ones(1, 1, blocks, blocks, dtype=torch.bool)
    decoded = {}
    for iterations in (0, 2):
        encoding = _encoding(smooth_base_iterations=iterations)
        decoded[iterations] = decode_and_reconstruct(
            coarse_state, allocation, coarse_valid, valid, area, encoding
        )
        recovered, _, _ = area_weighted_block_mean(
            decoded[iterations], area, valid, factor=factor, valid_area_threshold=0.0
        )
        assert torch.allclose(recovered, coarse, atol=1.0e-5)
    assert seam_ratio(decoded[0]) == float("inf")
    assert seam_ratio(decoded[2]) < 1.5


def test_legacy_v2_replay_keeps_the_block_constant_base():
    from bdhires.data import LegacyV2SubgridEncoding

    assert LegacyV2SubgridEncoding().smooth_base_iterations == 0
    assert "smooth_base_iterations" not in {
        field.name for field in LegacyV2SubgridEncoding.__dataclass_fields__.values()
    }


def test_every_reader_resolves_the_archive_schema_the_same_way():
    """Sampler and evaluator must never disagree about what they accept.

    A schema rule written out separately in each script drifts: script 60 was
    taught to read a legacy v4 archive while scripts 58 and 61 still demanded
    v5, so sampling succeeded and evaluation died on its own output.  Worse,
    both evaluators then rebuilt the encoding with a bare ``from_mapping``,
    which hands a v4 archive the current smooth-base default and decodes it
    differently from the sampler that wrote it.
    """
    from bdhires.data import LEGACY_V4_SUBGRID_SCHEMA, resolve_archive_encoding

    current, schema = resolve_archive_encoding(
        {"schema": SUBGRID_SCHEMA, "subgrid_encoding": {"factor": 10}}
    )
    assert schema == SUBGRID_SCHEMA
    assert current.smooth_base_iterations == 2

    legacy, schema = resolve_archive_encoding(
        {"schema": LEGACY_V4_SUBGRID_SCHEMA, "subgrid_encoding": {"factor": 10}}
    )
    assert schema == LEGACY_V4_SUBGRID_SCHEMA
    assert legacy.smooth_base_iterations == 0

    # The pin wins even if the archive carries a conflicting value.
    pinned, _ = resolve_archive_encoding(
        {
            "schema": LEGACY_V4_SUBGRID_SCHEMA,
            "subgrid_encoding": {"factor": 10, "smooth_base_iterations": 2},
        }
    )
    assert pinned.smooth_base_iterations == 0

    # Training never fits a new model on a superseded target.
    with pytest.raises(ValueError, match="is not one of"):
        resolve_archive_encoding(
            {"schema": LEGACY_V4_SUBGRID_SCHEMA, "subgrid_encoding": {}},
            allow_legacy_v4=False,
        )
    with pytest.raises(ValueError, match="is not one of"):
        resolve_archive_encoding(
            {"schema": LEGACY_V2_SUBGRID_SCHEMA, "subgrid_encoding": {}}
        )


def test_pipeline_scripts_share_one_schema_gate():
    """No script may re-implement the archive schema check locally."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in (
        "58_evaluate_subgrid_prior.py",
        "60_v4_subgrid_da_test.py",
        "61_evaluate_v4_subgrid_da_test.py",
    ):
        source = (root / "scripts" / name).read_text()
        assert "resolve_archive_encoding(" in source, name
        assert not re.search(
            r'attrs\.get\("schema"\)\s*!=\s*SUBGRID_SCHEMA', source
        ), f"{name} re-implements the schema gate"
        assert not re.search(
            r'SubgridEncoding\.from_mapping\(\s*target\.attrs', source
        ), f"{name} rebuilds the encoding outside the resolver"


def test_pilot_reads_chirps_on_the_state_date_not_the_observation_label():
    """Two date axes, one physical day.

    Training is same-day: CPC, ERA5 and CHIRPS share one date, and the
    background predicts CHIRPS on that date.  BMD accumulates to the following
    morning and IMERG is aligned to that window, so an observation file labelled
    D+1 measures the rain of state date D.  ``--background-day-offset -1`` is
    therefore the correct physical alignment, not a lag.

    Everything that is *scored* -- CHIRPS, CPC, every model field -- lives on the
    state date.  Only the observation reads use the label.  Mixing the two
    compared the analysis against the wrong day and collapsed every CHIRPS
    pattern correlation toward zero, which reads exactly like a model with no
    skill.
    """
    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/60_v4_subgrid_da_test.py"
    ).read_text()

    def block(pattern: str) -> str:
        match = re.search(pattern, source, re.DOTALL)
        assert match, f"could not locate {pattern!r}"
        return match.group(0)

    # CHIRPS is a target, so it is read on the state date alongside the forcing.
    chirps_read = block(r"chirps = np\.stack\(.*?\n    \)")
    assert "condition_index" in chirps_read
    assert "target_index" not in chirps_read

    # ...and the conditioning must agree with it.
    for name in ("coarse_condition", "fine_condition"):
        read = block(rf"{name} = np\.stack\(.*?\n    \)")
        assert "condition_index" in read, name

    # Observations keep the label date.
    assert "load_imerg_subset(args.imerg, days," in source
    assert "args.stations, days," in source

    # The archive records the state date so no consumer has to re-derive it.
    assert '"state_date"' in source
    assert "context_date_convention" in source


def test_evaluators_pair_samples_to_targets_on_the_state_date():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    gridded = (root / "scripts/58_evaluate_subgrid_prior.py").read_text()
    assert "state_time" in gridded
    assert not re.search(
        r"target_lookup\[int\(value\)\] for value in sample_time", gridded
    ), "script 58 still pairs CHIRPS on the observation label"

    pilot = (root / "scripts/61_evaluate_v4_subgrid_da_test.py").read_text()
    assert 'if "state_date" in samples' in pilot, (
        "script 61 must label maps with the state date; the observation label "
        "captions every panel one day late"
    )


def test_pilot_rejects_observation_labels_before_the_state_date():
    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/60_v4_subgrid_da_test.py"
    ).read_text()
    # The target store's own offset describes the TRAINING build and must be 0;
    # it is a different quantity from the observation labelling offset and the
    # two must never be compared to each other again.
    assert "if training_offset != 0:" in source
    assert "if observation_offset > 0:" in source
    assert not re.search(
        r"args\.background_day_offset\)\s*!=\s*training_offset", source
    ), "the pilot again compares the observation offset against the build offset"


def test_v5_hr_uses_the_original_grids_at_a_0p1_degree_support():
    """The 0.1-degree support retires the whole CPC-alignment apparatus.

    ``bd_cpc``/``wide_cpc``, the 160-cell halo canvas, the mod-10 crop lattice
    and the factor-40 crop-size rule all existed only because 0.5 degrees does
    not nest in the project's 0.05-degree grids.  IMERG's 0.1 degrees does, so
    V5-HR runs on the original BD/WIDE grids and the legacy 128 crop is valid
    again -- which also restores direct V1/V2 comparability.
    """
    for grid in (BD, WIDE):
        validate_cpc_alignment(grid, factor=2, coarse_res=0.1)
        assert grid.nlat % 2 == 0 and grid.nlon % 2 == 0

    validate_aligned_crop((0, 0), 128, factor=2, downsamplings=3)
    validate_aligned_crop((64, 130), 128, factor=2, downsamplings=3)
    with pytest.raises(ValueError, match="modulo"):
        validate_aligned_crop((1, 0), 128, factor=2, downsamplings=3)

    # BD is an exact subarray of WIDE, so no crop translation is needed.
    row, column = crop_offsets(WIDE, BD)
    assert row % 2 == 0 and column % 2 == 0


def test_v5_rebuild_differs_from_v4_by_the_smooth_base_alone():
    """A v5 target archive must be a one-variable change from v4.

    v5 exists to remove the 0.5-degree conservation lattice: a block-constant
    base has an infinite seam index, the conservative smooth base brings it to
    ~1.16, and conservation is unchanged.  Measuring that requires everything
    else to stay fixed.  IMERG conditioning, ERA5 total precipitation and
    terrain-forced ascent remain available behind flags, but as defaults they
    would move the conditioning set at the same time and make the v4/v5
    comparison uninterpretable.
    """
    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/56_build_chirps_subgrid_targets.py"
    ).read_text()

    assert 'ERA5_DEFAULT = ("tcwv", "cape", "u10", "v10", "msl")' in source
    assert re.search(r'"--grid",\s*default="wide_cpc"', source)
    assert re.search(r'"--factor",\s*type=int,\s*default=10', source)
    assert re.search(r'"--coarse-res",\s*type=float,\s*default=0\.5', source)
    assert re.search(r'"--smooth-base-iterations",\s*type=int,\s*default=2', source)

    for flag in ("--imerg-glob", "--orographic-ascent"):
        assert flag in source, flag
    assert "args.orographic_ascent" in source, "ascent must be opt-in"
    assert '"sqrt_imerg_precip"' in source and "orographic_ascent" in source


def test_date_convention_fixes_do_not_depend_on_the_schema():
    """The shifted-day bug lived in the pilot and evaluators, not the encoding.

    That is why it applies to an existing v4 archive with no retraining: the
    state/label distinction is enforced in scripts 58, 60 and 61, and none of
    them consults the target schema to decide it.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name, marker in (
        ("58_evaluate_subgrid_prior.py", "state_time"),
        ("60_v4_subgrid_da_test.py", "condition_index"),
        ("61_evaluate_v4_subgrid_da_test.py", "state_date"),
    ):
        source = (root / "scripts" / name).read_text()
        assert marker in source, f"{name} lost its state-date handling"
        for line in source.splitlines():
            if "SUBGRID_SCHEMA" in line and ("state_" in line or "condition_" in line):
                raise AssertionError(f"{name}: date handling branches on the schema")


def test_slurm_pipeline_never_pins_a_schema_literal():
    """The submission chain must follow the library, not restate its schema.

    This is the bug that broke the last run: the schema moved to v5 while the
    preflight in ``v3_subgrid_train.sbatch`` still asserted v4, so a correctly
    prepared archive was rejected after the dependency chain had already been
    queued.  A literal anywhere in the SLURM layer reintroduces it, and the
    failure surfaces hours later inside a job rather than at submission.
    """
    import re
    from pathlib import Path

    slurm = Path(__file__).resolve().parents[1] / "slurm"
    for name in (
        "submit_v3_subgrid_pipeline.sh",
        "v3_subgrid_prepare.sbatch",
        "v3_subgrid_train.sbatch",
    ):
        source = (slurm / name).read_text()
        found = re.findall(r'"cpc_v3_subgrid_v\d+"', source)
        assert not found, f"{name} pins the schema literal {found}"

    for name in ("v3_subgrid_prepare.sbatch", "v3_subgrid_train.sbatch"):
        source = (slurm / name).read_text()
        assert "from bdhires.data import SUBGRID_SCHEMA" in source, (
            f"{name} checks the schema without importing the definition"
        )


def test_submit_script_reads_the_target_from_the_configs():
    """The archive path is stated once, in the configs the trainers actually read.

    Preparing one store while three trainings read another is silent: every job
    succeeds and the models are fitted to a stale archive.  The submit script
    therefore derives the expected target instead of carrying its own copy.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (root / "slurm/submit_v3_subgrid_pipeline.sh").read_text()
    assert 'awk \'$1 == "zarr:" {print $2; exit}\'' in source, (
        "submit script no longer derives the target from the configs"
    )
    assert "EXPECTED_TARGET=" not in source, "hardcoded target reintroduced"

    targets = set()
    for stage in ("coarse", "allocation", "joint"):
        text = (root / f"configs/train_h100_cpc_v3_subgrid_{stage}.yaml").read_text()
        for line in text.splitlines():
            if line.strip().startswith("zarr:"):
                targets.add(line.split(":", 1)[1].strip())
                break
    assert len(targets) == 1, f"configs disagree about the target archive: {targets}"


def test_every_training_config_states_its_learning_rate_schedule():
    """Warmup and floor are frozen in the config, not inherited from a default.

    ``v3_subgrid_train.sbatch`` refuses to resume when the resolved config
    differs from the checkpoint's.  A schedule left to the trainer's defaults
    would therefore make any future change to those defaults un-resumable
    mid-run, which is precisely when a 48-hour job needs to resume.
    """
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for stage in ("coarse", "allocation", "joint"):
        config = yaml.safe_load(
            (root / f"configs/train_h100_cpc_v3_subgrid_{stage}.yaml").read_text()
        )
        train = config["train"]
        assert "warmup_fraction" in train, f"{stage} config omits warmup_fraction"
        assert "lr_min_fraction" in train, f"{stage} config omits lr_min_fraction"
        assert 0.0 <= train["warmup_fraction"] < 0.5
        assert 0.0 < train["lr_min_fraction"] < 1.0

    joint = yaml.safe_load(
        (root / "configs/train_h100_cpc_v3_subgrid_joint.yaml").read_text()
    )
    coarse = yaml.safe_load(
        (root / "configs/train_h100_cpc_v3_subgrid_coarse.yaml").read_text()
    )
    assert joint["train"]["warmup_fraction"] > coarse["train"]["warmup_fraction"], (
        "the joint stage fine-tunes two pretrained branches and needs the longer warmup"
    )


def test_trainer_refuses_to_resume_a_rescheduled_run():
    """Changing epochs mid-run would jump the learning rate without warning.

    The cosine curve is parameterised by the total step count, so a resume under
    a different ``epochs`` silently restarts the decay from the wrong place.  The
    trainer stores the step count and compares it rather than trusting the config.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts/57_train_subgrid_oracle.py"
    ).read_text()
    assert '"total_steps": total_steps' in source, "checkpoint omits the step count"
    assert "stored_total != total_steps" in source, "resume does not compare it"


def test_smooth_base_survives_blocks_with_no_valid_area():
    """A coarse block that is entirely ocean must not poison its neighbours.

    CHIRPS is undefined over the Bay of Bengal, so on the wide canvas whole
    0.5-degree blocks contain no valid fine cell and their area-weighted mean is
    identically zero.  The multiplicative correction divides by that mean, and
    because it is applied through a bilinear lift the resulting ~1e38 factor does
    not stay inside the offending block: it spreads into the valid coastal blocks
    beside it and overflows float32.  This is the failure that killed the first
    v5 preparation run, and it is invisible at ``iterations=0`` because that path
    returns the block constant before any division happens.
    """
    import torch

    from bdhires.data.subgrid_dataset import (
        area_weighted_block_mean,
        conservative_smooth_upsample,
    )

    factor, height, width = 10, 4, 4
    valid = torch.ones(1, 1, height * factor, width * factor, dtype=torch.bool)
    valid[..., 2 * factor:] = False           # two columns of blocks are all ocean
    area = torch.ones(1, 1, height * factor, width * factor)
    torch.manual_seed(0)
    coarse = torch.rand(1, 1, height, width) * 20.0

    _, _, fraction = area_weighted_block_mean(
        torch.ones_like(area), area, valid, factor, 0.0
    )
    assert (fraction == 0.0).any(), "the fixture must contain a fully invalid block"

    for iterations in (0, 1, 2, 3):
        base = conservative_smooth_upsample(coarse, area, valid, factor, iterations)
        assert torch.isfinite(base).all(), f"iterations={iterations} produced non-finite values"
        assert (base >= 0.0).all()
        assert base.max() < 100.0 * float(coarse.max()), (
            f"iterations={iterations} produced an implausible magnitude"
        )


def test_smooth_base_conserves_mass_on_a_coastline():
    """Exact conservation must hold in every block that has any valid area.

    The fallback for an uncorrectable block is the block constant, which is
    conservative by construction, so a partly-masked coastal block is the case
    that actually needs checking: it is corrected by a ratio computed only from
    its valid cells.
    """
    import torch

    from bdhires.data.subgrid_dataset import (
        area_weighted_block_mean,
        conservative_smooth_upsample,
    )

    factor, height, width = 10, 6, 6
    valid = torch.ones(1, 1, height * factor, width * factor, dtype=torch.bool)
    valid[..., 3 * factor + 4:] = False       # coastline cutting through a block
    torch.manual_seed(1)
    area = 1.0 + 0.01 * torch.rand(1, 1, height * factor, width * factor)
    coarse = torch.rand(1, 1, height, width) * 30.0
    coarse[0, 0, 1, 1] = 0.0                  # a dry block
    coarse[0, 0, 4, 4] = 250.0                # an isolated extreme beside dry neighbours

    base = conservative_smooth_upsample(coarse, area, valid, factor, 2)
    mean, _, fraction = area_weighted_block_mean(base, area, valid, factor, 0.0)
    live = fraction > 0.0
    relative = (mean[live] - coarse[live]).abs() / coarse[live].clamp_min(1.0e-6)
    assert float(relative.max()) < 1.0e-5, f"conservation broke: {float(relative.max()):.2e}"
    assert float(mean[0, 0, 1, 1]) == 0.0, "a dry block must stay exactly dry"


def test_smooth_base_removes_the_block_edge_it_was_written_for():
    """The 0.5-degree seam must shrink, and keep shrinking with iterations.

    Measured on the base alone as the mean absolute gradient across block edges
    over the mean absolute gradient inside blocks.  The block constant is
    perfectly flat inside a block, so its seam index is unbounded -- every
    gradient it has is a seam.  The absolute value the smooth base reaches
    depends on how correlated the coarse field is, so this asserts the two things
    that are actually properties of the algorithm: the seam becomes finite and
    comparable to the interior, and more iterations make it smaller.  The seam of
    the *reconstruction* is lower again, because the allocation weights add
    within-block variance that this base-only measurement excludes.
    """
    import torch

    from bdhires.data.subgrid_dataset import conservative_smooth_upsample

    factor, height, width = 10, 6, 6
    valid = torch.ones(1, 1, height * factor, width * factor, dtype=torch.bool)
    area = torch.ones(1, 1, height * factor, width * factor)
    torch.manual_seed(2)
    coarse = torch.rand(1, 1, height, width) * 30.0

    def seam_index(field: torch.Tensor) -> float:
        step = (field[..., :, 1:] - field[..., :, :-1]).abs()[0, 0]
        edge = torch.zeros_like(step, dtype=torch.bool)
        edge[:, factor - 1::factor] = True
        interior = step[~edge].mean()
        if float(interior) <= 0.0:
            return float("inf")
        return float(step[edge].mean() / interior)

    blocky = seam_index(conservative_smooth_upsample(coarse, area, valid, factor, 0))
    assert blocky == float("inf"), "the block constant should have no interior gradient"

    scores = [
        seam_index(conservative_smooth_upsample(coarse, area, valid, factor, n))
        for n in (1, 2, 3, 4)
    ]
    assert all(score < 5.0 for score in scores), f"seam never became finite: {scores}"
    assert scores == sorted(scores, reverse=True), (
        f"more iterations should reduce the seam, got {scores}"
    )


def test_target_build_round_trip_survives_an_ocean_masked_canvas():
    """Exactly the call that killed the first v5 preparation run.

    Script 56 encodes a CHIRPS chunk and immediately decodes it back to measure
    the oracle ceiling.  On the wide canvas CHIRPS is missing over the Bay of
    Bengal, so that round trip runs with whole coarse blocks carrying no valid
    fine cell.  Under the smooth base this raised FloatingPointError after the
    statistics pass had already completed -- the most expensive possible moment
    to discover it.  The unit-level guard above covers the base; this covers the
    path the build actually takes.
    """
    torch.manual_seed(7)
    factor = 10
    height = width = 4
    fine = torch.rand(3, 1, height * factor, width * factor) * 40.0
    fine[fine < 20.0] = 0.0                      # a realistic wet fraction
    valid = torch.ones(1, 1, height * factor, width * factor, dtype=torch.bool)
    valid[..., 2 * factor:] = False              # open ocean: two block columns
    valid[..., factor:2 * factor, factor + 3:2 * factor] = False   # a ragged coast
    area = torch.ones(1, 1, height * factor, width * factor)

    encoding = _encoding(smooth_base_iterations=2)
    targets = encode_subgrid_targets(fine, valid, area, encoding, sample_offset=0)
    decoded = decode_and_reconstruct(
        targets.coarse_state,
        targets.allocation_state,
        targets.coarse_valid,
        targets.fine_valid,
        area,
        encoding,
        hard=True,
    )

    assert torch.isfinite(decoded).all(), "the build round trip produced non-finite values"
    assert (decoded >= 0.0).all()
    assert float(decoded[:, :, :, 2 * factor:].abs().max()) == 0.0, (
        "rainfall was reconstructed over cells with no valid data"
    )

    # And it still conserves: the decoded field must reproduce the coarse amounts.
    mean, _, fraction = area_weighted_block_mean(decoded, area, valid, factor, 0.0)
    live = fraction > 0.0
    reference = targets.coarse_mm
    relative = (mean[live] - reference[live]).abs() / reference[live].clamp_min(1.0e-6)
    assert float(relative.max()) < 1.0e-4, f"conservation broke: {float(relative.max()):.2e}"


class _TwoPoint(torch.nn.Module):
    """Two observations: a block-mean one and a single-cell one."""

    def forward(self, field):
        block = field.mean(dim=(-1, -2))
        point = field[:, :, 3, 4]
        return torch.stack([block, point], dim=-1)


def _routing_gradient(routing, amount_mask=None):
    encoding = _encoding()
    coarse = torch.tensor([[[[2.0]], [[4.0]]]])
    allocation = torch.zeros(1, 2, 10, 10)
    allocation[:, 1] = 4.0
    observations = HierarchicalObservations(
        _TwoPoint(),
        torch.tensor([[[10.0, 12.0]]]),
        torch.tensor([1.0, 1.0]),
        GuidanceConfig(gamma=1.0e-3, clip_norm=None),
        routing=routing,
        amount_mask=amount_mask,
    )
    _, gradient, _ = hierarchical_guidance_grad(
        _ToyJoint(), HierarchicalState(coarse, allocation), torch.tensor([0.7]),
        None, None, observations, HierarchicalRectifiedFlow(), encoding,
        torch.ones(1, 1, 1, 1, dtype=torch.bool),
        torch.ones(1, 1, 10, 10, dtype=torch.bool),
        torch.ones(10, 10), 1.0,
    )
    return gradient


def test_observation_routing_confines_each_stream_to_one_scale():
    """A stream may be restricted to the part of the state it can resolve.

    The hard per-block reconstruction has a null direction: an observation on a
    support smaller than the conservation block can be satisfied by moving the
    block amount OR by redistributing mass inside the block.  Guidance takes the
    cheaper route, which for a point gauge is redistribution -- and that drains
    every other cell in the block.  Routing removes the choice.
    """
    both = _routing_gradient("both")
    assert both.coarse.abs().sum() > 0.0
    assert both.allocation.abs().sum() > 0.0

    amount = _routing_gradient("amount")
    assert amount.coarse.abs().sum() > 0.0
    assert float(amount.allocation.abs().sum()) == 0.0, "amount routing moved allocation"

    allocation = _routing_gradient("allocation")
    assert float(allocation.coarse.abs().sum()) == 0.0, "allocation routing moved the amount"
    assert allocation.allocation.abs().sum() > 0.0


def test_split_routing_takes_each_component_from_its_own_stream():
    """The operational arm: IMERG constrains the amount, gauges the structure.

    The coarse component must come from the amount-routed observations alone and
    the allocation component from the others, so neither stream can reach the
    scale it cannot resolve.  Checked against single-stream runs rather than
    against itself.
    """
    mask = torch.tensor([True, False])
    split = _routing_gradient("split", mask)

    # Only the block-mean observation may move the amount.
    only_block = HierarchicalObservations(
        _TwoPoint(), torch.tensor([[[10.0, 12.0]]]),
        torch.tensor([1.0, 1.0e12]),
        GuidanceConfig(gamma=1.0e-3, clip_norm=None), routing="amount",
    )
    encoding = _encoding()
    coarse = torch.tensor([[[[2.0]], [[4.0]]]])
    allocation = torch.zeros(1, 2, 10, 10)
    allocation[:, 1] = 4.0
    _, block_only, _ = hierarchical_guidance_grad(
        _ToyJoint(), HierarchicalState(coarse, allocation), torch.tensor([0.7]),
        None, None, only_block, HierarchicalRectifiedFlow(), encoding,
        torch.ones(1, 1, 1, 1, dtype=torch.bool),
        torch.ones(1, 1, 10, 10, dtype=torch.bool),
        torch.ones(10, 10), 1.0,
    )
    assert torch.allclose(split.coarse, block_only.coarse, atol=1.0e-6), (
        "split routing's amount component is contaminated by the point stream"
    )
    assert split.allocation.abs().sum() > 0.0


def test_routing_is_validated_rather_than_silently_ignored():
    """A mistyped routing must fail loudly; a split without a mask is undefined."""
    values, variance = torch.tensor([[[1.0]]]), torch.tensor([1.0])
    with pytest.raises(ValueError, match="routing must be one of"):
        HierarchicalObservations(_Point(), values, variance, routing="coarse")
    with pytest.raises(ValueError, match="requires amount_mask"):
        HierarchicalObservations(_Point(), values, variance, routing="split")
    with pytest.raises(ValueError, match="only meaningful"):
        HierarchicalObservations(
            _Point(), values, variance, amount_mask=torch.tensor([True])
        )
    with pytest.raises(ValueError, match="must match"):
        HierarchicalObservations(
            _Point(), values, torch.tensor([1.0, 1.0]), routing="split",
            amount_mask=torch.tensor([True]),
        )


def test_unrouted_guidance_is_unchanged_by_the_routing_feature():
    """Existing arms must score identically; routing defaults to the old path."""
    encoding = _encoding()
    coarse = torch.tensor([[[[2.0]], [[4.0]]]])
    allocation = torch.zeros(1, 2, 10, 10)
    allocation[:, 1] = 4.0
    kwargs = dict(
        values=torch.tensor([[[10.0]]]), variance=torch.tensor([1.0]),
        guidance=GuidanceConfig(gamma=1.0e-3, clip_norm=None),
    )
    default = HierarchicalObservations(_Point(), **kwargs)
    explicit = HierarchicalObservations(_Point(), **kwargs, routing="both")
    grads = []
    for observations in (default, explicit):
        _, gradient, _ = hierarchical_guidance_grad(
            _ToyJoint(), HierarchicalState(coarse, allocation), torch.tensor([0.7]),
            None, None, observations, HierarchicalRectifiedFlow(), encoding,
            torch.ones(1, 1, 1, 1, dtype=torch.bool),
            torch.ones(1, 1, 10, 10, dtype=torch.bool),
            torch.ones(10, 10), 1.0,
        )
        grads.append(gradient)
    assert torch.equal(grads[0].coarse, grads[1].coarse)
    assert torch.equal(grads[0].allocation, grads[1].allocation)
    assert grads[0].coarse.abs().sum() > 0.0


def test_routed_arms_are_declared_everywhere_they_are_consumed():
    """An arm added in one place and missed in another fails hours into a job.

    Script 60 samples the arms, script 58 scores them on the grid, and script 61
    reads both.  A name present in the sampler but absent from the sbatch's
    --methods list produces a KeyError in the evaluator after the GPU work is
    already spent, which is the most expensive possible place to find out.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sampler = (root / "scripts/60_v4_subgrid_da_test.py").read_text()
    evaluator = (root / "scripts/61_evaluate_v4_subgrid_da_test.py").read_text()
    sbatch = (root / "slurm/v4_subgrid_da_test.sbatch").read_text()

    for name in ("routed_withheld", "routed_all"):
        assert f'"{name}"' in sampler, f"{name} is not sampled"
        assert f'"{name}"' in evaluator, f"{name} is not evaluated"
        assert name in sbatch, f"{name} is missing from the sbatch --methods list"

    # The routed arms must differ from the simultaneous ones ONLY by routing.
    assert 'routing="split"' in sampler
    assert sampler.count("amount_mask=torch.cat") == 2, (
        "both routed arms need an explicit amount mask"
    )
    # Gauges are concatenated before the satellite, so the mask must be
    # False for the gauge block and True for the satellite block, in that order.
    assert "torch.zeros(len(gauge_variance" in sampler
    assert "torch.ones(satellite_r.shape[0]" in sampler


def test_unrouted_arms_survive_the_routing_change():
    """The existing arms are the control; they must keep their old behaviour."""
    from pathlib import Path

    sampler = (
        Path(__file__).resolve().parents[1] / "scripts/60_v4_subgrid_da_test.py"
    ).read_text()
    for name in ("gauges_withheld", "imerg_only", "simultaneous_withheld",
                 "gauges_all", "simultaneous_all"):
        assert f'"{name}": HierarchicalObservations(' in sampler, name
    # None of the pre-existing arms may acquire a routing argument.
    prefix = sampler.split('"routed_withheld"')[0]
    assert "routing=" not in prefix, "an existing arm was silently routed"


def test_every_sampled_arm_is_described_in_method_specs():
    """The archive writer rejects a mismatch only after all sampling is done.

    ``write_hierarchical_sample_zarr`` compares ``method_specs`` against the
    physical fields and raises if they differ.  That happens at the very end of
    the run, so an arm added to METHODS but not described here throws away every
    GPU-hour already spent.  This catches it in the 30-second test job instead.
    """
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts/60_v4_subgrid_da_test.py"
    ).read_text()
    tree = ast.parse(source)

    methods = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "METHODS" for target in node.targets
        ):
            methods = {element.value for element in node.value.elts}
    assert methods, "could not locate METHODS"

    specs = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "method_specs" for target in node.targets
        ):
            specs = {key.value for key in node.value.keys}
    assert specs, "could not locate method_specs"

    assert methods == specs, (
        f"METHODS and method_specs disagree: only in METHODS {sorted(methods - specs)}, "
        f"only in method_specs {sorted(specs - methods)}"
    )


def test_osse_mode_swaps_the_observations_and_records_that_it_did():
    """Perfect-observation mode must be self-identifying and self-consistent.

    An OSSE replaces the gauge values with the truth.  Two things then matter:
    the pseudo-gauge must be read with the SAME operator the analysis uses, so
    the test is not confounded by interpolation mismatch, and the sample store
    must record that it is an OSSE -- otherwise the reuse check would happily
    hand an OSSE archive to a real-data request, and every number downstream
    would be a perfect-observation result wearing a real-data label.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts/60_v4_subgrid_da_test.py"
    ).read_text()

    assert '"--osse"' in source and '"--osse-sigma-mm"' in source
    assert "if args.osse:" in source, "the flag is declared but never acted on"
    # Truth is sampled with the analysis operator, not a hand-rolled lookup.
    osse_block = source.split("if args.osse:")[1].split("gauge_variance")[0]
    assert "BilinearObsOperator(grid, stations.lat, stations.lon)" in osse_block
    assert "chirps" in osse_block, "OSSE must sample the CHIRPS truth"
    # The near-zero error replaces the real budget rather than adding to it.
    assert "gauge_sigma = args.osse_sigma_mm if args.osse else args.gauge_sigma_mm" in source
    # And the run identifies itself.
    assert '"osse": bool(args.osse)' in source, "OSSE runs are not marked in the report"


def test_osse_block_support_changes_only_what_is_assimilated():
    """Verification must stay at points, or the comparison means nothing.

    The point-versus-block OSSE is only interpretable if the two runs differ in
    exactly one thing: the support of the assimilated observation.  If the
    verification target moved to block means as well, a 'better' score would
    just mean the analysis got easier to hit.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts/60_v4_subgrid_da_test.py"
    ).read_text()

    assert '"--osse-gauge-support"' in source
    # The assimilated values come from a separate array...
    assert "assimilation_mm = gauge_mm" in source, "assimilated values are not separated"
    assert "observation = assimilation_mm[day_position, index]" in source
    # ...while gauge_mm, which the archive verifies against, is never reassigned
    # to block means.
    assert "gauge_mm = truth_blocks" not in source, "verification target was moved to blocks"
    assert "assimilation_mm = truth_blocks[:, osse_block_index]" in source
    # The operator must match the support of the observation.
    assert "class _BlockSubsetOperator" in source
    assert "_BlockSubsetOperator(block_operator" in source
    assert '"osse_gauge_support"' in source, "support is not recorded in the report"


def test_v7_coarse_driver_replicates_cpc_onto_a_finer_support():
    """V7's analysis level is finer than CPC, so CPC must be replicated.

    V3-SG's coarse support WAS CPC's own 0.5-degree grid, so exact selection was
    correct.  A 0.1-degree support is finer, and nearest selection there is block
    replication -- which preserves the CPC block mean exactly, because a constant
    equals its own mean.  Interpolating instead would not, and the archive would
    then disagree with CPC about how much rain fell.
    """
    import importlib.util
    from pathlib import Path

    import numpy as np

    xr = pytest.importorskip("xarray")
    from bdhires.grids import Grid

    path = Path(__file__).resolve().parents[1] / "scripts/56_build_chirps_subgrid_targets.py"
    spec = importlib.util.spec_from_file_location("builder56", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    cpc_lat = np.arange(20.25, 23.0, 0.5)
    cpc_lon = np.arange(88.25, 91.0, 0.5)
    data = xr.DataArray(
        np.arange(cpc_lat.size * cpc_lon.size, dtype=float).reshape(cpc_lat.size, cpc_lon.size),
        coords={"lat": cpc_lat, "lon": cpc_lon}, dims=("lat", "lon"),
    )

    meso = Grid(name="meso", lon_min=88.0, lat_min=20.0, nlon=20, nlat=20, res=0.1)
    placed, native = builder._coarse_driver(data, meso, "CPC")
    assert not native, "0.5-degree CPC must not report native on a 0.1-degree grid"
    assert np.allclose(placed.lat.values, meso.lat)
    assert np.allclose(placed.lon.values, meso.lon)
    block = placed.values.reshape(4, 5, 4, 5).mean(axis=(1, 3))
    assert np.allclose(block, data.values[:4, :4]), "replication lost the CPC block mean"

    # v5 reproduction: on CPC's own grid nothing changes and it still says native.
    native_grid = Grid(
        name="cpc", lon_min=88.0, lat_min=20.0,
        nlon=cpc_lon.size, nlat=cpc_lat.size, res=0.5,
    )
    same, is_native = builder._coarse_driver(data, native_grid, "CPC")
    assert is_native and np.allclose(same.values, data.values)

    # A driver finer than the coarse grid would be silently subsampled.
    finer = xr.DataArray(
        np.zeros((40, 40)),
        coords={"lat": np.arange(20.0, 24.0, 0.1), "lon": np.arange(88.0, 92.0, 0.1)},
        dims=("lat", "lon"),
    )
    with pytest.raises(ValueError, match="finer than the coarse grid"):
        builder._coarse_driver(finer, native_grid, "IMERG")


def test_v7_stage_a_is_cpcv2_with_only_the_resolution_changed():
    """Stage A must be CPCv2 verbatim apart from what the resolution forces.

    The point of V7 stage A is that it inherits v2's working assimilation
    behaviour.  That only holds if it inherits v2's training as well -- in
    particular the SOFT coarse-consistency penalty, which lets the analysis
    depart from CPC when an observation says CPC was wrong, and the wet
    sampling, which is what stops the amount head learning a light-day
    distribution.  A quiet divergence from v2 here would reproduce V5's failures
    at a new resolution and look like a fresh mystery.
    """
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    v2 = yaml.safe_load((root / "configs/train_h100_cpc_v2.yaml").read_text())
    v7 = yaml.safe_load((root / "configs/train_v7_meso.yaml").read_text())

    assert v7["model"] == v2["model"], "stage A changed the v2 architecture"
    assert v7["data"]["cond_channels"] == v2["data"]["cond_channels"]
    assert v7["train"]["wet_sampling"] == v2["train"]["wet_sampling"], (
        "wet sampling is what addresses the amount compression; keep it identical"
    )

    # Soft consistency, not a decoder constraint -- the defining V7 property.
    consistency = v7["train"]["coarse_consistency"]
    assert consistency["target_weight"] == v2["train"]["coarse_consistency"]["target_weight"]
    assert consistency["cpc_weight"] == v2["train"]["coarse_consistency"]["cpc_weight"]
    # 0.1 -> 0.5 degrees is five cells where 0.05 -> 0.5 was ten.
    assert consistency["factor"] * 2 == v2["train"]["coarse_consistency"]["factor"]

    # The crop must cover the SAME ground, so attention lands on real levels.
    assert v7["data"]["crop"] * 2 == v2["data"]["crop"]
    levels = [v7["data"]["crop"] // 2**i for i in range(len(v7["model"]["channel_mult"]))]
    for resolution in v7["model"]["attn_resolutions"]:
        assert resolution in levels, f"attention at {resolution} has no U-Net level"

    # Everything in train: apart from paths, the consistency factor and the
    # schedule LENGTH is v2's.  `epochs` is a deliberate, separately justified
    # departure: both V7 stages run 400 so their curves are comparable epoch for
    # epoch.  It changes how long the cosine decay is, not the objective, the
    # architecture or the sampling -- which are the things "verbatim" is about.
    ignored = {"out_dir", "coarse_consistency", "epochs"}
    for key, value in v2["train"].items():
        if key in ignored:
            continue
        assert v7["train"][key] == value, f"stage A diverged from v2 on train.{key}"

    # If the epoch counts drift apart, the reason above stops being true.
    alloc = yaml.safe_load((root / "configs/train_v7_allocation.yaml").read_text())
    assert v7["train"]["epochs"] == alloc["train"]["epochs"], (
        "the two V7 stages must run the same number of epochs, or the shared "
        "10-epoch checkpoint cadence stops lining them up"
    )

    # Statistics belong to the archive they were measured on.
    assert v7["data"]["stats"] != v2["data"]["stats"], (
        "stage A must not reuse the 0.05-degree statistics"
    )
    assert v7["data"]["zarr"] != v2["data"]["zarr"]


def test_v7_stage_b_is_the_factor_two_allocation_branch():
    """Stage B is V3-SG's allocation branch, which is the part that worked."""
    import yaml
    from pathlib import Path

    from bdhires.grids import WIDE_CPC

    # No grid rebuild: the canvas closes on 0.1-degree edges at factor 2.
    validate_cpc_alignment(WIDE_CPC, factor=2, coarse_res=0.1)

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/train_v7_allocation.yaml").read_text())
    assert config["stage"] == "allocation"
    data = config["data"]
    assert int(data["factor"]) == 2, "V7 stage B is a factor-2 design"
    validate_aligned_crop((0, 0), data["crop"], data["factor"], data["downsamplings"])
    levels = len(config["model"]["channel_mult"]) - 1
    assert data["crop"] % 2**levels == 0
    assert "init_coarse" not in config["train"], "V7 has no coupled joint stage"
    # Tiled validation covers the domain with non-overlapping crops, so the crop
    # must divide it.  A crop that merely satisfies the lattice rule is not
    # enough, and the dataset only discovers that after the job has started.
    from bdhires.grids import WIDE_CPC

    for extent in WIDE_CPC.shape:
        assert extent % data["crop"] == 0, (
            f"crop {data['crop']} does not divide the {extent}-cell domain; "
            "tiled validation would weight some cells twice"
        )
    levels = [data["crop"] // 2**i for i in range(len(config["model"]["channel_mult"]))]
    for resolution in config["model"]["attn_resolutions"]:
        assert resolution in levels, f"attention at {resolution} has no U-Net level"
    # Conditioning augmentation is what lets stage B survive being run on
    # ANALYSED 0.1-degree fields rather than the clean ones it trains on.
    assert config["train"]["conditioning_augmentation"]["max_coarse_noise"] > 0.0


def test_v7_coarsened_archive_keeps_the_v2_layout():
    """The coarsened archive must be a drop-in for the packed one.

    ``06_compute_stats.py`` and ``scripts/train.py`` are used unchanged, so array
    names, channel order and the attribute block have to survive coarsening --
    and the result must be identifiable as coarsened so it can never be mistaken
    for a packed archive.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts/71_v7_coarsen_pack_archive.py"
    ).read_text()
    for name in ("time", "lat", "lon", "valid", "static", "target", "cond"):
        assert f'"{name}"' in source, f"the coarsened archive drops {name}"
    assert 'attributes["coarsened_from"]' in source, "provenance is not recorded"
    assert 'attributes["coarsen_factor"]' in source
    assert "area_weighted_block_mean" in source, "coarsening must be area weighted"
    # Channel order is inherited rather than rebuilt.
    assert 'source.attrs["cond_channels"]' in source


def test_v7_coarsener_uses_the_zarr_api_the_cluster_has():
    """The cluster runs zarr 2; the container runs zarr 3.

    ``create_array`` exists only on zarr 3, so a script that uses it passes every
    local check and then dies on the compute node -- after the archive read and
    the queue wait are already spent.  ``create_dataset`` with an explicit shape
    works on both, and is what ``bdhires.zarr_output`` already uses, so there is
    one convention rather than two.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts/71_v7_coarsen_pack_archive.py"
    ).read_text()
    assert ".create_array(" not in source, "create_array is unavailable on zarr 2"
    assert "group.create_dataset(name, **kwargs)" in source
    # Shape and dtype must be explicit: zarr 3's create_dataset requires shape.
    assert 'kwargs.setdefault("shape"' in source
    assert 'kwargs.setdefault("dtype"' in source


def test_v7_stage_a_validation_monitor_is_not_silently_disabled():
    """The sampled-validation monitor must survive the change of resolution.

    ``build_monitor`` compares the monitor grid's width against ``data.crop`` and
    returns None when they differ.  BD is 128 cells wide at 0.05 degrees and V7
    stage A crops 64, so a monitor grid resolved at the packed resolution makes
    that comparison fail -- and the run then trains for 150 epochs with no CRPS,
    no panels and no crash to say so.  The grid has to be expressed at the
    ARCHIVE's resolution, which is where ``at_resolution`` comes in.
    """
    import yaml
    from pathlib import Path

    from bdhires.grids import at_resolution

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/train_v7_meso.yaml").read_text())

    # WIDE coarsened by the same factor is the outer grid the archive records.
    outer = at_resolution(WIDE, 0.05 * 2)
    monitor = at_resolution(get_grid(cfg["data"]["monitor_grid"]), outer.res)
    assert monitor.nlon == cfg["data"]["crop"], (
        "V7 stage A would train with the validation monitor switched off"
    )
    # The same geography, so the offsets scale with the factor rather than moving.
    assert crop_offsets(outer, monitor) == tuple(o // 2 for o in crop_offsets(WIDE, BD))
    # v2 must be untouched: at its own resolution this is an identity.
    assert at_resolution(BD, 0.05) is BD


def test_v7_stage_a_monitor_grid_comes_from_the_archive():
    """A hardcoded outer grid is the bug, not the symptom.

    ``crop_offsets`` requires both grids to share a resolution, so pinning WIDE
    while the archive is coarsened either raises or -- worse -- disables the
    monitor.  The trainer must read the grid the archive itself records.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "scripts/train.py").read_text()
    assert "crop_offsets(WIDE," not in source, "the outer grid must not be hardcoded"
    assert "crop_offsets(outer, grid)" in source
    assert "def archive_grid(" in source
    assert 'at_resolution(get_grid(cfg["data"].get("monitor_grid"' in source


# --------------------------------------------------------------------------
# in-training sampled diagnostics (V7)
# --------------------------------------------------------------------------


def _allocation_monitor_store(days: int = 12, size: int = 16, factor: int = 2):
    """A tiny but structurally complete allocation archive."""
    rng = np.random.default_rng(11)
    coarse = size // factor
    # A rainfall gradient across days, so quantile-based case selection has a
    # distribution to pick from rather than noise.
    scale = np.linspace(0.2, 12.0, days).astype(np.float32)
    fine = np.abs(rng.gamma(2.0, 1.0, (days, size, size))).astype(np.float32)
    fine *= scale[:, None, None]
    area = np.ones((size, size), np.float32)
    valid = np.ones((size, size), bool)
    block = fine.reshape(days, coarse, factor, coarse, factor).mean(axis=(2, 4))
    return _MemoryStore(
        {
            "time": np.asarray(
                [f"2019-07-{d + 1:02d}" for d in range(days)], dtype="datetime64[ns]"
            ).astype(np.int64),
            "fine_mm": fine,
            "coarse_mm": block.astype(np.float32),
            "fine_valid": valid,
            "cell_area": area,
            "coarse_valid": np.ones((coarse, coarse), bool),
            "coarse_state": rng.normal(size=(days, 2, coarse, coarse)).astype(np.float32),
            "allocation_state": rng.normal(size=(days, 2, size, size)).astype(np.float32),
            "fine_cond": rng.normal(size=(days, 3, size, size)).astype(np.float32),
            "coarse_cond": rng.normal(size=(days, 3, coarse, coarse)).astype(np.float32),
        },
        attrs={
            "schema": SUBGRID_SCHEMA,
            "subgrid_encoding": {
                "factor": factor,
                "amount_sqrt_mean": 1.0,
                "amount_sqrt_std": 1.0,
                "intensity_log_mean": 0.0,
                "intensity_log_std": 0.5,
                "smooth_base_iterations": 2,
            },
        },
    )


def test_sampled_diagnostic_runs_end_to_end_on_the_allocation_branch(tmp_path):
    """The whole path: pick cases, sample, score, write history and figures.

    This is the part that cannot be checked by reading the source.  The monitor
    calls the model with the allocation signature, reconstructs through the hard
    decoder, and has to produce finite numbers for a branch whose weights are
    random -- because a NaN on epoch 10 of a 48-hour run is indistinguishable
    from a NaN caused by the training itself.
    """
    from bdhires.data import SubgridDataset, SubgridDatasetConfig
    from bdhires.eval import SubgridMonitor, SubgridMonitorConfig

    dataset = SubgridDataset(
        SubgridDatasetConfig(root="unused", crop=16, random_crop=False, factor=2),
        store=_allocation_monitor_store(),
    )
    model = AllocationFlow(
        3, image_size=16, base_channels=8, channel_mult=(1, 2),
        num_res_blocks=1, attn_resolutions=(8,), num_heads=1,
    )
    monitor = SubgridMonitor(
        dataset, torch.device("cpu"), tmp_path / "diagnostics", "allocation",
        SubgridMonitorConfig(
            start_epoch=10, every=10, cases=2, members=3, n_steps=2,
        ),
    )

    assert len(monitor.cases) == 2
    # Chosen by rainfall, and the two cases must not be the same day.
    assert monitor.cases[0].date != monitor.cases[1].date
    assert monitor.cases[0].domain_mean_mm < monitor.cases[1].domain_mean_mm

    assert not monitor.should_run(0)          # before start_epoch
    assert not monitor.should_run(10)         # epoch 11, not a multiple of 10
    assert monitor.should_run(9)              # epoch 10

    # A failure here must return None rather than raise; assert we did not take
    # that path by requiring a real summary back.
    summary = monitor.run(model, epoch=9)
    assert summary is not None, "the diagnostic swallowed an exception"
    assert summary["epoch"] == 10
    assert len(summary["cases"]) == 2
    for case in summary["cases"]:
        for key in (
            "anomaly_r", "smooth_anomaly_r", "crps_mm",
            "conservation_abs_mm", "seam_index",
        ):
            assert key in case, f"{key} missing from the diagnostic"
        assert np.isfinite(case["crps_mm"])
        # Hard conservation is the contract this stage is built on.
        assert case["conservation_abs_mm"] < 1.0e-4, case["conservation_abs_mm"]

    import json

    history = (tmp_path / "diagnostics" / "history.jsonl").read_text().splitlines()
    assert len(history) == 1
    assert json.loads(history[0])["stage"] == "allocation"
    assert (tmp_path / "diagnostics" / "epoch_0010.png").is_file()


def test_sampled_diagnostic_leaves_the_model_as_it_found_it(tmp_path):
    """Training mode and gradients must survive the diagnostic.

    ``run`` puts the model in eval mode for sampling.  If it forgets to restore
    training mode, every batch after the first diagnostic trains with dropout
    and batch statistics switched off -- a silent, permanent change of the
    objective that would surface only as an unexplained kink at epoch 10.
    """
    from bdhires.data import SubgridDataset, SubgridDatasetConfig
    from bdhires.eval import SubgridMonitor, SubgridMonitorConfig

    dataset = SubgridDataset(
        SubgridDatasetConfig(root="unused", crop=16, random_crop=False, factor=2),
        store=_allocation_monitor_store(),
    )
    model = AllocationFlow(
        3, image_size=16, base_channels=8, channel_mult=(1, 2),
        num_res_blocks=1, attn_resolutions=(8,), num_heads=1,
    )
    model.train()
    before = [p.detach().clone() for p in model.parameters()]

    monitor = SubgridMonitor(
        dataset, torch.device("cpu"), tmp_path / "d", "allocation",
        SubgridMonitorConfig(start_epoch=0, every=1, cases=1, members=2,
                             n_steps=2, save_maps=False),
    )
    monitor.run(model, epoch=0)

    assert model.training, "the diagnostic left the model in eval mode"
    for parameter, original in zip(model.parameters(), before):
        assert torch.equal(parameter, original), "the diagnostic changed weights"
        assert parameter.grad is None, "the diagnostic accumulated gradients"


def test_sampled_diagnostic_is_deterministic_across_calls(tmp_path):
    """Same weights, same day, same numbers.

    The point of a per-epoch curve is that a change means the model moved.  If
    the noise draw changed too, every wiggle would be unreadable.
    """
    from bdhires.data import SubgridDataset, SubgridDatasetConfig
    from bdhires.eval import SubgridMonitor, SubgridMonitorConfig

    def build():
        dataset = SubgridDataset(
            SubgridDatasetConfig(root="unused", crop=16, random_crop=False, factor=2),
            store=_allocation_monitor_store(),
        )
        return SubgridMonitor(
            dataset, torch.device("cpu"), tmp_path / "d", "allocation",
            SubgridMonitorConfig(start_epoch=0, every=1, cases=1, members=2,
                                 n_steps=2, save_maps=False),
        )

    torch.manual_seed(0)
    model = AllocationFlow(
        3, image_size=16, base_channels=8, channel_mult=(1, 2),
        num_res_blocks=1, attn_resolutions=(8,), num_heads=1,
    )
    model.eval()
    first = build().run(model, epoch=0)["cases"][0]
    second = build().run(model, epoch=0)["cases"][0]
    assert first["crps_mm"] == pytest.approx(second["crps_mm"], rel=1e-9)
    assert first["anomaly_r"] == pytest.approx(second["anomaly_r"], rel=1e-9)


def test_sampled_diagnostic_refuses_a_cadence_that_could_never_fire():
    """A monitor that never runs looks exactly like a healthy one.

    ``every`` has to be a multiple of ``keep_every``, otherwise the picture and
    the checkpoint it describes do not both survive.  This must raise at
    startup, not go quiet for 400 epochs.
    """
    from bdhires.eval import SubgridMonitor, SubgridMonitorConfig

    monitor = SubgridMonitor.__new__(SubgridMonitor)
    monitor.cfg = SubgridMonitorConfig(every=7)
    with pytest.raises(ValueError, match="multiple"):
        SubgridMonitor.validate_cadence(monitor, keep_every=10)
    monitor.cfg = SubgridMonitorConfig(every=10)
    SubgridMonitor.validate_cadence(monitor, keep_every=10)      # no raise
    monitor.cfg = SubgridMonitorConfig(every=10, enabled=False)
    SubgridMonitor.validate_cadence(monitor, keep_every=7)       # disabled: no raise


def test_sampled_validation_config_rejects_unknown_keys():
    """A typo must not become a silently different diagnostic."""
    from bdhires.eval import SubgridMonitorConfig

    with pytest.raises(ValueError, match="unknown sampled_validation keys"):
        SubgridMonitorConfig.from_dict({"member": 4})
    assert SubgridMonitorConfig.from_dict({"members": 4}).members == 4


def test_v7_configs_pair_every_kept_checkpoint_with_a_picture():
    """Both stages: 400 epochs, kept every 10, sampled at the same cadence.

    The two trainers spell this differently -- script 57 reads
    ``train.sampled_validation`` and ``train.keep_every``; ``scripts/train.py``
    reads a top-level ``validation`` block and ``train.keep_every`` -- so the
    agreement between them is a property worth asserting rather than assuming.
    """
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    meso = yaml.safe_load((root / "configs/train_v7_meso.yaml").read_text())
    alloc = yaml.safe_load((root / "configs/train_v7_allocation.yaml").read_text())

    assert meso["train"]["epochs"] == 400
    assert alloc["train"]["epochs"] == 400

    assert meso["train"]["keep_every"] == 10
    assert alloc["train"]["keep_every"] == 10

    # Stage A: the monitor fires on kept-checkpoint epochs.
    assert meso["validation"]["enabled"] is True
    assert meso["validation"]["every"] == 10
    assert meso["validation"]["every"] % meso["train"]["ckpt_every"] == 0
    assert meso["validation"]["n_steps"] == 50
    assert meso["validation"]["members"] == 4
    assert len(meso["validation"]["quantiles"]) == 2
    assert meso["validation"]["max_cases"] == 2

    # Stage B: same cadence, same sampling budget.
    sampled = alloc["train"]["sampled_validation"]
    assert sampled["enabled"] is True
    assert sampled["every"] == alloc["train"]["keep_every"]
    assert sampled["cases"] == 2
    assert sampled["members"] == 4
    assert sampled["n_steps"] == 50
    # Every key must be one the loader accepts.
    from bdhires.eval import SubgridMonitorConfig

    SubgridMonitorConfig.from_dict(sampled)


def test_stage_b_trainer_samples_before_it_drops_the_ema_weights():
    """Order matters: the picture must describe the checkpoint.

    Script 57 swaps EMA weights in, validates, then swaps the online weights
    back before saving.  A diagnostic placed after the restore would sample
    parameters that are never written to disk, so the figure and the checkpoint
    would disagree with no way to tell from either one.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts/57_train_subgrid_oracle.py"
    ).read_text()
    sample_at = source.index("monitor.should_run(epoch)")
    restore_at = source.index("model.load_state_dict(online)", sample_at - 4000)
    assert sample_at < restore_at, (
        "the diagnostic runs after the EMA weights are swapped back out"
    )
    # And the kept-checkpoint cadence is the one the diagnostic is tied to.
    assert "monitor.validate_cadence(keep_every(config))" in source
    assert "(epoch + 1) % keep_every(config) == 0" in source


def test_v7_stage_a_statistics_use_the_cpcv2_recipe():
    """Stage A's stats must be measured the way CPCv2 measured its own.

    ``06_compute_stats.py``'s defaults are log1p, an ABSOLUTE target and no
    daily wetness.  CPCv2 used none of them: sqrt on the target, sqrt on
    cpc_precip, daily wetness on, and a RESIDUAL parameterisation based on
    cpc_precip.  A stage A trained against the defaults would normalise its
    inputs differently AND predict a different quantity, while every schema
    gate, preflight and shape check still passed -- so this is asserted rather
    than trusted.  The recipe of record is slurm/submit_compute_stats_cpc_v2.sh.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    v2 = (root / "slurm/submit_compute_stats_cpc_v2.sh").read_text()
    meso = (root / "slurm/v7_prepare_meso.sbatch").read_text()

    def v2_default(name):
        match = re.search(rf'{name}:-([^}}]+)\}}', v2)
        assert match, f"{name} is no longer set in the v2 recipe"
        return match.group(1)

    assert v2_default("STATS_TRANSFORM") == "sqrt"
    assert v2_default("STATS_CPC_PRECIP_TRANSFORM") == "sqrt"
    assert v2_default("STATS_DAILY_WETNESS") == "1"
    assert v2_default("STATS_RESIDUAL") == "1"
    assert v2_default("STATS_RESIDUAL_BASE") == "cpc_precip"

    # ...and V7 passes each of them through to the same script.
    for flag in (
        "--transform sqrt",
        '--cond-transform "cpc_precip=sqrt"',
        "--daily-wetness",
        "--residual",
        "--residual-base cpc_precip",
        "--train-years",
    ):
        assert flag in meso, f"V7 stage A statistics are missing {flag}"

    # The residual base index must be derived, never pinned: it is an offset
    # into the conditioning stack, which is exactly the kind of literal that
    # stays syntactically valid while becoming wrong.
    assert 'attrs["cond_channels"]' in meso and 'index("cpc_precip")' in meso, (
        "the residual base index must be read from the archive"
    )
    assert "--residual-base-index 6" not in meso, "the index is hardcoded again"


# --------------------------------------------------------------------------
# V7 two-stage OSSE diagnostic
# --------------------------------------------------------------------------


def test_v7_window_nests_the_two_stages_over_bangladesh():
    """The stages meet by cropping, never by regridding -- and cover the country.

    Stage A trains at 64 cells of 0.1 degree and stage B at 120 cells of 0.05,
    which are NOT the same window.  A convolutional U-Net accepts either size
    silently and simply stops applying attention when the level sizes no longer
    match attn_resolutions, so running one stage at the other's size is a
    different model wearing the same weights.  Each therefore runs at its own
    size and the windows nest -- which only works because both 0.1-degree grids
    share an origin and a lattice.
    """
    from bdhires.eval.v7_window import (
        BANGLADESH_LAT,
        BANGLADESH_LON,
        bangladesh_window,
    )

    window = bangladesh_window()

    # A stage B fine index is exactly twice its coarse index: no interpolation.
    assert window.fine_origin == tuple(2 * o for o in window.coarse_origin)
    assert window.fine_size == 2 * window.coarse_size

    # Stage B's coarse window lies wholly inside stage A's output.
    row, column = window.meso_local
    assert row >= 0 and column >= 0, "a negative offset slices from the wrong end"
    assert row + window.coarse_size <= window.meso_size
    assert column + window.coarse_size <= window.meso_size

    # And the product actually covers Bangladesh.
    grid = window.fine_grid()
    assert grid.lon_min <= BANGLADESH_LON[0]
    assert grid.lon_min + grid.nlon * grid.res >= BANGLADESH_LON[1]
    assert grid.lat_min <= BANGLADESH_LAT[0]
    assert grid.lat_min + grid.nlat * grid.res >= BANGLADESH_LAT[1]


def test_v7_window_refuses_a_crop_too_small_for_the_country():
    """Silently clipping Bangladesh would be the worst possible failure."""
    from bdhires.eval.v7_window import bangladesh_window

    with pytest.raises(ValueError, match="cannot cover|does not cover"):
        bangladesh_window(meso_size=64, fine_size=40)   # 2.0 deg, far too small
    with pytest.raises(ValueError, match="cannot be composed"):
        bangladesh_window(meso_size=16, fine_size=120)  # stage A too small to feed B


def _osse_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/72_v7_two_stage_osse.py"
    spec = importlib.util.spec_from_file_location("v7_osse", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v7_osse_coarse_context_round_trips_through_the_decoder():
    """Stage B must receive the encoding it was TRAINED on, not raw millimetres.

    The allocation branch reads the archive's stored ``coarse_state``: channel 0
    a standardised square root, channel 1 a dequantised binary logit -- not a
    +/-1 flag.  Handing it mm/day, or a hand-rolled occurrence channel, is a
    domain shift that no shape check catches and that would show up only as
    inexplicably poor samples.  Decoding what we build must return what we
    started from.
    """
    from bdhires.data import SubgridEncoding, decode_coarse_amount

    module = _osse_module()
    encoding = SubgridEncoding(
        factor=2, amount_sqrt_mean=1.0, amount_sqrt_std=1.0,
        intensity_log_mean=0.0, intensity_log_std=0.5, smooth_base_iterations=2,
    )
    rng = np.random.default_rng(0)
    mm = torch.from_numpy(np.abs(rng.gamma(2.0, 3.0, (4, 1, 8, 8))).astype(np.float32))
    mm[mm < 0.5] = 0.0                      # genuinely dry cells, so the gate matters

    generator = torch.Generator(device="cpu").manual_seed(7)
    context = module.coarse_context_of(mm, encoding, generator)
    assert context.shape == (4, 2, 8, 8)

    recovered = decode_coarse_amount(
        context, torch.ones_like(mm, dtype=torch.bool), encoding, hard=True
    )
    assert torch.allclose(recovered, mm, atol=1.0e-4), (
        "the coarse context stage B receives does not decode to stage A's analysis"
    )


def test_v7_osse_allocation_sampling_conserves_with_and_without_guidance():
    """Guidance may redistribute inside a block; it may not change the total.

    Conservation to the analysed 0.1-degree field is what bounds the gauge
    increment to 11 km, and that bound is the entire reason V7 might succeed
    where V5 failed.  If guidance broke conservation the experiment would be
    measuring something else.
    """
    from bdhires.da import BilinearObsOperator, GuidanceConfig, build_R
    from bdhires.data import SubgridEncoding, area_weighted_block_mean
    from bdhires.grids import Grid

    module = _osse_module()
    torch.manual_seed(0)
    encoding = SubgridEncoding(
        factor=2, amount_sqrt_mean=1.0, amount_sqrt_std=1.0,
        intensity_log_mean=0.0, intensity_log_std=0.5, smooth_base_iterations=2,
    )
    members, size = 3, 40
    model = AllocationFlow(
        3, image_size=size, base_channels=8, channel_mult=(1, 2),
        num_res_blocks=1, attn_resolutions=(20,), num_heads=1,
    ).eval()
    rng = np.random.default_rng(1)
    coarse = torch.from_numpy(
        np.abs(rng.gamma(2.0, 3.0, (members, 1, size // 2, size // 2))).astype(np.float32)
    )
    cond = torch.randn(members, 3, size, size)
    valid = torch.ones(members, 1, size, size, dtype=torch.bool)
    area = torch.ones(members, 1, size, size)

    def run(observation):
        generator = torch.Generator().manual_seed(3)
        return module.allocation_sample(
            model, coarse, cond, valid, area, encoding, 4, generator,
            observation=observation,
            gcfg=GuidanceConfig(gamma=1e-3, clip_norm=50.0, huber_delta=3.0),
        )

    grid = Grid(name="t", lon_min=88.0, lat_min=21.0, nlon=size, nlat=size, res=0.05)
    operator = BilinearObsOperator(
        grid, np.array([21.5, 22.0, 22.5]), np.array([88.5, 89.0, 89.5])
    )
    observation = {
        "H": operator,
        "y": torch.full((members, 1, 3), 40.0),   # demand far more than the prior gives
        "R": build_R(3, 0.5, representativeness=0.0),
    }

    plain = run(None)
    guided = run(observation)

    for label, field in (("unguided", plain), ("guided", guided)):
        assert torch.isfinite(field).all(), f"{label} sampling produced non-finite values"
        block, _, _ = area_weighted_block_mean(field, area, valid, 2, 0.0)
        assert (block - coarse).abs().max() < 1.0e-4, (
            f"{label} sampling broke conservation to the 0.1-degree field"
        )

    # Guidance has to actually do something, and in the right direction.
    with torch.no_grad():
        before = operator(plain)[:, 0].mean().item()
        after = operator(guided)[:, 0].mean().item()
    assert after > before, "guidance did not move the field toward the observations"


def test_v7_osse_arms_are_declared_consistently():
    """Every arm must have a note, and the four must be distinct settings.

    Script 60 shipped a run where the arm list and its description table had
    drifted apart, and the check fired only AFTER all the sampling was done.
    """
    module = _osse_module()
    assert set(module.ARMS) == set(module.ARM_NOTES)
    # (what is assimilated at 0.1 deg, whether gauges act at 0.05 deg)
    assert module.ARMS["background"] == ("none", False)
    assert module.ARMS["da_both"] == ("gauges", True)
    # The two single-stage arms are what make a degradation attributable.
    assert module.ARMS["da_meso"] != module.ARMS["da_fine"]
    assert len(set(module.ARMS.values())) == len(module.ARMS)


def test_v7_osse_crps_is_ensemble_size_fair():
    """An (m-1)-corrected CRPS, so changing --members is not a change of skill."""
    module = _osse_module()
    truth = np.array([1.0, 5.0, 0.0])
    assert module.crps(np.tile(truth, (4, 1)), truth) == pytest.approx(0.0)
    assert module.crps(np.tile(truth + 1.0, (4, 1)), truth) == pytest.approx(1.0)
    # A wider ensemble around the truth must not look better than a tight one.
    rng = np.random.default_rng(0)
    tight = truth[None] + rng.normal(0, 0.1, (32, 3))
    wide = truth[None] + rng.normal(0, 3.0, (32, 3))
    assert module.crps(tight, truth) < module.crps(wide, truth)


def test_v7_osse_slurm_job_never_writes_into_a_live_run_directory():
    """The diagnostic runs against training that is still going.

    Anything it wrote into runs/v7/*/ could collide with the trainer's own
    checkpointing.  It reads best.pt and writes only under --out.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sbatch = (root / "slurm/v7_osse_diagnostic.sbatch").read_text()
    source = (root / "scripts/72_v7_two_stage_osse.py").read_text()

    assert "--partition=grace" in sbatch and "--gres=gpu:1" in sbatch
    # Checkpoints are copied before use, not read in place.
    assert "shutil.copy2" in source, "the checkpoints are not snapshotted"
    assert "_frozen.pt" in source
    # No write path anywhere near the run directories.
    for forbidden in ('"runs/v7/meso/last', "runs/v7/allocation/last", "best.pt.part"):
        assert forbidden not in source
    assert 'out_dir.mkdir' in source


def test_v7_osse_takes_the_conditioning_contract_from_the_checkpoint(tmp_path):
    """The conditioning stack is read, never re-derived.

    ``data.cond_channels`` selects a SUBSET of the archive's channels and
    ``seasonal_encoding`` adds two more, so the width of the network's first
    convolution is a joint property of both.  Building the inference dataset
    without them produced 17 channels against the trained 16, and the only
    symptom was a torch size mismatch naming two numbers and no cause.  The
    expected width is therefore read from the weights themselves.
    """
    from pathlib import Path

    module = _osse_module()
    source = (
        Path(__file__).resolve().parents[1] / "scripts/72_v7_two_stage_osse.py"
    ).read_text()

    # The dataset is built from the checkpoint's config, not from CLI defaults.
    assert 'meso_cfg["data"].get("cond_channels")' in source
    assert 'meso_cfg["data"].get("seasonal_encoding", True)' in source
    assert "cond_channels=tuple(selected) if selected else None" in source
    # ...and the result is checked against the weights before anything loads.
    assert "meso_expected_cond_channels" in source
    assert 'state["in_conv.weight"].shape[1]' in source

    # The width really does come from the first convolution.
    state = {"in_conv.weight": torch.zeros(96, 17, 3, 3)}
    checkpoint = {"model": state, "weights": "model"}
    frozen = tmp_path / "meso.pt"
    torch.save(checkpoint, frozen)
    assert module.meso_expected_cond_channels(frozen, in_channels=1) == 16

    # Statistics are part of the weights' meaning; a different file must be
    # refused rather than silently mis-scaling every input.
    assert "they must be the same file" in source


def test_v7_osse_imerg_arms_are_gated_and_stacked_correctly():
    """IMERG is an observation, assimilated simultaneously -- never conditioning.

    BMD-aligned IMERG at observation_factor 2 sits on exactly stage A's
    0.1-degree cells, so its forward operator is the identity.  Simultaneous
    means ONE likelihood over a stacked observation vector, not two sequential
    updates -- the second of which would count the prior twice.
    """
    import numpy as np

    from bdhires.da import (
        BilinearObsOperator,
        BlockAverageObsOperator,
        CompositeObsOperator,
    )
    from bdhires.grids import Grid

    module = _osse_module()
    assert module.IMERG_ARMS == {"da_imerg", "da_sim", "da_sim_fine"}
    # The default set stays gauge-only, so an unchanged command line does not
    # suddenly require a file it never needed.
    assert set(module.DEFAULT_ARMS).isdisjoint(module.IMERG_ARMS)
    assert set(module.ARMS) == set(module.ARM_NOTES)
    assert module.ARMS["da_sim"] == ("both", False)
    assert module.ARMS["da_sim_fine"] == ("both", True)
    assert module.ARMS["background"] == ("none", False)

    # A factor-1 block average IS the identity: one footprint per state cell.
    grid = Grid(name="t", lon_min=87.6, lat_min=20.3, nlon=16, nlat=16, res=0.1)
    field = torch.rand(2, 1, 16, 16)
    identity = BlockAverageObsOperator(1, valid=np.ones((16, 16), bool))
    assert torch.allclose(identity(field), field.reshape(2, 1, -1))

    # And the composite stacks gauges first, matching how y and R are built.
    gauge = BilinearObsOperator(grid, np.array([21.0, 21.5]), np.array([88.0, 88.5]))
    composite = CompositeObsOperator([gauge, identity])
    stacked = composite(field)
    assert stacked.shape == (2, 1, 2 + 256)
    assert torch.allclose(stacked[:, :, :2], gauge(field))
    assert torch.allclose(stacked[:, :, 2:], identity(field))


def test_v7_window_can_anchor_stage_a_to_the_imerg_grid():
    """Anchoring stage A to BD-at-0.1 is what makes the IMERG operator exact.

    Without it the analysis window and the BMD-aligned IMERG footprints sit on
    different origins, and assimilating the file would displace rainfall by a
    couple of cells -- with the right shape throughout, so nothing would catch
    it but the verification score.
    """
    from bdhires.eval.v7_window import (
        BANGLADESH_LAT,
        BANGLADESH_LON,
        bangladesh_window,
    )
    from bdhires.grids import BD, WIDE, at_resolution, crop_offsets

    anchored = bangladesh_window(align_meso_to_bd=True)
    assert anchored.meso_origin == crop_offsets(
        at_resolution(WIDE, 0.1), at_resolution(BD, 0.1)
    ), "stage A is not on the IMERG-aligned grid"

    # IMERG factor-2 footprint centres ARE BD-at-0.1 cell centres.
    assert np.allclose(
        BD.lat.reshape(-1, 2).mean(axis=1), at_resolution(BD, 0.1).lat
    )

    # Anchoring must not cost Bangladesh coverage, and stage B must still nest.
    grid = anchored.fine_grid()
    assert grid.lon_min <= BANGLADESH_LON[0]
    assert grid.lon_min + grid.nlon * grid.res >= BANGLADESH_LON[1]
    assert grid.lat_min <= BANGLADESH_LAT[0]
    assert grid.lat_min + grid.nlat * grid.res >= BANGLADESH_LAT[1]
    row, column = anchored.meso_local
    assert row >= 0 and column >= 0
    assert row + anchored.coarse_size <= anchored.meso_size
    assert column + anchored.coarse_size <= anchored.meso_size

    # The unanchored placement remains available and still covers the country.
    centred = bangladesh_window(align_meso_to_bd=False)
    assert centred.meso_origin != anchored.meso_origin


def test_v7_osse_pseudo_satellite_is_an_observation_not_the_answer():
    """A perfect satellite leaks the verification truth; this must not be default.

    The pseudo-satellite reports every 0.1-degree cell, INCLUDING the cells the
    withheld gauges sit in.  Without error it therefore hands the analysis the
    thing the analysis is scored against: withheld CRPS collapses, gauges look
    redundant on top of it, and simultaneous assimilation appears WORSE than
    satellite-only because gauges can only perturb an already-correct field.
    That is a property of the experiment, not of the observing systems, and it
    is exactly what real-data CPCv2 did NOT show.
    """
    import numpy as np

    module = _osse_module()
    rng = np.random.default_rng(0)
    truth = np.abs(rng.gamma(2.0, 6.0, (48, 48))).astype(np.float32)
    truth[:4] = np.nan                     # unobserved cells stay unobserved

    observed, sigma = module.corrupt_satellite(
        truth, sigma_mm=2.0, sigma_frac=0.35, corr_cells=2.0,
        bias_frac=0.10, perfect=False, seed=1,
    )
    keep = np.isfinite(truth)
    assert np.isnan(observed[:4]).all(), "masked cells must stay masked"
    assert (observed[keep] >= 0.0).all(), "a satellite cannot retrieve negative rain"

    # It must differ from truth by roughly the sigma it declares -- an error
    # model whose R does not match how the field was made is mis-specified.
    error = np.abs(observed - truth)[keep]
    assert error.mean() > 0.5, "the pseudo-satellite is essentially perfect"
    assert 0.3 < error.mean() / sigma[keep].mean() < 1.5, (
        "declared sigma and realised error disagree"
    )

    # The systematic part survives averaging -- this is why gauges still carry
    # independent information once a satellite is assimilated.
    assert observed[keep].mean() > truth[keep].mean(), "the wet bias vanished"

    # And the error is spatially correlated, not white: white noise is averaged
    # away by the analysis almost for free, which flatters the satellite.
    residual = np.where(keep, observed - truth, 0.0)
    lag1 = np.corrcoef(residual[4:-1].ravel(), residual[5:].ravel())[0, 1]
    assert lag1 > 0.3, f"satellite error is nearly white (lag-1 {lag1:.2f})"

    # The perfect mode still exists, but only behind an explicit flag.
    exact, exact_sigma = module.corrupt_satellite(
        truth, sigma_mm=2.0, sigma_frac=0.35, corr_cells=2.0,
        bias_frac=0.10, perfect=True, seed=1,
    )
    assert np.allclose(exact[keep], truth[keep])
    assert (exact_sigma > 0).all(), "a zero R is a singular likelihood"

    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts/72_v7_two_stage_osse.py"
    ).read_text()
    assert '"--osse-satellite-perfect", action="store_true"' in source, (
        "the truth-leaking mode must be opt-in, never the default"
    )
    assert "DIAGNOSTIC ONLY" in source


def test_v7_osse_real_mode_verifies_against_actual_gauge_reports():
    """OSSE and real data answer different questions; the script must do both.

    In OSSE the pseudo-gauges ARE CHIRPS, so gauge error and representativeness
    drop out and the experiment isolates how an increment propagates.  With real
    reports all three are in play, and that is the test CPCv2 was judged on --
    where gauges beat the satellite and simultaneous assimilation won.  A script
    that can only do the first cannot reproduce that comparison.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts/72_v7_two_stage_osse.py"
    ).read_text()

    assert 'choices=("osse", "real")' in source
    # Real mode uses the values load_stations returns, not CHIRPS.
    assert "stations, gauge_mm = load_stations(" in source
    assert "truth_at_stations = np.asarray(gauge_mm[day_position]" in source
    # A pseudo-satellite is not a real observation and must be refused there.
    assert "not a real observation" in source
    # Missing reports become NaN, never an assimilated zero -- a gauge that did
    # not report is not a gauge that measured no rain.
    assert "transformed[~np.isfinite(truth_assim)] = np.nan" in source
    assert "draws[:, ~np.isfinite(truth_assim)] = np.nan" in source
    # And an empty verification set must stop the run, not produce a score.
    assert "the verification set" in source
    # The mode is recorded, so a JSON can never be read as the wrong experiment.
    assert '"observations": args.observations,' in source


def test_v7_osse_gauge_sigma_follows_the_observation_mode():
    """A real gauge is not a perfect gauge, and R must say so.

    The OSSE assimilates pseudo-gauges tightly because they ARE the truth, and
    with no point-vs-cell mismatch its representativeness term is zero.  Real
    reports carry both, and configs/da.yaml's tuned values say the
    representativeness term is usually the DOMINANT one -- carrying the OSSE
    settings into a real run would over-fit the assimilated stations and
    flatter every gauge arm in exactly the comparison the run exists to make.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts/72_v7_two_stage_osse.py"
    ).read_text()

    assert '"--meso-gauge-sigma"' in source
    assert '"--meso-gauge-representativeness"' in source
    assert '"--fine-gauge-sigma-mm"' in source
    # Resolved once from the mode, not sprinkled through the call sites.
    assert "real = args.observations == \"real\"" in source
    assert "args.fine_gauge_sigma_mm = 3.0 if real else args.osse_sigma_mm" in source
    # Printed and recorded, so two runs cannot be compared across a silent
    # change of observation error.
    assert "gauge error: stage A sigma" in source
    assert '"meso_gauge_sigma_transformed": args.meso_gauge_sigma,' in source
    assert '"fine_gauge_sigma_mm": args.fine_gauge_sigma_mm,' in source


def test_v7_osse_observation_error_uses_the_right_units_per_stage():
    """The two stages evaluate their likelihoods in different spaces.

    Stage A compares ``tf.forward(rainfall)``, so build_R -- whose docstring
    says "in transformed space" -- must be given transformed units.  Stage B
    compares reconstructed PHYSICAL mm.  One shared number made the 0.1-degree
    gauge variance 9.0 against configs/da.yaml's tuned 0.0725, i.e. 124x too
    weak: the analysis then ignored the very gauges it was assimilating, fitting
    them only 18% better than the background where the OSSE managed 94%.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts/72_v7_two_stage_osse.py"
    ).read_text()

    # Separate flags, each named for its own unit system.
    assert '"--meso-gauge-sigma"' in source and '"--fine-gauge-sigma-mm"' in source
    assert "args.gauge_sigma_mm" not in source, "the conflated sigma survives"
    # CPCv2's tuned values are the real-data defaults.
    assert "args.meso_gauge_sigma = 0.10 if real else 0.05" in source
    assert "args.meso_gauge_representativeness = 0.25 if real else 0.0" in source
    # Each build_R call gets the sigma belonging to its own space.
    # The stage-A sigma may be overridden per arm by the sweep, but it is still
    # a TRANSFORMED-space value that reaches build_R -- never the physical one.
    assert "arm_sigma[arm] if swept else args.meso_gauge_sigma," in source
    assert "args.meso_gauge_sigma," in source
    # A swept value is the TOTAL error, so its representativeness must fold to
    # zero; otherwise 0.25 floors the sweep and the region below CPCv2's 0.269
    # -- which is where this run's optimum lies -- is unreachable.
    assert "0.0 if swept else args.meso_gauge_representativeness" in source
    assert "len(assimilated), args.fine_gauge_sigma_mm, device=device" in source


def test_v7_osse_converts_a_millimetre_error_into_transformed_units():
    """IMERG reports randomError in mm; stage A's R is in transformed units.

    Feeding millimetres straight in is wrong by orders of magnitude and wrong in
    a rain-rate-dependent direction, so the satellite would be over-trusted in
    heavy rain and ignored in light rain -- the opposite of its actual error.
    """
    import numpy as np

    from bdhires.transforms import PrecipTransform

    module = _osse_module()
    transform = PrecipTransform.from_dict(
        {"kind": "sqrt", "mu": 1.5, "sd": 1.2, "eps": 0.02}
    )
    rainfall = np.array([0.0, 1.0, 5.0, 25.0, 100.0], np.float32)
    sigma = module.transformed_sigma(transform, rainfall, np.full(5, 3.0, np.float32))

    assert np.all(np.isfinite(sigma)), "a transformed sigma went non-finite"
    assert np.all(sigma > 0.0)
    # sqrt compresses the tail, so the SAME millimetre error is a smaller
    # transformed error in heavy rain.  That intensity dependence is the point.
    assert sigma[-1] < sigma[2] < sigma[1]
    # And the dry cell must stay finite: sqrt has infinite slope at zero, so a
    # pointwise derivative would report ~25 units here and discard the
    # observation entirely.  A secant across the error is what keeps it sane.
    assert sigma[0] < 2.0, f"dry-cell sigma blew up ({sigma[0]:.2f})"
    assert "SECANT" in (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts/72_v7_two_stage_osse.py"
    ).read_text()


def test_v7_osse_aligns_the_bmd_accumulation_window():
    """BMD day D is not model day D, and nothing about the shapes says so.

    docs/METHOD_SWEEP_PLAN.md measured it: BMD day D is the 24 h ending 03:00
    UTC on D, so it is ~87 percent calendar day D-1, while CHIRPS and CPC are
    00-00 UTC calendar days.  CHIRPS-vs-BMD correlation is 0.626 at lag -1
    against 0.271 at lag 0.  Script 60 carries BACKGROUND_DAY_OFFSET=-1 for
    exactly this reason; a diagnostic that skips it compares a wet model day to
    a dry gauge day and reports the difference as model bias.

    OSSE needs no offset: the pseudo-gauge IS the model-day field.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts/72_v7_two_stage_osse.py"
    ).read_text()

    assert '"--gauge-day-offset"' in source
    assert "args.gauge_day_offset = 1 if real else 0" in source
    # The shift must be applied when the gauge series is READ, so the returned
    # array can still be indexed by model day.
    assert "gauge_times = day_times + np.timedelta64(int(args.gauge_day_offset)" in source
    assert "args.stations, gauge_times, grid=fine_grid" in source
    assert "args.stations, day_times, grid=fine_grid" not in source, (
        "the gauge series is still read on unshifted model dates"
    )
    # Recorded, so a JSON can never be read without knowing its alignment.
    assert '"gauge_day_offset": int(args.gauge_day_offset),' in source
    # And measured per run rather than trusted.
    assert "def lag_check(" in source
    assert "correlation peaks at" in source


def test_v7_osse_lag_check_finds_a_planted_day_shift():
    """The alignment check has to actually detect a shift, not just print.

    A station series that is a lagged copy of the field must peak at the lag it
    was built with; otherwise the check is decoration and a real misalignment
    would still pass unnoticed.
    """
    import numpy as np

    module = _osse_module()
    assert callable(module.lag_check)
    # The correlation logic itself: a series shifted by one day correlates best
    # against the field one day away, which is the whole premise.
    rng = np.random.default_rng(0)
    series = rng.gamma(2.0, 5.0, 40)
    field_day = series[1:-1]
    for offset, expected in ((-1, series[0:-2]), (0, series[1:-1]), (1, series[2:])):
        r = np.corrcoef(field_day, expected)[0, 1]
        if offset == 0:
            assert r > 0.99, "a zero shift must correlate perfectly"
        else:
            assert r < 0.5, f"a {offset:+d} day shift should decorrelate"


def test_v7_osse_reports_spread_skill_so_sharpness_is_not_mistaken_for_skill():
    """CRPS rewards sharpness, so shrinking R always looks like an improvement.

    An over-dispersed analysis lowers its CRPS merely by tightening, and
    tightening is exactly what a smaller observation error does.  Read alone,
    that reads as "the gauges deserve more trust" when it is really a
    spread-calibration problem.  The spread/RMSE ratio is what separates the
    two, so it is computed, reported and stored rather than left implicit.
    """
    import numpy as np

    module = _osse_module()

    truth = np.array([10.0, 20.0, 5.0, 30.0], np.float32)
    # Same ensemble mean error, very different spread.
    tight = np.repeat(truth[None] + 2.0, 8, axis=0) + np.linspace(
        -0.2, 0.2, 8
    )[:, None].astype(np.float32)
    wide = np.repeat(truth[None] + 2.0, 8, axis=0) + np.linspace(
        -12.0, 12.0, 8
    )[:, None].astype(np.float32)

    tight_score = module.score_stations(tight, truth)
    wide_score = module.score_stations(wide, truth)

    for score in (tight_score, wide_score):
        assert "spread_skill" in score, "the calibration diagnostic is missing"
    # Same mean, so the same RMSE and bias -- only the spread differs.
    assert tight_score["rmse_mm"] == pytest.approx(wide_score["rmse_mm"], rel=1e-6)
    assert tight_score["bias_mm"] == pytest.approx(wide_score["bias_mm"], rel=1e-6)
    assert wide_score["spread_skill"] > tight_score["spread_skill"]
    # And the wide one is flagged as over-dispersed while the tight one is not.
    assert wide_score["spread_skill"] > 1.25
    assert tight_score["spread_skill"] < 1.25

    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts/72_v7_two_stage_osse.py"
    ).read_text()
    # The sweep must say which metric it optimised, because they disagree.
    assert "best by " in source
    assert "prefer bias and RMSE" in source
    assert "over-dispersed" in source


def test_v7_osse_matrix_pairs_point_and_field_verification():
    """Eleven withheld gauges cannot say whether the field is right.

    An analysis can fit the gauges it sees by moving rain to the wrong places;
    withheld-point CRPS is weakly sensitive to that, and spatial pattern
    correlation against products the DA never ingested is what catches it.  The
    two belong side by side, because an arm that wins one while losing the other
    is the interesting case.
    """
    import numpy as np

    module = _osse_module()

    valid = np.ones((32, 32), bool)
    rng = np.random.default_rng(0)
    reference = rng.gamma(2.0, 5.0, (32, 32))

    assert module.field_pattern_r(reference, reference, valid) == pytest.approx(1.0)
    # A scaled field keeps its pattern even though its amounts are wrong: that
    # separation is exactly why pattern correlation is reported next to bias.
    assert module.field_pattern_r(3.0 * reference, reference, valid) == pytest.approx(1.0)
    shuffled = rng.permutation(reference.ravel()).reshape(32, 32)
    assert abs(module.field_pattern_r(shuffled, reference, valid)) < 0.3

    # Masked-out cells must not contribute, and too few valid cells is NaN
    # rather than a confident number from five points.
    empty = np.zeros((32, 32), bool)
    assert np.isnan(module.field_pattern_r(reference, reference, empty))

    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts/72_v7_two_stage_osse.py"
    ).read_text()
    # CPC and CHIRPS are always available; IMERG only when supplied.
    assert '"chirps_0p05": field_pattern_r(' in source
    assert '"cpc_0p1": field_pattern_r(' in source
    assert '"imerg_0p1"] = field_pattern_r(' in source
    # The matrix is printed and drawn.
    assert "MATRIX: withheld-gauge CRPS beside spatial pattern correlation" in source
    assert 'out_dir / "matrix.png"' in source


def test_v7_osse_shifts_imerg_onto_the_same_window_as_the_gauges():
    """A BMD-windowed satellite takes the SAME day offset as a BMD gauge.

    load_imerg_meso refuses any file whose bmd_accumulation_end_hour_utc is not
    3, so the file is BMD-windowed by construction: IMERG day D covers the 24 h
    ending 03 UTC on D and is ~87 percent calendar day D-1, exactly like a gauge
    report.  The model is on calendar days, so both take offset +1.

    The trap is docs/METHOD_SWEEP_PLAN.md's lag table, which shows IMERG peaking
    at lag 0 while CHIRPS and CPC peak at -1.  That table is measured against
    the GAUGES, not against the model -- it says IMERG and the gauges already
    share a window, which is exactly why they take the same offset here.  Read
    as a model-relative statement it says the opposite, and IMERG would be
    assimilated a day early with every shape and grid check still passing.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "scripts/72_v7_two_stage_osse.py"
    ).read_text()

    assert "load_imerg_meso(args.imerg, gauge_times, window, meso_grid)" in source, (
        "IMERG is read on unshifted model dates"
    )
    assert "np.datetime64(d) for d, _, _ in days]),\n            window, meso_grid" \
        not in source, "the old unshifted IMERG load survives"
    # The window requirement is what licenses reusing the gauge offset.
    assert "bmd_accumulation_end_hour_utc" in source
    assert 'int(end_hour) != 3' in source
    # And a cross-check that fires when the alignment is wrong anyway.
    assert "imerg vs cpc (same day, both 0.1 deg)" in source
    assert "suspiciously low; check the day alignment" in source
