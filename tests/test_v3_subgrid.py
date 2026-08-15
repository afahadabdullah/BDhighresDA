"""Scientific invariants for the V3-SG hierarchical implementation."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bdhires.da import (  # noqa: E402
    AreaWeightedBlockObsOperator,
    GuidanceConfig,
    HierarchicalObservations,
    authority_decomposition,
    hierarchical_guidance_grad,
)
from bdhires.data import (  # noqa: E402
    SubgridEncoding,
    aligned_production_canvas,
    area_weighted_block_mean,
    encode_subgrid_targets,
    reconstruct_from_amount,
    validate_aligned_crop,
    validate_cpc_alignment,
)
from bdhires.grids import BD, BD_CPC, WIDE_CPC, crop_offsets  # noqa: E402
from bdhires.models import (  # noqa: E402
    AllocationFlow,
    CoarseHurdleFlow,
    CoupledSubgridFlow,
    HierarchicalRectifiedFlow,
    HierarchicalState,
)


def _encoding(**kwargs):
    return SubgridEncoding(
        factor=10,
        amount_sqrt_mean=0.0,
        amount_sqrt_std=1.0,
        dequant_noise=0.0,
        **kwargs,
    )


def test_v3_domains_close_on_cpc_edges_and_preserve_legacy_bd_crop():
    validate_cpc_alignment(BD_CPC)
    validate_cpc_alignment(WIDE_CPC)
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
