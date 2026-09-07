"""G0 engineering checks; synthetic results are not rainfall skill evidence."""
# ruff: noqa: E402 -- torch-dependent imports follow importorskip

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from bdhires.da.guidance import GuidanceConfig, guidance_grad
from bdhires.da.observation import (
    CompositeObsOperator,
    PhysicalBilinearObsOperator,
    PhysicalBlockAverageObsOperator,
    perturb_observations,
)
from bdhires.da.sampler import SamplerConfig, assimilate, sample
from bdhires.eval.graphflow import response_metrics, single_observation_response
from bdhires.grids import Grid
from bdhires.models import (
    EMA,
    GraphFlowUNet,
    RectifiedFlow,
    UNet,
    VelocityOnly,
    build_model,
    flow_matching_loss,
    model_from_checkpoint,
    model_metadata,
    predict_dry_logit,
    select_weights,
    split_prediction,
)
from bdhires.models.graph_processor import MultimeshProcessor
from bdhires.models.multimesh import MeshCache, build_multimesh
from bdhires.models.unet import ResBlock, fourier_time_embedding
from bdhires.transforms import PrecipTransform, ResidualSpec

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_kwargs():
    return dict(in_channels=1, cond_channels=3, out_channels=1, base_channels=8,
                channel_mult=(1, 2, 3, 4), num_res_blocks=2, image_size=32,
                attn_resolutions=(4, 8), dropout=0.0, num_heads=1,
                multiscale_conditioning=True)


def tiny_model(out_channels=1, **graph):
    kwargs = tiny_kwargs()
    kwargs["out_channels"] = out_channels
    return GraphFlowUNet(**kwargs, graph=dict(hidden_dim=16, num_blocks=2, **graph))


def activate(model):
    # The base U-Net also zero-initializes its output. Exercise the actual
    # nonzero network Jacobian instead of passing a trivial all-zero-head test.
    with torch.no_grad():
        model.out_conv.weight.normal_(0, 0.03)
        if getattr(model, "graph_processor", None) is not None:
            model.graph_processor.output_projection.weight.normal_(0, 0.03)


@pytest.mark.parametrize("shape", [(32, 32), (7, 11), (8, 12), (1, 7), (1, 1)])
@pytest.mark.parametrize("diagonal", [True, False])
def test_multimesh_topology(shape, diagonal):
    height, width = shape
    mesh = build_multimesh(height, width, diagonal_edges=diagonal)
    again = build_multimesh(height, width, diagonal_edges=diagonal)
    assert torch.equal(mesh.src, again.src)
    assert torch.equal(mesh.edge_features, again.edge_features)
    assert ((mesh.src >= 0) & (mesh.src < height * width)).all()
    assert ((mesh.dst >= 0) & (mesh.dst < height * width)).all()
    pairs = list(zip(mesh.src.tolist(), mesh.dst.tolist(), strict=False))
    pair_set = set(pairs)
    assert len(pairs) == len(pair_set)
    assert all((b, a) in pair_set for a, b in pairs)
    assert (mesh.degree > 0).all() and torch.isfinite(mesh.degree).all()
    raw_degree = torch.bincount(mesh.dst, minlength=height * width).clamp_min(1)
    assert torch.equal(mesh.degree, raw_degree.float())
    reachable = {0}
    local_pairs = []
    for (src, dst), features in zip(pairs, mesh.edge_features.tolist(), strict=False):
        y, x = divmod(src, width)
        y2, x2 = divmod(dst, width)
        stride = [1, 2, 4, 8][round(features[3] * 3)]
        assert y % stride == x % stride == y2 % stride == x2 % stride == 0
        assert abs(y2 - y) in (0, stride) and abs(x2 - x) in (0, stride)
        assert max(abs(y2 - y), abs(x2 - x)) == stride
        if not diagonal:
            assert abs(y2 - y) + abs(x2 - x) == stride
        assert features[:2] == pytest.approx([(x2 - x) / 8, (y2 - y) / 8])
        if stride == 1:
            local_pairs.append((src, dst))
    for _ in range(height * width):
        reachable.update(b for a, b in local_pairs if a in reachable)
        if len(reachable) == height * width:
            break
    assert len(reachable) == height * width


def test_cache_reuse_eviction_and_inference_to_autograd():
    cache = MeshCache(max_entries=2)
    with torch.inference_mode():
        first = cache.get(8, 8, (1, 2), True, "cpu")
    assert not torch.is_inference(first.src)
    assert first is cache.get(8, 8, (1, 2), True, torch.device("cpu"))
    cache.get(8, 12, (1, 2), True, "cpu")
    cache.get(12, 12, (1, 2), True, "cpu")
    assert len(cache._entries) == 2
    assert first is not cache.get(8, 8, (1, 2), True, "cpu")
    model = tiny_model()
    with torch.inference_mode():
        model(torch.randn(1, 1, 32, 32), torch.ones(1), torch.randn(1, 3, 32, 32))
    activate(model)
    x = torch.randn(1, 1, 32, 32, requires_grad=True)
    model(x, torch.ones(1), torch.randn(1, 3, 32, 32)).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert not any("cache" in key for key in model.state_dict())
    model.to(dtype=torch.float64)
    assert not model.graph_processor.cache._entries


@pytest.mark.parametrize("shape", [(128, 128), (64, 96)])
@pytest.mark.parametrize("channels", [1, 2])
def test_shapes_conditioning_and_hurdle(shape, channels):
    model = tiny_model(out_channels=channels).eval()
    x, cond, t = torch.randn(2, 1, *shape), torch.randn(2, 3, *shape), torch.rand(2)
    seen = []
    handle = model.graph_processor.register_forward_pre_hook(lambda _, args: seen.append(args[0].shape))
    with torch.no_grad():
        output = model(x, t, cond)
    handle.remove()
    assert output.shape == (2, channels, *shape)
    assert seen == [torch.Size([2, 24, shape[0] // 4, shape[1] // 4])]
    assert torch.isfinite(output).all()
    velocity, dry = split_prediction(output, channels == 2)
    assert velocity.shape == x.shape
    if channels == 2:
        assert dry.shape == x.shape
        assert predict_dry_logit(model, x, cond[:1]).shape == x.shape
    else:
        assert dry is None


def old_forward(model, x, t, cond):
    """Pre-G0 forward path, independent of the new encoder hook."""
    emb = model.time_embed(fourier_time_embedding(t, model.base_channels))
    features = model.condition_encoder(cond) if model.condition_encoder is not None else None
    h = model.in_conv(torch.cat([x, cond], dim=1) if cond is not None else x)
    skips = [h]
    for index, block in enumerate(model.down):
        for layer in block:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
                if features is not None:
                    h = h + model.down_condition_projections[index](features[model.down_condition_levels[index]])
            else:
                h = layer(h)
        skips.append(h)
    for index, layer in enumerate(model.mid):
        h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        if index == 0 and features is not None:
            h = h + model.mid_condition_projection(features[-1])
    for index, block in enumerate(model.up):
        h = torch.cat([h, skips.pop()], dim=1)
        for layer in block:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
                if features is not None:
                    h = h + model.up_condition_projections[index](features[model.up_condition_levels[index]])
            else:
                h = layer(h)
    return model.out_conv(torch.nn.functional.silu(model.out_norm(h)))


@pytest.mark.parametrize("multiscale", [False, True])
@pytest.mark.parametrize("shape", [(32, 32), (40, 48)])
def test_legacy_regression_and_nontrivial_zero_init_equivalence(multiscale, shape):
    kwargs = tiny_kwargs()
    kwargs["multiscale_conditioning"] = multiscale
    baseline = UNet(**kwargs).eval()
    # Nonzero weights throughout, including every originally zero residual/head.
    with torch.no_grad():
        for p in baseline.parameters():
            p.uniform_(-0.1, 0.1)
    graph = GraphFlowUNet(**kwargs, graph=dict(hidden_dim=16, num_blocks=2)).eval()
    missing, unexpected = graph.load_state_dict(baseline.state_dict(), strict=False)
    assert missing and all(key.startswith("graph_processor.") for key in missing)
    assert not unexpected
    x, cond, t = torch.randn(2, 1, *shape), torch.randn(2, 3, *shape), torch.rand(2)
    with torch.no_grad():
        reference = old_forward(baseline, x, t, cond)
        assert reference.abs().max() > 0
        torch.testing.assert_close(baseline(x, t, cond), reference, rtol=0, atol=0)
        torch.testing.assert_close(graph(x, t, cond), reference, rtol=0, atol=0)
    disabled = GraphFlowUNet(**kwargs, graph=dict(enabled=False))
    disabled.load_state_dict(baseline.state_dict(), strict=True)
    torch.testing.assert_close(disabled(x, t, cond), reference, rtol=0, atol=0)


@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_processor_gradients(checkpoint_blocks):
    processor = MultimeshProcessor(8, hidden_dim=12, num_blocks=2, zero_init_output=False,
                                  checkpoint_blocks=checkpoint_blocks)
    x = torch.randn(2, 8, 7, 9, requires_grad=True)
    processor(x).square().mean().backward()
    for name, p in processor.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    assert torch.isfinite(x.grad).all()


def observation_case(size=32, device="cpu"):
    grid = Grid("synthetic", 88, 21, size, size, 0.05)
    tf = PrecipTransform(kind="sqrt")
    gauge = PhysicalBilinearObsOperator(grid, np.array([grid.lat[size // 2]]),
                                       np.array([grid.lon[size // 2]]), tf)
    satellite = PhysicalBlockAverageObsOperator(8, tf)
    H = CompositeObsOperator([gauge, satellite], component_spread_cells=[6.0, 0.0]).to(device)
    residual = ResidualSpec(enabled=True, mean=0.2, std=0.7)
    base = torch.full((1, 1, size, size), 2.0, device=device)
    def decode(x):
        return residual.decode(x, base)
    y = H(decode(torch.zeros(1, 1, size, size, device=device))).detach()
    R = torch.full((y.shape[-1],), 0.10**2 + 0.25**2, device=device)
    return H, y + 0.1, R, decode


@pytest.mark.parametrize("hurdle", [False, True])
def test_checkpoint_ema_loss_guidance_and_sampler(tmp_path, hurdle):
    torch.manual_seed(17)
    model = tiny_model(out_channels=2 if hurdle else 1, checkpoint_blocks=True)
    activate(model)
    ema = EMA(model, decay=0.9)
    x, cond = torch.randn(2, 1, 32, 32), torch.randn(2, 3, 32, 32)
    loss = flow_matching_loss(model, x, cond, RectifiedFlow(), cond_dropout=0,
                              dry_target=(x < 0).float() if hurdle else None)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    ema.update(model)
    checkpoint_data = dict(model=model.state_dict(), ema=ema.state_dict(), weights="ema",
                           model_config=model_metadata(model))
    path = tmp_path / "graphflow.pt"
    torch.save(checkpoint_data, path)
    loaded = torch.load(path, weights_only=False)
    restored = model_from_checkpoint(loaded, cond_channels=3, image_size=64)
    assert restored.image_size == 32  # saved attention placement, variable forward canvas
    assert isinstance(restored, GraphFlowUNet)
    assert model_metadata(restored) == model_metadata(model)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, ema.shadow[key])
    assert select_weights(loaded) is loaded["ema"]
    restored = VelocityOnly(restored)
    H, y, R, decode = observation_case()
    u, grad = guidance_grad(x, torch.full((2,), 0.5), restored, RectifiedFlow(), cond,
                            H, y, R, GuidanceConfig(gamma=0.01), to_precip=decode)
    assert u.shape == grad.shape == x.shape
    assert torch.isfinite(grad).all() and grad.norm() > 0
    assert all(p.grad is None for p in restored.parameters())
    cfg = SamplerConfig(n_steps=3, n_corrections=1, corrector_max_step=0.02, seed=9)
    first = sample(restored, cond[:1], x.shape, "cpu", cfg=cfg)
    torch.testing.assert_close(first, sample(restored, cond[:1], x.shape, "cpu", cfg=cfg))
    yy = torch.tensor(perturb_observations(y[0, 0].numpy(), R, 2, seed=5), dtype=x.dtype)[:, None]
    assert not torch.equal(yy[0], yy[1])
    kwargs = dict(H=H, y=yy, R=R, cfg=cfg, to_precip=decode, gcfg=GuidanceConfig(gamma=0.01))
    with torch.no_grad():
        guided = assimilate(restored, cond[:1], x.shape, "cpu", **kwargs)
    assert torch.isfinite(first).all() and torch.isfinite(guided).all()
    torch.testing.assert_close(guided, assimilate(restored, cond[:1], x.shape, "cpu", **kwargs))
    assert not torch.equal(first, guided)
    kwargs["y"] = y
    assert not torch.equal(guided, assimilate(restored, cond[:1], x.shape, "cpu", **kwargs))
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        guidance_grad(x, torch.full((2,), 0.5), restored, RectifiedFlow(), cond,
                      H, y, R, GuidanceConfig(), to_precip=decode)


def test_legacy_checkpoint_without_metadata(tmp_path):
    baseline = UNet(**tiny_kwargs()).eval()
    activate(baseline)
    cfg = dict(model={k: v for k, v in tiny_kwargs().items()
                      if k not in {"in_channels", "cond_channels", "out_channels", "image_size"}},
               data=dict(crop=32))
    ck = dict(cfg=cfg, ema=baseline.state_dict())  # pre-weights/architecture schema
    path = tmp_path / "legacy.pt"
    torch.save(ck, path)
    restored = model_from_checkpoint(torch.load(path, weights_only=False))
    assert type(restored) is UNet
    x, t, cond = torch.randn(2, 1, 32, 32), torch.rand(2), torch.randn(2, 3, 32, 32)
    torch.testing.assert_close(restored(x, t, cond), baseline(x, t, cond), rtol=0, atol=0)


def test_no_condition_and_amp_cpu():
    kwargs = tiny_kwargs()
    kwargs["cond_channels"] = 0
    model = GraphFlowUNet(**kwargs, graph=dict(hidden_dim=16, num_blocks=2))
    activate(model)
    x = torch.randn(2, 1, 32, 32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = flow_matching_loss(model, x, None, RectifiedFlow())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_amp_and_guidance():
    model = tiny_model().cuda()
    activate(model)
    x, cond = torch.randn(2, 1, 32, 32, device="cuda"), torch.randn(2, 3, 32, 32, device="cuda")
    H, y, R, decode = observation_case(device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
        loss = flow_matching_loss(model, x, cond, RectifiedFlow())
        _, grad = guidance_grad(x, torch.full((2,), 0.5, device="cuda"), model,
                                 RectifiedFlow(), cond, H, y, R, GuidanceConfig(), to_precip=decode)
    loss.backward()
    assert torch.isfinite(grad).all() and torch.isfinite(loss)


def test_config_is_frozen_cpcv2_except_backbone_and_directory():
    base = yaml.safe_load((ROOT / "configs/train_h100_cpc_v2.yaml").read_text())
    new = yaml.safe_load((ROOT / "configs/train_h100_cpc_graphflow_g0.yaml").read_text())
    assert new["model"].pop("architecture") == "graphflow_unet"
    graph = new["model"].pop("graph")
    assert graph["hidden_dim"] == 256 and graph["num_blocks"] == 8 and graph["level"] == 2
    assert new["train"]["out_dir"] != base["train"]["out_dir"]
    new["train"]["out_dir"] = base["train"]["out_dir"]
    assert new == base


@pytest.mark.parametrize("strides", [(2, 4), (1, 1, 2), (1, 0), (1, 2.5), ()])
def test_invalid_strides(strides):
    with pytest.raises(ValueError, match="mesh_strides"):
        build_multimesh(4, 4, strides)


def test_bad_architecture_settings_fail():
    with pytest.raises(ValueError, match="unknown velocity"):
        build_model(architecture="gnn")
    with pytest.raises(ValueError, match="graph settings"):
        build_model(graph={})
    with pytest.raises(ValueError, match="graph level"):
        tiny_model(level=10)
    with pytest.raises(ValueError, match="unknown graph"):
        tiny_model(wind_edges=True)
    with pytest.raises(ValueError, match="mean aggregation"):
        tiny_model(aggregation="sum")
    with pytest.raises(ValueError, match="boolean"):
        tiny_model(enabled="false")
    model = tiny_model()
    with pytest.raises(ValueError, match="pyramid factor"):
        model(torch.randn(1, 1, 33, 32), torch.rand(1), torch.randn(1, 3, 33, 32))
    ck = dict(model_config=model_metadata(model), model=model.state_dict())
    with pytest.raises(ValueError, match="conditioning channel"):
        model_from_checkpoint(ck, cond_channels=9)
    broken = deepcopy(ck)
    broken["model_config"]["graph"]["hidden_dim"] = 19
    with pytest.raises(RuntimeError, match="size mismatch"):
        model_from_checkpoint(broken)


def test_graph_update_reaches_next_downsample_and_corresponding_skip():
    model = tiny_model().eval()
    activate(model)
    seen = {}

    def record_output(_module, _args, output):
        seen["graph"] = output.detach().clone()

    def record_downsample(_module, args):
        seen["downsample"] = args[0].detach().clone()

    def record_skip(_module, args):
        seen["skip"] = args[0][:, -24:].detach().clone()

    hooks = [model.graph_processor.register_forward_hook(record_output)]
    # Last level-2 block precedes downsample (2 residual blocks / level).
    down_index = next(i for i, level in enumerate(model.down_condition_levels)
                      if level == -1 and model.down_condition_levels[i - 1] == 2)
    hooks.append(model.down[down_index][0].register_forward_pre_hook(record_downsample))
    up_index = model.up_condition_levels.index(2)
    hooks.append(model.up[up_index][0].register_forward_pre_hook(record_skip))
    model(torch.randn(1, 1, 32, 32), torch.rand(1), torch.randn(1, 3, 32, 32))
    for hook in hooks:
        hook.remove()
    torch.testing.assert_close(seen["graph"], seen["downsample"], rtol=0, atol=0)
    torch.testing.assert_close(seen["graph"], seen["skip"], rtol=0, atol=0)


def test_online_checkpoint_selection_and_cfg_only_graph_roundtrip():
    model = tiny_model()
    metadata = model_metadata(model)
    cfg_model = {k: v for k, v in metadata.items()
                 if k not in {"in_channels", "cond_channels", "out_channels", "image_size"}}
    ck = dict(cfg=dict(model=cfg_model, data=dict(crop=32)), model=model.state_dict(),
              ema={k: value + 1 for k, value in model.state_dict().items()}, weights="model")
    restored = model_from_checkpoint(ck)
    assert isinstance(restored, GraphFlowUNet)
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, ck["model"][name], rtol=0, atol=0)


def test_sensitivity_metrics_distinguish_isotropic_and_directional_response():
    y, x = np.indices((33, 33))
    circle = np.exp(-((x - 16)**2 + (y - 16)**2) / 8.)
    horizontal = np.exp(-((x - 16)**2 / 40. + (y - 16)**2 / 2.))
    a, b = (response_metrics(field, 16, 16, 12) for field in (circle, horizontal))
    assert a["anisotropy"] < .001 and b["anisotropy"] > .8
    assert b["maximum_distant_response"] > a["maximum_distant_response"]
    assert a["l2_norm"] == pytest.approx(np.linalg.norm(circle))
    assert sum(a["radial_energy"]) == pytest.approx(np.square(circle).sum())


def test_sensitivity_gradient_sees_graph_backbone_and_zero_innovation():
    model = tiny_model().eval()
    activate(model)
    grid = Grid("synthetic", 88, 21, 32, 32, .05)
    x, cond = torch.randn(1, 1, 32, 32), torch.randn(1, 3, 32, 32)
    kwargs = dict(grid=grid, transform=PrecipTransform(kind="sqrt"),
                  residual=ResidualSpec(enabled=True, base="cpc_precip"),
                  base=torch.full_like(x, 2), mask=torch.ones_like(x), row=16., col=16.)
    response = single_observation_response(model, x, cond, **kwargs)
    assert np.isfinite(response["raw_gradient"]).all()
    assert np.linalg.norm(response["raw_gradient"]) > 0
    zero = single_observation_response(model, x, cond, **kwargs, innovation=0)
    assert np.all(zero["raw_gradient"] == 0)
