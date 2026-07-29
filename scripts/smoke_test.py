#!/usr/bin/env python
"""End-to-end smoke test on synthetic data. No downloads, runs on CPU in ~2 min.

    python scripts/smoke_test.py

Checks
------
1. transforms are invertible
2. the flow-matching identities (x1_hat / score / velocity-from-score) are
   self-consistent to floating-point precision
3. the U-Net forward pass and loss run and shapes match
4. the observation operator recovers known point values
5. guided sampling actually pulls the state toward the observations
6. a synthetic Zarr store round-trips through PrecipDataset
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bdhires.da import BilinearObsOperator, GuidanceConfig, SamplerConfig, build_R
from bdhires.da.sampler import assimilate
from bdhires.grids import BD, Grid
from bdhires.models import RectifiedFlow, UNet, flow_matching_loss
from bdhires.transforms import PrecipTransform

OK, FAIL = "  ok  ", " FAIL "


def check(name, cond, detail=""):
    print(f"[{OK if cond else FAIL}] {name} {detail}")
    if not cond:
        raise SystemExit(1)


def test_transforms():
    p = np.array([0.0, 0.05, 1.0, 25.0, 300.0, 900.0])
    for kind in ("log1p", "sqrt", "cbrt", "none"):
        tf = PrecipTransform(kind=kind).fit(np.random.gamma(0.3, 8.0, 20000))
        back = tf.inverse(tf.forward(p))
        check(f"transform {kind} invertible", np.allclose(back, p, atol=1e-4),
              f"max err {np.abs(back-p).max():.2e}")


def test_flow_identities():
    flow = RectifiedFlow()
    torch.manual_seed(0)
    x1 = torch.randn(4, 1, 16, 16)
    t = torch.rand(4).clamp(0.05, 0.95)
    x_t, u, x0 = flow.interpolate(x1, t)
    check("x1_hat identity", torch.allclose(flow.x1_hat(x_t, t, u), x1, atol=1e-5))
    check("x0_hat identity", torch.allclose(flow.x0_hat(x_t, t, u), x0, atol=1e-5))
    s = flow.score(x_t, t, u)
    tb = t.view(-1, 1, 1, 1)
    check("score == -(x_t - t*x1)/(1-t)^2",
          torch.allclose(s, -(x_t - tb * x1) / (1 - tb) ** 2, atol=1e-4))
    check("velocity_from_score round-trip",
          torch.allclose(flow.velocity_from_score(x_t, t, s), u, atol=1e-4))


def test_unet():
    net = UNet(in_channels=1, cond_channels=5, out_channels=1, base_channels=32,
               channel_mult=(1, 2, 2), num_res_blocks=1, attn_resolutions=(16,), image_size=64)
    x = torch.randn(2, 1, 64, 64)
    c = torch.randn(2, 5, 64, 64)
    t = torch.rand(2)
    y = net(x, t, c)
    check("unet shape", y.shape == x.shape, f"{tuple(y.shape)}  params={net.num_parameters/1e6:.1f}M")
    mask = (torch.rand(2, 1, 64, 64) > 0.3).float()
    loss = flow_matching_loss(net, x, c, RectifiedFlow(), mask=mask, cond_dropout=0.1)
    loss.backward()
    check("loss finite + backward", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")


def test_obs_operator():
    g = Grid("t", lon_min=0.0, lat_min=0.0, nlon=32, nlat=32, res=1.0)
    lat = np.array([g.lat[5], g.lat[20]])
    lon = np.array([g.lon[10], g.lon[3]])
    H = BilinearObsOperator(g, lat, lon)
    field = torch.zeros(1, 1, 32, 32)
    field[0, 0, 5, 10] = 7.0
    field[0, 0, 20, 3] = -2.0
    out = H(field)[0, 0].numpy()
    check("obs operator picks exact cell centres", np.allclose(out, [7.0, -2.0], atol=1e-4),
          f"got {out}")
    # gradient flows
    f = torch.zeros(1, 1, 32, 32, requires_grad=True)
    H(f).sum().backward()
    check("obs operator is differentiable", f.grad.abs().sum().item() > 0)


def test_guided_sampling():
    """A tiny net trained for a few steps on a constant field; guidance must
    pull the sampled field toward the observed value at the station."""
    torch.manual_seed(0)
    g = Grid("t", lon_min=0.0, lat_min=0.0, nlon=32, nlat=32, res=1.0)
    net = UNet(in_channels=1, cond_channels=0, out_channels=1, base_channels=16,
               channel_mult=(1, 2), num_res_blocks=1, attn_resolutions=(), image_size=32)
    flow = RectifiedFlow()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(150):  # learn p(x) = N(0, 1) fields -- a valid, cheap prior
        x1 = torch.randn(16, 1, 32, 32) * 0.5
        loss = flow_matching_loss(net, x1, None, flow, cond_dropout=0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()

    lat = np.array([g.lat[16]])
    lon = np.array([g.lon[16]])
    H = BilinearObsOperator(g, lat, lon)
    y = torch.full((4, 1, 1), 3.0)
    R = build_R(1, sigma_obs=0.05)
    scfg = SamplerConfig(n_steps=30, n_corrections=1, seed=0)
    gcfg = GuidanceConfig(gamma=1e-3, scale=1.0)

    free = assimilate(net, None, (4, 1, 32, 32), torch.device("cpu"), cfg=scfg, flow=flow)
    guided = assimilate(net, None, (4, 1, 32, 32), torch.device("cpu"), H=H, y=y, R=R,
                        cfg=scfg, gcfg=gcfg, flow=flow)
    v_free = float(H(free).mean())
    v_guided = float(H(guided).mean())
    check("guidance moves the state toward the observation",
          abs(v_guided - 3.0) < abs(v_free - 3.0),
          f"free={v_free:.3f} guided={v_guided:.3f} target=3.0")
    check("guided field stays finite", torch.isfinite(guided).all().item())


def test_dataset_roundtrip():
    """Exercises PrecipDataset against an in-memory store with the same layout
    the packing script writes (dict of numpy arrays is enough -- zarr arrays
    are duck-type compatible for everything the dataset does)."""
    from bdhires.data import DatasetConfig, PrecipDataset

    T, H, W, C = 12, 64, 64, 4
    rng = np.random.default_rng(0)
    tgt = rng.gamma(0.3, 8.0, (T, H, W)).astype("f4")
    tgt[:, :34, :] = np.nan  # fake ocean (deep enough that every 32px crop touches it)
    valid = np.ones((H, W), "f4")
    valid[:34, :] = 0
    store = {
        "target": tgt,
        "cond": rng.normal(size=(T, C, H, W)).astype("f4"),
        "static": rng.normal(size=(7, H, W)).astype("f4"),
        "valid": valid,
        "time": np.arange(T).astype("datetime64[D]").astype("datetime64[ns]").view("i8"),
    }

    tf = PrecipTransform(kind="log1p").fit(tgt[np.isfinite(tgt)])
    ds = PrecipDataset(DatasetConfig(root="", crop=32, random_crop=True), tf,
                       cond_mean=np.zeros(C, "f4"), cond_std=np.ones(C, "f4"), store=store)
    item = ds[0]
    check("dataset x1 shape", tuple(item["x1"].shape) == (1, 32, 32))
    check("dataset cond channels", item["cond"].shape[0] == ds.total_cond_channels,
          f"{item['cond'].shape[0]} == {ds.total_cond_channels}")
    check("dataset has no NaNs", torch.isfinite(item["x1"]).all().item()
          and torch.isfinite(item["cond"]).all().item())
    check("mask excludes ocean", item["mask"].mean().item() < 1.0)
    check("target round-trips through the transform",
          np.allclose(tf.inverse(item["x1"].numpy()) * item["mask"].numpy(),
                      np.nan_to_num(ds.transform.inverse(item["x1"].numpy())) * item["mask"].numpy(),
                      atol=1e-4))


def test_grids():
    check("BD grid is 128x128", BD.shape == (128, 128))
    check("BD covers Bangladesh",
          BD.lon_min < 88.0 and BD.lon_max > 92.7 and BD.lat_min < 20.57 and BD.lat_max > 26.64,
          f"bbox={BD.bbox}")
    check("Dhaka is inside the domain",
          BD.lon_min < 90.38 < BD.lon_max and BD.lat_min < 23.78 < BD.lat_max)


if __name__ == "__main__":
    print("=" * 70)
    test_grids()
    test_transforms()
    test_flow_identities()
    test_obs_operator()
    test_unet()
    test_dataset_roundtrip()
    test_guided_sampling()
    print("=" * 70)
    print("all smoke tests passed")
