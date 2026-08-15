# Review of `VERSION3_SUBGRID_MODEL_IDEA.md` (V3-SG) — round 5

Reviewed against the 50.9 KB revision. Significant items only; three remain.

All five round-4 items are closed, two of them better than I proposed:

- On the frozen thresholds, I suggested raising a threshold to the measured MDE.
  That was wrong, and the document's rule is correct: widening a scientifically
  meaningful non-inferiority margin to whatever the data can resolve makes the
  gate pass without making the claim true. Increasing dates/members where that
  improves precision, and otherwise marking the gate unresolved and lowering the
  claim ceiling, is the right ordering.
- On observation authority, I suggested reporting `m`/`z` increment shares. The
  symmetric two-factor decomposition in **physical units** is materially better —
  latent norms are not comparable across branches, and `C_m + C_z = x_aa − x_bb`
  makes the attribution exact. I checked the algebra; it is correct, and because
  `R` is linear in `m` for fixed `z`, both components reduce to interpretable
  quantities (`C_m ∝ (m_a − m_b)` times a mean allocation pattern, `C_z ∝` the
  mean amount times the allocation change).

---

## 1. `coupled_oracle` uses naive replacement sampling, so it measures the sampler, not the architecture

This undermines the fix just added for gate 2.

The arm clamps the coarse state at every stage to

```text
m_t = (1-t) * epsilon_m + t * m_truth
```

with pinned noise, and evolves only the fine branch. That is the **replacement**
(a.k.a. naive inpainting) method, and it does not sample `p(z | m_1 = m_truth)`.

The reason: at intermediate `t`, the network is not told that the terminal coarse
field *is* `m_truth`. It is shown `m_t` and internally infers a posterior over
`m_1`, which at moderate `t` is diffuse. So the fine branch is conditioned on a
progressively **smeared** version of `m_truth`, not on `m_truth`. The bias is
largest exactly where the coupling is strongest — which is the regime the whole
design is built around. This is the well-known failure of replacement guidance
that resampling schemes exist to correct.

Consequence for gate 2: if `coupled_oracle` is artificially depressed by sampler
bias, then

- the reported "architecture/coupling cost" (`oracle_flow` → `coupled_oracle`)
  is inflated and attributed to the wrong cause; and
- the one-third retention threshold is measured against a **depressed**
  reference, making gate 2 easier to pass than intended. The gate becomes lax in
  the direction that matters least.

The existing test — "recovering exact `m_truth` at the terminal state" — only
verifies that the clamp is applied. It cannot detect this.

**Add a decisive and cheap validation:** clamp `m` to a coarse field the coupled
flow itself sampled, and check that the resulting `z` distribution matches the
unclamped joint sample's `z` distribution (allocation entropy, wet fraction,
residual variance by scale, and the `z`-marginal at matched `m`). If they agree,
the replacement scheme is adequate for this model and `coupled_oracle` is
trustworthy. If they do not, either

1. use a resampling / back-and-forth conditional sampler for the clamped arm; or
2. train the coupled flow with coarse-branch dropout so it accepts a clean
   clamped `m` as a conditioning mode it has actually seen.

Whichever is used must be the same scheme for `coupled_oracle` and for any other
clamped-state diagnostic, and it should be named in the document rather than left
as "conditional/inpainting sampling."

---

## 2. Gate 6 tests the direction that is nearly structural, not the quantity that is interesting

`S_m(IMERG impulse) − S_m(gauge impulse) > 0` is very likely to hold almost by
construction. `H_imerg` is a 0.4° area mean, which is largely determined by the
overlapping 0.5° amounts; `H_gauge` is a point sample, which the prior can
satisfy locally. A reviewer will say so.

The genuinely novel quantity — the one the mismatched-support design exists to
create — is the **magnitude of IMERG's allocation share**, `1 − S_m(IMERG)`.
Because 0.4° footprints straddle 0.5° blocks, a coarse area-mean innovation *can*
be absorbed by sub-block restructuring without changing any `m`. How much of it
the learned prior chooses to absorb that way is not predictable from the operator
geometry alone, and it is the number that demonstrates a coarse observation
carrying genuine subgrid information.

**Strengthen gate 6** by making the primary authority statement two-sided:

- the contrast `S_m(IMERG) − S_m(gauge)` is positive with its interval excluding
  zero (keep as-is, as a sanity condition); **and**
- `1 − S_m(IMERG)` is materially greater than the value obtained under a control
  in which the mismatch is removed — e.g. the same impulse experiment run with
  an IMERG operator aligned to the 0.5° blocks, where sub-block restructuring
  cannot change the observed mean.

That control is cheap (one alternative observation operator on the same
backgrounds and seeds) and converts gate 6 from "the area observation behaves
like an area observation" into a measurement of what the deliberate support
mismatch actually buys. It also gives the Purpose section's claim that the
mismatch "makes this claim testable rather than definitionally true" a designated
test, which it currently lacks.

---

## 3. The fine branch has the same interface problem the coarse branch just had

The coarse-branch transfer was fixed properly: Model A is now a rectified flow
over hurdle latents with a matching velocity interface, and there is a test for
identical pre-fine-tuning velocities on a pinned batch.

The fine branch has not had the same treatment, and it has a specific mismatch.
Model B is trained in Phase 2 with **conditioning augmentation**: corrupted
`m_truth` plus *the corruption level as a conditioning scalar*. In the coupled
flow, `m` is state rather than conditioning, so that scalar input channel has no
counterpart. On transfer it is either dropped — leaving unmatched parameters on
the first layer that consumes it, which the "no unmatched intended parameters"
test would flag as a failure — or retained and fed a constant that was never a
training mode.

Two related statements are also now stale:

- Phase 3: "Model A's held-out error distribution sets the final
  conditioning-augmentation calibration." In the coupled flow there is nothing to
  calibrate; the joint model learns the amount–allocation dependence directly.
  The sentence describes the sequential ablation, not the operational path.
- Phase 2's augmentation work therefore serves three purposes — the sequential
  ablation, the oracle arms, and fine-branch initialisation — and only the third
  touches the operational model.

**Resolve it the same way as the coarse branch:** state the fine-branch state and
conditioning interface explicitly, say what happens to the corruption-level
channel on transfer (drop it and reinitialise the affected layer, or retain it
pinned at the calibrated operational level), and extend the pinned-batch
velocity-equality test to the fine branch. Then correct the Phase-3 sentence to
scope conditioning-augmentation calibration to the sequential generator.

---

## Nothing else

I re-derived the `C_m`/`C_z` decomposition, the `bd_cpc` crop indices and the
conservation identity under area weighting; all correct. The straight-through
occurrence estimator with quantitative soft-to-hard `O−A` bounds, the Phase −1
gate-precision audit, the ERA5-ablation probe, the compute freeze and the V2
comparability freeze need nothing further from me.

---

## Sources

- [Aich et al. (2026), conditional diffusion for precipitation downscaling and bias correction — GMD](https://doi.org/10.5194/gmd-19-1791-2026)
- [Schmidt et al. (2025), probabilistic spatiotemporally coherent climate downscaling with a diffusion prior — npj Clim. Atmos. Sci.](https://doi.org/10.1038/s41612-025-01157-y)
