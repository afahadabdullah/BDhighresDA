# BDhighresDA — manuscript review and what to do next

Written against `BDhighresDA_arxiv.tex` (rewritten 2 Aug 2026). Ordered by what
would change a reviewer's verdict, not by effort.

---

## Part 1 — What I changed, and why

| Change | Reason |
|---|---|
| Retitled; abstract cut to one full-width paragraph leading with the claim | The old title had two nouns doing the same job ("Development and Process Validation"). The old abstract buried the OSSE result behind three sentences of setup. |
| Added §2 Related work | The paper had **zero** positioning against CorrDiff, DGMR, Harris et al., DPS, or score-based DA. A reviewer's first question is "how is this different from CorrDiff?" — it now has an answer (they ingest observations as predictors; this one doesn't). |
| Added Appendix A: derivation of the (1−t)/t score↔velocity factor | The old text asserted this conversion with no support. It's exact for a linear interpolant and takes half a page to prove. This is now the most defensible technical content in the paper. |
| Derived `V_t = R + Γ(1−t)²/t² I` instead of calling it "an early-time uncertainty term" | Inverting the interpolant under a flat prior gives `Var(x₁|x_t) = (1−t)²/t² I` exactly. What looked ad hoc is actually principled — say so. |
| Added §6.3 "How much can five days actually decide?" | **The most important change.** The old paper reported a 0.240 mm/day CRPS gap with no uncertainty and let it decide the method. |
| Showed that gate G2 is vacuous | Under H₀, wins ~ Binomial(5, ½), so P(≥3 of 5 wins) = 16/32 = **0.50**. A "three of five folds" criterion is met half the time by chance. Stating this yourself is far stronger than having a referee find it. |
| Softened "we select gauge-only" → "gauge-only is provisional; fusion is unadjudicated, not refuted" | The data cannot support the stronger claim, and the weaker one is more interesting. |
| Added 18 citations (Liu, Dhariwal & Nichol, Karras, Chung, Esser, Ho & Salimans, Ravuri, Harris, Leinonen, Mardani, Rozet NeurIPS, Hamill, Hersbach, Ferro, Zamo & Naveau, Politis & Romano, Perez, Wu & He) | "ADM-style", "Heun", "rectified flow", "logit-normal t", "classifier-free guidance", "fair CRPS" were all used uncited. |
| Consolidated the ablation table into Block A / Block B with a comparability note | The old table mixed incomparable rows behind a caption disclaimer. |
| Replaced 14 empty figure frames with Table 6 (planned figures + producing script + release gate) | 14 grey boxes on arXiv read as unfinished. A gated figure plan reads as disciplined. |
| Added an explicit "What this paper does not claim" paragraph | Pre-empts the three objections a referee would otherwise lead with. |
| Added an Availability section and a "computational cost — to be recorded" row | Both are expected; both are currently blank. |
| Rebuilt Figure 1 | See Part 2. |
| Layout: `lmodern` + `T1`, full-width abstract, figure moved to p.3, bibliography before appendices, appendix tables de-floated | The old file also would not compile with `microtype` once sans-serif entered the figure — `lmodern` fixes that permanently. |

Backup of the original is at `BDhighresDA_arxiv_v1_original.tex`.

---

## Part 2 — Figure 1

The old diagram was a single flat row squeezed through `\resizebox`, so its
text rendered smaller than the caption, with a dashed arc swinging outside the
frame and no legend.

The rebuild makes the paper's central architectural claim *visible* rather than
merely stated: two banded phases, a hard rule between them, and exactly **one
arrow crossing it**, labelled "weights are the only thing that crosses."
Everything else follows from that. It also now carries:

- the notation the equations use (`c`, `x₁`, `x_t`, `H`, `R`, `V_t`, `∇log p(y|x_t)`), so the figure and the method section reference each other;
- the exact IMERG window and the `D−1` background offset, i.e. the two things §6.1 says mattered most;
- the temporal split with the leakage caveat printed inside Phase 1;
- a five-entry legend, with "assimilated observation" visually distinct because that colour appears *only* in Phase 2 — which is the whole point;
- native TikZ at true size (no `\resizebox`), so type matches the document.

---

## Part 3 — What the work needs. Prioritised.

### Tier 1 — blocking. Without these the central comparison is undecidable.

**1. Run the configured full-May experiment and attach paired block-bootstrap intervals.**
Everything else is downstream of this. 930 station-days instead of 150.
Circular block bootstrap over ΔCRPS with block length ≥ 3 days (rainfall
autocorrelation), ≥ 10,000 resamples, paired by station-day, report 95% CI on
*gauges − simultaneous*. If that interval straddles zero — which is plausible —
the honest conclusion is "indistinguishable," and that is publishable.
Retire gate G2 in favour of the interval.

**2. Retrain with 2018 excluded, then rerun unchanged.**
This is the single line separating "process validation" from any skill claim.
One training run. Do not touch a hyperparameter between the two runs, or the
comparison is lost.

**3. Fit the IMERG bias correction (`scripts/07_bias_correct_imerg.py`) on 2001–2017.**
The evidence points straight at it: the simultaneous arm's bias is
+3.122 mm/day versus +1.913 for gauges. A Gaussian likelihood assumes an
unbiased observation, and IMERG over South Asia is not — it over-detects
drizzle and misses orographic rain on the Meghalaya barrier. Your own
`METHODOLOGY.md` §4.4 calls skipping this "the main way this design goes
wrong," and the current results are consistent with exactly that. Fusion may
well win once this is fitted.

**4. Fill in affiliation, repository URL, and computational cost.**
Training wall-clock, per-day guided-sampling wall-clock, and GPU type. Reviewers
ask, and the guided/unguided cost ratio is a selling point for the design.

### Tier 2 — the difference between a lab notebook and a paper.

**5. Add non-generative baselines to the headline table.** *(biggest remaining gap)*
Table 5 compares four DA arms **only against each other**. Nothing there shows
that the generative prior is necessary. A referee will ask directly: could
inverse-distance weighting of the same 24 gauges, or CPC bilinearly
interpolated, or optimal interpolation with a fitted covariance, have produced
4.6 mm/day? You compute these baselines already
(`19_compare_bmd_5day_sensitivity.py`) — put them in the primary table, not in
a supporting figure. Add a deterministic U-Net regression baseline trained on
identical data if you can afford the run; it isolates "generative" from
"learned downscaling."

**6. Add a domain and observation-geometry figure (planned Fig. 2).**
There is currently **no map anywhere in the paper**. For a geoscience
readership that is disqualifying on its own, and it is the cheapest figure on
the list — nothing needs to run.

**7. Do the "climate of the model" check** (Manshausen Appendix C; already in
your ROADMAP Phase 2). Time-mean maps and PDFs of unconditional samples versus
CHIRPS. Cheap, standard, and it either validates the prior or finds a bug.

**8. Address the dispersion failures rather than only reporting them.**
Both OSSE regimes fail spread–skill and rank histograms. Ordered by leverage:
Γ sweep (`configs/da.yaml` already lists `[1e-4, 1e-3, 1e-2]`), prior
temperature sweep (`[1.0 … 2.0]`), ensemble size, then post-hoc inflation
calibrated out-of-sample as a clearly-labelled last resort. Even a negative
result ("no setting in this sweep achieves calibration") is worth a figure.

**9. Lead harder on the two-failure-modes finding.**
"Fine-scale power ratio 5.98 with spread–skill 0.51" — too rough *and* too
narrow simultaneously — is the most novel diagnostic observation in the paper
and it is currently one paragraph. Give it a spectral figure (planned Fig. 7).
It generalises beyond Bangladesh and beyond precipitation, and it is the kind
of thing that gets a paper cited.

### Tier 3 — scientific upside once Tier 1 lands.

**10. Station-density sweep** (5/10/20/35/100 gauges; already in ROADMAP Phase 3).
Directly answers "how many gauges does Bangladesh need to get a 5 km analysis,
and where?" — which is the policy-relevant question and a strong standalone
result. Your OSSE framework can run it today without new data.

**11. Extreme-event case studies** — 2022 Sylhet flood, 2017 haor flash floods.
Only meaningful after the leakage-free retrain, since both must sit outside
the training window.

**12. Seasonal weighting.** Your own open question #4 — the optimal IMERG weight
almost certainly differs between JJAS and DJF. One σ_IMERG for the whole year is
an assumption worth testing.

**13. CPC v2 versus v1** on identical seeds and folds, once v1 has a clean baseline.

---

## Part 4 — Verify before posting

Things I wrote from your docs and could not check against data:

- **All reported scores** are transcribed from remote run reports; I could not access `data/` or `runs/`. Re-check every number in Tables 3–5 against the JSON outputs before posting.
- **Parameter counts** (51.8 M / 57.5 M) come from the source specification, not from counting a loaded checkpoint. Count them.
- **Γ = 1e-3** — verified against `configs/da.yaml` line 76. ✅
- **50 sampler steps, 16 members** — verified against `configs/da.yaml`. ✅
- **"Wetherell 2026" (arXiv:2606.00281)**, cited in `README.md` and `METHODOLOGY.md`, is *not* in the manuscript bibliography. I did not add it because I could not verify it exists. Check the arXiv ID; if real, it belongs in §2 next to `liu2023`.
- **`mardani2024`** is cited as the arXiv version. If the journal version is out, update it.
- The **affiliation** line says George Mason University, inferred from your project path. Correct it if wrong, and add a department and ORCID.

## Part 5 — Venue

As it stands the paper is well-suited to arXiv and poorly suited to JAMES/MWR,
for one reason: it has no positive product claim, and journals in this space
expect one. Two workable routes:

- **Post now as a preprint.** The negative results — dense-satellite
  over-constraint, the accumulation-window bug, the vanishing fusion win — are
  genuinely useful to anyone building a similar system, and nobody publishes
  them. Then submit the journal version after Tier 1.
- **Or hold and reframe** once the full-month and leakage-free runs land, with
  the OSSE as the methodological core and the real-data section as a deployment
  reality check.

I'd post now. The current framing is honest and the failed ablations are the
part of this work least likely to exist anywhere else.
