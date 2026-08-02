# What "downscaling skill" means here, and how we measure it

## The question this document answers

> The OSSE has a product that uses 0.1° for the DA. But for testing the
> sub-grid-scale downscaling we compare just background downscaling with the
> target, right? Or some other method?

Comparing the background against the target is **necessary but not sufficient**,
and on its own it is not interpretable. "Background RMSE against CHIRPS = 4.2"
is not a downscaling claim, because it does not say what the alternative was.
Every skill statement needs a null model, and the honest null is different for
each of the three things we actually want to claim.

The instinct is right — the background *is* where the downscaling claim lives,
because the background never sees an observation. What is missing is (a) the
null to compare it against, and (b) a second, stronger claim that the analysis
supports and the background cannot.

---

## The three claims

### Claim A — downscaling gain. *Does the prior beat the coarse input?*

- **Field scored:** background (never the analysis).
- **Null:** `coarse_base_mm`, the 0.5° conditioning precipitation on the fine
  grid. It has no genuine variance below its own resolution.
- **Means:** the generative prior synthesises real structure between 0.5° and
  0.05°.

Scored on the background *only*. If we scored the analysis against this null,
assimilated observations would be counted as downscaling skill — which is the
most common way papers in this class overstate their result. The prior must earn
this one unaided.

### Claim B — sub-footprint gain. *Does the analysis beat perfect satellite information?*

- **Field scored:** the residual after removing each field's own 0.1° block mean.
- **Null:** the **truth's own 0.1° block mean**, upsampled. Perfect coarse
  information, exactly zero structure inside each footprint.
- **Means:** the analysis placed rainfall correctly below the satellite
  footprint. In the satellite-only arm no assimilated observation resolves
  this component. In gauge and simultaneous arms, point gauges may constrain
  it locally, so attribution must be stratified by distance from gauges.

This is the load-bearing result. The null is deliberately unfair to us: it is
handed better satellite information than any real retrieval provides. Beating it
is not "we fit the footprints well" — fitting footprints is trivial and is
reported separately as a sanity check under `footprint_component`. Beating it
means the prior correctly allocated rain *inside* footprints it was only told
the average of.

This is also why the answer to the original question is "both, not either".
Claim A shows the prior can downscale. Claim B shows the downscaling survives
contact with the assimilation, which is the claim a reader of a DA paper
actually cares about.

### Claim C — texture realism. *Do the members look right, or just look sharp?*

- **Diagnostics:** power spectra, effective resolution, variogram, FSS.
- **Never RMSE**, which rewards blurring — an ensemble mean always beats a
  member on RMSE while being physically wrong for precipitation.

Claim C is a guard, not a claim. A field can beat the nulls on located skill and
still have unusable texture, and it can have beautiful texture and no located
skill at all. `test_right_amplitude_wrong_location_earns_no_skill` in
`tests/test_scale_metrics.py` pins exactly this: a sign-flipped field has a
member energy ratio of exactly 1.0 and negative skill.

---

## Why a scale ladder rather than a single number

`scale_ladder()` decomposes every field at aggregation factors 1, 2, 4, 8
(0.05°, 0.1°, 0.2°, 0.4°) and scores the two components separately:

- **aggregated** — how good the field is at that scale *and coarser*
- **residual** — how good it is *below* that scale

Reading the two together is what distinguishes downscaling from bias correction.
A model that improves only the aggregated component has not downscaled anything;
it has corrected a mean. The residual curve is the one that has to move.

## Two failure modes that must not be merged

The OSSE reports a member/truth fine-scale power ratio near 6 alongside a
spread–skill ratio near 0.5. These are opposite defects and are usually
collapsed into one word, "calibration":

| Symptom | Meaning | Fix direction |
|---|---|---|
| power ratio ≫ 1 | an individual member is **too rough** — it paints texture the truth does not have | reduce sampler temperature, more steps, stronger coarse consistency |
| spread/skill < 1 | the ensemble is **too narrow** — members agree too much about where it rained | inflate the prior, raise Γ, more members |

Applying a fix for one while blind to the other makes the other worse. The
report keeps them in separate fields for that reason, and the paper-suite
ranking treats spread/skill and energy ratio as **targets of 1.0**, not as
quantities to maximise — otherwise the least calibrated arm wins the table.

## Effective resolution

Reported in km: the shortest wavelength down to which the field's power stays
within a factor of two of CHIRPS. This is the product's honest resolution —
features finer than it are either damped away or invented. It is the single most
useful number for a reader deciding whether a "5 km product" is really 5 km, and
it is the one comparison where the coarse input should look obviously worse.

Caveat recorded in the code: spectra are computed on zero-filled, Hann-tapered
fields. Gap filling adds a little spurious power near the coast. Interpolating
instead would *suppress* high-wavenumber power — an error in the direction that
flatters the model — so zero-filling is the conservative choice.

---

## Pipeline

```
10_osse.py                  runs the OSSE, dumps ensembles + coarse_base_mm
   |
22_evaluate_osse_downscaling.py   claims A/B/C for ONE arm
   |                              -> downscaling.json, curves.npz, spatial.nc
   |
24_osse_paper_suite.py      all arms -> tables, figures, CSV, RESULTS.md
```

Run everything with:

```bash
bash slurm/submit_osse_paper.sh
```

`24` recomputes nothing. It selects, arranges and renders, so every number in
the manuscript traces back to exactly one `downscaling.json` and the tables
cannot drift away from the text.

## Note on `coarse_base_mm`

Claim A was unmeasurable before this change: the OSSE dump did not carry the
conditioning field, so there was no null to compare the background against.
`10_osse.py` now stores it, decoded with `residual.fill` rather than zeros —
`decode()` adds the standardisation mean back, so feeding zeros would offset the
whole field by μ_r and bias every claim-A score. Dumps predating this change
still work; claim A reports `unavailable` with instructions rather than
silently disappearing.
