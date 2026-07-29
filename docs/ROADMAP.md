# Roadmap

## Phase 0 — scaffold ✅

- [x] Domains, transforms, U-Net, rectified flow, EMA
- [x] Score ↔ velocity identities linking flow matching to SDA guidance
- [x] Differentiable station operator + block-average (IMERG) operator
- [x] Guided sampler with Langevin correctors
- [x] Dataset with random-crop augmentation and land masking
- [x] Verification metrics (point + spatial)
- [x] Smoke test passing end to end on CPU

## Phase 1 — data (the long pole)

- [ ] ERA5 download, 1981–2025 (CDS queue is the bottleneck; submit early,
      chunk by month, expect days-to-weeks)
- [ ] CHIRPS download + subset, 1981–2025
- [ ] IMERG Final download, 2000-06–2025
- [ ] DEM (GMTED2010 or SRTM) → static channels
- [ ] Pack to Zarr; **verify time alignment** by correlating ERA5 tp and CHIRPS
      at lags −2…+2 days — the peak must be at lag 0
- [ ] BMD gauge CSV from BMD; QC; pseudo-station archive
- [ ] Sanity plots: CHIRPS vs IMERG vs ERA5 climatology, monsoon cycle,
      Meghalaya gradient

## Phase 2 — prior

- [ ] Train ERA5-only prior, 1981–2018 (`configs/train_era5only.yaml`)
- [ ] "Climate of the model" check (Manshausen Appendix C): time-mean maps and
      PDFs of unconditional samples vs CHIRPS
- [ ] Transform ablation: `log1p` vs `sqrt`, focused on the upper tail
- [ ] Deterministic U-Net baseline for comparison

## Phase 3 — assimilation

- [ ] Pseudo-observation experiments (known truth): tune Γ, σ_gauge, σ_IMERG
- [ ] IMERG → CHIRPS quantile map + empirical error model (script 07)
- [ ] Three-way ablation: IMERG `condition` vs `assimilate` vs off
- [ ] Station-density sweep: 5 / 10 / 20 / 35 / 100 gauges
- [ ] Real BMD gauges, 3-fold CV, 2021–2025
- [ ] Calibration: spread/skill, rank histograms; fix under-dispersion if present

## Phase 4 — product and paper

- [ ] Produce 1981–2025 daily 5 km ensemble reanalysis over the BD grid
- [ ] CF-compliant NetCDF/Zarr release + DOI
- [ ] Compare against CHIRPS, IMERG, ERA5, MSWEP at withheld gauges
- [ ] Extreme-event case studies (2022 Sylhet flood; 2020 Amphan;
      2017 haor flash floods)
- [ ] Paper draft

## Phase 5 — extensions

- [ ] Multivariate: add Tmax/Tmin channels (Mishra South Asia 5 km as target);
      assimilate temperature gauges with no code change
- [ ] Sub-daily via IMERG half-hourly
- [ ] 4D / sequence prior (assimilate a window rather than a snapshot)
- [ ] Real-time mode with IMERG Early (4 h latency) and inflated R

## Open questions

1. Is ~14k daily fields enough, or does the prior need a South-Asia-wide
   training domain? Decide from the Phase 2 climate check.
2. Does the log1p transform's upper tail hold up at the 100 mm/day threshold,
   or does it need a heavy-tail-aware treatment (Pandey et al. 2024)?
3. How much observation-error correlation is there between BMD gauges and
   CHIRPS (which already blends GTS gauges)? If a station is already in
   CHIRPS, assimilating it is partly circular — worth checking the CHIRPS
   station list before choosing the evaluation set.
4. Does the optimal IMERG weighting differ by season? Likely yes (JJAS ≫ DJF).
