# CLAUDE.md — ML Centrality Determination at FXT Energies

## Project at a glance

A machine-learning method for determining collision centrality (impact parameter, centrality percentile, with calibrated per-event uncertainty) from final-state observables in low-multiplicity heavy-ion collisions. Demonstrated on SMASH transport simulations at STAR fixed-target (FXT) energies. Designed to outperform classical RefMult/Glauber methods in the high-baryon-density, low-multiplicity regime where volume fluctuations dominate cumulant-analysis systematic uncertainty.

**Target venue:** Phys. Rev. C (PRC).
**Timeline:** 6–8 weeks solo.
**Methodology paper, no real STAR data, no detector emulation, no STAR GPC approval needed.**

---

## Author

Mathias Labonté, UC Davis nuclear physics PhD candidate (STAR Collaboration), graduating ~2027.

## Working style notes

- No fluff. Do exactly what is asked, no more.
- Honest competitiveness assessments — flag risks, prior art, and limitations directly.
- Verify before asserting — check current state of code/files before claiming behavior.

---

## Scientific motivation

In low-energy heavy-ion collisions (√sNN < ~10 GeV), event-by-event multiplicity is small (~100s of charged particles per central event). Classical centrality determination via charged-multiplicity quantiles (RefMult) or Glauber-tuned multiplicity fits has poor resolution in this regime. The resulting volume fluctuations are the dominant systematic uncertainty on the higher-order cumulant ratios used to search for the QCD critical point in STAR's Beam Energy Scan II program. Improved centrality resolution directly improves the BES-II critical-point search signal-to-noise, and benefits all centrality-dependent observables (flow, spectra, fluctuations).

## Contribution: five differentiators vs Mallick et al. (arXiv:2103.01736)

Mallick used boosted decision trees and shallow neural networks on AMPT events at √sNN = 200 GeV to LHC energies. We differentiate on five axes:

1. **FXT energy regime.** √sNN = 3.2, 3.5, 3.9, 4.5 GeV (vs Mallick's 200 GeV → 5.02 TeV). An order of magnitude lower multiplicity, transport-dominated dynamics, baryon stopping matters, system far from boost-invariant. This is the regime where the volume-fluctuation problem is most severe.

2. **Modern permutation-invariant architectures.** Compare four architectures (decision 2026-05-17): MLP-mean-pool, DeepSets, Set Transformer, and EFN (Energy Flow Network — Komiske–Metodiev–Thaler). All four take the *same* per-particle inputs (pT, η_lab, φ, charge in the lab frame, 0 < η_lab < 2 charged-only filter) and the same event-level features (√sNN, mult_lab, mean_pT_lab, total_pT_lab) — only the inductive bias differs. The CBM PointNet line of work (arXiv:2009.01584 etc.) already used a permutation-invariant network at fixed-target energies; we extend that on multi-energy joint training, calibrated UQ, and downstream cumulant impact (see project_scoop_risk memory).

3. **Joint multi-energy training.** A single network spanning all 4 FXT energies with √sNN as a feature, plus cross-energy generalization tests (train on 3 energies, predict on held-out 4th). Mallick trained per-energy and re-tuned Glauber per energy.

4. **Calibrated per-event uncertainty.** Evidential regression heads producing per-event posterior on impact parameter, *plus* deep ensembles (5 seeds) as a complementary UQ method robust to the Amini-NIG "mirage" critique (arXiv:2510.18322). Mallick gave only point estimates.

5. **Downstream physics impact.** Show ML centrality reduces volume-fluctuation contamination in net-proton C₂ through C₄ ratios on the same SMASH events. Mallick stops at "we predict b better."

## What we do NOT do in v1 (possible follow-ups)

- **Minimal detector emulation IS included** (decision 2026-05-17) — charged tracks only, pT > 50 MeV/c cut, Gaussian pT smearing (~1% + 0.5%·pT), ~90% tracking efficiency. This is acceptance + resolution + efficiency only, not a full GEANT/CBMRoot stack. The truth-level pipeline is also kept so the paper reports both. Full detector simulation (GEANT, calorimeter response, EMC reconstruction, vertex resolution, etc.) is still deferred.
- No real STAR data application.
- No CIGAR (Wang & Luo arXiv:2505.03666) direct comparison.
- No joint inference of (b, N_part, N_coll, V) — just b and percentile for v1.
- No cumulant-aware training loss — standard regression objective.
- No cross-transport-model validation (no UrQMD or AMPT cross-check).

---

## Data

**Existing SMASH simulations (already generated, ready to use):**

- 4 FXT collision energies: √sNN = 3.2, 3.5, 3.9, 4.5 GeV
- Event counts: 50k at 3.2 GeV, 100k each at 3.5 / 3.9 / 4.5 GeV (350k total)
- Au+Au, minimum bias (full centrality range)
- Mean-field setting (single EoS choice)
- Nucleon final states only (no light nuclei)
- Truth-level output (no detector smearing or acceptance applied)
- Truth labels available per event:
  - **b** (fm) — direct, in `impact_param` branch.
  - **N_part** — derived: `N_part = 2A − N_spectators = 394 − count(final-state nucleons with ncoll == 0)`. (Direct participant count from `(|pdg|∈{2112,2212}) & (ncoll>0)` undercounts by ~0–24 because some participants are converted to Δ/N* resonances and decay to non-nucleon final states; spectator count is unambiguous.) Verified: corr(N_part, b) ≈ −0.94 at 3.5 GeV.
  - **N_coll** — *not cleanly derivable* from per-particle `ncoll` (which counts all scatterings, including with produced mesons; not the Glauber binary NN count). Either approximate (upper bound from sum-of-ncoll-over-participants) or rerun SMASH with collision logging enabled. Default v1: skip N_coll as a truth label; Glauber baseline produces its own internally.

**Data path:** `~/phd/SMASH_Analysis/OUT/particle_lists_AuAu_{3.2,3.5,3.9,4.5}GeV.root`

## SMASH output format

**Reference frame.** SMASH dumps 4-momenta in the **center-of-mass frame**, not the lab/target-at-rest frame. Verified empirically: spectator protons cluster at y ≈ ±y_beam_cm = ±arccosh(√sNN / 2m_N). Consequence: η computed from raw (px,py,pz) is already CM-frame η, so |η|<0.5 corresponds to CM mid-rapidity directly — no boost needed.

**Hyperon feed-down.** SMASH does **not** perform weak decays. Final-state Λ/Σ±/Σ⁰/Ξ⁻ are present as stable particles in the output. Final-state protons (pdg=2212) are therefore "direct" relative to weak-decay feed-down — no Λ→pπ contamination — and downstream net-proton cumulants can use the raw 2212 count without a feed-down correction. (This is *not* what STAR data looks like after detector tracking — but this is a truth-level methodology paper, so the truth-level convention is correct here.)

**N_part definition caveat.** Our N_part = 2A − N_spectators with spectator = (nucleon, ncoll==0) is an upper bound vs the Glauber convention, because SMASH's per-particle `ncoll` counts elastic scatterings too, while Glauber N_part counts only nucleons with at least one *inelastic* NN collision. At FXT energies the bias is small but non-zero. Document this in the paper; do not silently report N_part as if it were the Glauber quantity.

**Mass shell.** p0² = px² + py² + pz² + m² holds to ≤1.5×10⁻⁷ relative precision across the sample. The `mass` branch is reliable and can be used directly.

ROOT TTree (one file per energy, ~0.7–1.5 GB each). Read with `uproot` (no pyROOT needed). Single tree `tree` per file, with:

- **Event-level scalars (one per event):** `event_id`, `nparticles`, `impact_param` (fm), `empty_event`, `scattering`.
- **Per-particle jagged arrays (length = nparticles):**
  `pdg`, `ID`, `charge`, `baryon_number`, `strangeness`, `mass`,
  4-momentum `p0`, `px`, `py`, `pz` (GeV),
  freeze-out position `t`, `x`, `y`, `z` (fm),
  history fields `ncoll` (collisions this particle has had), `form_time`, `xsecfac`, `proc_id_origin`, `proc_type_origin`, `time_last_coll`, `pdg_mother1`, `pdg_mother2`.

The ingestion pipeline produces per-event records containing:
- Truth label: b (fm).
- Particle list: (pdg, pT, η, φ, mass, charge) for each final-state particle, derived from the 4-momentum and PDG metadata.
- Event-level summary scalars: charged multiplicity in η windows, mean pT per species, proton/antiproton counts, forward-energy proxy.

Standardize as HDF5 (preferred) with one row per event for downstream training. Persist SMASH version and ingestion-script git commit in HDF5 attributes for reproducibility.

### N_part derivation, N_coll deferral

**N_part is derivable** (see truth-labels section above) and should be computed during ingestion.

**N_coll is not cleanly derivable** from the current SMASH output. The per-particle `ncoll` counts every scattering a particle was involved in — including elastic, resonance excitation, and scatterings off produced mesons — so summing it does not give the Glauber binary-NN-collision count. Options for v1:

1. **Skip N_coll as a v1 truth label.** The Glauber baseline constructs its own N_coll internally. The paper's headline claims target b and centrality percentile, not N_coll. (Default.)
2. **Rerun SMASH** with extended collision logging enabled to dump per-event N_coll directly. Cleanest but costs simulation time.
3. **Approximate** N_coll ≈ ½ × sum(ncoll over participants), with a documented caveat that this is an upper bound contaminated by produced-hadron scatterings.

Stick with option 1 unless `physics-reviewer` says otherwise.

---

## Implementation roadmap

### Task 1 — Lock decisions (Day 1)

Confirm:
- Architectures to compare: MLP (feature-vector baseline), DeepSets, Set Transformer.
- Output heads: impact parameter regression + centrality percentile classification + evidential uncertainty.
- Uncertainty quantification: evidential regression (Amini et al. 2020) as default; mixture density network as alternative.
- Train/val/test split: 80/10/10 within each energy.

### Task 2 — Data pipeline

Convert SMASH output to a standardized HDF5/Parquet dataset with one row per event:
- Truth labels (b, N_part, N_coll)
- Per-particle arrays (pT, η, φ, m, charge) — variable-length, padded for batching
- Event-level summary features (charged multiplicity in η windows, mean pT per species, forward-energy proxy)
- Energy label (one of the 4 FXT energies)

Quick sanity plots: multiplicity-vs-b at each energy, b-distribution (should be triangular for min-bias), N_part-vs-b.

### Task 3 — Classical baselines

Terminology note: in STAR, **RefMult** is the *observable* (charged-particle count in TPC acceptance), not a centrality method. The standard STAR centrality method is **Glauber-MC fit to the RefMult distribution**, with b read off the Glauber model per bin. Our truth-tuned baseline is *not* what STAR does — it's an oracle that uses SMASH truth-b that an experiment cannot access. Both are kept as v1 baselines.

Two baselines:
1. **Truth-tuned percentile baseline** (`src/baselines/truth.py`): bin events by the RefMult-equivalent observable (`mult_eta05`, charged multiplicity in CM-frame |η|<0.5), and assign each bin the *SMASH-truth* mean b. Represents the upper bound on what any single-observable classical method can extract — ML beating this is the meaningful claim because no purely multiplicity-based classical method can do better.
2. **Glauber-MC tuned to RefMult** (`src/baselines/glauber.py`): the STAR-style method. Generate Glauber events (Woods–Saxon Au, geometric NN cross-section overlap), sample multiplicity via the two-component NBD ansatz n_ch = NBD(N_a · n_pp, N_a · k) with N_a = x · N_part/2 + (1−x) · N_coll. Fit (n_pp, k, x) per energy to match the SMASH multiplicity distribution. Predicted b per centrality bin is the *Glauber-model* mean b.

Both baselines bin on the same observable and use the same percentile edges (0–5%, 5–10%, 10–20%, …, 70–80%), so the two are directly comparable. Centrality resolution σ(b_pred − b_true) vs true b at each energy is the bar ML must beat.

**Important FXT finding from Task 3**: ~47% of events at all 4 energies have zero charged particles in |η|<0.5 in the CM frame (spectator-dominated peripherals). Adjacent percentile bins collapse to the same multiplicity threshold of 0, producing an *unresolved peripheral region* for b ≳ 11 fm in both classical baselines. This is the volume-fluctuation problem made concrete and is the strongest motivation for the ML approach.

### Task 4 — Feature-vector MLP baseline ML model

Inputs: scalar summary features (~10–20 numbers per event).
Hidden: 2–4 fully connected layers, ReLU, dropout.
Output heads:
- b regression with evidential output (NIG distribution: μ, ν, α, β).
- Centrality percentile classification with categorical output over 10–20 bins.

Train end-to-end with combined loss. Document hyperparameters.

### Task 5 — DeepSets and Set Transformer

Both architectures operate on the per-particle feature lists (variable length, permutation invariant).

**DeepSets:** φ(particle) → pool → ρ(pooled) → output heads.
**Set Transformer:** stacked self-attention blocks (SAB/PMA from Lee et al. 2019) on particle features, then output heads.

Same output heads as the MLP. Compare against the MLP baseline.

### Task 6 — Calibration diagnostics

- Coverage plots for per-event uncertainty (does the predicted 68% interval contain truth ~68% of the time?).
- Centrality-resolution vs true b: ML methods vs both classical baselines.
- Confusion matrix for centrality percentile classification.

### Task 7 — Cross-energy generalization

Three scenarios:
- Train on 3.2, 3.5, 3.9; test on 4.5 (interpolation/extrapolation up).
- Train on 3.5, 3.9, 4.5; test on 3.2 (extrapolation down).
- Train on all 4; test on held-out events within each (within-distribution baseline).

Compare ML's joint generalization to Glauber's per-energy retuning requirement. This is one of the paper's headline results.

### Task 8 — Downstream cumulant impact

For each centrality method (ML and the two classical baselines):
- Bin events by predicted centrality.
- Compute net-proton C₁, C₂, C₃, C₄ in each centrality bin.
- Report ratios C₂/C₁, Sσ (= C₃/C₂), and κσ² (= C₄/C₂).
- Compare to the "oracle" version using truth-level b for binning.

Show that ML centrality gets closer to the oracle than RefMult or Glauber. This is the physics-utility headline plot.

### Task 9 — Publication figures and writeup

Five planned figures:
1. Centrality-resolution vs true b (ML vs baselines, at one energy).
2. Coverage plot for calibrated uncertainty.
3. Cross-energy generalization (ML vs Glauber, multi-energy).
4. Ablation: MLP vs DeepSets vs Set Transformer.
5. Downstream cumulant accuracy (ML vs baselines vs oracle).

PRC submission target.

---

## Tooling and dependencies

**Required Python packages:**
- `torch` (latest stable, GPU-capable)
- `numpy`, `pandas`, `h5py` or `pyarrow` for data
- `scikit-learn` for the MLP baseline / decision-tree comparisons
- `matplotlib` for figures
- `tqdm` for progress

**Optional / recommended:**
- `lightning` (PyTorch Lightning) for training loop boilerplate
- `wandb` or `tensorboard` for experiment tracking
- `evidential-deep-learning` or implement NIG output directly

**Hardware:** Single GPU sufficient (4M events total in training set; models are small).

**Repository structure (suggested):**

```
fxt-centrality-ml/
├── CLAUDE.md                  # this file
├── README.md                  # short user-facing summary
├── data/
│   ├── raw/                   # original SMASH outputs (gitignored)
│   └── processed/             # HDF5/Parquet event records (gitignored)
├── scripts/
│   ├── ingest_smash.py
│   ├── train_mlp.py
│   ├── train_deepsets.py
│   ├── train_settransformer.py
│   ├── eval_calibration.py
│   ├── eval_cross_energy.py
│   ├── eval_cumulants.py
│   └── make_figures.py
├── src/
│   ├── models/
│   │   ├── mlp.py
│   │   ├── deepsets.py
│   │   └── set_transformer.py
│   ├── baselines/
│   │   ├── refmult.py
│   │   └── glauber.py
│   ├── losses/
│   │   └── evidential.py
│   └── data/
│       └── dataset.py
├── notebooks/
│   └── sanity_plots.ipynb
├── configs/
│   └── default.yaml
├── tests/
└── pyproject.toml or requirements.txt
```

---

## Key references

- **Closest prior art:** Mallick, Tripathy, Behera, et al. "Estimation of impact parameter and transverse spherocity in heavy-ion collisions using machine learning" — arXiv:2103.01736 — BDT centrality at top RHIC/LHC. Cite as the baseline approach to differentiate from.

- **Volume-fluctuation context:** Wang & Luo, "Centrality-independent framework for revealing genuine higher-order cumulants in heavy-ion collisions" (CIGAR) — arXiv:2505.03666. Alternative approach to the same problem (sidesteps centrality determination via Edgeworth expansion + Bayesian inference). Worth comparing in discussion; possible follow-up paper.

- **Bzdak-Koch volume fluctuation corrections:** Bzdak, Koch, Skokov — Phys. Rev. C 87, 014901. Standard analytical correction for volume fluctuations.

- **Permutation-invariant architectures:**
  - Zaheer et al., "Deep Sets" — NeurIPS 2017.
  - Lee et al., "Set Transformer" — ICML 2019.

- **Evidential regression for uncertainty:** Amini et al., "Deep Evidential Regression" — NeurIPS 2020.

- **SMASH:** Weil et al. (SMASH Collaboration), "Particle production and equilibrium properties within a new hadron transport approach for heavy-ion collisions" — Phys. Rev. C 94, 054905 (2016).

- **STAR FXT BES-II program context:** STAR's recent BES-II light-nuclei paper arXiv:2512.05295 and net-proton cumulant papers — for framing the physics-utility argument.

---

## Risks and pre-empts

1. **"Mallick already did ML centrality."** Differentiation must lean on all 5 axes (FXT regime, modern architectures, multi-energy, calibrated uncertainty, downstream cumulant impact) — none alone is decisive.

2. **"Why no detector emulation? Real data has detector effects."** Methodological paper, deliberately experiment-agnostic. Detector-specific applications are explicit future work.

3. **"Truth-level results may not transfer to real data."** Acknowledge this in the limitations section; the cross-transport-model robustness (UrQMD) and detector-emulation extensions are obvious follow-ups.

4. **"Glauber baseline isn't the right comparison."** STAR analysts may push back. Make sure the Glauber implementation is faithful to what STAR actually does (per-energy negative-binomial tuning).

5. **STAR analysts may publish ML centrality on real data via GPC before we publish.** Methods paper on simulation can be drafted faster than a STAR GPC analysis paper — push to submit promptly.

---

## What to start with on day 1

1. Confirm SMASH data path and format. Run a quick ingestion sanity check.
2. Make the multiplicity-vs-b scatter plot at each energy. Eyeball whether multiplicity alone is a meaningful centrality estimator at FXT energies (expectation: significantly noisier than at top RHIC).
3. Start a fresh git repo with the suggested structure.
4. Implement Task 3 (classical baselines) before any ML work — establishes the bar to beat.
