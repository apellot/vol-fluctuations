# Modelling plan — UrQMD + detector + b<14

Companion to `docs/restructure_urqmd_detector_b14.md`. Locks the modelling-phase
decisions made 2026-05-28. Primary data: `data/processed/cached/urqmd_padded_det.h5`
(b<14, native Npart, per-particle `(pT, η_lab, φ, charge)` + event scalars
`(√sNN, mult_lab, mean_pT_lab, total_pT_lab)`).

## Locked decisions (2026-05-28)

1. **Centrality is derived from Npart** — no classification head. Rank events by
   Npart (predicted Npart for the ML method, **true Npart** for the oracle),
   DESCENDING (highest Npart = most central), cut the standard percentile edges
   (`DEFAULT_BIN_EDGES`). The confusion matrix (Task 6) is computed on these derived
   bins. Rationale: Npart is the participant/volume measure that drives particle
   production and the volume fluctuations this paper targets — binning by Npart
   controls volume more tightly than b (at fixed b, Npart still has a Glauber
   spread), and it matches the Glauber-centrality convention. `corr(Npart,b)≈−0.98`
   confirms it is a clean monotonic centrality variable. Percentile binning is
   rank-based, so the Npart-provenance caveat (possible A=208/Pb scale) does NOT
   affect the binning — only absolute ⟨Npart⟩ reporting. Classical baselines still
   bin on their multiplicity observable (unchanged); the cumulant **oracle is
   true-Npart** percentile binning.
2. **`b` and `Npart` each get an evidential NIG head** — calibrated per-event UQ for
   both targets.
3. **Deep ensembles are phased** — single-model evidential first (Phase 1), 5-seed
   ensembles after (Phase 2).
4. **Keep φ; enforce rotational invariance by augmentation.** φ carries centrality
   information through the flow-vector magnitudes |Qₙ| (v₁ large at FXT, v₂ small/
   negative) — only the absolute reaction-plane orientation Ψ_RP is the uniform
   nuisance. The per-event random φ-rotation augmentation preserves |Qₙ| (it only
   rotates the unobserved Ψ_RP) and forces the network to learn rotation-invariant
   features. We do NOT reconstruct the reaction plane (EP resolution is poor at FXT
   multiplicities). See "φ handling" below.

## Inputs & φ handling

Per-particle: `(pT, η_lab, φ, charge)`; event scalars: `(√sNN, mult_lab,
mean_pT_lab, total_pT_lab)`. Same inputs for all architectures.

- **φ-rotation augmentation, ON by default in the canonical trainer.** Port
  `apply_phi_rotation` (`src/data/lab_cached_dataset.py:112`) into `train_cached.py`
  (currently only `train_lab.py` applies it — a real gap: models trained via
  `train_cached.py` today overfit the random Ψ_RP). Independent Δφ per event,
  preserves all relative angles ⇒ |Qₙ| unchanged.
- **Input representation option (to ablate):** feed φ as **(cos φ, sin φ)** instead
  of the raw angle. An MLP handles the ±π wraparound badly; sin/cos makes a rotation
  a linear map and lets the network form Qₙ = Σ e^{inφ} far more easily. Track as a
  representation flag and ablate against raw-φ on one architecture.
- `MLP-pool` only sees `mean(φ)` (rotation-non-invariant junk) — do not read flow
  sensitivity into it; the architectures that can exploit anisotropy are
  DeepSets / SetTransformer / PFN / EFN / GNN.

## Heads (`src/models/heads.py`)

Replace `OutputHeads` with **two NIG heads, no classifier**:
- `b`  → (μ, ν, α, β)
- `Npart` → (μ, ν, α, β)
- Returns `{b_mu,b_nu,b_alpha,b_beta, np_mu,np_nu,np_alpha,np_beta}`.

All set models call `self.heads(h)` → updated automatically; `mlp.py` has inline
heads → update separately. Drop `n_centrality_bins` from every config (no classifier).

## Targets & standardization

Standardize `b` and `Npart` (z-score on the **train split only**); train each NIG in
standardized space; invert μ and scale σ back to fm / participants for reporting.
Store `(mean, std)` per target in the checkpoint. Standardization makes the two NIG
losses comparable in magnitude.

## Loss (`src/losses/evidential.py` reused)

`loss = evidential(b_std) + w_np · evidential(npart_std)`. Drop the cross-entropy
term entirely. `w_np` default 1.0 (both standardized); `coeff_reg=1e-2` per head.

## Canonical trainer (consolidate on `train_cached.py`)

- Add `Npart` to the batch and targets (cache already stores it).
- Add **event-feature z-score normalization** (port from `train_lab.py`), stored in
  the checkpoint.
- **Port φ-rotation augmentation** (on by default; `--no-phi-aug` to disable).
- Add **input-representation flag**: `--phi-encoding {raw,sincos}` (default `raw`
  initially, flip after the ablation).
- **Drop the `truth_dir` / `centrality_bin` dependency** — training no longer needs
  truth labels (`run_truth` stays only for the *baseline*). Big simplification.
- Per-energy prediction files: `b_pred`+UQ, `npart_pred`+UQ, `is_train/val/test`,
  plus a derived `centrality_bin` (from `b_pred`) for convenience.
- Deprecate `train_lab.py` / `train_set_model.py` (fold into `train_cached.py`).
  Keep `train_mlp.py` only if the 9-feature PID-augmented MLP baseline is wanted;
  otherwise route the feature-MLP through the canonical trainer on event scalars.

## Centrality derivation (new shared util)

`assign_centrality(estimator, edges, higher_is_central)` — percentile binning with
the same `DEFAULT_BIN_EDGES`; events beyond 80% → -1 (out of scope), mirroring the
now-fixed `truth.assign_bins`. Direction flag because the estimators differ:
- **Npart** (ML predicted / oracle true): `higher_is_central=True` (descending Npart).
- multiplicity (classical baselines): `higher_is_central=True`.
- b (if ever binned on b): `higher_is_central=False`.

Used with **predicted Npart** for the ML method and **true Npart** for the oracle;
classical baselines keep binning on their multiplicity observable. Wire into
`compute_cumulants.py` and `validate_models.py`. (Equivalent to the multiplicity
path in `truth.assign_bins`, generalized so b and Npart can share one function.)

## Architectures

Train all seven: MLP (feature baseline), MLP-pool, DeepSets, Set Transformer, EFN,
PFN, GNN. **Headline ablation (Fig 4): MLP vs DeepSets vs Set Transformer**;
EFN / PFN / GNN as extended comparison.

## Phases

- **Phase 1 — single-model, dual evidential heads.** Heads + standardization + loss +
  canonical trainer + φ-aug + centrality-from-b util; train all 7 on
  `urqmd_padded_det.h5` (80/10/10, seed 0); evaluate (resolution vs true b, coverage,
  derived-centrality confusion, cumulants).
- **Phase 2 — deep ensembles.** 5-seed wrapper per architecture; predictive mean +
  (NIG + ensemble) variance; UQ-robustness section vs the NIG-"mirage" critique.
- **Phase 3 — cross-energy generalization.** `--hold-out-energy` split (train 3,
  test 1), both directions (Task 7).

## Ablations to run (one architecture, e.g. DeepSets)

- φ-aug **on vs off** (does enforcing rotational invariance help generalization?).
- φ encoding **raw vs (cos φ, sin φ)**.
- (later) with vs without φ entirely, to quantify the flow contribution.

## Training defaults (starting point)

lr 3e-4, batch 256, patience 5; **bump epochs** from 18 (cheap — full UrQMD-det
build trains fast) and pick by the val curve. Device auto (mps→cuda→cpu).

## Open items / caveats

- **Npart provenance** (native UrQMD Glauber, max≈411 ⇒ possibly A=208/Pb) — confirm
  the nucleus before reporting Npart as a physical Au count; `b` is unaffected. If it
  proves Pb-based, fall back to b-only reported target. (See restructure doc.)
- N_coll remains deferred (needs cluster re-ingestion).
