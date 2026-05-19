# Project handoff — ML centrality at FXT energies

Mathias Labonté · UC Davis / STAR · `mathias.labonte@gmail.com`
Last updated: 2026-05-18

This is a methodology paper for *Phys. Rev. C*. The deep design doc lives in [`CLAUDE.md`](CLAUDE.md) — read that for the full motivation, prior-art comparison, and v1 scope. This file is the short "come up to speed" version for collaborators.

---

## 1. What the project is

We're building a machine-learning method to determine **collision centrality** (impact parameter `b`, centrality percentile, and a calibrated per-event uncertainty) from final-state particles in **low-multiplicity heavy-ion collisions** at the STAR fixed-target (FXT) energies √sNN = 3.2, 3.5, 3.9, 4.5 GeV.

The volume-fluctuation contamination from poor classical centrality resolution is the dominant systematic on STAR BES-II higher-order net-proton cumulants — i.e. on the QCD critical-point search. If the ML method beats RefMult / Glauber in this regime, it directly improves the BES-II signal-to-noise and benefits every centrality-dependent observable downstream.

Closest prior art: Mallick et al. (arXiv:2103.01736), BDTs at √sNN ≥ 200 GeV. We differentiate on FXT regime + permutation-invariant architectures + multi-energy joint training + calibrated uncertainty + downstream cumulant impact. See [`CLAUDE.md`](CLAUDE.md) §"Contribution" and the [scoop-risk memory](.claude/projects/-Users-mlabonte-phd-papers-volFluc/memory/project_scoop_risk.md) for the honest differentiator scorecard.

**Truth-level study only in v1.** Minimal detector emulation (charged-only, pT > 50 MeV/c, ~1% pT smearing, ~90% tracking efficiency) is included; full GEANT/detector reconstruction and real STAR data are deferred.

---

## 2. Status snapshot (2026-05-18)

| Task | Status | Notes |
|---|---|---|
| 1. Lock decisions | done | Architectures + UQ choice locked; see [`CLAUDE.md`](CLAUDE.md) §"Implementation roadmap" |
| 2. SMASH ingestion | done | `scripts/ingest_smash.py` → `data/processed/auau_*GeV.h5` |
| 3. Classical baselines | done | Truth-tuned + Glauber-NBD; both trained and evaluated per energy |
| 4. MLP baseline | done | `scripts/train_mlp.py`, checkpoint in `checkpoints/mlp_baseline_v1/` |
| 5. DeepSets | done | Checkpoint in `checkpoints/deepsets_baseline_v1/` |
| 5. Set Transformer | **broken** | See §4 — training crashes the laptop, code has mismatched interfaces |
| 5. EFN | not started | |
| 6. Calibration diagnostics | partial | Some sanity plots in `figures/`; no coverage plot yet |
| 7. Cross-energy generalization | not started | |
| 8. Downstream cumulant impact | not started | The headline figure |
| 9. PRC figures + writeup | not started | |

Existing artifacts:
- Per-event prediction HDF5s for MLP / MLP-pool / DeepSets / truth / Glauber under `data/processed/<method>/`
- Sanity + baseline figures under `figures/`
- Trained checkpoints under `checkpoints/` (`set_transformer_v1_truth/` is empty — never completed a run)

---

## 3. What's interesting so far

A non-trivial fraction of the SMASH events have zero charged particles in the analysis window — but the dominant cause is **simulation geometry, not physics**: the production runs sampled `b` up to ~20 fm, well past the geometric overlap cutoff (~16 fm for Au+Au), so roughly **37% of events are pure misses with N_part = 0**. These are filtered out at cache-build time (`scripts/build_padded_cache.py` drops `N_part == 0`).

After that filter, in the current lab-frame analysis window (charged + spectator-removed, η_lab ∈ [0, 1.5]), the spectator-dominated peripherals still contribute a low-multiplicity tail that collapses adjacent percentile bins in both classical baselines — that's the volume-fluctuation problem the ML method has to beat. We need a clean updated plot of the post-filter multiplicity distribution and the peripheral-resolution loss before quoting numbers in the paper. (The CLAUDE.md §"Task 3" 47%-at-|η_CM|<0.5 figure is from the older CM-frame, no-filter analysis and is no longer current.)

---

## 4. Current blocker — Set Transformer training

When training the Set Transformer locally on M-series Mac, the process either errors or eats all unified memory and forces a reboot. Three compounding issues identified today:

1. **`data/processed/cached/all_padded.h5` is stale.** Built with the old schema (`max_particles=896`, 5 features incl. mass + CM-frame η, spectators included). The current `scripts/build_padded_cache.py` produces a different schema: `MAX_PARTICLES=512`, 4 features (pT, η_lab, φ, charge), charged + spectator-removed + η_lab ∈ [0, 1.5]. The dataset class is reading old-schema data the new models can't consume.

2. **`scripts/train_cached.py` is out of sync with the model interface.** Script calls `model(cont, pdg_idx, mask, sqrtsNN)` (4 args). Models in `src/models/*.py` define `forward(cont, mask, event_feats)` (3 args, where `event_feats` is a 4-vector `(sqrtsNN, mult_lab, mean_pT_lab, total_pT_lab)`).

3. **Attention memory at L=896 is brutal on 16 GB.** `batch=256 × heads=4 × 896 × 896 × 4 B ≈ 3.3 GB` per attention tensor. Forward+backward ≈ 6–10 GB. Once swap kicks in the OS hangs.

**Fix order:**
1. Rebuild the cache with current `build_padded_cache.py` (drops max length; charged-only filter shrinks peripherals to ~0–10 particles).
2. Update `train_cached.py` to use the 3-arg model signature and build `event_feats` from the cache columns.
3. (Optional but big win) Add a length-bucketed batch sampler so most batches pad to tens, not 512.
4. Migrate training to UCD cluster regardless — same fixes apply, just more RAM headroom.

---

python scripts/ingest_smash.py --input /Users/andreanpellot/Downloads/volFluc/root_data/particle_lists_AuAu_3p2GeV.root --output data/processed/auau_3p2GeV.h5 --energy 3.2

python scripts/ingest_smash.py --input /Users/andreanpellot/Downloads/volFluc/root_data/particle_lists_AuAu_3p5GeV.root --output data/processed/auau_3p5GeV.h5 --energy 3.5

python scripts/ingest_smash.py --input /Users/andreanpellot/Downloads/volFluc/root_data/particle_lists_AuAu_3p9GeV.root --output data/processed/auau_3p9GeV.h5 --energy 3.9

python scripts/ingest_smash.py --input /Users/andreanpellot/Downloads/volFluc/root_data/particle_lists_AuAu_4p5GeV.root --output data/processed/auau_4p5GeV.h5 --energy 4.5

---

## 6. Repository map

```
volFluc/
├── CLAUDE.md                       # full design doc (read this)
├── HANDOFF.md                      # this file
├── requirements.txt
├── scripts/
│   ├── ingest_smash.py             # ROOT → per-energy HDF5
│   ├── build_padded_cache.py       # per-energy HDF5s → single padded cache for training
│   ├── sanity_plots.py
│   ├── run_truth.py                # truth-tuned percentile baseline
│   ├── fit_glauber.py              # fit Glauber-NBD (n_pp, k, x) per energy
│   ├── run_glauber.py              # apply fitted Glauber → per-event b prediction
│   ├── train_mlp.py                # MLP on event-level scalars
│   ├── train_lab.py                # variable-length pipeline (alternative to cached)
│   ├── train_cached.py             # cached pipeline (DeepSets / Set Transformer) — broken, see §4
│   ├── plot_baseline_resolution.py
│   └── compare_baselines.py
├── src/
│   ├── data/
│   │   ├── particle_dataset.py     # variable-length Dataset over per-energy HDF5s
│   │   └── cached_dataset.py       # fixed-shape Dataset over the padded cache
│   ├── models/
│   │   ├── mlp.py                  # event-level scalars only
│   │   ├── mlp_pool.py             # mean-pool + dense
│   │   ├── deepsets.py
│   │   ├── set_transformer.py
│   │   ├── efn.py                  # not yet trained
│   │   └── heads.py                # shared NIG + classifier heads, masked_mean
│   ├── losses/evidential.py        # Amini NIG loss
│   └── baselines/                  # truth.py, glauber.py, refmult.py
├── tests/                          # pytest — kinematics + evidential loss
├── configs/                        # (empty placeholder)
├── notebooks/                      # (empty placeholder)
├── figures/                        # sanity + baseline plots (PNG + PDF)
└── checkpoints/                    # trained model weights (small — included in archive)
```

---

## 7. Where help would be most useful

1. **Unblock the Set Transformer training** — issues in §4. Highest priority.
2. **Cross-energy generalization scaffolding** (Task 7) — train on 3 energies, evaluate on the held-out 4th, both directions. Needs a clean train/eval-split utility that can leave one energy fully out.
3. **Downstream net-proton cumulant impact** (Task 8) — bin events by each centrality method (truth, Glauber, ML) and compute C₁–C₄ and ratios per centrality bin; compare to truth-b oracle. This is the physics-utility headline figure.
4. **EFN architecture** (Komiske–Metodiev–Thaler) implementation in `src/models/efn.py`. Scaffold exists, training script does not.
5. **Calibration coverage plots** — does the predicted 68% NIG interval cover truth ~68% of the time? Implementation in `src/losses/evidential.py`; plotting code does not exist yet.

If you pick one up, ping Mathias first — happy to brief on context, conventions, and the bits of physics that aren't obvious from the code.

---

## 8. References

Pinned in [`CLAUDE.md`](CLAUDE.md) §"Key references". Most important:
- Mallick et al. arXiv:2103.01736 — closest prior art (BDTs at high √sNN).
- Amini et al., NeurIPS 2020 — evidential regression loss.
- Lee et al., ICML 2019 — Set Transformer (SAB / PMA).
- SMASH: Weil et al., PRC 94, 054905 (2016).
