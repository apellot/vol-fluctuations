# Detector-emulated cache verification (Phase B)

Status: built and verified 2026-05-27.

Two new caches built alongside the truth-level ones:

| cache | path | size | events |
|---|---|---:|---:|
| SMASH-truth | `data/processed/cached/smash_padded_v2.h5` | 1.7 GB | 106,249 |
| SMASH-det   | `data/processed/cached/smash_padded_v2_det.h5` | 631 MB | 106,249 |
| UrQMD-truth | `data/processed/cached/urqmd_padded.h5` | 3.9 GB | 246,872 |
| UrQMD-det   | `data/processed/cached/urqmd_padded_det.h5` | 1.5 GB | 246,872 |

Same events in both caches per generator (the event filter `b<11 AND N_part>0` is unchanged); only the per-particle stage and event scalars differ. Truth caches now also carry `attrs["detector_emulation"] = False` so consumers can branch unambiguously.

## Locked detector parameters (in HDF5 attrs of `_det.h5`)

| effect | value |
|---|---|
| η acceptance | `0 < η_lab < 1.5` |
| pT min | `pT_smeared > 0.050 GeV/c` |
| pT smearing | Gaussian, `σ(pT)/pT = 0.01 + 0.005·pT`, clip > 1e-6 |
| efficiency | logistic, `ε(pT) = 0.90 / (1 + exp(-(pT-0.075)/0.015))`, Bernoulli on smeared pT |
| stored pT | SMEARED value |
| seed | 42 (default; CLI `--seed` to override) |

## (a) Multiplicity drop

| √sNN | ⟨m⟩ SMASH truth | ⟨m⟩ SMASH det | ratio | ⟨m⟩ UrQMD truth | ⟨m⟩ UrQMD det | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 3.2 | 192.0 | 66.7 | 0.347 | 226.6 | 73.9 | 0.326 |
| 3.5 | 210.6 | 69.1 | 0.328 | 250.9 | 77.9 | 0.310 |
| 3.9 | 231.5 | 71.0 | 0.307 | 281.2 | 81.5 | 0.290 |
| 4.5 | 253.9 | 72.0 | 0.284 | 325.7 | 85.6 | 0.263 |

Ratio ~0.26–0.35 (vs the plan's 0.45–0.55 estimate). The one-sided `η > 0` cut is the largest contributor — at FXT energies the lab-frame distribution is broad in (-y_cm, +2y_cm) and we keep only the +η half within 1.5. The ratio shrinks with √sNN as the soft-pT tail and the η>1.5 forward jet both grow.

## (b) ⟨pT⟩ shift

| √sNN | ⟨pT⟩ SMASH truth | ⟨pT⟩ SMASH det | Δ (GeV/c) | ⟨pT⟩ UrQMD truth | ⟨pT⟩ UrQMD det | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 3.2 | 0.392 | 0.464 | +0.072 | 0.350 | 0.432 | +0.082 |
| 3.5 | 0.401 | 0.470 | +0.068 | 0.357 | 0.433 | +0.076 |
| 3.9 | 0.416 | 0.480 | +0.064 | 0.363 | 0.432 | +0.070 |
| 4.5 | 0.444 | 0.501 | +0.057 | 0.368 | 0.431 | +0.063 |

Shifts upward by 60–80 MeV/c. This is the pT_min cut removing the soft tail (Boltzmann-like spectrum with truth ⟨pT⟩~350–450 MeV); smearing itself is symmetric and doesn't move the mean.

## (c) Efficiency curve recovery — primary correctness test

Restrict truth cache to `0 < η_lab < 1.5`, charged; histogram pT in narrow bins; compare to detector-cache pT histogram. The ratio should follow the analytic logistic `ε(pT)`. SMASH 3.5 GeV:

| pT bin (GeV/c) | h_truth | h_det | ratio | ε(pT) | diff |
|---|---:|---:|---:|---:|---:|
| [0.05, 0.07] | 20224 | 5297 | 0.262 | 0.242 | +0.020 |
| [0.07, 0.09] | 33456 | 17713 | 0.529 | 0.524 | +0.005 |
| [0.09, 0.10] | 22122 | 15771 | 0.713 | 0.712 | +0.001 |
| [0.10, 0.12] | 53758 | 43955 | 0.818 | 0.820 | -0.003 |
| [0.12, 0.15] | 102691 | 90743 | 0.884 | 0.884 | -0.000 |
| [0.15, 0.20] | 202654 | 181926 | 0.898 | 0.899 | -0.001 |
| [0.20, 0.30] | 410087 | 369248 | 0.900 | 0.900 | +0.000 |
| [0.30, 0.50] | 631688 | 568529 | 0.900 | 0.900 | +0.000 |
| [0.50, 0.80] | 527272 | 474333 | 0.900 | 0.900 | -0.000 |
| [0.80, 1.20] | 278664 | 250862 | 0.900 | 0.900 | +0.000 |
| [1.20, 2.00] | 88253 | 79574 | 0.902 | 0.900 | +0.002 |

Agreement within ±2% across the entire pT range. The +2% bias at the lowest bin (50–70 MeV) is from pT smearing folding events upward from below the threshold. **PASS.**

## (d) Cross-generator parity on detector caches

| √sNN | n SMASH-det | n UrQMD-det | ⟨Npart⟩ S | ⟨Npart⟩ U | ΔNp | ⟨m⟩ S | ⟨m⟩ U | Δm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.2 | 15180 | 61850 | 261.8 | 267.4 | +5.6 | 66.7 | 73.9 | +7.2 |
| 3.5 | 30380 | 61886 | 267.9 | 268.9 | +1.0 | 69.1 | 77.9 | +8.7 |
| 3.9 | 30375 | 61639 | 275.2 | 269.5 | −5.7 | 71.0 | 81.5 | +10.5 |
| 4.5 | 30314 | 61497 | 282.4 | 270.1 | −12.3 | 72.0 | 85.6 | +13.6 |

ΔNpart and Δmult similar to truth-mode parity (the same transport-model systematic; detector ops don't change Npart since spectator removal uses truth pdg/ncoll). Cross-generator MAE will be driven by the mult offset, as expected.

## (e) RNG determinism

Built `det_test_A.h5`, `det_test_B.h5` with `--seed 42` and `det_test_C.h5` with `--seed 43` on the 3.2 GeV SMASH file:

- `cont` array byte-equal between A and B: **True**
- `mult_lab` equal: **True**, `mean_pT_lab` equal: **True**
- `cont` differs between A and C: **True**
- `mult_lab` differs: **True**

Cache is bit-reproducible given `(inputs, --chunk-events, --seed)`. **PASS.**

## (f) `MAX_PARTICLES` headroom

| cache | max observed | cap | headroom |
|---|---:|---:|---:|
| SMASH-det | 178 | 384 | 46% |
| UrQMD-det | 217 | 384 | 44% |

Comfortably under the 384 cap. No truncation. The cap can be lowered further (e.g. to 256) for a smaller cache, but 384 leaves room for higher-multiplicity events at higher √sNN if we later add 7.7 GeV. **PASS.**

## Determinism contract

The detector cache is bit-identical when these all match: input files, file order, `--chunk-events`, `--seed`. All are persisted in HDF5 attrs (`source_files`, `chunk_events`, `detector_seed`). Changing any of them invalidates the cache.

## Files touched

- `src/data/cuts.py` — added detector constants, `smear_pt`, `efficiency_at_pt`, `detector_keep_mask`; rewrote the EXCLUDED docstring block as DETECTOR.
- `scripts/build_padded_cache.py` — `--detector`/`--seed` CLI, RNG init, chunk-loop branch, attrs, `MAX_PARTICLES_DETECTOR=384`.
- `scripts/build_padded_cache_urqmd.py` — same.
- `src/data/features.py` — `detector`/`seed` kwargs on `load_features`; `detector` field on `EventDataset`; safety check in `stack_energies` against mixing modes.
- Existing truth caches patched in-place with `detector_emulation=False` attr.

## Out of scope (follow-ups)

- Wiring `--detector` into training/eval scripts (`train_*.py`, `run_truth.py`, `run_glauber.py`, `validate_models.py`). Routine arg passthroughs.
- Refactoring the two near-mirror builders into a shared `src/data/cache_builder.py` module.
- Update CLAUDE.md / `project_scope_v2.md` to reflect Phase B has landed.
