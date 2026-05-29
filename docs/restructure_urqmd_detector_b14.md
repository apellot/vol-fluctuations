# Restructure 2026-05-28 — UrQMD + detector + b<14 as the headline study

This note records a scope reversal. It **supersedes** the earlier defaults
(SMASH-primary, truth-level default, b<11) documented in `CLAUDE.md`, the
`project_scope_v2` memory, and `docs/npart_reconciliation.md`.

## Decisions (locked with the user 2026-05-28)

1. **Primary generator: UrQMD.** SMASH is demoted to a secondary cross-transport
   comparison and is not re-run for now. UrQMD has more events (100k/energy) and
   is the headline transport.
2. **Primary caches: detector-emulated** (`*_padded_det.h5`). The detector
   emulation (η window 0<η_lab<1.5, pT smearing, pT_min, logistic efficiency;
   see `docs/detector_cache_check.md`) is the main study, not the truth-level
   cache.
3. **Full impact-parameter range: b < 14 fm.** `B_MAX_FM` in `src/data/cuts.py`
   raised 11 → 14 for both generators. The temporary b<11 tightening existed
   only to dodge the negative-Npart tail produced by the spectator-rule recompute
   on UrQMD; that recompute is no longer the UrQMD label source (see #4), so the
   full range is kept. SMASH's `ncoll` inherits correctly, so b<14 is safe there.
4. **UrQMD Npart: native Glauber-header value.** The UrQMD cache builder now
   reads the raw file's stored `Npart` (UrQMD 4.0 `Participants_Glauber`, always
   ≥0) instead of recomputing via `cuts.npart_from_spectators`. The spectator-rule
   recompute only mattered to make SMASH and UrQMD Npart comparable for joint
   training, which is deprioritized. `npart_from_spectators` is retained for the
   SMASH ingest and as a diagnostic.
5. **N_coll: still deferred.** UrQMD's Glauber emits a binary-collision count, but
   it was not extracted during ingestion, and the raw `.f13` + `ingest_urqmd.py`
   live on a remote STAR cluster (not local). Adding N_coll as a target would
   require a cluster re-ingestion pass. Out of scope for now — consistent with the
   original CLAUDE.md N_coll deferral. Targets remain `b` + `Npart`.

## Open caveat to verify before reporting Npart in the paper

The native UrQMD `Npart` has `max ≈ 411` at 3.5 GeV, and the ingest note computes
its diagnostic with `2*208` (Pb), whereas Au is A=197 (`2A = 394`). A participant
count above 394 is impossible for Au+Au. Yet the SMASH-rule recompute with A=197
matched SMASH within ~3 units (`npart_reconciliation.md`), which points back at Au.
**Before reporting native Npart as a physical Au participant count, confirm the
projectile/target nuclei and the A used by UrQMD's Glauber header** (check the
UrQMD input decks / data author). `b` — the headline target — is unaffected
regardless. If native Npart turns out Pb-based on Au events, fall back to a
`b`-only reported target.

## Physics note (no action) — the CM→lab boost is not nucleon-specific

The boost rapidity `y_cm = arccosh(√sNN / 2 m_N)` uses `m_N` only to set the
**frame** velocity of the NN center of mass (a beam property). The resulting
longitudinal Lorentz boost applies correctly to every species; per-particle mass
already enters in the energy reconstruction `p0 = √(pT²·cosh²η + m²)` and the code
recomputes η_lab from the boosted 4-vector rather than shifting pseudorapidity
directly (which would be wrong for slow, heavy nucleons). No bug.

## Implementation touchpoints

- `src/data/cuts.py`: `B_MAX_FM = 14`; docstrings updated.
- `scripts/build_padded_cache_urqmd.py`: native `h["Npart"]`; `npart_source` attr.
- `scripts/build_padded_cache.py`: docstring notes SMASH is secondary; b from `cuts.B_MAX_FM`.
- Default flips to UrQMD + detector in `run_truth.py`, `fit_glauber.py`,
  `run_glauber.py` (`--transport urqmd` default, `*_det.h5` caches),
  `train_cached.py` (`--truth-dir` default `…/urqmd`), and the usage examples in
  `validate_models.py` / `compute_cumulants.py`.
- `scripts/compare_glauber_vs_models.py`: marked stale (old full/lab-split SMASH
  study); use `validate_models.py` on `urqmd_padded_det.h5` for the new comparison.

## Rebuild sequence (data — run after code changes)

1. Rebuild UrQMD detector cache at b<14 + native Npart:
   `python scripts/build_padded_cache_urqmd.py --inputs data/raw/urqmd_auau_{3p2,3p5,3p9,4p5}GeV.h5 --output data/processed/cached/urqmd_padded_det.h5 --detector`
2. `python scripts/run_truth.py` (defaults to UrQMD now) → centrality-bin labels.
3. `python scripts/fit_glauber.py` then `run_glauber.py` for all 4 energies
   (fits to detector multiplicity; σ_NN unchanged).
4. Retrain all architectures on the new cache.
5. Re-run `validate_models.py` / `compute_cumulants.py`; regenerate figures.
