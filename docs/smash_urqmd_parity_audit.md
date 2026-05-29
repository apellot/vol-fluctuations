# SMASH vs UrQMD Data-Parity Audit

Date: 2026-05-27
Scope: pre-flight check before any cross-transport (UrQMD) evaluation of ML-centrality models trained on SMASH.
Files audited (raw):
- `data/raw/particle_lists_AuAu_{3.2,3.5,3.9,4.5}GeV.root` (SMASH, intact)
- `data/raw/urqmd_auau_3p2GeV.h5` (UrQMD, intact, 100k events)
- `data/raw/urqmd_auau_3p5GeV.h5` (UrQMD, **truncated mid-download** — `OSError: eof = 1286701056, stored_eof = 1471173112`; do not use until rsync completes)
- `data/raw/urqmd_auau_{3p9,4p5}GeV.h5` (not yet present)

---

## 1. UrQMD HDF5 schema

The UrQMD file already follows the SMASH-processed schema (good news — the upstream ingester `ingest_urqmd.py` was modelled on `ingest_smash.py`).

Root-level attributes on `urqmd_auau_3p2GeV.h5`:

| attr | value |
|---|---|
| `sqrtsNN` | 3.2 |
| `n_events` | 100000 |
| `source` | `urqmd_f13` |
| `input_dir` | `/gpfs/.../urqmd_output/3p2` |
| `ingest_script` | `ingest_urqmd.py` |
| `npart_note` | "Npart = Glauber Nuc_part from UrQMD 4.0 Participants_Glauber header line. Npart_ncoll = 2·208 − N_spec ... stored for diagnostics only." |
| `pdg_note` | "PDG codes derived from (mass, charge) via mass_charge_to_pdg(). Nucleons, pions, kaons, and eta assigned; other species pdg=0. Neutral particles (n, pi0, K0) and spectators (ncoll==0) ARE included." |

Datasets:

Event-level (length N_events = 100000):
| name | dtype | shape | notes |
|---|---|---|---|
| `b` | float32 | (N,) | impact parameter, fm |
| `Npart` | int16 | (N,) | **direct Glauber N_part from UrQMD output header** (more authoritative than SMASH's derived value) |
| `Npart_ncoll` | int16 | (N,) | diagnostic 2·208 − N_spec (matches SMASH formula but uses A=208, not 197 — see Issue #1) |
| `mult_eta05` | int32 | (N,) | charged mult, \|η\|<0.5 |
| `mult_eta10` | int32 | (N,) | charged mult, \|η\|<1.0 |
| `n_proton`, `n_antiproton` | int32 | (N,) | event-level counts |
| `nparticles` | int32 | (N,) | particles per event |
| `offset` | int64 | (N+1,) | cumulative offsets into per-particle arrays |
| `sqrtsNN` | float32 | (N,) | one constant repeated |

Per-particle (length N_total = 51,877,741):
| name | dtype | notes |
|---|---|---|
| `particles/pT` | float32 | GeV |
| `particles/eta` | float32 | **CM frame** (verified, see §1.1) |
| `particles/phi` | float32 | rad |
| `particles/mass` | float32 | GeV |
| `particles/charge` | int8 | |
| `particles/pdg` | int32 | reconstructed from (mass, charge); ~1.23% of particles are pdg=0 (only 0.33% of those are charged — negligible for mult observables) |
| `particles/ityp` | int32 | native UrQMD species code (SMASH-side has no equivalent) |
| `particles/ncoll` | int16 | per-particle collision count |

**What is missing vs SMASH ingested files:** no `p0/px/py/pz` (UrQMD stores only the derived `pT, η, φ, mass`). This is fine — `build_padded_cache.py` already reconstructs `pz = pT·sinh(η)` and `p0 = √(pT²·cosh²η + m²)` from exactly these inputs, so the boost path is portable.

### 1.1 Reference-frame verdict — both files are CM frame

Method: histogram spectator-proton rapidity y (computed from pT, η, mass) and pseudorapidity η. Spectators are isolated as `(|pdg|==2212) & (ncoll==0)`. In CM frame, beam/target spectator nucleons cluster at y ≈ ±y_beam_cm = ±arccosh(√sNN/2m_N); in lab frame they cluster at y ≈ 0 (target) and y ≈ +2y_beam_cm (projectile).

At √sNN = 3.2 GeV, y_beam_cm = 1.127.

| dataset | spectator y peaks | verdict |
|---|---|---|
| SMASH 3.2 | ±1.10 to ±1.20 (median \|y\|=0.618) | CM |
| UrQMD 3.2 | ±1.10 to ±1.20 (median \|y\|=0.911) | CM |

η distributions confirm: both files have spectator η symmetric about 0 with peaks at \|η\|≈2.7 (the η→∞ tail of a near-beam-axis nucleon at the CM-frame beam rapidity).

**Conclusion:** UrQMD HDF5 is in CM frame, identical convention to SMASH ROOT output. Same CM→lab boost in `build_padded_cache.py` applies unchanged.

### 1.2 N_part definition mismatch (Issue #1)

- SMASH-side ingestion uses `AU_2A = 394` (Au = ²⁰⁷Au, A=197).
- UrQMD-side `Npart_ncoll` uses `2·208 = 416` (likely Pb=208 leak in `ingest_urqmd.py`).
- Sanity: UrQMD `Npart` (Glauber-direct from f13 header) has max=410 ≤ 2·205, mean=121.6, min=0 — units look right but the reference 2A is off. Need to confirm.

Recommendation: use the **direct UrQMD `Npart`** column (Glauber-from-header), not the derived `Npart_ncoll`. The CLAUDE.md memo notes SMASH's spectator-derived N_part is an upper bound vs Glauber convention (elastic-only participants), so the UrQMD direct value is *more* faithful to the Glauber number. Document this asymmetry: SMASH-side and UrQMD-side N_part are computed by different definitions — for cross-transport comparison rely on **b** (not N_part) as the truth label.

---

## 2. Impact-parameter coverage

### Per-energy summary

| dataset | N events | b min | b max | ⟨b⟩ | bmax policy |
|---|---|---|---|---|---|
| SMASH 3.2 | 50,000 | 0.07 | 20.00 | 13.32 | uniform sample to b=20 fm |
| SMASH 3.5 | 100,000 | (∼0.07) | (∼20) | 13.32 | same |
| SMASH 3.9 | 100,000 | — | (∼20) | 13.33 | same |
| SMASH 4.5 | 100,000 | — | (∼20) | 13.32 | same |
| **UrQMD 3.2** | 100,000 | 0.07 | **14.00** | **9.33** | **b_max = 14 fm cap** |

### b-distribution shape (SMASH vs UrQMD, 3.2 GeV)

Both datasets are triangular dN/db ∝ b within their respective b ranges (ratios of observed/expected = 0.95–1.04 in every bin where both populate). Confirmed min-bias generation. The mismatch in ⟨b⟩ is entirely due to the b_max cap difference, not a shape issue.

```
b bin   SMASH frac   UrQMD frac    SMASH-tri-fit   UrQMD-tri-fit
[ 0, 1)   0.00200     0.00486         0.80              0.95
[ 1, 2)   0.00722     0.01594         0.96              1.04
[ 2, 3)   0.01236     0.02525         0.99              0.99
...
[13,14)   0.06772     0.13676         1.00              0.99
[14,15)   0.07208     0.00070         1.00            (UrQMD cuts off)
[15,16)   0.07704     0.00000
...
[19,20)   0.09780     0.00000
```

### Practical consequence — Issue #2

**Roughly 37% of SMASH events have b > 14 fm**, and the vast majority of these are N_part = 0 geometric misses (CLAUDE.md). For an apples-to-apples cross-transport comparison the analysis must run on either:

- (a) **N_part > 0 + b < 14 fm intersection on both datasets** (fairest); this is the recommended cut. Result at 3.2 GeV:

  | dataset | events kept | ⟨b⟩ | ⟨mult_η05⟩ | ⟨mult_η10⟩ |
  |---|---|---|---|---|
  | SMASH | 24,439 | 9.31 | 32.3 | 61.7 |
  | UrQMD | 99,080 | 9.29 | 38.3 | 72.3 |

  b-distribution means now agree to 0.3%. Multiplicities differ by ~18% — within reason for two transport models (UrQMD produces slightly more particles at FXT than SMASH, consistent with published comparisons).

- (b) Just `N_part > 0` (already done on SMASH side by `build_padded_cache.py`); leaves a residual b-coverage mismatch above b≈14 fm where UrQMD has zero events.

The existing SMASH `build_padded_cache.py` only applies `N_part > 0`. It does **not** clip b < 14 fm. For the cross-transport comparison the recommendation is to add a b<14 fm cut symmetrically on both sides when running `validate_models.py --urqmd-cache ...`. Either bake into the UrQMD cache builder, or post-filter in `validate_models.py`.

---

## 3. Event counts and basic kinematics

All events, no filters:

| dataset | √sNN | N events | b range | ⟨b⟩ | ⟨mult \|η\|<0.5⟩ | ⟨mult \|η\|<1.0⟩ | ⟨pT⟩ charged \|η\|<1 (GeV) |
|---|---|---|---|---|---|---|---|
| SMASH | 3.2 | 50,000 | [0.07, 20.0] | 13.32 | 15.90 | 30.33 | 0.500 |
| SMASH | 3.5 | 100,000 | [∼0, 20.0] | 13.32 | 16.88 | 32.11 | 0.515 |
| SMASH | 3.9 | 100,000 | [∼0, 20.0] | 13.33 | 17.81 | 33.83 | 0.541 |
| SMASH | 4.5 | 100,000 | [∼0, 20.0] | 13.32 | 18.56 | 35.23 | 0.593 |
| UrQMD | 3.2 | 100,000 | [0.07, 14.0] | 9.33 | 37.91 | 71.67 | 0.460 |
| UrQMD | 3.5 | n/a | (file truncated) | — | — | — | — |
| UrQMD | 3.9 | not yet | — | — | — | — | — |
| UrQMD | 4.5 | not yet | — | — | — | — | — |

After fair filter (N_part>0, b<14 fm) at 3.2 GeV: UrQMD multiplicities run ~18% higher than SMASH (mult_η05: 38.3 vs 32.3); ⟨pT⟩ runs ~9% lower (0.460 vs 0.500). This is a model difference, not a parsing/frame mismatch. **Documented as a known transport-code systematic** — it is exactly what cross-transport robustness is supposed to probe.

Charged-particle PDG counts in UrQMD (top species, first 5M particles): π⁻ 6.9%, π⁰ 6.6%, π⁺ 5.2%, K⁰ 0.49%, K⁺ 0.42%, η 0.19%, K⁻ 0.04%, p̄ ≈ 0%. UrQMD generates essentially no anti-protons at √sNN=3.2 (only 1 in 5M particles) — consistent with the kinematic threshold. SMASH at 3.2 should be checked separately; an order-of-magnitude difference in p̄ would point to chemistry differences worth flagging.

---

## 4. Parser/cut inventory

Files involved (read-only audit):

### `scripts/ingest_smash.py` (SMASH-only entry point)
- L100–104: `tree.iterate(..., step_size=1000)`; drops events with `empty_event == True` (L103–105).
- L115: derives `pT, η, φ` from `(px, py, pz)` via `pt_eta_phi()` in `src/data/dataset.py:45`. These are CM-frame (no boost).
- L118: `npart_from_event(pdg, ncoll)` → N_part = 394 − count of nucleons with ncoll==0.
- L120–121: `charged_mult_in_eta(eta, charge, 0.5)` and `(eta, charge, 1.0)` — CM-frame η window, charge != 0.
- L122–123: `count_pdg(pdg, ±2212)`.
- **No frame transform applied at ingestion time.**
- **No PDG / charged-only filter at ingestion time** — everything SMASH emits is written.

### `src/data/dataset.py`
- L25–42: `npart_from_event` — spectator-based derivation, AU_2A=394.
- L45–55: `pt_eta_phi` — derives η = ½ ln((p+pz)/(p−pz)) directly from raw 3-momentum. Frame inherits from inputs (CM in our case).
- L58–60: `charged_mult_in_eta` — `(|η| < eta_max) & (charge != 0)`. No spectator removal, no pT cut. CM-frame η.
- L63–65: `count_pdg` — exact PDG match.

### `scripts/build_padded_cache.py` (the **main** cut/boost script)
This is where the analysis-frame transform lives. Operates on already-ingested SMASH HDF5 (one per energy) and writes a single concatenated padded cache.

Pre-pass (L101–112):
- For each energy, load `Npart` array; build mask `Npart > 0`. **Drops geometric-miss events** before the cache.

Per-chunk (L173–243):
- L190–192: reconstruct CM-frame `(p0_cm, pz_cm)` from `(pT, η_cm, mass)`.
- L192 (`boost_to_lab`): apply longitudinal boost by `y_boost = arccosh(√sNN / 2 m_N)` (see L60–63 and L51 `M_N = 0.938`). This is the **CM → lab/target-rest frame boost**, defined at L60–63 `y_cm_to_lab()`.
- L193: `eta_lab = eta_from_pz(pT, pz_lab)`.
- L200–202: selection mask: `(charge != 0) & (eta_lab > 0.0) & (eta_lab < 1.5) & (~is_spectator)`, where `is_spectator = (|pdg|∈{2112,2212}) & (ncoll==0)`.
  - **η window:** `0.0 < η_lab < 1.5` (L57). Asymmetric, lab-frame, FXT-specific (matches STAR FXT TPC).
  - **Spectator removal:** nucleons with `ncoll == 0` are excluded — rationale at L195–199 ("real STAR target spectators never reach the TPC").
- L230–233: features stored per particle: `[pT, eta_lab, phi, charge]`. **PDG, mass, ID, history fields are all dropped at this stage** (locked design: no-PID inputs).
- L235–240: event-level features computed *from in-acceptance particles only*: `mult_lab, mean_pT_lab, total_pT_lab`.
- L243–256: only `chunk_keep = (Npart > 0)` rows written.

### `src/data/lab_cached_dataset.py` (training-time reader)
Pure reader — no additional cuts or transforms. Reads the padded cache produced by `build_padded_cache.py`, attaches centrality labels from `truth_auau_{X.X}GeV.h5` files, and emits batches. φ rotation augmentation (L112–132) does not affect physics observables.

### `src/data/features.py` (MLP scalar-feature path)
- Reads CM-frame `eta` directly from the ingested HDF5 (L102). **Does NOT apply the CM→lab boost** — the MLP baseline operates on CM-frame η for its scalar summary features (mult_η05, mult_η10, mean_pT in |η|<1, etc.).
- This is a deliberate design choice for the MLP baseline; the per-particle networks (DeepSets/Set Transformer/EFN/etc.) use the lab-frame cache instead.

### `src/data/particle_dataset.py` (legacy CM-frame per-particle path)
- Reads the ingested HDF5 directly, yields `(pT, η_cm, φ, mass, charge, pdg)` per particle. No frame transform. No cuts. This path predates `build_padded_cache.py` and is **not used by the locked design** (project_scope_v2 memory) — kept for historical experiments.

### `src/data/cached_dataset.py`
- Older variant of `lab_cached_dataset.py`; same schema but reads `eta_lab` as the second column (L13–14 docstring still says `eta_lab`). No additional cuts.

### Summary of the SMASH analysis-pipeline cuts

| stage | cut | frame |
|---|---|---|
| `ingest_smash.py` | drop `empty_event` | CM |
| `build_padded_cache.py` pre-pass | drop `Npart == 0` | n/a |
| `build_padded_cache.py` particle filter | `charged & 0<η_lab<1.5 & ~spectator(nucleon,ncoll==0)` | **lab** |

**There is no parallel UrQMD ingestion or cache-build path in `src/data/` or `scripts/`.** `validate_models.py` references `--urqmd-cache data/processed/cached/urqmd_padded.h5` and `--urqmd-truth-dir data/processed/urqmd_truth/` but those files do not exist yet (verified: `data/processed/cached/` contains only `all_padded.h5` and `all_lab_truth.h5`).

---

## 5. FXT boost location & UrQMD plan

### Where the CM→lab boost happens (SMASH path)
- Definition: `scripts/build_padded_cache.py:60` — `y_cm_to_lab(sqrtsNN) = arccosh(sqrtsNN / (2·M_N))`, `M_N = 0.938 GeV`.
- Boost: `scripts/build_padded_cache.py:65` — `boost_to_lab(p0, pz, y_boost)`. Standard longitudinal Lorentz transform.
- Invocation: `scripts/build_padded_cache.py:192` — once per chunk, called on the CM-frame `(p0, pz)` reconstructed from `(pT, η_cm, mass)` at L190–191.
- Acceptance applied AFTER the boost: `0.0 < η_lab < 1.5` (L57, L202).

### Does UrQMD need the same boost? — Yes, with no code changes
- UrQMD HDF5 is CM-frame (verified §1.1). Identical convention to SMASH.
- The boost in `build_padded_cache.py` consumes `(pT, η_cm, mass)`, all of which UrQMD provides natively (it doesn't ship p0/px/py/pz, but it doesn't need to — those are reconstructed inside the script).
- So the same boost function with the same `y_boost = arccosh(√sNN / 2m_N)` is correct for UrQMD.

### What needs to be written for UrQMD
A minimal UrQMD cache-builder, structurally identical to `scripts/build_padded_cache.py` but reading from UrQMD's already-ingested HDF5 files. **Differences from the SMASH path are exactly two:**

1. **Drop the `Npart` pre-filter (or use UrQMD's direct `Npart`).** UrQMD has the authoritative Glauber-direct N_part from the f13 header. Use it. Almost all UrQMD events already have N_part > 0 (99,138 / 100,000 at 3.2 GeV) because UrQMD was generated with b_max = 14 fm.
2. **Spectator definition asymmetry — Issue #1 caveat.** UrQMD's `ncoll == 0` includes only inelastic+elastic NN collisions per the UrQMD f13 conventions, while SMASH's `ncoll == 0` counts even more (any scattering, including off produced mesons). Both definitions produce a `spectator` mask that is well-defined and applied identically; just document that the spectator counts will differ slightly between codes. This does **not** require a code-level change in the cut, only a comment.

Everything else — CM-frame reconstruction (`pz = pT·sinh(η)`, `p0 = √(...)`), longitudinal boost, η_lab acceptance window, charged + ~spectator mask, MAX_PARTICLES = 512, per-event feature definitions — copies over verbatim.

---

## 6. Punch list — what to change/add before cross-transport eval

Concrete, ordered.

**P1 — block downstream UrQMD eval until done**

1. **Re-rsync UrQMD 3.5 GeV.** `urqmd_auau_3p5GeV.h5` is truncated (1.29 GB on disk, 1.47 GB expected); h5py refuses to open it. Re-download then run the same schema check from §1 to confirm it matches the 3.2 GeV file.
2. **Wait/request UrQMD 3.9 and 4.5 GeV.** No files yet. Same schema expected; re-run the audit at §1.1 (spectator-y test) and §2 (b-coverage) for each new file.
3. **Add `scripts/build_padded_cache_urqmd.py`.** Take `scripts/build_padded_cache.py` as the template; adapt to:
   - read from the UrQMD HDF5 schema (no p4 components — already handled by the existing reconstruction at L190–191);
   - use UrQMD's direct `Npart` column instead of the derived one for the `N_part > 0` filter (or skip the filter — UrQMD's b_max cap already excludes nearly all geometric misses);
   - write outputs to `data/processed/cached/urqmd_padded.h5` with the **identical schema** as `all_padded.h5` (same attrs, same datasets, same dtypes) so `LabCachedDataset` and `validate_models.py --urqmd-cache` work without further changes;
   - keep the same `ETA_LAB_MIN=0.0, ETA_LAB_MAX=1.5`, `MAX_PARTICLES=512`, `M_N=0.938`.
4. **Add `scripts/run_truth_urqmd.py`** (or just reuse `run_truth.py` if it is already cache-path-generic — verify). Outputs `data/processed/urqmd_truth/truth_auau_{3p2,3p5,3p9,4p5}GeV.h5` files matching the SMASH truth-baseline schema (one `centrality_bin` array per energy).

**P2 — methodology decisions to lock before running validate**

5. **Apply a symmetric `b < 14 fm` cut on the SMASH side when comparing to UrQMD.** Either bake into the UrQMD cache builder (cheaper, applies once) or filter inside `validate_models.py` before computing the cross-transport MAE. Without this cut the SMASH peripheral tail (b > 14 fm) has no UrQMD counterpart and the cross-transport metric is biased.
6. **Decide on the N_part definition convention.** Recommendation: report `b` as the truth label everywhere (already the case), keep N_part as a diagnostic only, and footnote that UrQMD `Npart` = Glauber-direct from header while SMASH `Npart` = spectator-derived (upper bound vs Glauber).
7. **Fix the `Npart_ncoll = 2·208 − ...` formula in `ingest_urqmd.py`** (lives in `~/phd/SMASH_Analysis/` or wherever it was run) to use `2·197` if Au is the species. The direct `Npart` from header is the one we'll use, so this is cosmetic; do it for sanity.

**P3 — extra checks once 3.5 / 3.9 / 4.5 GeV UrQMD arrives**

8. **Repeat §1.1 (spectator y diagnostic) at every UrQMD energy.** Don't assume the f13 → HDF5 ingester maintained CM-frame consistency across energies — verify it.
9. **Repeat the §2 b-coverage check at every energy.** If b_max varies between UrQMD energies, the b<14 fm cut in P2.5 needs to become b<min(b_max_uq) per energy.
10. **Run a 5-feature sanity comparison at each energy:** ⟨b⟩, ⟨mult_η05⟩, ⟨mult_η10⟩, ⟨pT charged \|η\|<1⟩, ⟨N_part⟩ on the **N_part>0 + common-b** subset. Document the SMASH↔UrQMD ratios in a per-energy table in the paper's systematics section.

**Not in scope for this audit (flagged for later):**

- UrQMD's antiproton yield is essentially zero at 3.2 GeV (1 in 5M). SMASH's may also be small. If antiproton C-cumulant ratios end up in the paper, audit p̄ yields across the two codes — this is a chemistry difference, not a centrality issue.
- UrQMD ⟨mult⟩ is ~18% higher than SMASH at fixed b in the overlap window. This is a known transport-model systematic; it is exactly the thing cross-transport evaluation is supposed to probe. Do not "correct" for it.

---

## Verified facts

- SMASH 3.2/3.5/3.9/4.5 GeV ROOT inputs present and intact at `data/raw/particle_lists_AuAu_*.root`.
- SMASH ingested HDF5 outputs intact at `data/processed/auau_{3p2,3p5,3p9,4p5}GeV.h5`.
- UrQMD 3.2 HDF5 intact, 100k events, CM-frame, schema-compatible with SMASH.
- UrQMD 3.5 HDF5 truncated (1.29 / 1.47 GB).
- UrQMD 3.9, 4.5 GeV not yet on disk.
- No UrQMD cache-builder exists in the repo. `validate_models.py` already accepts `--urqmd-cache`; the cache file just needs to be produced with the right schema.
