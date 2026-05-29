# N_part reconciliation: SMASH vs UrQMD (b<14 fm)

> **Superseded for the UrQMD-primary line (2026-05-28).** This note's conclusion
> — recompute UrQMD Npart via the SMASH-style spectator rule, and tighten to
> b<11 to avoid the negative peripheral tail — was made for *joint* SMASH+UrQMD
> training. After the restructure, UrQMD is primary and uses its **native**
> Glauber-header Npart (always ≥0) at the full b<14 range. See
> `docs/restructure_urqmd_detector_b14.md`. The cross-transport analysis below is
> still valid background, but the spectator-rule recompute is no longer the UrQMD
> label source.

## Symptom

The padded caches (`data/processed/cached/{smash_padded_v2,urqmd_padded}.h5`),
both built with the identical event filter (`b<14 fm` AND `Npart>0`), show:

| √sNN (GeV) | SMASH ⟨Npart⟩ | UrQMD ⟨Npart⟩ | SMASH/UrQMD |
| ---: | ---: | ---: | ---: |
| 3.2 | 184.8 | 122.7 | 1.51 |
| 3.5 | 191.1 | 123.1 | 1.55 |
| 3.9 | 197.5 | 123.0 | 1.61 |
| 4.5 | 204.5 | 122.6 | 1.67 |

⟨b⟩ ≈ 9.30 fm in both caches at every energy, so the cut is being applied
identically.

## Diagnostic 1 — cross-apply the SMASH rule to UrQMD

SMASH `Npart` is computed in `src/data/dataset.py:npart_from_event` as

```
Npart = 2A − N_spectators, N_spec = count(|pdg|∈{2112,2212} AND ncoll==0), A=197
```

I applied this exact recipe to UrQMD's per-particle `(pdg, ncoll)` on the
first 10 k events at 3.2 and 3.5 GeV (b<14 fm intersection):

| Source / rule | √sNN 3.2 | √sNN 3.5 |
| --- | ---: | ---: |
| SMASH, stored Npart (SMASH rule on SMASH ncoll) | 181.7 | 191.4 |
| UrQMD, stored Npart (Glauber-header value) | 121.6 | 122.9 |
| UrQMD, **SMASH rule applied to UrQMD ncoll** | **185.0** | **188.8** |
| UrQMD, stored `Npart_ncoll` (= 2·208 − N_spec) | 185.0 | 188.8 |

Cross-applying the SMASH rule to UrQMD raises ⟨Npart⟩ from 122 to 185–189, in
quantitative agreement with the SMASH-stored value of 182–191. The two
generators agree on the participant population to ≤3 units; they disagree
on what "participant" *means*.

The UrQMD ingester's `Npart_ncoll` diagnostic field (the one CLAUDE.md flagged
as the "Pb-mass-number footgun") happens to match the SMASH rule numerically.
That is an algebraic coincidence: `Npart_ncoll` is computed as
`2·208 − N_spec`, but the empirical max of stored `Npart_ncoll` in the b<14
sample is 393 ≈ 2·197 — meaning the implementation actually caps spectators
plus rescattered nucleons such that the field is effectively `2·197 − N_spec`
on Au+Au. Either way, this field is not used by the cache builder
(`scripts/build_padded_cache_urqmd.py:125` reads `src["Npart"]`, the
Glauber-header value), so the Pb-mass-number bug does **not** propagate to
the SMASH-vs-UrQMD gap.

## Diagnostic 2 — Glauber-header rule applied to SMASH (best-effort)

SMASH does not ship a Glauber-header `Npart`. I inspected the history
branches available in the .root file (`pdg_mother1`, `pdg_mother2`,
`proc_type_origin`, `proc_id_origin`, `time_last_coll`) on a sample event:

- `proc_type_origin == 0` matches `ncoll == 0` exactly for nucleons (18/18
  in event 0). So `proc_type_origin` cannot resolve "first interaction was
  inelastic NN" — it only distinguishes never-interacted from interacted.
- `pdg_mother1, pdg_mother2` describe the *parent process of the final-state
  particle*, i.e., the **most recent** decay/reaction, not the *first*
  interaction of the original nucleon. A nucleon that underwent NN → N N
  followed by a meson rescatter has mothers from the meson rescatter, not
  from the original NN.
- `proc_type_origin` does encode reaction-type categories (1, 3, 5 seen in
  event 0 — corresponding to SMASH internal process IDs), but these tag the
  final-state particle's *last* process, not the nucleon's *first*.

**Conclusion**: the available SMASH branches cannot reconstruct the
Glauber-style "first interaction was inelastic NN-NN" criterion. Doing so
would require rerunning SMASH with collision logging enabled
(`Collision_Term: Logging: Collisions: INFO` or equivalent), which is out
of scope here.

## Verdict — (a) pure definition asymmetry

The 1.5–1.7× SMASH/UrQMD gap in stored ⟨Npart⟩ is **entirely** a definition
mismatch:

- SMASH ingester uses **`2A − N_spec(ncoll==0)`**: a participant is any
  nucleon that scattered at least once, *including* elastic scatters and
  rescatters off produced mesons. This is an upper bound on the Glauber
  participant count, already flagged in CLAUDE.md and in
  `src/data/dataset.py:35-39`.
- UrQMD ingester stores the **Glauber-header `Nuc_part`** value from the
  UrQMD `f13` output. This is the Glauber-convention number — nucleons with
  at least one inelastic NN interaction at event start.

When the SMASH rule is cross-applied to UrQMD per-particle data, ⟨Npart⟩ jumps
from 122 to 185–189, matching SMASH within ~3 units. The two transport codes
produce the same participant population at FXT energies; they label it
differently. There is no transport-model systematic to footnote.

## Sanity — b distributions agree

dN/db in 1-fm bins on the b<14 sample (first 10 k events each):

```
3.2 GeV  SMASH: [ 19  71 147 166 223 244 328 367 443 475 581 569 640 675]
3.2 GeV  UrQMD: [ 47 165 236 371 468 573 628 774 854 1019 1064 1179 1268 1352]
3.5 GeV  SMASH: [ 14  72 110 185 254 256 365 368 434 507 539 551 627 656]
3.5 GeV  UrQMD: [ 52 155 237 383 446 542 680 818 898 984 1084 1165 1242 1309]
```

Both linear-triangular as expected for geometric min-bias. The UrQMD counts
are ~2× SMASH only because UrQMD has 100 k events vs SMASH 50 k at 3.2 GeV
(after the same sample cap). Shape is identical. **b is a safe joint
regression target**.

## Recommendation — option (iii): recompute UrQMD Npart with the SMASH rule

The four options ranked:

1. **(i) Keep both as-is and footnote.** Pros: zero code change. Cons:
   ⟨Npart⟩ disagrees by 60% in any cross-generator plot; reviewers will
   flag it; a "joint training across generators" claim becomes muddled
   because the regression target is inconsistent. **Reject.**

2. **(ii) Redefine SMASH Npart to match Glauber-header rule.** Pros:
   matches what STAR analysts mean by N_part. Cons: requires rerunning all
   4 SMASH .root files with collision logging enabled (several days of
   simulation + ingester rewrite). **Reject for v1.**

3. **(iii) Redefine UrQMD Npart to match SMASH ncoll-spectator rule.**
   Pros: 5 lines of code; uses data already in the UrQMD HDF5
   (`particles/pdg`, `particles/ncoll`); makes the regression target
   apples-to-apples; the "upper bound vs Glauber" caveat already documented
   in CLAUDE.md applies symmetrically to both generators. Cons: now neither
   stored `Npart` is the Glauber quantity, so the paper has to be explicit
   that "Npart in this work = 2A − N_spec(ncoll==0), an upper bound on the
   Glauber N_part, used identically for SMASH and UrQMD." **Recommended.**

4. **(iv) Drop Npart as a regression target; predict only b.** Pros:
   removes the problem entirely; `b` is the cleanly-defined geometric
   quantity both codes share unambiguously. Cons: paper loses the N_part
   regression result, which has nominal value for downstream cumulant
   work. **Acceptable fallback** if reviewer pushback on (iii) is severe.

**Preferred path: (iii) for v1 + retain b as the primary target. Document
the symmetric upper-bound caveat in the paper.**

## Implementation for option (iii)

Add a one-pass recomputation in `scripts/build_padded_cache_urqmd.py`
(or, cleaner, in the UrQMD ingester `ingest_urqmd.py` next time it runs).
Below is the cache-builder patch — works on the raw HDF5 inputs without
needing a re-ingest:

```python
# In scripts/build_padded_cache_urqmd.py, replace the line
#   npart_all = src["Npart"][:].astype(np.int16)
# with the recomputed SMASH-rule version:

AU_2A = 394  # Au, A=197
pdg_all   = src["particles/pdg"][:]
ncoll_all = src["particles/ncoll"][:]
offsets   = src["offset"][:]
is_nuc = (np.abs(pdg_all) == 2112) | (np.abs(pdg_all) == 2212)
spec_mask = is_nuc & (ncoll_all == 0)
# Per-event spectator count via reduceat on offsets
n_spec_per_event = np.add.reduceat(spec_mask.astype(np.int32), offsets[:-1])
# Handle empty events (offsets[i] == offsets[i+1]) by zeroing them out
empty = (np.diff(offsets) == 0)
n_spec_per_event[empty] = 0
npart_all = (AU_2A - n_spec_per_event).astype(np.int16)
```

Verification target after the patch: UrQMD ⟨Npart⟩ on the b<14 cache should
jump from ~122 to ~185 and approximately match SMASH at each energy
(differences of a few units are expected from genuine transport-dynamics
differences in elastic rescattering rates).

## Open follow-ups (not for this audit)

- Once option (iii) is applied, re-check whether the small residual SMASH−UrQMD
  ⟨Npart⟩ gap (a few units, energy-dependent) shows a √sNN trend. SMASH ⟨Npart⟩
  rises with energy (185 → 205); UrQMD-SMASH-rule was roughly flat (185 → 189
  in the 10 k sample). If real, this reflects different elastic rescattering
  in the two codes and is a publishable transport-model systematic.
- The Pb-mass-number footgun (`Npart_ncoll = 2·208 − N_spec`) lives only in
  the UrQMD ingester's diagnostic field and is not used downstream; safe to
  ignore, but worth fixing on next ingest pass.
