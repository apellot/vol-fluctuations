"""Train a centrality model to predict approximate N_coll.

N_coll_approx ≡ ½ × Σ(per-particle ncoll, summed over nucleon participants),
where nucleon participant = abs(pdg) ∈ {2112, 2212} AND ncoll > 0.

Caveats (document in paper):
  * SMASH's per-particle ncoll counts ALL scatterings (elastic, inelastic,
    resonance excitations, scatterings off produced mesons) — not Glauber
    binary NN collisions.  N_coll_approx is therefore an upper bound on the
    Glauber N_coll.
  * At FXT energies the bias is moderate; cross-checks against UrQMD or
    SMASH with collision logging enabled are deferred to future work.

Architecture and particle-level inputs are identical to train_cached.py
(same padded cache, same per-particle features).  Only the regression target
changes: the evidential NIG head targets N_coll_approx instead of b.
The classification head is left in place but its loss weight is forced to 0.

Usage (typical, with throttling):
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 python3 scripts/train_ncoll.py \\
        --arch deepsets \\
        --cache data/processed/cached/all_padded.h5 \\
        --inputs data/processed/auau_3p2GeV.h5 data/processed/auau_3p5GeV.h5 \\
                 data/processed/auau_3p9GeV.h5 data/processed/auau_4p5GeV.h5 \\
        --output-tag ncoll_v1 \\
        --device mps --batch-size 256 --epochs 20 --num-workers 2

Pass --ncoll-cache to save/reload the computed labels and skip recomputation
on subsequent runs:
    ... --ncoll-cache data/processed/ncoll_labels.h5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

# Set thread caps BEFORE importing torch so the libs pick them up.
def _set_thread_caps(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(n))


_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--num-threads", type=int, default=4)
_pre_args, _ = _pre_parser.parse_known_args()
_set_thread_caps(_pre_args.num_threads)

import torch  # noqa: E402
from torch import nn, Tensor  # noqa: E402
from torch.utils.data import DataLoader, Dataset, Subset  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.losses.evidential import evidential_loss, nig_to_moments  # noqa: E402
from src.models.deepsets import DeepSets, DeepSetsConfig  # noqa: E402
from src.models.set_transformer import SetTransformer, SetTransformerConfig  # noqa: E402
from src.models.efn import EFN, EFNConfig  # noqa: E402
from src.models.gnn import GNN, GNNConfig  # noqa: E402
from src.models.pfn import PFN, PFNConfig  # noqa: E402


# ---------------------------------------------------------------------------
# N_coll label computation
# ---------------------------------------------------------------------------

def compute_ncoll_labels(input_h5s: list[Path], cache_path: Path) -> np.ndarray:
    """Compute approximate N_coll per event, aligned to the padded cache index.

    N_coll_approx = ½ × Σ(ncoll over nucleon participants).
    Nucleon participant: abs(pdg) ∈ {2112, 2212} AND ncoll > 0.

    Applies the same Npart > 0 event filter used by build_padded_cache.py, so
    the returned array is one-to-one with the cache's event index.

    Args:
        input_h5s:  Ingested per-energy HDF5 files (same order as
                    cache attrs['source_files']).
        cache_path: Pre-padded cache; used only to verify event-count alignment.

    Returns:
        ncoll: (N_cache_events,) float32
    """
    with h5py.File(cache_path, "r") as c:
        n_events_per_energy: list[int] = list(c.attrs["n_events_per_energy"])

    out_arrays: list[np.ndarray] = []
    for e_idx, in_path in enumerate(input_h5s):
        print(f"  [{e_idx+1}/{len(input_h5s)}] {in_path.name} ...", end=" ", flush=True)
        t0 = time.time()
        with h5py.File(in_path, "r") as h:
            npart_arr  = h["Npart"][:]                            # (n_events,)
            keep_mask  = npart_arr > 0                            # same filter as cache builder
            offsets    = h["offset"][:].astype(np.int64)          # (n_events+1,)
            pdg_flat   = h["particles/pdg"][:].astype(np.int32)   # (total_particles,)
            ncoll_flat = h["particles/ncoll"][:].astype(np.int32) # (total_particles,)

        # Vectorised: mark nucleon participants and sum their ncoll per event.
        #
        # Each NN collision contributes +1 to the ncoll counter of BOTH
        # participants, so the per-event sum over participants = 2 × N_coll.
        # Dividing by 2 gives the approximation.  The result is still an
        # upper bound because SMASH ncoll includes elastic scatterings and
        # scatterings off produced mesons (not just binary NN inelastic).
        is_nucleon    = (np.abs(pdg_flat) == 2112) | (np.abs(pdg_flat) == 2212)
        is_participant = is_nucleon & (ncoll_flat > 0)
        weighted      = (ncoll_flat * is_participant).astype(np.int64)

        # np.add.reduceat(a, starts) sums a[starts[i]:starts[i+1]] for each i.
        # offsets[:-1] are event start positions; ends are offsets[1:] implicitly.
        ncoll_sum    = np.add.reduceat(weighted, offsets[:-1].astype(np.intp))
        ncoll_approx = (ncoll_sum / 2.0).astype(np.float32)  # (n_events,)

        # Apply the Npart > 0 filter to match the cache layout.
        ncoll_kept = ncoll_approx[keep_mask]
        expected   = n_events_per_energy[e_idx]
        if len(ncoll_kept) != expected:
            raise RuntimeError(
                f"N_coll count mismatch for energy index {e_idx} ({in_path.name}): "
                f"got {len(ncoll_kept)}, expected {expected} (from cache attrs)."
            )
        out_arrays.append(ncoll_kept)

        kept = int(keep_mask.sum())
        print(
            f"done in {time.time()-t0:.1f}s  "
            f"(kept {kept:,}/{len(npart_arr):,} events, "
            f"median N_coll_approx={float(np.median(ncoll_kept)):.1f}, "
            f"max={float(ncoll_kept.max()):.0f})"
        )

    return np.concatenate(out_arrays, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset and collate
# ---------------------------------------------------------------------------

@dataclass
class NCollBatch:
    cont:    Tensor  # (B, MAX_PARTICLES, 4) float32 — pT, eta_lab, phi, charge
    mask:    Tensor  # (B, MAX_PARTICLES) bool
    sqrtsNN: Tensor  # (B,) float32
    ncoll:   Tensor  # (B,) float32 — regression target (N_coll_approx)
    b:       Tensor  # (B,) float32 — reference truth, NOT used in the loss


class NCollCachedDataset(Dataset):
    """Padded-cache dataset using N_coll_approx as the regression target."""

    def __init__(self, cache_path: Path, ncoll_labels: np.ndarray) -> None:
        self.cache_path   = Path(cache_path)
        self.ncoll_labels = ncoll_labels  # (N,) float32

        with h5py.File(self.cache_path, "r") as h:
            self.n_events      = int(h["b"].shape[0])
            self.max_particles = int(h.attrs["max_particles"])
            self.energy_sizes  = list(h.attrs["n_events_per_energy"])

        if len(self.ncoll_labels) != self.n_events:
            raise ValueError(
                f"ncoll_labels length ({len(ncoll_labels)}) != "
                f"cache n_events ({self.n_events}). "
                "Ensure --inputs are in the same order as the cache's source_files."
            )
        self._h: h5py.File | None = None

    def _ensure_open(self) -> None:
        if self._h is None:
            self._h = h5py.File(self.cache_path, "r", swmr=True)

    def __len__(self) -> int:
        return self.n_events

    def __getitem__(self, idx: int) -> dict:
        self._ensure_open()
        n    = int(self._h["length"][idx])
        cont = self._h["cont"][idx]  # (MAX_PARTICLES, 4) float32
        mask = np.zeros(self.max_particles, dtype=bool)
        mask[:n] = True
        return {
            "cont":    cont,
            "mask":    mask,
            "sqrtsNN": float(self._h["sqrtsNN"][idx]),
            "ncoll":   float(self.ncoll_labels[idx]),
            "b":       float(self._h["b"][idx]),
        }


def collate_ncoll(batch: list[dict]) -> NCollBatch:
    return NCollBatch(
        cont    = torch.from_numpy(np.stack([b["cont"] for b in batch])),
        mask    = torch.from_numpy(np.stack([b["mask"] for b in batch])),
        sqrtsNN = torch.tensor([b["sqrtsNN"] for b in batch], dtype=torch.float32),
        ncoll   = torch.tensor([b["ncoll"]   for b in batch], dtype=torch.float32),
        b       = torch.tensor([b["b"]       for b in batch], dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(arch: str) -> nn.Module:
    # n_centrality_bins=1: classification head present but loss weight is 0.
    cfg = dict(n_centrality_bins=1)
    if arch == "deepsets":
        return DeepSets(DeepSetsConfig(**cfg))
    if arch == "settransformer":
        return SetTransformer(SetTransformerConfig(
            **cfg, n_sab=1, d_model=64, n_heads=4, ff_hidden=128,
        ))
    if arch == "efn":
        return EFN(EFNConfig(**cfg))
    if arch == "pfn":
        return PFN(PFNConfig(**cfg))
    if arch == "gnn":
        return GNN(GNNConfig(**cfg))
    raise ValueError(f"Unknown arch: {arch!r}")


def pick_device(requested: str) -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested


# ---------------------------------------------------------------------------
# Event-level feature vector (identical to train_cached.py)
# ---------------------------------------------------------------------------

def _build_event_feats(cont: Tensor, mask: Tensor, sqrtsNN: Tensor) -> Tensor:
    """(sqrtsNN, mult_lab, mean_pT_lab, total_pT_lab) → (B, 4)."""
    mask_f   = mask.float()
    pT       = cont[..., 0]
    n_real   = mask_f.sum(dim=1).clamp(min=1.0)
    total_pT = (pT * mask_f).sum(dim=1)
    mean_pT  = total_pT / n_real
    return torch.stack([sqrtsNN, n_real, mean_pT, total_pT], dim=-1)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: str,
    coeff_reg: float,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        cont    = batch.cont.to(device)
        mask    = batch.mask.to(device)
        sqrtsNN = batch.sqrtsNN.to(device)
        y       = batch.ncoll.to(device)

        event_feats = _build_event_feats(cont, mask, sqrtsNN)
        out  = model(cont, mask, event_feats)
        loss = evidential_loss(y, out["mu"], out["nu"], out["alpha"], out["beta"],
                               coeff_reg=coeff_reg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.no_grad()
def eval_split(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    coeff_reg: float,
) -> tuple[float, float]:
    """Returns (mean_loss, mean_MAE_in_N_coll_units)."""
    model.eval()
    losses: list[float] = []
    maes:   list[float] = []
    for batch in loader:
        cont    = batch.cont.to(device)
        mask    = batch.mask.to(device)
        sqrtsNN = batch.sqrtsNN.to(device)
        y       = batch.ncoll.to(device)

        event_feats = _build_event_feats(cont, mask, sqrtsNN)
        out  = model(cont, mask, event_feats)
        loss = evidential_loss(y, out["mu"], out["nu"], out["alpha"], out["beta"],
                               coeff_reg=coeff_reg)
        losses.append(float(loss))
        maes.append(float((out["mu"] - y).abs().mean()))
    return float(np.mean(losses)), float(np.mean(maes))


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------

def split_indices(
    n: int, *, seed: int, train: float = 0.8, val: float = 0.1
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(n * train))
    n_val   = int(round(n * val))
    return {
        "train": np.sort(perm[:n_train]),
        "val":   np.sort(perm[n_train:n_train + n_val]),
        "test":  np.sort(perm[n_train + n_val:]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--arch", required=True,
                   choices=["deepsets", "settransformer", "efn", "pfn", "gnn"])
    p.add_argument("--cache", required=True, type=Path,
                   help="Pre-padded cache built by build_padded_cache.py")
    p.add_argument("--inputs", required=True, nargs="+", type=Path,
                   help="Ingested per-energy HDF5 files (same order as cache source_files attr)")
    p.add_argument("--output-tag", default="ncoll_v1")
    p.add_argument("--ncoll-cache", type=Path, default=None,
                   help="Optional path to save/load computed N_coll labels (avoids recomputing)")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--coeff-reg",  type=float, default=1e-2,
                   help="Regularisation coefficient in the evidential loss")
    p.add_argument("--patience",   type=int,   default=5)
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--num-workers",type=int,   default=2)
    p.add_argument("--num-threads",type=int,   default=4,
                   help="Intra-op thread cap (OMP/MKL/etc.). Applied at import time.")
    args = p.parse_args()

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"arch: {args.arch}  device: {device}  threads: {args.num_threads}")

    # ------------------------------------------------------------------
    # N_coll labels
    # ------------------------------------------------------------------
    ncoll_cache = args.ncoll_cache
    if ncoll_cache is not None and Path(ncoll_cache).exists():
        print(f"Loading precomputed N_coll labels from {ncoll_cache}")
        with h5py.File(ncoll_cache, "r") as f:
            ncoll_labels = f["ncoll_approx"][:]
    else:
        print("Computing N_coll_approx labels from ingested HDF5 files ...")
        ncoll_labels = compute_ncoll_labels(
            [Path(x) for x in args.inputs],
            args.cache.expanduser(),
        )
        if ncoll_cache is not None:
            Path(ncoll_cache).parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(ncoll_cache, "w") as f:
                f.create_dataset("ncoll_approx", data=ncoll_labels)
                f.attrs["definition"] = (
                    "N_coll_approx = 0.5 * sum(ncoll over nucleon participants). "
                    "Upper bound; includes elastic and produced-hadron scatterings."
                )
            print(f"Saved N_coll labels to {ncoll_cache}")

    print(
        f"N_coll_approx stats — "
        f"min={ncoll_labels.min():.0f}  "
        f"median={float(np.median(ncoll_labels)):.1f}  "
        f"mean={ncoll_labels.mean():.1f}  "
        f"max={ncoll_labels.max():.0f}  "
        f"std={ncoll_labels.std():.1f}"
    )

    # ------------------------------------------------------------------
    # Dataset and DataLoader
    # ------------------------------------------------------------------
    ds = NCollCachedDataset(args.cache.expanduser(), ncoll_labels)
    n_events    = len(ds)
    energy_sizes = ds.energy_sizes
    print(f"Dataset: {n_events:,} events across {len(energy_sizes)} energies ({energy_sizes})")

    splits = split_indices(n_events, seed=args.seed)
    print(
        f"Split: train={len(splits['train']):,}  "
        f"val={len(splits['val']):,}  "
        f"test={len(splits['test']):,}"
    )

    dl_kw = dict(
        num_workers=args.num_workers,
        collate_fn=collate_ncoll,
        persistent_workers=(args.num_workers > 0),
    )
    dl_train = DataLoader(
        Subset(ds, splits["train"].tolist()),
        batch_size=args.batch_size, shuffle=True, **dl_kw,
    )
    dl_val = DataLoader(
        Subset(ds, splits["val"].tolist()),
        batch_size=args.batch_size, shuffle=False, **dl_kw,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(args.arch).to(device)
    n_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"model: {n_params:,} parameters")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir  = Path("checkpoints") / f"{args.arch}_ncoll_{args.output_tag}"
    ckpt_path = ckpt_dir / "best.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val  = float("inf")
    since_best = 0
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(args.epochs):
        t_ep = time.time()
        tr_loss = train_one_epoch(model, dl_train, opt, device, args.coeff_reg)
        va_loss, va_mae = eval_split(model, dl_val, device, args.coeff_reg)
        dt = time.time() - t_ep

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "val_ncoll_mae": va_mae,
            "sec": dt,
        })
        improved = va_loss < best_val
        if improved:
            best_val = va_loss
            since_best = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": args.arch,
                    "config": asdict(model.cfg),
                    "epoch": epoch,
                    "target": "ncoll_approx",
                },
                ckpt_path,
            )
        else:
            since_best += 1

        flag = "*" if improved else " "
        print(
            f"  epoch {epoch:3d}  {dt:5.1f}s  "
            f"train {tr_loss:.4f}  val {va_loss:.4f}  "
            f"val_ncoll_MAE {va_mae:.1f}  {flag}",
            flush=True,
        )
        if since_best >= args.patience:
            print(f"  early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f} s, best val loss {best_val:.4f}")

    # ------------------------------------------------------------------
    # Inference on all events — emit per-energy HDF5s
    # ------------------------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dl_full = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_ncoll,
        persistent_workers=(args.num_workers > 0),
    )

    mus, nus, alphas, betas = [], [], [], []
    ncoll_true_all, b_true_all = [], []

    with torch.no_grad():
        for batch in dl_full:
            cont    = batch.cont.to(device)
            mask    = batch.mask.to(device)
            sqrtsNN = batch.sqrtsNN.to(device)
            event_feats = _build_event_feats(cont, mask, sqrtsNN)
            out = model(cont, mask, event_feats)
            mus.append(out["mu"].cpu().numpy())
            nus.append(out["nu"].cpu().numpy())
            alphas.append(out["alpha"].cpu().numpy())
            betas.append(out["beta"].cpu().numpy())
            ncoll_true_all.append(batch.ncoll.numpy())
            b_true_all.append(batch.b.numpy())

    mu         = np.concatenate(mus)
    nu         = np.concatenate(nus)
    alpha      = np.concatenate(alphas)
    beta       = np.concatenate(betas)
    ncoll_true = np.concatenate(ncoll_true_all)
    b_true_np  = np.concatenate(b_true_all)

    moments = nig_to_moments(
        torch.from_numpy(mu),  torch.from_numpy(nu),
        torch.from_numpy(alpha), torch.from_numpy(beta),
    )

    cum      = np.cumsum([0] + list(energy_sizes))
    is_train = np.zeros(n_events, dtype=bool); is_train[splits["train"]] = True
    is_val   = np.zeros(n_events, dtype=bool); is_val[splits["val"]]     = True
    is_test  = np.zeros(n_events, dtype=bool); is_test[splits["test"]]   = True

    out_root = Path("data/processed") / f"{args.arch}_ncoll" / args.output_tag
    out_root.mkdir(parents=True, exist_ok=True)

    print()
    for e_idx, in_path in enumerate(args.inputs):
        lo, hi   = int(cum[e_idx]), int(cum[e_idx + 1])
        out_path = out_root / f"{args.arch}_ncoll_pred_{in_path.stem}.h5"
        with h5py.File(out_path, "w") as h:
            h.create_dataset("ncoll_pred",    data=mu[lo:hi].astype(np.float32))
            h.create_dataset("ncoll_true",    data=ncoll_true[lo:hi].astype(np.float32))
            h.create_dataset("b_true",        data=b_true_np[lo:hi].astype(np.float32))
            h.create_dataset("nu",            data=nu[lo:hi].astype(np.float32))
            h.create_dataset("alpha",         data=alpha[lo:hi].astype(np.float32))
            h.create_dataset("beta",          data=beta[lo:hi].astype(np.float32))
            h.create_dataset("total_var",     data=moments["total_var"][lo:hi].numpy().astype(np.float32))
            h.create_dataset("aleatoric_var", data=moments["aleatoric_var"][lo:hi].numpy().astype(np.float32))
            h.create_dataset("epistemic_var", data=moments["epistemic_var"][lo:hi].numpy().astype(np.float32))
            h.create_dataset("is_train",      data=is_train[lo:hi])
            h.create_dataset("is_val",        data=is_val[lo:hi])
            h.create_dataset("is_test",       data=is_test[lo:hi])
            h.attrs["n_events"]        = hi - lo
            h.attrs["source_h5"]       = str(in_path)
            h.attrs["checkpoint"]      = str(ckpt_path)
            h.attrs["arch"]            = args.arch
            h.attrs["target"]          = "ncoll_approx"
            h.attrs["ncoll_definition"] = (
                "N_coll_approx = 0.5 * sum(ncoll over nucleon participants). "
                "Upper bound; includes elastic and produced-hadron scatterings."
            )

        mae_test = float(
            np.abs(mu[lo:hi][is_test[lo:hi]] - ncoll_true[lo:hi][is_test[lo:hi]]).mean()
        )
        n_test = int(is_test[lo:hi].sum())
        print(
            f"  energy {e_idx} ({in_path.stem}): "
            f"test N_coll-MAE = {mae_test:.1f}  (N_test = {n_test:,})"
        )

    # ------------------------------------------------------------------
    # Metrics JSON
    # ------------------------------------------------------------------
    with open(out_root / "train_metrics.json", "w") as f:
        json.dump(
            {
                "args": vars(args)
                | {
                    "inputs": [str(x) for x in args.inputs],
                    "cache": str(args.cache),
                    "ncoll_cache": str(args.ncoll_cache) if args.ncoll_cache else None,
                },
                "metrics": {
                    "history": history,
                    "best_val_loss": best_val,
                    "elapsed_sec": elapsed,
                    "n_parameters": n_params,
                    "ncoll_approx_stats": {
                        "min":    float(ncoll_labels.min()),
                        "max":    float(ncoll_labels.max()),
                        "mean":   float(ncoll_labels.mean()),
                        "median": float(np.median(ncoll_labels)),
                        "std":    float(ncoll_labels.std()),
                    },
                },
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nWrote predictions and metrics to {out_root}/")


if __name__ == "__main__":
    main()
