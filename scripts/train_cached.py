"""Train the permutation-invariant architectures (and the MLP baselines) from the
pre-padded cache, with DUAL evidential heads for b and Npart.

2026-05-28 modelling restructure (docs/modelling_plan.md):
  * Targets: b AND Npart, each via its own evidential NIG head; both z-standardized
    on the train split (predicted in standardized space, inverted for reporting).
  * Loss: evidential(b) + w_npart * evidential(Npart). No classification head —
    centrality is DERIVED downstream from predicted Npart.
  * Event-level scalar features are z-normalized (stats from the train split).
  * φ-rotation augmentation ON by default (enforces rotational invariance so the
    network learns |Q_n| / flow magnitudes, not the random reaction-plane angle).
  * No truth_dir / centrality_bin dependency — training reads only the cache.

Usage:
    python scripts/train_cached.py --arch deepsets \\
        --cache data/processed/cached/urqmd_padded_det.h5 \\
        --inputs data/raw/urqmd_auau_{3p2,3p5,3p9,4p5}GeV.h5 \\
        --output-tag urqmd_v1 --device auto --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


def _set_thread_caps(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(n))


_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--num-threads", type=int, default=4)
_pre_args, _ = _pre_parser.parse_known_args()
_set_thread_caps(_pre_args.num_threads)

import h5py  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cached_dataset import CachedParticleDataset, collate_cached  # noqa: E402
from src.data.lab_cached_dataset import apply_phi_rotation  # noqa: E402
from src.losses.evidential import evidential_loss, nig_to_moments  # noqa: E402
from src.baselines.truth import centrality_from_values  # noqa: E402
from src.models.deepsets import DeepSets, DeepSetsConfig  # noqa: E402
from src.models.set_transformer import SetTransformer, SetTransformerConfig  # noqa: E402
from src.models.efn import EFN, EFNConfig  # noqa: E402
from src.models.gnn import GNN, GNNConfig  # noqa: E402
from src.models.pfn import PFN, PFNConfig  # noqa: E402
from src.models.mlp_pool import MLPPool, MLPPoolConfig  # noqa: E402
from src.models.mlp import MLPHead, MLPConfig  # noqa: E402
from src.models.heads import N_EVENT_FEATURES  # noqa: E402

ARCHES = ["deepsets", "settransformer", "efn", "pfn", "gnn", "mlp_pool", "mlp"]


def split_indices(n: int, *, seed: int, train: float = 0.8, val: float = 0.1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    return {
        "train": np.sort(perm[:n_train]),
        "val":   np.sort(perm[n_train:n_train + n_val]),
        "test":  np.sort(perm[n_train + n_val:]),
    }


def pick_device(requested: str) -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested


def build_model(arch: str) -> nn.Module:
    if arch == "deepsets":
        return DeepSets(DeepSetsConfig())
    if arch == "settransformer":
        # Smaller defaults for the cached pipeline: the attention tensor is large,
        # so lower n_sab/d_model to fit comfortably in unified memory at batch=128.
        return SetTransformer(SetTransformerConfig(n_sab=1, d_model=64, n_heads=4, ff_hidden=128))
    if arch == "efn":
        return EFN(EFNConfig())
    if arch == "pfn":
        return PFN(PFNConfig())
    if arch == "gnn":
        return GNN(GNNConfig())
    if arch == "mlp_pool":
        return MLPPool(MLPPoolConfig())
    if arch == "mlp":
        # Feature-vector baseline: consumes only the event-level scalars.
        return MLPHead(MLPConfig(n_features=N_EVENT_FEATURES))
    raise ValueError(f"Unknown arch: {arch!r}")


def build_event_feats(cont: torch.Tensor, mask: torch.Tensor, sqrtsNN: torch.Tensor) -> torch.Tensor:
    """event_feats = (sqrtsNN, mult_lab, mean_pT_lab, total_pT_lab), built from the
    kept-particle list so it matches the cache's stored event scalars."""
    mask_f   = mask.float()
    pT       = cont[..., 0]
    n_real   = mask_f.sum(dim=1).clamp(min=1.0)
    total_pT = (pT * mask_f).sum(dim=1)
    mean_pT  = total_pT / n_real
    return torch.stack([sqrtsNN, n_real, mean_pT, total_pT], dim=-1)  # (B, 4)


def run_model(model, arch: str, cont, mask, event_feats) -> dict:
    """Dispatch on input signature: the feature-MLP takes only event_feats."""
    if arch == "mlp":
        return model(event_feats)
    return model(cont, mask, event_feats)


class Standardizer:
    """Per-target z-standardization. Holds tensors on the training device."""

    def __init__(self, b_mean, b_std, np_mean, np_std, ef_mean, ef_std, device):
        # float32 throughout — MPS does not support float64.
        f32 = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
        self.b_mean = f32(float(b_mean))
        self.b_std = f32(max(float(b_std), 1e-6))
        self.np_mean = f32(float(np_mean))
        self.np_std = f32(max(float(np_std), 1e-6))
        self.ef_mean = f32(np.asarray(ef_mean, dtype=np.float32))
        self.ef_std = f32(np.where(np.asarray(ef_std) > 1e-6, ef_std, 1.0).astype(np.float32))

    def norm_ef(self, ef):       return (ef - self.ef_mean) / self.ef_std
    def std_b(self, b):          return (b - self.b_mean) / self.b_std
    def std_np(self, n):         return (n - self.np_mean) / self.np_std
    def inv_b(self, mu):         return mu * self.b_std + self.b_mean
    def inv_np(self, mu):        return mu * self.np_std + self.np_mean


def _losses(out, yb_std, ynp_std, coeff_reg, w_npart):
    l_b = evidential_loss(yb_std, out["b_mu"], out["b_nu"], out["b_alpha"], out["b_beta"],
                          coeff_reg=coeff_reg)
    l_np = evidential_loss(ynp_std, out["np_mu"], out["np_nu"], out["np_alpha"], out["np_beta"],
                           coeff_reg=coeff_reg)
    return l_b + w_npart * l_np, l_b, l_np


def train_one_epoch(model, arch, loader, opt, device, std, coeff_reg, w_npart, augment_phi):
    model.train()
    losses = []
    for batch in loader:
        cont = batch.cont.to(device)
        mask = batch.mask.to(device)
        if augment_phi:
            cont = apply_phi_rotation(cont, mask)
        sqrtsNN = batch.sqrtsNN.to(device)
        yb_std = std.std_b(batch.b.to(device))
        ynp_std = std.std_np(batch.Npart.to(device))

        event_feats = std.norm_ef(build_event_feats(cont, mask, sqrtsNN))
        out = run_model(model, arch, cont, mask, event_feats)
        loss, _, _ = _losses(out, yb_std, ynp_std, coeff_reg, w_npart)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.no_grad()
def eval_split(model, arch, loader, device, std, coeff_reg, w_npart):
    model.eval()
    losses, b_maes, np_maes = [], [], []
    for batch in loader:
        cont = batch.cont.to(device)
        mask = batch.mask.to(device)
        sqrtsNN = batch.sqrtsNN.to(device)
        b_true = batch.b.to(device)
        np_true = batch.Npart.to(device)
        event_feats = std.norm_ef(build_event_feats(cont, mask, sqrtsNN))
        out = run_model(model, arch, cont, mask, event_feats)
        loss, _, _ = _losses(out, std.std_b(b_true), std.std_np(np_true), coeff_reg, w_npart)
        losses.append(float(loss))
        # MAEs reported in PHYSICAL units (fm, participants) for interpretability.
        b_maes.append(float((std.inv_b(out["b_mu"]) - b_true).abs().mean()))
        np_maes.append(float((std.inv_np(out["np_mu"]) - np_true).abs().mean()))
    return float(np.mean(losses)), float(np.mean(b_maes)), float(np.mean(np_maes))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=ARCHES)
    p.add_argument("--cache", required=True, type=Path,
                   help="Pre-padded cache built by build_padded_cache{,_urqmd}.py")
    p.add_argument("--inputs", required=True, nargs="+", type=Path,
                   help="Per-energy ingested HDF5s, in cache order (used for output file naming)")
    p.add_argument("--output-tag", default="urqmd_v1")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--coeff-reg", type=float, default=1e-2)
    p.add_argument("--w-npart", type=float, default=1.0,
                   help="Weight on the Npart evidential loss (both targets standardized).")
    p.add_argument("--no-phi-aug", action="store_true", help="Disable φ-rotation augmentation")
    p.add_argument("--phi-encoding", choices=["raw", "sincos"], default="raw",
                   help="Per-particle φ representation. 'sincos' is the tracked ablation "
                        "(needs per-particle feature-count parametrization) — not yet wired.")
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--num-threads", type=int, default=4)
    args = p.parse_args()

    if args.phi_encoding == "sincos":
        raise NotImplementedError(
            "phi-encoding=sincos is not yet wired — it requires expanding the "
            "per-particle feature count from 4 to 5 and parametrizing each model's "
            "input layer. Tracked as the φ-encoding ablation in docs/modelling_plan.md.")

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"arch: {args.arch}  device: {device}  threads: {args.num_threads}  "
          f"phi_aug: {not args.no_phi_aug}")

    ds = CachedParticleDataset(args.cache.expanduser())
    n_events = len(ds)
    energy_sizes = ds.energy_sizes
    print(f"Cached dataset: {n_events:,} events across {len(energy_sizes)} energies ({energy_sizes})")

    splits = split_indices(n_events, seed=args.seed)
    tr_idx = splits["train"]
    print(f"Split: train={len(tr_idx):,}, val={len(splits['val']):,}, test={len(splits['test']):,}")

    # --- standardization stats from the TRAIN split (read scalars straight from cache) ---
    with h5py.File(args.cache.expanduser(), "r") as h:
        b_all   = h["b"][:].astype(np.float64)
        np_all  = h["Npart"][:].astype(np.float64)
        ef_all  = np.stack([h["sqrtsNN"][:], h["mult_lab"][:],
                            h["mean_pT_lab"][:], h["total_pT_lab"][:]], axis=1).astype(np.float64)
    std = Standardizer(
        b_mean=b_all[tr_idx].mean(),  b_std=b_all[tr_idx].std(),
        np_mean=np_all[tr_idx].mean(), np_std=np_all[tr_idx].std(),
        ef_mean=ef_all[tr_idx].mean(axis=0), ef_std=ef_all[tr_idx].std(axis=0),
        device=device,
    )
    print(f"standardize: b ~ N({float(std.b_mean):.2f}, {float(std.b_std):.2f}) fm, "
          f"Npart ~ N({float(std.np_mean):.1f}, {float(std.np_std):.1f})")

    dl_kw = dict(num_workers=args.num_workers, collate_fn=collate_cached,
                 persistent_workers=(args.num_workers > 0))
    dl_train = DataLoader(Subset(ds, tr_idx.tolist()), batch_size=args.batch_size,
                          shuffle=True, **dl_kw)
    dl_val = DataLoader(Subset(ds, splits["val"].tolist()), batch_size=args.batch_size,
                        shuffle=False, **dl_kw)

    model = build_model(args.arch).to(device)
    n_params = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    print(f"model: {n_params:,} parameters")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ckpt_path = Path("checkpoints") / f"{args.arch}_{args.output_tag}" / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf"); since_best = 0
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        t_ep = time.time()
        tr = train_one_epoch(model, args.arch, dl_train, opt, device, std,
                             args.coeff_reg, args.w_npart, augment_phi=not args.no_phi_aug)
        va, b_mae, np_mae = eval_split(model, args.arch, dl_val, device, std,
                                       args.coeff_reg, args.w_npart)
        dt = time.time() - t_ep
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va,
                        "val_b_mae": b_mae, "val_npart_mae": np_mae, "sec": dt})
        improved = va < best_val
        if improved:
            best_val = va; since_best = 0
            torch.save({
                "state_dict": model.state_dict(),
                "arch": args.arch,
                "config": asdict(model.cfg),
                "standardize": {"b_mean": float(std.b_mean), "b_std": float(std.b_std),
                                "np_mean": float(std.np_mean), "np_std": float(std.np_std),
                                "ef_mean": std.ef_mean.cpu().tolist(),
                                "ef_std": std.ef_std.cpu().tolist()},
                "epoch": epoch,
            }, ckpt_path)
        else:
            since_best += 1
        flag = "*" if improved else " "
        print(f"  epoch {epoch:3d}  {dt:5.1f}s  train {tr:.4f}  val {va:.4f}  "
              f"b_MAE {b_mae:.3f} fm  Npart_MAE {np_mae:.1f}  {flag}", flush=True)
        if since_best >= args.patience:
            print(f"  early stop at epoch {epoch}")
            break
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f} s, best val loss {best_val:.4f}")

    # --- predict on every event, emit per-energy HDF5s ---
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    cum = np.cumsum([0] + list(energy_sizes))
    is_train = np.zeros(n_events, dtype=bool); is_train[splits["train"]] = True
    is_val   = np.zeros(n_events, dtype=bool); is_val[splits["val"]]     = True
    is_test  = np.zeros(n_events, dtype=bool); is_test[splits["test"]]   = True

    out_root = Path("data/processed") / args.arch / args.output_tag
    out_root.mkdir(parents=True, exist_ok=True)

    dl_full = DataLoader(ds, batch_size=args.batch_size, shuffle=False, **dl_kw)
    keys = ["b_mu", "b_nu", "b_alpha", "b_beta", "np_mu", "np_nu", "np_alpha", "np_beta"]
    acc = {k: [] for k in keys}
    b_true, np_true = [], []
    with torch.no_grad():
        for batch in dl_full:
            cont = batch.cont.to(device); mask = batch.mask.to(device)
            event_feats = std.norm_ef(build_event_feats(cont, mask, batch.sqrtsNN.to(device)))
            out = run_model(model, args.arch, cont, mask, event_feats)
            for k in keys:
                acc[k].append(out[k].cpu().numpy())
            b_true.append(batch.b.numpy()); np_true.append(batch.Npart.numpy())
    arr = {k: np.concatenate(v) for k, v in acc.items()}
    b_true_np = np.concatenate(b_true); np_true_np = np.concatenate(np_true)

    # Invert standardization → physical units, and propagate variance (× std²).
    b_pred  = arr["b_mu"] * float(std.b_std) + float(std.b_mean)
    np_pred = arr["np_mu"] * float(std.np_std) + float(std.np_mean)
    b_mom  = nig_to_moments(*[torch.from_numpy(arr[f"b_{s}"]) for s in ("mu", "nu", "alpha", "beta")])
    np_mom = nig_to_moments(*[torch.from_numpy(arr[f"np_{s}"]) for s in ("mu", "nu", "alpha", "beta")])
    b_var  = b_mom["total_var"].numpy()  * float(std.b_std) ** 2
    np_var = np_mom["total_var"].numpy() * float(std.np_std) ** 2

    print()
    for e_idx, in_path in enumerate(args.inputs):
        lo, hi = int(cum[e_idx]), int(cum[e_idx + 1])
        # Centrality derived per-energy from PREDICTED Npart (higher = more central).
        cent = centrality_from_values(np_pred[lo:hi], higher_is_central=True)
        out_path = out_root / f"{args.arch}_pred_{in_path.stem}.h5"
        with h5py.File(out_path, "w") as h:
            h.create_dataset("b_pred",         data=b_pred[lo:hi].astype(np.float32))
            h.create_dataset("b_var",          data=b_var[lo:hi].astype(np.float32))
            h.create_dataset("npart_pred",     data=np_pred[lo:hi].astype(np.float32))
            h.create_dataset("npart_var",      data=np_var[lo:hi].astype(np.float32))
            h.create_dataset("centrality_bin", data=cent.astype(np.int16))
            h.create_dataset("is_train",       data=is_train[lo:hi])
            h.create_dataset("is_val",         data=is_val[lo:hi])
            h.create_dataset("is_test",        data=is_test[lo:hi])
            h.attrs["n_events"]   = hi - lo
            h.attrs["source_h5"]  = str(in_path)
            h.attrs["checkpoint"] = str(ckpt_path)
            h.attrs["arch"]       = args.arch
            h.attrs["centrality_from"] = "predicted_Npart"
        te = is_test[lo:hi]
        b_mae  = float(np.abs(b_pred[lo:hi][te]  - b_true_np[lo:hi][te]).mean())
        np_mae = float(np.abs(np_pred[lo:hi][te] - np_true_np[lo:hi][te]).mean())
        print(f"  energy {e_idx}: test b-MAE = {b_mae:.3f} fm  Npart-MAE = {np_mae:.1f}  "
              f"(N_test = {int(te.sum()):,})")

    with open(out_root / "train_metrics.json", "w") as f:
        json.dump({"args": vars(args) | {"inputs": [str(x) for x in args.inputs],
                                         "cache": str(args.cache)},
                   "metrics": {"history": history, "best_val_loss": best_val,
                               "elapsed_sec": elapsed, "n_parameters": n_params}},
                  f, indent=2, default=str)
    print(f"\nWrote predictions to {out_root}/")


if __name__ == "__main__":
    main()
