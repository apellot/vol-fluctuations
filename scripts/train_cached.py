"""Train DeepSets or Set Transformer from the pre-padded cache.

Fixed-shape input means the DataLoader does no padding and the GPU memory
footprint is predictable across batches — both major wins over the variable-
length pipeline. Designed to be run with throttled resources (nice, thread
caps) so the user's machine remains responsive while training.

Outputs match the schema in scripts/train_set_model.py so downstream
evaluation code does not need to branch on training pipeline.

Usage (typical, with throttling):
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 python3 scripts/train_cached.py \\
        --arch deepsets \\
        --cache data/processed/cached/all_padded.h5 \\
        --inputs data/processed/auau_{3p2,3p5,3p9,4p5}GeV.h5 \\
        --output-tag baseline_v1 \\
        --device mps --batch-size 256 --epochs 18 --num-workers 2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
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
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cached_dataset import CachedParticleDataset, collate_cached  # noqa: E402
from src.losses.evidential import evidential_loss, nig_to_moments  # noqa: E402
from src.models.deepsets import DeepSets, DeepSetsConfig  # noqa: E402
from src.models.set_transformer import SetTransformer, SetTransformerConfig  # noqa: E402


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


def build_model(arch: str, n_centrality_bins: int) -> nn.Module:
    if arch == "deepsets":
        return DeepSets(DeepSetsConfig(n_centrality_bins=n_centrality_bins))
    if arch == "settransformer":
        # Smaller defaults for the cached pipeline: with MAX=896 the attention
        # tensor (B × heads × 896 × 896) is large, so we lower n_sab and keep
        # n_heads=4 to fit comfortably in M3 unified memory at batch=128.
        return SetTransformer(SetTransformerConfig(
            n_centrality_bins=n_centrality_bins, n_sab=1, d_model=64, n_heads=4, ff_hidden=128,
        ))
    raise ValueError(f"Unknown arch: {arch!r}")


def train_one_epoch(model, loader, opt, ce, device, coeff_reg, w_cls):
    model.train()
    losses = []
    for batch in loader:
        cont = batch.cont.to(device); pdg_idx = batch.pdg_idx.to(device)
        mask = batch.mask.to(device); sqrtsNN = batch.sqrtsNN.to(device)
        y = batch.b.to(device); z = batch.centrality_bin.to(device)
        out = model(cont, pdg_idx, mask, sqrtsNN)
        l_reg = evidential_loss(y, out["mu"], out["nu"], out["alpha"], out["beta"], coeff_reg=coeff_reg)
        l_cls = ce(out["logits"], z)
        loss = l_reg + w_cls * l_cls
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.no_grad()
def eval_split(model, loader, ce, device, coeff_reg, w_cls):
    model.eval()
    losses, maes = [], []
    for batch in loader:
        cont = batch.cont.to(device); pdg_idx = batch.pdg_idx.to(device)
        mask = batch.mask.to(device); sqrtsNN = batch.sqrtsNN.to(device)
        y = batch.b.to(device); z = batch.centrality_bin.to(device)
        out = model(cont, pdg_idx, mask, sqrtsNN)
        l_reg = evidential_loss(y, out["mu"], out["nu"], out["alpha"], out["beta"], coeff_reg=coeff_reg)
        l_cls = ce(out["logits"], z)
        losses.append(float(l_reg + w_cls * l_cls))
        maes.append(float((out["mu"] - y).abs().mean()))
    return float(np.mean(losses)), float(np.mean(maes))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=["deepsets", "settransformer"])
    p.add_argument("--cache", required=True, type=Path,
                   help="Path to the pre-padded cache built by build_padded_cache.py")
    p.add_argument("--inputs", required=True, nargs="+", type=Path,
                   help="Original ingested HDF5s (used only for per-energy output file naming)")
    p.add_argument("--output-tag", default="baseline_v1")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--coeff-reg", type=float, default=1e-2)
    p.add_argument("--w-cls", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-centrality-bins", type=int, default=9)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--num-threads", type=int, default=4,
                   help="Cap on intra-op thread count (set OMP/MKL/etc.). Already applied at module load.")
    args = p.parse_args()

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"arch: {args.arch}  device: {device}  threads: {args.num_threads}")

    ds = CachedParticleDataset(args.cache.expanduser())
    n_events = len(ds)
    energy_sizes = ds.energy_sizes
    print(f"Cached dataset: {n_events:,} events across {len(energy_sizes)} energies ({energy_sizes})")

    splits = split_indices(n_events, seed=args.seed)
    print(f"Split: train={len(splits['train']):,}, val={len(splits['val']):,}, test={len(splits['test']):,}")

    dl_train = DataLoader(Subset(ds, splits["train"].tolist()), batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers, collate_fn=collate_cached,
                          persistent_workers=(args.num_workers > 0))
    dl_val = DataLoader(Subset(ds, splits["val"].tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, collate_fn=collate_cached,
                        persistent_workers=(args.num_workers > 0))

    model = build_model(args.arch, args.n_centrality_bins).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {n_params:,} parameters")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    ckpt_path = Path("checkpoints") / f"{args.arch}_{args.output_tag}" / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf"); since_best = 0
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        t_ep = time.time()
        tr = train_one_epoch(model, dl_train, opt, ce, device, args.coeff_reg, args.w_cls)
        va, mae = eval_split(model, dl_val, ce, device, args.coeff_reg, args.w_cls)
        dt = time.time() - t_ep
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va, "val_b_mae": mae, "sec": dt})
        improved = va < best_val
        if improved:
            best_val = va; since_best = 0
            torch.save({
                "state_dict": model.state_dict(),
                "arch": args.arch,
                "config": asdict(model.cfg),
                "epoch": epoch,
            }, ckpt_path)
        else:
            since_best += 1
        flag = "*" if improved else " "
        print(f"  epoch {epoch:3d}  {dt:5.1f}s  train {tr:.4f}  val {va:.4f}  val_b_MAE {mae:.3f} fm  {flag}", flush=True)
        if since_best >= args.patience:
            print(f"  early stop at epoch {epoch}")
            break
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f} s, best val loss {best_val:.4f}")

    # Predict on every event and emit per-energy HDF5s.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    cum = np.cumsum([0] + list(energy_sizes))
    is_train = np.zeros(n_events, dtype=bool); is_train[splits["train"]] = True
    is_val = np.zeros(n_events, dtype=bool); is_val[splits["val"]] = True
    is_test = np.zeros(n_events, dtype=bool); is_test[splits["test"]] = True

    out_root = Path("data/processed") / args.arch / args.output_tag
    out_root.mkdir(parents=True, exist_ok=True)

    # Sequential pass over the whole cache to gather predictions in index order.
    dl_full = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collate_cached,
                         persistent_workers=(args.num_workers > 0))
    mus, nus, alphas, betas, logits_all, b_true = [], [], [], [], [], []
    with torch.no_grad():
        for batch in dl_full:
            cont = batch.cont.to(device); pdg_idx = batch.pdg_idx.to(device)
            mask = batch.mask.to(device); sqrtsNN = batch.sqrtsNN.to(device)
            out = model(cont, pdg_idx, mask, sqrtsNN)
            mus.append(out["mu"].cpu().numpy()); nus.append(out["nu"].cpu().numpy())
            alphas.append(out["alpha"].cpu().numpy()); betas.append(out["beta"].cpu().numpy())
            logits_all.append(out["logits"].cpu().numpy())
            b_true.append(batch.b.numpy())
    mu = np.concatenate(mus); nu = np.concatenate(nus)
    alpha = np.concatenate(alphas); beta = np.concatenate(betas)
    logits = np.concatenate(logits_all); b_true_np = np.concatenate(b_true)
    moments = nig_to_moments(torch.from_numpy(mu), torch.from_numpy(nu),
                             torch.from_numpy(alpha), torch.from_numpy(beta))

    print()
    for e_idx, in_path in enumerate(args.inputs):
        lo, hi = int(cum[e_idx]), int(cum[e_idx + 1])
        out_path = out_root / f"{args.arch}_pred_{in_path.stem}.h5"
        with h5py.File(out_path, "w") as h:
            h.create_dataset("b_pred", data=mu[lo:hi].astype(np.float32))
            h.create_dataset("nu", data=nu[lo:hi].astype(np.float32))
            h.create_dataset("alpha", data=alpha[lo:hi].astype(np.float32))
            h.create_dataset("beta", data=beta[lo:hi].astype(np.float32))
            h.create_dataset("total_var", data=moments["total_var"][lo:hi].numpy().astype(np.float32))
            h.create_dataset("aleatoric_var", data=moments["aleatoric_var"][lo:hi].numpy().astype(np.float32))
            h.create_dataset("epistemic_var", data=moments["epistemic_var"][lo:hi].numpy().astype(np.float32))
            h.create_dataset("centrality_bin", data=np.argmax(logits[lo:hi], axis=1).astype(np.int16))
            h.create_dataset("logits", data=logits[lo:hi].astype(np.float32))
            h.create_dataset("is_train", data=is_train[lo:hi])
            h.create_dataset("is_val", data=is_val[lo:hi])
            h.create_dataset("is_test", data=is_test[lo:hi])
            h.attrs["n_events"] = hi - lo
            h.attrs["source_h5"] = str(in_path)
            h.attrs["checkpoint"] = str(ckpt_path)
            h.attrs["arch"] = args.arch
        mae_test = float(np.abs(mu[lo:hi][is_test[lo:hi]] - b_true_np[lo:hi][is_test[lo:hi]]).mean())
        print(f"  energy {e_idx}: test b-MAE = {mae_test:.3f} fm  (N_test = {int(is_test[lo:hi].sum()):,})")

    with open(out_root / "train_metrics.json", "w") as f:
        json.dump({"args": vars(args) | {"inputs": [str(p) for p in args.inputs],
                                          "cache": str(args.cache)},
                   "metrics": {"history": history, "best_val_loss": best_val,
                               "elapsed_sec": elapsed, "n_parameters": n_params}},
                  f, indent=2, default=str)
    print(f"\nWrote predictions to {out_root}/")


if __name__ == "__main__":
    main()
