"""Train one of {mlp_pool, deepsets, set_transformer, efn} from the lab-frame cache.

All four architectures share the same training loop, loss, dataloader, and
prediction-output schema. The only thing that varies between runs is --arch.

Outputs:
  checkpoints/<arch>_<tag>/best.pt
  data/processed/<arch>/<tag>/<arch>_pred_auau_<E>GeV.h5
  data/processed/<arch>/<tag>/train_metrics.json

Usage (single seed, throttled to 4 cores and low priority):
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 python3 scripts/train_lab.py \\
        --arch deepsets \\
        --cache data/processed/cached/all_lab_truth.h5 \\
        --truth-preds data/processed/truth_lab/truth_auau_{3p2,3p5,3p9,4p5}GeV.h5 \\
        --output-tag v1_truth \\
        --device mps --batch-size 256 --epochs 18 --num-workers 2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path


def _set_thread_caps(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(n))


# Set thread caps before importing torch so the math libraries pick them up.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--num-threads", type=int, default=4)
_pre_args, _ = _pre_parser.parse_known_args()
_set_thread_caps(_pre_args.num_threads)

import h5py    # noqa: E402
import numpy as np    # noqa: E402
import torch    # noqa: E402
from torch import nn    # noqa: E402
from torch.utils.data import DataLoader, Subset    # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.lab_cached_dataset import (  # noqa: E402
    EVENT_FEATURE_NAMES, LabCachedDataset, apply_phi_rotation, collate_lab,
)
from src.losses.evidential import evidential_loss, nig_to_moments  # noqa: E402
from src.models.deepsets import DeepSets, DeepSetsConfig  # noqa: E402
from src.models.efn import EFN, EFNConfig  # noqa: E402
from src.models.mlp_pool import MLPPool, MLPPoolConfig  # noqa: E402
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


def build_model(arch: str, n_bins: int) -> nn.Module:
    if arch == "mlp_pool":
        return MLPPool(MLPPoolConfig(n_centrality_bins=n_bins))
    if arch == "deepsets":
        return DeepSets(DeepSetsConfig(n_centrality_bins=n_bins))
    if arch == "set_transformer":
        return SetTransformer(SetTransformerConfig(n_centrality_bins=n_bins))
    if arch == "efn":
        return EFN(EFNConfig(n_centrality_bins=n_bins))
    raise ValueError(f"Unknown arch: {arch!r}")


def normalize_event_feats(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (values - mean) / std


def train_one_epoch(model, loader, opt, ce, device, coeff_reg, w_cls, ef_mean, ef_std, augment_phi):
    model.train()
    losses = []
    for batch in loader:
        cont = batch.cont.to(device); mask = batch.mask.to(device)
        ef = normalize_event_feats(batch.event_feats.to(device), ef_mean, ef_std)
        y = batch.b.to(device); z = batch.centrality_bin.to(device)

        if augment_phi:
            cont = apply_phi_rotation(cont, mask)

        out = model(cont, mask, ef)
        l_reg = evidential_loss(y, out["mu"], out["nu"], out["alpha"], out["beta"], coeff_reg=coeff_reg)
        l_cls = ce(out["logits"], z)
        loss = l_reg + w_cls * l_cls
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.no_grad()
def eval_split(model, loader, ce, device, coeff_reg, w_cls, ef_mean, ef_std):
    model.eval()
    losses, maes = [], []
    for batch in loader:
        cont = batch.cont.to(device); mask = batch.mask.to(device)
        ef = normalize_event_feats(batch.event_feats.to(device), ef_mean, ef_std)
        y = batch.b.to(device); z = batch.centrality_bin.to(device)
        out = model(cont, mask, ef)
        l_reg = evidential_loss(y, out["mu"], out["nu"], out["alpha"], out["beta"], coeff_reg=coeff_reg)
        l_cls = ce(out["logits"], z)
        losses.append(float(l_reg + w_cls * l_cls))
        maes.append(float((out["mu"] - y).abs().mean()))
    return float(np.mean(losses)), float(np.mean(maes))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=["mlp_pool", "deepsets", "set_transformer", "efn"])
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--truth-preds", required=True, nargs="+", type=Path,
                   help="Per-energy truth-baseline prediction files (for centrality labels)")
    p.add_argument("--output-tag", default="v1_truth")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--coeff-reg", type=float, default=1e-2)
    p.add_argument("--w-cls", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-centrality-bins", type=int, default=7,
                   help="Matches the truth baseline's bin count (Kimelman thesis Tab. 4.1).")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--no-phi-aug", action="store_true", help="Disable φ augmentation")
    args = p.parse_args()

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"arch: {args.arch}  device: {device}  threads: {args.num_threads}  phi_aug: {not args.no_phi_aug}")

    ds = LabCachedDataset(args.cache.expanduser(), [p.expanduser() for p in args.truth_preds])
    n_events = len(ds)
    print(f"Cached dataset: {n_events:,} events, energies={ds.sqrts_per_energy}, sizes={ds.energy_sizes}")

    splits = split_indices(n_events, seed=args.seed)
    print(f"Split: train={len(splits['train']):,}, val={len(splits['val']):,}, test={len(splits['test']):,}")

    # Compute event-feature normalisation from the train split only (no test/val peek).
    with h5py.File(args.cache, "r") as h:
        ef_arr = np.column_stack([
            h["sqrtsNN"][:], h["mult_lab"][:], h["mean_pT_lab"][:], h["total_pT_lab"][:]
        ]).astype(np.float32)
    ef_train = ef_arr[splits["train"]]
    ef_mean_np = ef_train.mean(axis=0); ef_std_np = ef_train.std(axis=0)
    ef_std_np = np.where(ef_std_np > 1e-6, ef_std_np, 1.0)
    ef_mean = torch.from_numpy(ef_mean_np).to(device)
    ef_std = torch.from_numpy(ef_std_np).to(device)
    print(f"event-feat mean: {ef_mean_np}, std: {ef_std_np}")

    dl_train = DataLoader(Subset(ds, splits["train"].tolist()), batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers, collate_fn=collate_lab,
                          persistent_workers=(args.num_workers > 0))
    dl_val = DataLoader(Subset(ds, splits["val"].tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, collate_fn=collate_lab,
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
        tr = train_one_epoch(model, dl_train, opt, ce, device,
                             args.coeff_reg, args.w_cls, ef_mean, ef_std,
                             augment_phi=not args.no_phi_aug)
        va, mae = eval_split(model, dl_val, ce, device, args.coeff_reg, args.w_cls, ef_mean, ef_std)
        dt = time.time() - t_ep
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va, "val_b_mae": mae, "sec": dt})
        improved = va < best_val
        if improved:
            best_val = va; since_best = 0
            torch.save({
                "state_dict": model.state_dict(),
                "arch": args.arch,
                "config": asdict(model.cfg),
                "ef_mean": ef_mean_np.tolist(),
                "ef_std": ef_std_np.tolist(),
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

    # Final predictions on every event.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    cum = np.cumsum([0] + ds.energy_sizes)
    is_train = np.zeros(n_events, dtype=bool); is_train[splits["train"]] = True
    is_val = np.zeros(n_events, dtype=bool); is_val[splits["val"]] = True
    is_test = np.zeros(n_events, dtype=bool); is_test[splits["test"]] = True

    out_root = Path("data/processed") / args.arch / args.output_tag
    out_root.mkdir(parents=True, exist_ok=True)

    dl_full = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collate_lab,
                         persistent_workers=(args.num_workers > 0))
    mus, nus, alphas, betas, logits_all, b_true_all = [], [], [], [], [], []
    with torch.no_grad():
        for batch in dl_full:
            cont = batch.cont.to(device); mask = batch.mask.to(device)
            ef = normalize_event_feats(batch.event_feats.to(device), ef_mean, ef_std)
            out = model(cont, mask, ef)
            mus.append(out["mu"].cpu().numpy()); nus.append(out["nu"].cpu().numpy())
            alphas.append(out["alpha"].cpu().numpy()); betas.append(out["beta"].cpu().numpy())
            logits_all.append(out["logits"].cpu().numpy())
            b_true_all.append(batch.b.numpy())
    mu = np.concatenate(mus); nu = np.concatenate(nus)
    alpha = np.concatenate(alphas); beta = np.concatenate(betas)
    logits = np.concatenate(logits_all); b_true = np.concatenate(b_true_all)
    moments = nig_to_moments(torch.from_numpy(mu), torch.from_numpy(nu),
                             torch.from_numpy(alpha), torch.from_numpy(beta))

    print()
    for e_idx, sqrtsNN in enumerate(ds.sqrts_per_energy):
        lo, hi = int(cum[e_idx]), int(cum[e_idx + 1])
        tag = f"{sqrtsNN:.1f}".replace(".", "p")
        out_path = out_root / f"{args.arch}_pred_auau_{tag}GeV.h5"
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
            h.attrs["sqrtsNN"] = sqrtsNN
            h.attrs["n_events"] = hi - lo
            h.attrs["arch"] = args.arch
            h.attrs["checkpoint"] = str(ckpt_path)
        mae_test = float(np.abs(mu[lo:hi][is_test[lo:hi]] - b_true[lo:hi][is_test[lo:hi]]).mean())
        print(f"  {sqrtsNN} GeV: test b-MAE = {mae_test:.3f} fm  (N_test = {int(is_test[lo:hi].sum()):,})")

    with open(out_root / "train_metrics.json", "w") as f:
        json.dump({"args": vars(args) | {"cache": str(args.cache),
                                          "truth_preds": [str(p) for p in args.truth_preds]},
                   "metrics": {"history": history, "best_val_loss": best_val,
                               "elapsed_sec": elapsed, "n_parameters": n_params}},
                  f, indent=2, default=str)
    print(f"\nWrote predictions to {out_root}/")


if __name__ == "__main__":
    main()
