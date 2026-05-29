"""Train a permutation-invariant model (DeepSets or Set Transformer) jointly on all 4 energies.

Both architectures share the same training loop, loss (evidential + cross-entropy),
data loader, and prediction-output schema — the only choice the user makes is
--arch {deepsets, settransformer}.

Outputs:
  checkpoints/<arch>_<tag>/best.pt
  data/processed/<arch>/<tag>/<arch>_pred_<event-stem>.h5
  data/processed/<arch>/<tag>/train_metrics.json

Usage:
    python scripts/train_set_model.py --arch deepsets \\
        --inputs data/processed/auau_{3p2,3p5,3p9,4p5}GeV.h5 \\
        --centrality-preds data/processed/truth/smash/truth_auau_{3p2,3p5,3p9,4p5}GeV.h5 \\
        --output-tag baseline_v1
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.particle_dataset import ParticleEventDataset, collate_particles  # noqa: E402
from src.losses.evidential import evidential_loss, nig_to_moments  # noqa: E402
from src.models.deepsets import DeepSets, DeepSetsConfig  # noqa: E402
from src.models.set_transformer import SetTransformer, SetTransformerConfig  # noqa: E402


def device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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


def build_model(arch: str, n_centrality_bins: int) -> nn.Module:
    if arch == "deepsets":
        return DeepSets(DeepSetsConfig(n_centrality_bins=n_centrality_bins))
    if arch == "settransformer":
        return SetTransformer(SetTransformerConfig(n_centrality_bins=n_centrality_bins))
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
    p.add_argument("--inputs", required=True, nargs="+", type=Path)
    p.add_argument("--centrality-preds", required=True, nargs="+", type=Path,
                   help="Per-energy HDF5s containing the centrality_bin label (e.g., truth baseline)")
    p.add_argument("--output-tag", default="baseline_v1")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--coeff-reg", type=float, default=1e-2)
    p.add_argument("--w-cls", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-centrality-bins", type=int, default=9)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()

    if len(args.inputs) != len(args.centrality_preds):
        raise SystemExit("--inputs and --centrality-preds must have the same length and order")

    torch.manual_seed(args.seed)
    device = device_str()
    print(f"arch: {args.arch}, device: {device}")

    # Per-energy datasets — kept separate so HDF5 handles live in worker processes safely.
    per_energy = [ParticleEventDataset(ip.expanduser(), cp.expanduser())
                  for ip, cp in zip(args.inputs, args.centrality_preds)]
    energy_sizes = [len(d) for d in per_energy]
    full = ConcatDataset(per_energy)
    n_events = len(full)
    print(f"Joint dataset: {n_events:,} events across {len(per_energy)} energies")

    splits = split_indices(n_events, seed=args.seed)
    print(f"Split: train={len(splits['train']):,}, val={len(splits['val']):,}, test={len(splits['test']):,}")

    dl_train = DataLoader(Subset(full, splits["train"].tolist()), batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers, collate_fn=collate_particles,
                          persistent_workers=(args.num_workers > 0))
    dl_val = DataLoader(Subset(full, splits["val"].tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, collate_fn=collate_particles,
                        persistent_workers=(args.num_workers > 0))

    model = build_model(args.arch, args.n_centrality_bins).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {n_params:,} parameters")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss(ignore_index=-1)  # -1 = beyond-80% events (out of centrality scope)

    ckpt_path = Path("checkpoints") / f"{args.arch}_{args.output_tag}" / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf"); since_best = 0
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        tr = train_one_epoch(model, dl_train, opt, ce, device, args.coeff_reg, args.w_cls)
        va, mae = eval_split(model, dl_val, ce, device, args.coeff_reg, args.w_cls)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va, "val_b_mae": mae})
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
        print(f"  epoch {epoch:3d}  train {tr:.4f}  val {va:.4f}  val_b_MAE {mae:.3f} fm  {flag}")
        if since_best >= args.patience:
            print(f"  early stop at epoch {epoch}")
            break
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f} s, best val loss {best_val:.4f}")

    # Predict on every event and emit per-energy HDF5s matching the MLP schema.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    # Pre-compute event-index → energy mapping using ConcatDataset's cumulative_sizes.
    cum = np.cumsum([0] + energy_sizes)
    is_train = np.zeros(n_events, dtype=bool); is_train[splits["train"]] = True
    is_val = np.zeros(n_events, dtype=bool); is_val[splits["val"]] = True
    is_test = np.zeros(n_events, dtype=bool); is_test[splits["test"]] = True

    out_root = Path("data/processed") / args.arch / args.output_tag
    out_root.mkdir(parents=True, exist_ok=True)

    # Predict per energy by re-iterating each individual dataset (avoids re-padding across energies).
    print()
    with torch.no_grad():
        for e_idx, ds in enumerate(per_energy):
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_particles,
                            persistent_workers=(args.num_workers > 0))
            mus, nus, alphas, betas, logits_all, b_true = [], [], [], [], [], []
            for batch in dl:
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

            # Slice the global flags down to this energy's contiguous block.
            lo, hi = int(cum[e_idx]), int(cum[e_idx + 1])
            it = is_train[lo:hi]; iv = is_val[lo:hi]; ite = is_test[lo:hi]

            out_path = out_root / f"{args.arch}_pred_{args.inputs[e_idx].stem}.h5"
            with h5py.File(out_path, "w") as h:
                h.create_dataset("b_pred", data=mu.astype(np.float32))
                h.create_dataset("nu", data=nu.astype(np.float32))
                h.create_dataset("alpha", data=alpha.astype(np.float32))
                h.create_dataset("beta", data=beta.astype(np.float32))
                h.create_dataset("total_var", data=moments["total_var"].numpy().astype(np.float32))
                h.create_dataset("aleatoric_var", data=moments["aleatoric_var"].numpy().astype(np.float32))
                h.create_dataset("epistemic_var", data=moments["epistemic_var"].numpy().astype(np.float32))
                h.create_dataset("centrality_bin", data=np.argmax(logits, axis=1).astype(np.int16))
                h.create_dataset("logits", data=logits.astype(np.float32))
                h.create_dataset("is_train", data=it); h.create_dataset("is_val", data=iv)
                h.create_dataset("is_test", data=ite)
                h.attrs["sqrtsNN"] = ds.sqrtsNN
                h.attrs["n_events"] = ds.n_events
                h.attrs["source_h5"] = str(args.inputs[e_idx])
                h.attrs["checkpoint"] = str(ckpt_path)
                h.attrs["arch"] = args.arch

            mae_test = float(np.abs(mu[ite] - b_true_np[ite]).mean())
            print(f"  {ds.sqrtsNN} GeV: test b-MAE = {mae_test:.3f} fm  (N_test = {int(ite.sum()):,})")

    with open(out_root / "train_metrics.json", "w") as f:
        json.dump({"args": vars(args) | {"inputs": [str(p) for p in args.inputs],
                                          "centrality_preds": [str(p) for p in args.centrality_preds]},
                   "metrics": {"history": history, "best_val_loss": best_val, "elapsed_sec": elapsed,
                               "n_parameters": n_params}},
                  f, indent=2, default=str)
    print(f"\nWrote predictions to {out_root}/")


if __name__ == "__main__":
    main()
