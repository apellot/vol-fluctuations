"""Train the feature-vector MLP baseline jointly on all four FXT energies.

Per CLAUDE.md Task 4 and the ml-modeling agent spec:
  * Inputs: ~10 event-level scalar features (src/data/features.py).
  * Architecture: 3 hidden FC layers, ReLU, dropout 0.1, hidden 128.
  * Heads: NIG evidential (b regression) + softmax (centrality percentile).
  * Combined loss = w_reg · evidential + w_cls · cross-entropy. Defaults 1:1.
  * 80/10/10 split per energy, with a global seed so the same events end up in
    train/val/test across re-runs.
  * Joint training across all 4 energies, with √sNN as a feature.

Outputs:
  - checkpoints/mlp_<tag>/best.pt        — model weights at min val loss.
  - data/processed/mlp/mlp_pred_<tag>.h5 — per-event predictions on the full set
    with an is_test flag; matches the schema of the classical baseline outputs.
  - data/processed/mlp/<tag>_metrics.json — bookkeeping for the run.

Usage:
    python scripts/train_mlp.py \\
        --inputs data/processed/auau_{3p2,3p5,3p9,4p5}GeV.h5 \\
        --truth-preds data/processed/truth/smash/truth_auau_{3p2,3p5,3p9,4p5}GeV.h5 \\
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
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.features import N_FEATURES, EventDataset, load_features, stack_energies  # noqa: E402
from src.losses.evidential import evidential_loss, nig_to_moments  # noqa: E402
from src.models.mlp import MLPConfig, MLPHead, count_parameters  # noqa: E402


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


def standardize(features: np.ndarray, idx_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize each feature column using train-set statistics only.

    Standard practice — without it, features with very different scales (multiplicity
    ~100 vs ⟨pT⟩ ~0.5) make optimisation badly conditioned and the NIG head
    sensitive to initialisation.
    """
    mean = features[idx_train].mean(axis=0).astype(np.float32)
    std = features[idx_train].std(axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0)  # don't divide by zero on constant features
    return (features - mean) / std, mean, std


def device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train(
    dataset: EventDataset,
    splits: dict[str, np.ndarray],
    cfg: MLPConfig,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    coeff_reg: float,
    w_cls: float,
    patience: int,
    seed: int,
    ckpt_path: Path,
) -> dict:
    torch.manual_seed(seed)
    device = device_str()
    print(f"  device: {device}")

    # Standardize features using train-set statistics; save the scaler for inference.
    x_norm, x_mean, x_std = standardize(dataset.features, splits["train"])
    y = dataset.b.astype(np.float32)
    z = dataset.centrality_bin.astype(np.int64)

    def loader_for(split: str, shuffle: bool) -> DataLoader:
        idx = splits[split]
        ds = TensorDataset(
            torch.from_numpy(x_norm[idx]),
            torch.from_numpy(y[idx]),
            torch.from_numpy(z[idx]),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    dl_train = loader_for("train", shuffle=True)
    dl_val = loader_for("val", shuffle=False)

    model = MLPHead(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    print(f"  model: {count_parameters(model):,} parameters")

    best_val = float("inf")
    epochs_since_best = 0
    history = []
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb, zb in dl_train:
            xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
            out = model(xb)
            l_reg = evidential_loss(yb, out["mu"], out["nu"], out["alpha"], out["beta"], coeff_reg=coeff_reg)
            l_cls = ce(out["logits"], zb)
            loss = l_reg + w_cls * l_cls
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach()))

        # Validation pass — no grad, fixed-seed dropout off.
        model.eval()
        val_losses, val_b_mae = [], []
        with torch.no_grad():
            for xb, yb, zb in dl_val:
                xb, yb, zb = xb.to(device), yb.to(device), zb.to(device)
                out = model(xb)
                l_reg = evidential_loss(yb, out["mu"], out["nu"], out["alpha"], out["beta"], coeff_reg=coeff_reg)
                l_cls = ce(out["logits"], zb)
                val_losses.append(float(l_reg + w_cls * l_cls))
                val_b_mae.append(float((out["mu"] - yb).abs().mean()))
        val_loss = float(np.mean(val_losses))
        val_mae = float(np.mean(val_b_mae))

        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)),
                        "val_loss": val_loss, "val_b_mae": val_mae})
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            epochs_since_best = 0
            torch.save({
                "state_dict": model.state_dict(),
                "cfg": asdict(cfg),
                "x_mean": x_mean.tolist(),
                "x_std": x_std.tolist(),
                "feature_names": dataset.feature_names,
                "epoch": epoch,
            }, ckpt_path)
        else:
            epochs_since_best += 1

        flag = "*" if improved else " "
        print(f"  epoch {epoch:3d}  train {history[-1]['train_loss']:.4f}  "
              f"val {val_loss:.4f}  val_b_MAE {val_mae:.3f} fm  {flag}")

        if epochs_since_best >= patience:
            print(f"  early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break

    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f} s, best val loss {best_val:.4f}")
    return {"history": history, "best_val_loss": best_val, "elapsed_sec": elapsed}


def predict_full(ckpt_path: Path, dataset: EventDataset, splits: dict[str, np.ndarray]) -> dict:
    """Run the best-checkpoint model on every event and return per-event arrays."""
    device = device_str()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = MLPConfig(**ckpt["cfg"])
    model = MLPHead(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    x = ((dataset.features - x_mean) / np.where(x_std > 1e-6, x_std, 1.0)).astype(np.float32)

    # Batch through the full set to keep memory bounded.
    mus, nus, alphas, betas, logits_all = [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            xb = torch.from_numpy(x[start:start + 4096]).to(device)
            out = model(xb)
            mus.append(out["mu"].cpu().numpy())
            nus.append(out["nu"].cpu().numpy())
            alphas.append(out["alpha"].cpu().numpy())
            betas.append(out["beta"].cpu().numpy())
            logits_all.append(out["logits"].cpu().numpy())
    mu = np.concatenate(mus); nu = np.concatenate(nus)
    alpha = np.concatenate(alphas); beta = np.concatenate(betas)
    logits = np.concatenate(logits_all)
    pred_bin = np.argmax(logits, axis=1).astype(np.int16)

    is_test = np.zeros(len(x), dtype=bool); is_test[splits["test"]] = True
    is_val = np.zeros(len(x), dtype=bool); is_val[splits["val"]] = True
    is_train = np.zeros(len(x), dtype=bool); is_train[splits["train"]] = True
    return {
        "mu": mu, "nu": nu, "alpha": alpha, "beta": beta,
        "logits": logits, "pred_bin": pred_bin,
        "is_train": is_train, "is_val": is_val, "is_test": is_test,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True, nargs="+", type=Path,
                   help="Ingested SMASH HDF5 files (one per energy)")
    p.add_argument("--truth-preds", required=True, nargs="+", type=Path,
                   help="Truth-tuned baseline prediction HDF5s (one per energy, matching --inputs order)")
    p.add_argument("--output-tag", default="baseline_v1",
                   help="Tag for output directories (checkpoints/mlp_<tag>, data/processed/mlp/<tag>/)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--coeff-reg", type=float, default=1e-2, help="Amini NIG regularization λ")
    p.add_argument("--w-cls", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-centrality-bins", type=int, default=9)
    args = p.parse_args()

    if len(args.inputs) != len(args.truth_preds):
        raise SystemExit("--inputs and --truth-preds must have the same length and order")

    # Load per-energy features; concatenate into one joint training pool.
    per_energy = [load_features(in_p.expanduser(), pred_p.expanduser())
                  for in_p, pred_p in zip(args.inputs, args.truth_preds)]
    dataset = stack_energies(per_energy)
    print(f"Joint dataset: {dataset.n_events:,} events, {dataset.features.shape[1]} features")

    splits = split_indices(dataset.n_events, seed=args.seed)
    print(f"Split: train={len(splits['train']):,}, val={len(splits['val']):,}, test={len(splits['test']):,}")

    cfg = MLPConfig(n_features=N_FEATURES, n_centrality_bins=args.n_centrality_bins)
    ckpt_path = Path("checkpoints") / f"mlp_{args.output_tag}" / "best.pt"

    metrics = train(
        dataset, splits, cfg,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        coeff_reg=args.coeff_reg, w_cls=args.w_cls, patience=args.patience,
        seed=args.seed, ckpt_path=ckpt_path,
    )

    # Predict on the full event set and save predictions per energy in the same
    # shape as the classical baselines so the evaluator can treat all three uniformly.
    preds = predict_full(ckpt_path, dataset, splits)
    out_root = Path("data/processed/mlp") / args.output_tag
    out_root.mkdir(parents=True, exist_ok=True)

    # Re-split predictions per source energy file.
    offset = 0
    for in_p, ed in zip(args.inputs, per_energy):
        end = offset + ed.n_events
        slc = slice(offset, end)
        out_path = out_root / f"mlp_pred_{in_p.stem}.h5"
        moments = nig_to_moments(
            torch.from_numpy(preds["mu"][slc]),
            torch.from_numpy(preds["nu"][slc]),
            torch.from_numpy(preds["alpha"][slc]),
            torch.from_numpy(preds["beta"][slc]),
        )
        with h5py.File(out_path, "w") as h:
            h.create_dataset("b_pred", data=preds["mu"][slc].astype(np.float32))
            h.create_dataset("nu", data=preds["nu"][slc].astype(np.float32))
            h.create_dataset("alpha", data=preds["alpha"][slc].astype(np.float32))
            h.create_dataset("beta", data=preds["beta"][slc].astype(np.float32))
            h.create_dataset("total_var", data=moments["total_var"].numpy().astype(np.float32))
            h.create_dataset("aleatoric_var", data=moments["aleatoric_var"].numpy().astype(np.float32))
            h.create_dataset("epistemic_var", data=moments["epistemic_var"].numpy().astype(np.float32))
            h.create_dataset("centrality_bin", data=preds["pred_bin"][slc])
            h.create_dataset("logits", data=preds["logits"][slc].astype(np.float32))
            h.create_dataset("is_train", data=preds["is_train"][slc])
            h.create_dataset("is_val", data=preds["is_val"][slc])
            h.create_dataset("is_test", data=preds["is_test"][slc])
            h.attrs["sqrtsNN"] = ed.sqrtsNN
            h.attrs["n_events"] = ed.n_events
            h.attrs["source_h5"] = str(in_p)
            h.attrs["checkpoint"] = str(ckpt_path)
            h.attrs["feature_names"] = list(ed.feature_names)
        offset = end
        # Quick eval print: test-set b-MAE per energy.
        mu_e = preds["mu"][slc]
        b_true_e = ed.b
        test_mask = preds["is_test"][slc]
        mae_test = float(np.abs(mu_e[test_mask] - b_true_e[test_mask]).mean())
        print(f"  {ed.sqrtsNN} GeV: test b-MAE = {mae_test:.3f} fm  (N_test = {test_mask.sum():,})")

    with open(out_root / "train_metrics.json", "w") as f:
        json.dump({"args": vars(args) | {"inputs": [str(p) for p in args.inputs],
                                          "truth_preds": [str(p) for p in args.truth_preds]},
                   "metrics": metrics}, f, indent=2, default=str)
    print(f"\nWrote predictions to {out_root}/ and metrics to {out_root/'train_metrics.json'}")


if __name__ == "__main__":
    main()
