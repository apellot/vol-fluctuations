"""Truth-tuned percentile centrality baseline (an oracle for what classical
multiplicity binning can achieve on this observable).

This is NOT the STAR experimental method — STAR uses Glauber-MC fit to the
measured RefMult distribution and reads b off the model, because real
experiments have no access to truth b. Our truth-tuned baseline does have access
to truth b, and reports the *SMASH-truth* mean b per percentile bin. It therefore
represents the **upper bound on what any classical method binning on this single
observable can extract** — beating it with ML is the meaningful claim, because
no purely multiplicity-based classical method can do better.

Mechanically: sort events by a charged-multiplicity observable (the RefMult
equivalent), bin into percentile classes (0–5 %, 5–10 %, 10–20 %, …), and report
the bin each event falls into. The "predicted" b for each bin is the mean truth
b of training-half events that landed there — the train/test split keeps the
held-out half clean of any peek at calibration.

The observable here is `mult_eta05` (charged multiplicity in CM-frame |η|<0.5),
already stored by the ingestion pipeline. The choice is deliberate — total
`nparticles` is contaminated by the 394 Au nucleons SMASH dumps as spectators,
which floors it at 394 for peripheral events and makes it useless as a centrality
probe in that regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# STAR FXT centrality bin scheme per Kimelman thesis Table 4.1 (p. 37): seven
# bins with widening at peripheral to keep per-bin statistics reasonable in the
# low-multiplicity regime where the FXTMult observable resolves slowly.
DEFAULT_BIN_EDGES = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80])
DEFAULT_BIN_LABELS = ["0-5%", "5-10%", "10-20%", "20-30%", "30-40%",
                      "40-60%", "60-80%"]


@dataclass
class TruthCalibration:
    """A fitted RefMult mapping for one energy.

    `mult_thresholds` are descending charged-multiplicity values that demarcate
    the bin boundaries: the most-central bin is `mult >= mult_thresholds[0]`,
    the next is `mult_thresholds[1] <= mult < mult_thresholds[0]`, etc.
    `b_means` is the mean truth b in each bin, evaluated on the training data
    used to define the thresholds.
    """
    bin_edges: np.ndarray         # percentile fractions of length n_bins+1
    bin_labels: list[str]         # human-readable labels, length n_bins
    mult_thresholds: np.ndarray   # descending multiplicity boundaries, length n_bins
    b_means: np.ndarray           # mean truth b in each bin, length n_bins
    b_stds: np.ndarray            # std of truth b in each bin (for the resolution diagnostic)
    n_train: int                  # number of training events used


def fit(mult_train: np.ndarray, b_train: np.ndarray,
        bin_edges: np.ndarray = DEFAULT_BIN_EDGES,
        bin_labels: list[str] | None = None) -> TruthCalibration:
    """Calibrate the multiplicity-to-centrality mapping from training events.

    Translates percentile edges in [0, 1) into multiplicity thresholds via
    np.quantile. Centrality goes from MOST central (highest mult) to LEAST
    central (lowest mult); percentile fractions are conventionally written as
    "0 % = most central." We therefore use (1 − percentile_edge) when calling
    np.quantile so the thresholds come out in descending order.
    """
    if bin_labels is None:
        bin_labels = list(DEFAULT_BIN_LABELS)

    n_bins = len(bin_edges) - 1
    if len(bin_labels) != n_bins:
        raise ValueError(f"bin_labels length {len(bin_labels)} != n_bins {n_bins}")

    # Multiplicity quantiles: bin_edges[1] = 0.05 corresponds to "the top 5 %",
    # i.e. multiplicity ≥ the 95th percentile of the training distribution.
    upper_quantiles = 1.0 - bin_edges[:-1]   # length n_bins, descending from 1.0
    lower_quantiles = 1.0 - bin_edges[1:]    # length n_bins, descending

    # mult_thresholds[i] = lower edge of bin i (in multiplicity space).
    mult_thresholds = np.quantile(mult_train, lower_quantiles)

    # Assign each training event a bin index and compute (mean, std) of truth b per bin.
    train_bin = assign_bins(mult_train, mult_thresholds)
    b_means = np.zeros(n_bins, dtype=np.float64)
    b_stds = np.zeros(n_bins, dtype=np.float64)
    for k in range(n_bins):
        sel = train_bin == k
        if sel.sum() == 0:
            b_means[k] = np.nan
            b_stds[k] = np.nan
        else:
            b_means[k] = float(b_train[sel].mean())
            b_stds[k] = float(b_train[sel].std())

    return TruthCalibration(
        bin_edges=bin_edges,
        bin_labels=bin_labels,
        mult_thresholds=mult_thresholds,
        b_means=b_means,
        b_stds=b_stds,
        n_train=int(mult_train.size),
    )


def assign_bins(mult: np.ndarray, mult_thresholds: np.ndarray) -> np.ndarray:
    """Vectorized assignment of multiplicity values to centrality bin indices.

    Bin 0 is the most central (largest mult); bin n_bins-1 is the least central
    inside the calibrated range. Events with multiplicity below the last
    threshold (more peripheral than 80 %) are assigned bin n_bins-1 with a flag
    in the caller's interpretation — we do not introduce a separate "out of
    range" bin because peripheral classification beyond 80 % is not in scope.
    """
    n_bins = len(mult_thresholds)
    # np.searchsorted on a DESCENDING array: invert sign to make it ascending.
    # mult_thresholds is descending, so -mult_thresholds is ascending.
    asc = -mult_thresholds
    idx = np.searchsorted(asc, -mult, side="right") - 1
    # Clip to valid range. Events with mult above the top threshold land in bin 0;
    # events with mult below all thresholds (more peripheral than 80 %) land in n_bins-1.
    return np.clip(idx, 0, n_bins - 1).astype(np.int16)


def predict(mult: np.ndarray, calib: TruthCalibration) -> tuple[np.ndarray, np.ndarray]:
    """Apply a calibrated mapping to new events.

    Returns (bin_index, predicted_b) — the predicted b is the calibration's
    bin-mean. This is the simplest possible RefMult point estimate; per-event
    uncertainty for RefMult is conventionally taken as the bin's std (calib.b_stds).
    """
    bins = assign_bins(mult, calib.mult_thresholds)
    b_pred = calib.b_means[bins]
    return bins, b_pred.astype(np.float32)


def train_test_split_indices(n: int, train_frac: float = 0.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Half-and-half split by default — we want the train half large enough that
    bin-mean estimates are stable, but not so large that the resolution
    diagnostic on the held-out half is statistics-limited. For 100k events at a
    given energy, 50k/50k leaves ~5k events per centrality bin in each half.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(n * train_frac))
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])
