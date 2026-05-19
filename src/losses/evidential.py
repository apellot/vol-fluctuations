"""Deep Evidential Regression loss (Amini, Schwarting, Soleimany, Rus — NeurIPS 2020).

The regression head outputs four parameters per event (μ, ν, α, β) of a
Normal-Inverse-Gamma distribution:
    y    | σ²    ∼ Normal(μ, σ²/ν)
    σ²           ∼ InverseGamma(α, β)

Marginalising σ² gives a Student-t predictive on y with location μ, scale
sqrt(β(1+ν)/(αν)), and 2α degrees of freedom. From the (μ, ν, α, β) tuple we
read off:
    aleatoric variance   = β / (α − 1)            (irreducible data noise)
    epistemic variance   = β / (ν(α − 1))         (lack-of-evidence)
    total variance       = aleatoric + epistemic

The loss is two terms:
  1. NLL of the Student-t predictive at the observed y (the standard log-evidence
     term derived in Amini's appendix).
  2. A regularisation term  λ · |y − μ| · (2ν + α)  that penalises *evidence*
     accumulated on miscalibrated examples, preventing the network from
     reporting unjustifiably narrow uncertainty on points it gets wrong.

Reference: Amini et al., "Deep Evidential Regression" (NeurIPS 2020), Eq. 8 + 9.

Parameter constraints during training (enforced by the head's activation, not here):
    ν > 0, α > 1, β > 0    — μ is unconstrained.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def nig_nll(y: Tensor, mu: Tensor, nu: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
    """Negative log-likelihood of the Student-t marginal predictive at observation y.

    Vectorized; returns a tensor of per-event NLLs of the same shape as y. Caller
    should reduce (mean / sum) as appropriate.
    """
    # Derivation: Eq. 8 of Amini 2020, rewritten using torch.lgamma. The constant terms
    # are kept inline so a future reader can trace this back to the paper without an
    # external reference.
    two_b_lambda = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(two_b_lambda)
        + (alpha + 0.5) * torch.log(nu * (y - mu) ** 2 + two_b_lambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return nll


def nig_regularizer(y: Tensor, mu: Tensor, nu: Tensor, alpha: Tensor) -> Tensor:
    """Evidence-penalty term  |y − μ| · (2ν + α)  from Amini 2020 Eq. 9.

    Forces the network to express low evidence on examples where its mean
    prediction misses the target. The multiplier λ (set by the caller) trades
    off NLL against this penalty — Amini's default is λ = 1e-2.
    """
    return torch.abs(y - mu) * (2.0 * nu + alpha)


def evidential_loss(
    y: Tensor,
    mu: Tensor,
    nu: Tensor,
    alpha: Tensor,
    beta: Tensor,
    coeff_reg: float = 1e-2,
    reduction: str = "mean",
) -> Tensor:
    """Combined Amini NIG loss = NLL + coeff_reg · regularizer."""
    loss = nig_nll(y, mu, nu, alpha, beta) + coeff_reg * nig_regularizer(y, mu, nu, alpha)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unknown reduction: {reduction!r}")


def nig_to_moments(mu: Tensor, nu: Tensor, alpha: Tensor, beta: Tensor) -> dict[str, Tensor]:
    """Convenience: derive predictive mean, aleatoric / epistemic / total variance.

    For inference and calibration plots — not used by the training loop directly.
    """
    aleatoric = beta / (alpha - 1.0)
    epistemic = beta / (nu * (alpha - 1.0))
    return {
        "mean": mu,
        "aleatoric_var": aleatoric,
        "epistemic_var": epistemic,
        "total_var": aleatoric + epistemic,
    }
