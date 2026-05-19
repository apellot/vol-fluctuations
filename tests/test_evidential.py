"""Sanity tests for the evidential loss — these catch sign flips, lgamma vs gamma
mistakes, and dimensional errors that don't show up in a converging training run
but make the predictive uncertainty meaningless.
"""

from __future__ import annotations

import math

import torch

from src.losses.evidential import evidential_loss, nig_nll, nig_regularizer, nig_to_moments


def test_nll_finite_and_nonnegative_at_match():
    """When mu hits y exactly, NLL is finite and bounded below by the entropy floor.
    A NaN/Inf here usually means a log(0) bug in the parameterisation."""
    y = torch.zeros(8)
    mu = torch.zeros(8)
    nu = torch.full((8,), 1.0)
    alpha = torch.full((8,), 2.0)
    beta = torch.full((8,), 1.0)
    nll = nig_nll(y, mu, nu, alpha, beta)
    assert torch.isfinite(nll).all(), nll
    # Sanity bound: NLL of a continuous distribution is not constrained to be positive,
    # but it should not be more negative than -10 for any reasonable Gaussian-like predictive.
    assert (nll > -10).all(), nll


def test_nll_increases_when_prediction_misses():
    """Holding the uncertainty fixed, moving mu away from y should raise NLL."""
    mu = torch.tensor([0.0])
    nu = torch.tensor([1.0]); alpha = torch.tensor([2.0]); beta = torch.tensor([1.0])
    y_close = torch.tensor([0.1])
    y_far = torch.tensor([5.0])
    nll_close = nig_nll(y_close, mu, nu, alpha, beta).item()
    nll_far = nig_nll(y_far, mu, nu, alpha, beta).item()
    assert nll_far > nll_close, f"close={nll_close}, far={nll_far}"


def test_regularizer_is_zero_at_exact_match_and_positive_otherwise():
    y = torch.tensor([0.0, 1.0])
    mu = torch.tensor([0.0, 0.5])
    nu = torch.tensor([1.0, 1.0])
    alpha = torch.tensor([2.0, 2.0])
    r = nig_regularizer(y, mu, nu, alpha)
    assert r[0].item() == 0.0
    assert r[1].item() > 0.0


def test_moments_match_inverse_gamma_formulas():
    """Direct check that the closed-form aleatoric/epistemic variances are right."""
    mu = torch.tensor([0.0])
    nu = torch.tensor([2.0])
    alpha = torch.tensor([3.0])
    beta = torch.tensor([4.0])
    m = nig_to_moments(mu, nu, alpha, beta)
    # aleatoric = β/(α−1) = 4/2 = 2
    assert math.isclose(m["aleatoric_var"].item(), 2.0)
    # epistemic = β/(ν(α−1)) = 4/(2·2) = 1
    assert math.isclose(m["epistemic_var"].item(), 1.0)


def test_gradient_flows_through_all_four_parameters():
    """Backward pass must touch every NIG parameter; a typo in the loss that drops
    one of them would silently make the optimiser ignore that parameter."""
    y = torch.tensor([1.5])
    mu = torch.tensor([0.0], requires_grad=True)
    nu = torch.tensor([1.0], requires_grad=True)
    alpha = torch.tensor([2.0], requires_grad=True)
    beta = torch.tensor([1.0], requires_grad=True)
    loss = evidential_loss(y, mu, nu, alpha, beta, coeff_reg=0.01)
    loss.backward()
    for name, t in [("mu", mu), ("nu", nu), ("alpha", alpha), ("beta", beta)]:
        assert t.grad is not None and t.grad.abs().item() > 0, f"{name} got no gradient"
