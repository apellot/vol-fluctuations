"""Numerical sanity tests for kinematic identities in the SMASH output.

The test asserts that p0 = sqrt(p² + m²) to high precision over a small sample
read straight from the ROOT file. The point is to catch silent corruption of
the mass branch or the 4-momentum branches in a future SMASH version.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import uproot

SMASH_FILE = Path("~/phd/SMASH_Analysis/OUT/particle_lists_AuAu_3.5GeV.root").expanduser()


@pytest.mark.skipif(not SMASH_FILE.exists(), reason="SMASH ROOT file not present locally")
def test_mass_shell_to_one_in_a_million() -> None:
    with uproot.open(str(SMASH_FILE)) as f:
        arrs = f["tree"].arrays(["px", "py", "pz", "p0", "mass"], entry_stop=50, library="np")

    devs = []
    for i in range(len(arrs["px"])):
        px, py, pz = arrs["px"][i], arrs["py"][i], arrs["pz"][i]
        p0, m = arrs["p0"][i], arrs["mass"][i]
        expected = np.sqrt(px * px + py * py + pz * pz + m * m)
        devs.append(np.abs(p0 - expected) / np.maximum(expected, 1e-9))
    devs = np.concatenate(devs)

    # 1e-6 leaves a comfortable margin; observed in our sample is ~1.5e-7.
    assert devs.max() < 1e-6, f"max relative mass-shell deviation = {devs.max():.2e}"
