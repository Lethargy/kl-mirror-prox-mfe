r"""Arctangent example (Section 6.1) for the KL Mirror-Prox MFE paper.

Cost operator on Omega = [-1, 1]:

    J(mu, x) = x^2 + \int_Omega arctan(x - y) mu(dy).

The arctan kernel is antisymmetric, so the operator is Lasry-Lions monotone
with a vanishing monotonicity pairing; the Minty gap and the VI gap therefore
coincide. The unique equilibrium is mu* = delta_{-1/2}.

Running this script writes two figures to ./plots:

    arctangent_averaged_nu_8.png  -- averaged predictor nu-bar on the M=8 mesh
    arctangent_vi_gap.png         -- VI gap, Minty gap, and the Theorem bound

No randomness is involved; the output is fully deterministic.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# --------------------------------------------------------------------------- #
# Output location: ./plots relative to this file (matches \includegraphics).
# --------------------------------------------------------------------------- #
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUTDIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
LAM = 0.49        # step size; step-size condition here is 0 < lambda < 1/2
K = 200           # number of iterations
SNAPSHOTS = [2, 10, 50]
MESHES = [8, 1001]  # M=8 for the measure plot, M=1001 for the gap plot


def build_operator(grid):
    """Return vectorized cost helpers for a fixed mesh.

    kernel[i, j] = arctan(grid[i] - grid[j]).
    """
    kernel = np.arctan(grid[:, None] - grid[None, :])

    def cost(rho):
        """Cost vector (J(rho, x_i))_i on the mesh."""
        return grid**2 + kernel @ rho

    def J_at(rho, x):
        """Scalar cost J(rho, x) at an arbitrary point x (used for the gap)."""
        return x**2 + np.sum(rho * np.arctan(x - grid))

    def dJ_at(rho, x):
        """d/dx J(rho, x); J(rho, .) is strictly convex, so it has one root."""
        return 2.0 * x + np.sum(rho / (1.0 + (x - grid) ** 2))

    return cost, J_at, dJ_at


def run_kl_mirror_prox(grid, lam, num_iters):
    """KL Mirror-Prox iterates from the uniform initialization.

    Returns the array of predictor measures nu_k, shape (num_iters, M).
    """
    M = grid.size
    cost, _, _ = build_operator(grid)

    p = np.ones(M) / M
    nu_history = np.empty((num_iters, M))

    for k in range(num_iters):
        q = p * np.exp(-lam * cost(p))          # prediction at mu_k
        q /= q.sum()
        p = p * np.exp(-lam * cost(q))           # correction at nu_k
        p /= p.sum()
        nu_history[k] = q

    return nu_history


def ergodic_average(nu_history):
    """Running average nu-bar_K = (1/K) sum_{k<K} nu_k for every horizon K."""
    horizons = np.arange(1, nu_history.shape[0] + 1)[:, None]
    return np.cumsum(nu_history, axis=0) / horizons


def plot_averaged_measure(grid, nu_bar, path):
    fig, ax = plt.subplots(ncols=3, figsize=(8, 3), sharey=True, dpi=128)
    for panel, k in zip(ax, SNAPSHOTS):
        panel.scatter(grid, nu_bar[k])
        panel.vlines(grid, ymin=0, ymax=nu_bar[k])
        panel.scatter(x=-0.5, y=1, color="C1", label="MFE")
        panel.vlines(x=-0.5, ymin=0, ymax=1, color="C1", ls="--")
        panel.set_title(f"{k} iterations")
        panel.set_xlabel(r"$x$")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", transparent=False)
    plt.close(fig)


def plot_gaps(grid, nu_bar, lam, path):
    cost, J_at, dJ_at = build_operator(grid)
    M = grid.size
    num_iters = nu_bar.shape[0]

    gap = np.empty(num_iters)
    bound = np.empty(num_iters)
    for i in range(num_iters):
        rho = nu_bar[i]
        # VI gap = <J(rho,.), rho> - min_x J(rho, x).
        # For the antisymmetric kernel the self-pairing of the arctan term
        # vanishes, so <J(rho,.), rho> = sum_i rho_i x_i^2; this is exactly
        # rho @ cost(rho), which we use directly.
        x_min = brentq(lambda x: dJ_at(rho, x), -1.0, 1.0)
        gap[i] = rho @ cost(rho) - J_at(rho, x_min)
        bound[i] = np.log(M) / (lam * (i + 1)) + 5.0 / (M - 1)

    fig, ax = plt.subplots(ncols=2, figsize=(7, 3), dpi=128, sharey=True)
    # The two gaps coincide analytically for this example, so both panels
    # show the same computed curve (labels follow the paper caption:
    # VI gap on the left, Minty gap on the right).
    ax[0].plot(gap, label="VI gap")
    ax[0].plot(bound, label="upper bound")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("error")
    ax[0].legend()
    ax[1].plot(gap, label="Minty gap")
    ax[1].plot(bound, label="upper bound")
    ax[1].set_xlabel("iteration")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", transparent=False)
    plt.close(fig)


def main():
    for M in MESHES:
        grid = np.linspace(-1.0, 1.0, M)
        nu_bar = ergodic_average(run_kl_mirror_prox(grid, LAM, K))
        if M == 8:
            plot_averaged_measure(
                grid, nu_bar, os.path.join(OUTDIR, "arctangent_averaged_nu_8.png")
            )
        else:
            plot_gaps(
                grid, nu_bar, LAM, os.path.join(OUTDIR, "arctangent_vi_gap.png")
            )


if __name__ == "__main__":
    main()
