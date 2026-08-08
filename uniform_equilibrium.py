r"""Uniform-equilibrium example (Section 6.3) for the KL Mirror-Prox MFE paper.

Cost operator on Omega = [-1, 1]:

    J(mu, x) = x^2 / 2 - \int_Omega |x - y| mu(dy).

This operator is strongly Lasry-Lions monotone with alpha = 1, and the
continuous equilibrium is the uniform law on [-1, 1]. On an equispaced mesh the
finite-mesh equilibrium mu_h* is the trapezoidal discretization of the uniform
density (interior mass 1/(M-1), endpoint mass 1/(2(M-1))).

Running this script writes two figures to ./plots:

    strong_lasry_lions_evolution.png     -- last iterates mu_{k,h} vs mu_h*
    strong_lasry_lions_last_iterate.png  -- last-iterate W1 error vs the bound

It also prints the certified entry index k_0 and related constants used in
item (3) of the finite-mesh last-iterate theorem.

No randomness is involved; the output is fully deterministic.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Output location: ./plots relative to this file (matches \includegraphics).
# --------------------------------------------------------------------------- #
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(OUTDIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
M = 8
K = 2000
LAM = 0.49

# Problem constants for this cost on [-1, 1]: alpha = L_m = 1, D_Omega = 2.
ALPHA = 1.0
L_M = 1.0
D_OMEGA = 2.0
SNAPSHOTS = [0, 20, 200]


def main():
    grid = np.linspace(-1.0, 1.0, M)
    dx = grid[1] - grid[0]

    # Full-support initialization  mu_{0,h}(x_i) = (1 + 0.9 x_i) / M.
    p = 1.0 + 0.9 * grid
    p = p / p.sum()

    # Finite-mesh equilibrium: trapezoidal weights for the uniform law.
    mu_h_star = np.full(M, 1.0 / (M - 1))
    mu_h_star[[0, -1]] /= 2.0

    # J(rho) = x^2/2 - \int |x - y| rho(dy), evaluated on the mesh.
    distance = np.abs(grid[:, None] - grid[None, :])

    def J(rho):
        return 0.5 * grid**2 - distance @ rho

    def W1(rho, sigma):
        """Exact W1 distance for measures on this ordered uniform grid."""
        return dx * np.abs(np.cumsum(rho - sigma)[:-1]).sum()

    # KL Mirror-Prox last iterates.
    p_history = np.empty((K + 1, M))
    p_history[0] = p
    for k in range(K):
        q = p * np.exp(-LAM * J(p))
        q = q / q.sum()
        p = p * np.exp(-LAM * J(q))
        p = p / p.sum()
        p_history[k + 1] = p

    # --- Figure 1: evolution of the last iterate toward mu_h*. -------------- #
    fig, ax = plt.subplots(ncols=3, figsize=(8, 3), sharey=True, dpi=128)
    for panel, k in zip(ax, SNAPSHOTS):
        panel.scatter(grid, p_history[k])
        panel.vlines(grid, ymin=0, ymax=p_history[k])
        panel.scatter(grid, mu_h_star, color="C1", marker="x", label=r"$\mu_h^\ast$")
        panel.set_title(f"{k} iterations")
        panel.set_xlabel(r"$x$")
    ax[0].set_ylabel("mass")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "strong_lasry_lions_evolution.png"),
                bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: last-iterate W1 error vs. the geometric bound. ----------- #
    w1_error = np.array([W1(rho, mu_h_star) for rho in p_history])

    delta_h = dx
    m_h_star = mu_h_star.min()
    gamma_h = 0.25 * min(
        LAM * ALPHA * delta_h**2 / 2.0,
        1.0 - LAM * L_M * D_OMEGA,
    )
    contraction = 1.0 - gamma_h * m_h_star / 2.0

    # Certified entry index: first iterate inside the local reverse-KL region
    # D_KL(mu_h* || mu_{k,h}) <= (m_h*)^2 / 8 from item (3) of the theorem.
    kl_error = np.array(
        [np.sum(mu_h_star * np.log(mu_h_star / rho)) for rho in p_history]
    )
    threshold = m_h_star**2 / 8.0
    indices = np.flatnonzero(kl_error <= threshold)
    if indices.size == 0:
        raise RuntimeError("Increase K: the local convergence region was not reached.")
    k_0 = int(indices[0])

    iterations = np.arange(K + 1)
    bound_iterations = iterations[k_0:]
    upper_bound = (
        D_OMEGA * m_h_star / 2.0 * contraction ** ((bound_iterations - k_0) / 2.0)
    )

    fig, ax = plt.subplots(figsize=(4.5, 3), dpi=128)
    ax.plot(iterations, w1_error, label="last-iterate error")
    ax.plot(bound_iterations, upper_bound, label="upper bound")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$W_1(\mu_{k,h},\mu_h^\ast)$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "strong_lasry_lions_last_iterate.png"),
                bbox_inches="tight")
    plt.close(fig)

    print(f"gamma_h = {gamma_h:.6f}")
    print(f"local threshold = {threshold:.9f}")
    print(f"first certified iterate k_0 = {k_0}")
    print(f"D_k_0,h = {kl_error[k_0]:.9f}")
    print(f"contraction factor = {contraction:.10f}")


if __name__ == "__main__":
    main()
