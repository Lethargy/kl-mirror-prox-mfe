r"""Double-well example (Section 6.2) for the KL Mirror-Prox MFE paper.

Cost operator on Omega = [-2, 2]:

    J(mu, x) = (x^2 - 1)^2 + alpha * mean(mu) * x,   mean(mu) = \int y mu(dy).

This is Lasry-Lions monotone (pairing = alpha (mean(mu) - mean(nu))^2) but not
strongly monotone. The equilibrium is mu* = (1/2) delta_{-1} + (1/2) delta_{1}
(unique, checked directly).

Running this script writes two figures to ./plots:

    double_well_averaged_nu_8.png  -- averaged predictor nu-bar on the M=8 mesh
    double_well_gap5001.png        -- VI gap and Minty gap with their bounds

Closed forms used for the M=5001 gap plot:
  * Minty gap:  sup_eta <J(eta,.), nu-bar - eta>. The maximizer is supported on
    the two wells {-1, +1} (where the potential vanishes) with mean m*=mean/2,
    giving  <potential, nu-bar> + alpha * mean(nu-bar)^2 / 4.
  * VI gap:  <J(nu-bar,.), nu-bar> - min_x J(nu-bar, x); the inner minimizer is
    a root of 4x^3 - 4x + alpha*mean, compared against the endpoints +-2.

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
LAM = 0.24        # step size; condition here is 0 < lambda < 1/(8 alpha) = 1/4
K = 100           # number of iterations
ALPHA = 0.5
SNAPSHOTS = [2, 10, 50]
MESHES = [8, 5001]  # M=8 for the measure plot, M=5001 for the gap plot


def make_cost(grid):
    """Return the vectorized cost vector map rho |-> (J(rho, x_i))_i."""
    potential = (grid**2 - 1.0) ** 2

    def cost(rho):
        mean = grid @ rho
        return potential + ALPHA * mean * grid

    return cost


def run_kl_mirror_prox(grid, lam, num_iters):
    """KL Mirror-Prox iterates from the uniform initialization.

    Returns the array of predictor measures nu_k, shape (num_iters, M).
    """
    M = grid.size
    cost = make_cost(grid)

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
        # Row k-1 is the average over exactly k predictors.
        panel.scatter(grid, nu_bar[k - 1])
        panel.vlines(grid, ymin=0, ymax=nu_bar[k - 1])
        panel.scatter(x=[-1, 1], y=[0.5, 0.5], color="C1", label="MFE")
        panel.vlines(x=[-1, 1], ymin=[0, 0], ymax=[0.5, 0.5], color="C1", ls="--")
        panel.set_title(f"{k} iterations")
        panel.set_xlabel(r"$x$")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", transparent=False)
    plt.close(fig)


def continuous_min_cost(mean):
    """min_x [ (x^2-1)^2 + alpha*mean*x ] over x in [-2, 2]."""
    roots = np.roots([4.0, 0.0, -4.0, ALPHA * mean])   # 4x^3 - 4x + alpha*mean
    candidates = [-2.0, 2.0]
    candidates += [
        r.real for r in roots if abs(r.imag) < 1e-10 and -2.0 <= r.real <= 2.0
    ]
    candidates = np.asarray(candidates)
    costs = (candidates**2 - 1.0) ** 2 + ALPHA * mean * candidates
    return costs.min()


def plot_gaps(grid, nu_bar, lam, path):
    potential = (grid**2 - 1.0) ** 2
    M = grid.size
    num_iters = nu_bar.shape[0]
    mesh_error = (48.0 + 12.0 * ALPHA) / (M - 1)

    minty_gap = np.empty(num_iters)
    minty_ub = np.empty(num_iters)
    vi_gap = np.empty(num_iters)
    vi_ub = np.full(num_iters, np.nan)

    for i in range(num_iters):
        rho = nu_bar[i]
        mean = grid @ rho
        avg_potential = potential @ rho

        minty_gap[i] = avg_potential + ALPHA * mean**2 / 4.0
        minty_ub[i] = np.log(M) / (lam * (i + 1)) + mesh_error

        vi_gap[i] = avg_potential + ALPHA * mean**2 - continuous_min_cost(mean)
        if minty_ub[i] <= 16.0 * ALPHA:
            vi_ub[i] = 8.0 * np.sqrt(ALPHA * minty_ub[i])

    fig, ax = plt.subplots(ncols=2, figsize=(7, 3), dpi=128, sharey=True)
    ax[0].plot(minty_gap, label="minty gap")
    ax[0].plot(minty_ub, label="upper bound")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("error")
    ax[0].legend()
    ax[1].plot(vi_gap, label="vi gap")
    ax[1].plot(vi_ub, label="upper bound")
    ax[1].set_xlabel("iteration")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", transparent=False)
    plt.close(fig)


def main():
    for M in MESHES:
        grid = np.linspace(-2.0, 2.0, M)
        nu_bar = ergodic_average(run_kl_mirror_prox(grid, LAM, K))
        if M == 8:
            plot_averaged_measure(
                grid, nu_bar, os.path.join(OUTDIR, "double_well_averaged_nu_8.png")
            )
        else:
            plot_gaps(
                grid, nu_bar, LAM, os.path.join(OUTDIR, f"double_well_gap{M}.png")
            )


if __name__ == "__main__":
    main()
