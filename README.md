# KL Mirror-Prox for Mean-Field Equilibria — numerical experiments

Code reproducing the numerical experiments (Section 6) of the paper
*KL Mirror-Prox for Measure-Valued Variational Inequalities and Mean-Field
Equilibria*.

<!-- TODO: add arXiv link and full author list once available. -->

Each script implements the finite-dimensional KL Mirror-Prox iteration
(exponential-tilt prediction and correction steps, both centered at the current
iterate) on an equispaced mesh, and regenerates the figures for one example.
The arctangent gap plot uses `M=1001`, and the double-well plots use
`alpha=0.5` and `lambda=0.24`, matching the manuscript.

## Contents

| Script | Example | Figures written to `plots/` |
| --- | --- | --- |
| `arctangent.py` | Arctangent (§6.1), Lasry–Lions monotone | `arctangent_averaged_nu_8.png`, `arctangent_vi_gap.png` |
| `double_well.py` | Double well (§6.2), Lasry–Lions monotone | `double_well_averaged_nu_8.png`, `double_well_gap5001.png` |
| `uniform_equilibrium.py` | Uniform equilibrium (§6.3), strongly monotone | `strong_lasry_lions_evolution.png`, `strong_lasry_lions_last_iterate.png` |

## Requirements

```
pip install -r requirements.txt
```

Tested with numpy 2.4, scipy 1.17, matplotlib 3.10 (Python 3.12); the pinned
minimums in `requirements.txt` are conservative.

## Reproducing the figures

Run each script from the repository root; figures are written to `plots/`
(created automatically):

```
python arctangent.py
python double_well.py
python uniform_equilibrium.py
```

`uniform_equilibrium.py` also prints the constants used in the last-iterate
bound, including the certified entry index `k_0 = 123`.

The experiments involve no randomness, so the output is fully deterministic and
does not depend on a seed.
