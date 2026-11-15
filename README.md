# Hard-Constrained Sampling on Embedded Riemannian Manifolds via Adjoint Schrödinger Bridges

MATLAB code for the numerical experiments of the paper.
In the code, `ASBS-M` denotes R-ASBS (Algorithm 1) or Extended R-ASBS (Algorithm 2).

| Script | Experiment | Sampler | Baselines |
|---|---|---|---|
| `earth_v2.m` | Earthquake density on S^2 | R-ASBS | Riemannian ULA / MALA, Riemannian flow matching |
| `prova_stiefel_v5.m` | Expected energy on the Stiefel manifold V(4,2) | Extended R-ASBS | Geodesic Monte Carlo, Riemannian ULA, exact reference |
| `comparison_alg2_robotics_v3.m` | Closed-loop 10-DOF chain with obstacles | Extended R-ASBS | HardFlow, constrained MCMC |
| `comparison_wahba_so3_v3.m` | Robust Wahba problem on SO(3) | Extended R-ASBS, R-ASBS | Riemannian MALA, multistart RGD, exact target |

**Requirements.** MATLAB R2022a or later with the Deep Learning Toolbox.

**Data.** `earth_v2.m` reads `query.csv`, a USGS ComCat export of worldwide earthquakes with
magnitude M >= 6.5 (1526-2026), placed in the same folder (columns `latitude`, `longitude` are used). Note `cosmic_ray_ex.m` requires the dataset specified in the paper, which is not uploaded here due to its size.

**Usage.** Run each script on its own. Metrics are printed to the console, and the earthquake,
Stiefel and Wahba scripts also save them as `.csv` / `.mat` files.

## Citation

If you use this code for your research, please cite the associated paper:

```bibtex
@article{mosso2026hardconstrained,
  title={Hard-Constrained Sampling on Embedded Riemannian Manifold via Adjoint Schrödinger Bridges},
  author={Mosso, Mattia and Choi, Jaemoo and Yang, Heng},
  journal={arXiv preprint arXiv:2608.25838},
  year={2026}
}

