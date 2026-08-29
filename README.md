# Hard-Constrained Sampling via Adjoint Schrödinger Bridges (ASBS)

This repository contains the MATLAB implementation for the paper "Hard-Constrained Sampling on Embedded Riemannian Manifold via Adjoint Schrödinger Bridges". 

The codebase provides a framework for sampling from unnormalized Boltzmann distributions on embedded Riemannian manifolds by defining controlled diffusion processes intrinsically on the manifold. This approach enforces hard constraints at the state-space level, bypassing the need for projection-based penalties.

## Repository Contents

The repository is organized into two primary algorithmic approaches:

### 1. ASBS-M (Algorithm 1)
Implemented for manifolds with known analytical heat kernels and parallel transport (e.g., $\mathbb{S}^n, \mathbb{T}^n, SO(3)$).
* `asbs_sphere_sampler.m`: Benchmarks the sampler against a bi-modal distribution on $\mathbb{S}^2$.
* `earthquake_sphere_ex.m`: Applies the sampler to global earthquake distribution data.
* `cosmic_ray_ex.m`: Generates arrival direction samples constrained to the hemisphere $\mathbb{S}_{+}^2$.

### 2. Extended ASBS-M (Algorithm 2)
Implemented for general embedded Riemannian manifolds ($c(x)=0$) where analytical transport is unavailable. Uses Projection-as-Transport (PAT) and a Projected Chord Corrector.
* `alg2_stiefel.m`: Models probability distributions of orthogonal matrices on the Stiefel manifold.
* `alg2_robotics.m`: Solves a high-dimensional inverse kinematics problem for a 10-DOF planar robot subject to hard loop-closure constraints and obstacle avoidance.
* `asbs_m_wahba_so3.m`: Solves the Robust Wahba problem, serving as a stochastic optimizer (the csv file reports the related results).

## Dependencies and Usage

* **MATLAB R2025b**
* **Deep Learning Toolbox**: Required for `dlnetwork`, automatic differentiation, and training loops.

Each experiment is contained within a standalone script. To run an experiment:
1. Open the desired `.m` file in MATLAB.
2. Ensure the working directory contains all project files.
3. Execute the script via the MATLAB editor or command window.
Note `cosmic_ray_ex.m` requires the dataset specified in the paper.

## Citation

If you use this code for your research, please cite the associated paper:

```bibtex
@article{mosso2026hardconstrained,
  title={Hard-Constrained Sampling on Embedded Riemannian Manifold via Adjoint Schrödinger Bridges},
  author={Mosso, Mattia and Choi, Jaemoo and Yang, Heng},
  journal={arXiv preprint arXiv:2608.25838},
  year={2026}
}
