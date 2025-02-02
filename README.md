# FNO based Foundational Models for Phase Field Dynamics

This study implements a variant of Fourier Neural Operators (FNO) [1] to solve the one-dimensional Allen-Cahn equation. The Allen-Cahn equation is a fundamental partial differential equation (PDE) used to model phase separation processes in materials science. A primary challenge in this task is ensuring that the trained model can generalize to unseen initial conditions, a critical requirement for robust neural operator-based PDE solvers.

The FNO architecture is designed to learn mappings between function spaces, making it particularly well-suited for solving PDEs. Unlike traditional numerical solvers, FNOs operate in the frequency domain, enabling efficient learning of complex spatial and temporal dependencies and invariance to the mesh. The implementation of the model is structured across two primary files: lib/model.py, which defines the neural network architecture, and lib/layers.py, which contains the necessary layer definitions.

This work evaluates the performance of the trained model in predicting the solution of the Allen-Cahn equation for various initial conditions, including those not seen during training. The results highlight the model's ability to generalize and provide insight into the applicability of neural operators for PDE-driven problems.

A detailed write-up of the project is available in [this report](https://github.com/benedict-armstrong/FNO-for-Phase-Field-Dynamics/blob/main/writeup/report.pdf).

[1] Li, et al. "Fourier neural operator for parametric partial differential equations." 2020 Available: 
https://arxiv.org/abs/2010.08895