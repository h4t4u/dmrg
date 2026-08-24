### DMRG (Density Matrix Renormgroup) optimization

## Description
[DMRG](https://arxiv.org/abs/1008.3477) is an algorithm for solving a problem of quadratic form optimization. Originally it has been introduced as a method for finding minimal energy state for a quantum system encoded as MPS (Matrix Product State) with Hamiltonian encoded by MPO (Matrix Product Operator). It is one of the simplest quantum-inspired optimization algorithms using the power of tensor networks. It allows one to keep low ranks (and thus less memory used) while looking for a solution that minimizes some quadratic form, significantly decreasing the search space with case-dependent loss of accuracy. For some specific problems (as in examples) the exact solution is low-rank. The development of DMRG is [AMEn](https://arxiv.org/pdf/1304.1222) algorithm for optimization of quadratic functional, that solves some of the issues DMRG has and generalizes it. This version uses the [Tensor Train](https://www.researchgate.net/publication/220412263_Tensor-Train_Decomposition) approach developed by Ivan Oceledets. 

## The algorithm
The main idea is to 


This code was written by me as homework for Artem Melnikov's course "Tensor Networks and Their Applications" and later rewritten (without AI tools) in my free time using publicly available information.
