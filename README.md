### DMRG (Density Matrix Renormgroup) optimization

## Description
[DMRG](https://arxiv.org/abs/1008.3477) is an algorithm for solving a problem of quadratic form optimization. Originally it has been introduced as a method for finding minimal energy state for a quantum system encoded as MPS (Matrix Product State) with Hamiltonian encoded by MPO (Matrix Product Operator). It is one of the simplest quantum-inspired optimization algorithms using the power of tensor networks. It allows one to keep low ranks (and thus less memory used) while looking for a solution that minimizes some quadratic form, significantly decreasing the search space with case-dependent loss of accuracy. For some specific problems (as in examples) the exact solution is low-rank. The development of DMRG is [AMEn](https://arxiv.org/pdf/1304.1222) algorithm for optimization of quadratic functional, that solves some of the issues DMRG has and generalizes it. This version uses the [Tensor Train](https://www.researchgate.net/publication/220412263_Tensor-Train_Decomposition) approach developed by Ivan Oseledets. 

## The problem
Suppose we are given a space of huge tensors of shape $n_1 \times ... \times n_d$, whish we may identify as vector space with $\mathbb{R}^{n_1 \times ... \times n_d}$. Let there be also defined a quadratic form on $\mathbb{R}^{n_1 \times ... \times n_d}$, that we will denote by matrix $H$. The task is to find a tensor $x \in \mathbb{R}^{n_1 \times ... \times n_d}$ of norm 1: $x^Tx = ||x||^2 = 1$, such that $x^THx$ is minimal (of all norm-1 tensors). For small sizes the problem is easy to solve, but it suffers from dimensionality curse.

## The algorithm
The solution is actually pretty straightforward: Let us formulate the problem using **Lagrange multipliers method**. Introduce the functional $\mathcal{L}(x, \lambda)$:

$$\mathcal{L}(x, \lambda) = x^THx - \lambda(x^Tx -1)$$

To optimise it, take partial derivative by $x$:

$$\frac{\partial \mathcal{L}(x, \lambda)}{\partial x^T} = Hx - \lambda x = 0 \text{ at minimum.} \Rightarrow Hx = \lambda x$$

An example of places where this problem cannot be solved exactly comes from the area that inspired the method: If $H$ represents a hamiltonian of a quantum system, the size of tensor grows exponentially with size of it and classical matrix methods do not work. But we can try to find an approximate solution using MPOs ant tensor trains, especially if $H$ can be decently approximated by MPO (which is true in many cases). Define $x := x(C_1, ...C_d)$, so now $x$ is a function of cores, as well as the Lagrange functional:

$$\mathcal{L}(x, \lambda) = \mathcal{L}(C_1, ..., C_d, \lambda)$$

In Penrose notation 
```
							 ╭────╮  ╭────╮  ╭────╮      ╭────╮
							 │ C1 │──│ C2 │──│ C3 │── … ─│ Cd │
							 ╰─┬──╯  ╰─┬──╯  ╰─┬──╯      ╰─┬──╯         ⎛  ╭────╮  ╭────╮  ╭────╮      ╭────╮   		⎞		
	                           │       │       │           │            │  │ C1 │──│ C2 │──│ C3 │── … ─│ Cd │			│
							 ╭─┴──╮  ╭─┴──╮  ╭─┴──╮      ╭─┴──╮         │  ╰─┬──╯  ╰─┬──╯  ╰─┬──╯      ╰─┬──╯  			│
L(C_1, ..., C_d, \lambda) =  │ H1 │──│ H2 │──│ H3 │── … ─│ Hd │  ---  λ │    │       │       │           │       --- 1 	│
							 ╰─┬──╯  ╰─┬──╯  ╰─┬──╯      ╰─┬──╯         │  ╭─┴──╮  ╭─┴──╮  ╭─┴──╮      ╭─┴──╮			│
							   │       │       │           │            │  │ C1 │──│ C2 │──│ C3 │── … ─│ Cd │			│
							 ╭─┴──╮  ╭─┴──╮  ╭─┴──╮      ╭─┴──╮         ⎝  ╰────╯  ╰────╯  ╰────╯      ╰────╯ 			⎠
							 │ C1 │──│ C2 │──│ C3 │── … ─│ Cd │
							 ╰────╯  ╰────╯  ╰────╯      ╰────╯ 
```
Here $H_1, ... H_d$ is MPO representation of $H$, Now we may try to optimize the functional with respect to one core at a time, moving through them consequently in **sweeps**. For step $i$ of sweep, when we are optimizing the core $C_i$, the infimum condition on derivative looks like:
```
							      ╭────╮     ╭────╮          ╭────╮     ╭────╮
							      │ C1 │─ … ─│Ci-1│──      ──│Ci+1│─ … ─│ Cd │
							      ╰─┬──╯     ╰─┬──╯          ╰─┬──╯     ╰─┬──╯        ╭────╮     ╭────╮         ╭────╮     ╭────╮   
	                                │          │       │       │          │           │ C1 │─ … ─│Ci-1│─       ─│Ci+1│─ … ─│ Cd │
							      ╭─┴──╮     ╭─┴──╮  ╭─┴──╮  ╭─┴──╮     ╭─┴──╮        ╰─┬──╯     ╰─┬──╯         ╰─┬──╯     ╰─┬──╯  
∂L(C_1, ..., C_d, \lambda)/∂Ci =  │ H1 │─ … ─│Hi-1│──│ Hi │──│Hi+1│─ … ─│ Hd │ ---  λ   │          │       │      │          │  =  0
							      ╰─┬──╯     ╰─┬──╯  ╰─┬──╯  ╰─┬──╯     ╰─┬──╯        ╭─┴──╮     ╭─┴──╮  ╭─┴──╮ ╭─┴──╮     ╭─┴──╮
								    │          │       │       │          │           │ C1 │─ … ─│Ci-1│──│ Ci │─│Ci+1│─ … ─│ Cd │
								  ╭─┴──╮     ╭─┴──╮  ╭─┴──╮  ╭─┴──╮     ╭─┴──╮        ╰────╯     ╰────╯  ╰────╯ ╰────╯     ╰────╯ 
								  │ C1 │─   ─│Ci-1│──│ Ci │──│Ci+1│─ … ─│ Cd │
								  ╰────╯     ╰────╯  ╰────╯  ╰────╯     ╰────╯ 
```
Reshaping accordingly, this may be formulated as eigenvector condition:

$$\frac{\partial\mathcal{L}(C_1, ..., C_d, \lambda)}{\partial C_i} = A C_i - \lambda B C_i = 0$$

Finding solution with minimal $\lambda$ (so that is would correspond to minimal value of $x^THx$) we continue the sweep, optimizing one core at a step and approaching the minimum. 


*This code was written by me as homework for Artem Melnikov's course "Tensor Networks and their Applications" and later rewritten (without AI written code) in my free time using publicly available information.*
