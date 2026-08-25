import numpy as np
from source.core.tensor_train import TensorTrain
from source.core.matrix_product_operator import MatrixProductOperator
from source.dmrg.dmrg import dmrg_optimize

'''
This example uses DMRG to find minimal energy of Ising spin system with transverse magnetic field. 
The functional H defines the energy of state and is defined as 

J sum_i(sigma_z^i ⊗ sigma_z^i+1) + h sum_i(sigma_x^i)

This hamiltonian admits a well-known low rank MPO representation which we will use in this example.
'''

# Physical constants
J = 1
h = .1
d = 5

# Computational constants
rank = 10
nswp = 6

#Useful tensors
sigma_z = np.array([[1, 0],[0, -1]])
sigma_x = np.array([[0, 1],[1, 0]])
I = np.eye(2)
O = np.zeros((2, 2))

H1 = np.array([[h*sigma_x, J*sigma_z, I]])
H2 = np.array([
    [I, O, O],
    [sigma_z, O, O],
    [h*sigma_x, J*sigma_z, I]
])
H3 = np.array([[I, sigma_z, h*sigma_x]])

if __name__ == '__main__':
	# Initializing quadratic form representing energy of the system
	H_cores = [H1.transpose(0, 2, 3, 1)] + [H2.transpose(0, 2, 3, 1)]*(d-2) + [H3.transpose(0, 2, 3, 1).T]
	H = MatrixProductOperator(H_cores)
	print('The Hamiltonian:', H)

	#Running DMRG
	state = dmrg_optimize(H, n_sweeps = nswp, rank = rank, verbose=True)
	print('Raw optimized state:', state)
	print('Rounded optimized state:', state.rounded(max_rel_eps=1e-3))