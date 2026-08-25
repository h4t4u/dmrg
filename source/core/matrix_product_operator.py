from __future__ import annotations
import numpy as np
import copy

from tensor_train import TensorTrain

'''
This class describes a matrix product operator (MPO), that is a finite-dimensional operator of shape (n_1, ... n_d)x(n_1, ... n_d)  encoded using finite number of cores. 
Using Penrose diagram notation, it can be written as 

   │ m1    │ m2    │ m3              │ md 
 ╭─┴──╮r1╭─┴──╮r2╭─┴──╮r3   r_{d-1}╭─┴──╮
 │ C1 │──│ C2 │──│ C3 │── … ───────│ Cd │
 ╰─┬──╯  ╰─┬──╯  ╰─┬──╯            ╰─┬──╯
   │ n1    │ n2    │ n3              │ nd

Where Ci is named core and is a tensor of of shape (r_{i-1}, m_i, n_i, r_i). The r_{0} and r_d are taken to be 1. 
For d = 1 the MPO is just a matrix. 
This data format allows one to perform memory-saving operations on tensor trains.
This realisation uses real-valued tensors.
'''

class MatrixProductOperator:

    cores: list[np.ndarray]

    @property
    def ranks(self) -> list[int]:
        return [core.shape[0] for core in self.cores[1:]]

    @property
    def ns(self) -> list[int]:
        return [core.shape[1] for core in self.cores]

    @property
    def ms(self) -> list[int]:
        return [core.shape[2] for core in self.cores]

    def __init__(self, cores: list[np.ndarray]):
        '''
        The default constructor, initialising matrix product operator from list of its cores.

        Args: 
            cores: list[np.ndarray]
                List of shape d of np.arrays, each of shape (r_{i-1}, m_i, n_i, r_i). The r_0 and r_d are taken to be 1. 
        '''

        ranks_left  = [core.shape[0] for core in cores]
        ranks_right = [core.shape[3] for core in cores]

        if not ranks_left[0] == 1:
            raise ValueError(f"The first tensor of core must have shape (1, n1, m1, r1), instead has shape{cores[0].shape}.")

        if not ranks_right[-1] == 1:
            raise ValueError(f"The last tensor of core must have shape (r_{{d-1}}, nd, md, 1), instead has shape{cores[-1].shape}.")

        if not ranks_left[1:] == ranks_right[:-1]:
            bad_indices = np.where(np.array(ranks_left[1:]) != np.array(ranks_right[:-1]))
            raise ValueError(f"Rank mismatch: cores with numbers {bad_indices[0]+1} do not match ranks with their subsequents.")

        self.cores = cores

    def __matmul__(self, other):
        if isinstance(other, TensorTrain):
            return self.mult_by_tensor_train(other)
        elif isinstance(other, MatrixProductOperator):
            return self.mult_by_mpo(other)
        else:
            return NotImplemented

    def mult_by_tensor_train(self, tensor_train: TensorTrain) -> TensorTrain:
        '''
        Performs matrix-vector multiplication of TT by MPO. On the contrary to exponentially large amount of operations in full representation, it takes only O(n^2 x d x r^4) operations.
        
        Args: 
            tensor_train: TensorTrain
                The tensor train of shape (n_1, ... n_d), coinciding with self.ns.

        Returns:
            product: TensorTrain 
            The result of matrix-vector multiplication of TT by MPO, representing tensor of shape (m_1, ... m_d). 
            In (invariant) Einstein notation the result for MPO M and tensor train T is 
            M@T_{j_1, ...j_d} = M_{i_1, i_2, ... i_d, j_1, ...j_d}T_{i_1, i_2, ... i_d}, where:
                For M i_. correspond to n indices and j_. -- to m indices.

        '''

        cores = []
        if self.ns != tensor_train.ns:
            raise ValueError(f"The shape of tensor train must be compatible with shape of MPO!")

        for i in range(len(self.cores)):

            new_core = np.einsum('ijkl, mkn -> imjln', self.cores[i], tensor_train.cores[i])
            new_core = new_core.reshape(self.cores[i].shape[0]*tensor_train.cores[i].shape[0], self.cores[i].shape[2], self.cores[i].shape[3]*tensor_train.cores[i].shape[2])
            cores += [new_core]

        return TensorTrain(cores)


    def mult_by_mpo(self, matrix_product_operator: MatrixProductOperator) -> MatrixProductOperator:
        '''
        Performs matrix multiplication of MPOs. On the contrary to exponentially large amount of operations in full representation, it takes only O(n^3 x d x r^4) operations.
        
        Args: 
            matrix_product_operator: MatrixProductOperator
                The matrix product operator with ms (m_1, ... m_d), coinciding with self.ns.

        Returns:
            product: TensorTrain 
            The result of matrix multiplication of MPOs, representing MPO with ns = matrix_product_operator.ns and ms = self.ms. 
            In (invariant) Einstein notation the result for MPO M and MPO N is defined by 
            M@N_{k_1, ..., k_d, j_1, ...j_d} = M_{i_1, i_2, ... i_d, j_1, ...j_d}N_{k_1, ..., k_d, i_1, i_2, ... i_d}, where:
                For M, i_. correspond to n indices and j -- to m indices.
                For N, k_. correspond to n indices and i -- to m indices.

        '''

        if self.ns != matrix_product_operator.ms:
            raise ValueError(f"The shapes of MPO must be compatible!")

        cores = []

        for i in range(len(self.cores)):

            new_core = np.einsum('ijkl, mkno -> imjnlo', self.cores[i], matrix_product_operator.cores[i])
            new_core = new_core.reshape(self.cores[i].shape[0]*matrix_product_operator.cores[i].shape[0], self.cores[i].shape[1], matrix_product_operator.cores[i].shape[2], self.cores[i].shape[3]*matrix_product_operator.cores[i].shape[3])
            cores += [new_core]

        return MatrixProductOperator(cores)

    def __rmatmul__(self, tensor_train: TensorTrain) -> TensorTrain:
        '''
        Performs matrix-vector multiplication of TT by MPO. On the contrary to exponentially large amount of operations in full representation, it takes only O(n^2 x d x r^4) operations.
        
        Args: 
            tensor_train: TensorTrain
                The tensor train of shape (m_1, ... m_d), coinciding with self.ms.

        Returns:
            product: TensorTrain 
            The result of matrix-vector of TT by MPO, representing tensor of shape (n_1, ... n_d). 
            In (invariant) Einstein notation the result for MPO M and tensor train T is 
            T@M_{i_1, i_2, ... i_d} = M_{i_1, i_2, ... i_d, j_1, ...j_d}T_{j_1, j_2, ... j_d}, where:
                For M i_. correspond to n indices and j_. -- to m indices.

        '''

        cores = []
        if self.ms != tensor_train.ns:
            raise ValueError(f"The shape of tensor train must be compatible with shape of MPO!")

        for i in range(len(self.cores)):

            new_core = np.einsum('ijkl, mjn -> imkln', self.cores[i], tensor_train.cores[i])
            new_core = new_core.reshape(self.cores[i].shape[0]*tensor_train.cores[i].shape[0], self.cores[i].shape[1], self.cores[i].shape[3]*tensor_train.cores[i].shape[2])
            cores += [new_core]

        return TensorTrain(cores)

