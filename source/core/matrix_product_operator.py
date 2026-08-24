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

Where Ci is named core and is a tensor of of shape (r_{i-1}, n_i, m_i, r_i). The r_{0} and r_d are taken to be 1. 
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

    def __init__(self, cores: list[np.ndarray]):
        '''
        The default constructor, initialising matrix product operator from list of its cores.

        Args: 
            cores: list[np.ndarray]
                List of shape d of np.arrays, each of shape (r_{i-1}, n_i, m_i, r_i). The r_0 and r_d are taken to be 1. 
        '''

        ranks_left  = [core.shape[0] for core in cores]
        ranks_right = [core.shape[2] for core in cores]
        ns          = [core.shape[1] for core in cores]

        if not ranks_left[0] == 1:
            raise ValueError(f"The first tensor of core must have shape (1, n1, m1, r1), instead has shape{cores[0].shape}.")

        if not ranks_right[-1] == 1:
            raise ValueError(f"The last tensor of core must have shape (r_\{d-1\}, nd, md, 1), instead has shape{cores[-1].shape}.")

        if not ranks_left[1:] == ranks_right[:-1]:
            raise ValueError(f"Rank mismatch: cores with numbers {np.where(np.array(ranks_left[1:]) != np.array(ranks_right))+1} do not match ranks with their subsequents.")

        self.cores = cores

    def matmul(self, tensor_train: TensorTrain):
        pass

