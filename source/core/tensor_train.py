from __future__ import annotations
import numpy as np
import copy

'''
This class describes a tensor train (TT), that is a finite-dimensional tensor of shape (n_1, ... n_d) encoded using finite number of cores. 
Using Penrose diagram notation, it can be written as 

 ╭────╮r1╭────╮r2╭────╮r3   r_{d-1}╭────╮
 │ C1 │──│ C2 │──│ C3 │── … ───────│ Cd │
 ╰─┬──╯  ╰─┬──╯  ╰─┬──╯            ╰─┬──╯
   │ n1    │ n2    │ n3              │ nd

Where Ci is named core and is a tensor of of shape (r_{i-1}, n_i, r_i). The r_{0} and r_d are taken to be 1. 
The simplest nontrivial example of TT is d=2, r1=1, giving the one-rank matrix. 
This data format allows one to have a useful approximation to a full tensor, taking O(n_max x r_max^2 x d) memory instead of O(n_max^d) for full tensor.
However, the main reason why it is being used is that it allows to make operations such as contraction, norm taking or addition also take less memory and time.
The approach has been developed by Ivan Oseledets in https://www.researchgate.net/publication/220412263_Tensor-Train_Decomposition.
This realisation uses real-valued tensors.
'''

class TensorTrain:

	cores: list[np.ndarray]

	@property
	def ranks(self) -> list[int]:
		return [core.shape[0] for core in self.cores[1:]]

	@property
	def ns(self) -> list[int]:
		return [core.shape[1] for core in self.cores]
	

	def __init__(self, cores: list[np.ndarray]):
		'''
		The default constructor, initialising tensor train from list of its cores.

		Args: 
			cores: list[np.ndarray]
				List of shape d of np.arrays, each of shape (r_{i-1}, n_i, r_i). The r_0 and r_d are taken to be 1. 
		'''

		ranks_left  = [core.shape[0] for core in cores]
		ranks_right = [core.shape[2] for core in cores]

		if not ranks_left[0] == 1:
			raise ValueError(f"The first tensor of core must have shape (1, n1, r1), instead has shape{cores[0].shape}.")

		if not ranks_right[-1] == 1:
			raise ValueError(f"The last tensor of core must have shape (r_{{d-1}}, nd, 1), instead has shape{cores[-1].shape}.")

		if not ranks_left[1:] == ranks_right[:-1]:
			bad_indices = np.where(np.array(ranks_left[1:]) != np.array(ranks_right[:-1]))
			raise ValueError(f"Rank mismatch: cores with numbers {bad_indices[0]+1} do not match ranks with their subsequents.")

		self.cores = cores


	def __str__(self):
		return f"Tensor train with dimensions {self.ns} and ranks {self.ranks}"
	

	def full(self) -> np.ndarray:
		'''
		Returns the full tensor of shape (n_1, ... n_d) from TT representation. May take a lot of memory and time.
		
		Returns:
			full_tensor: np.ndarray
				The tensor represented by this TT, shape (n_1, ... n_d).

		'''

		full_tensor = self.cores[0]

		if len(self.cores) >= 2:
			for core in self.cores[1:]:
				full_tensor = np.einsum('ijk, klm -> ijlm', full_tensor, core)
				full_tensor = full_tensor.reshape(1, full_tensor.shape[1]*full_tensor.shape[2], full_tensor.shape[3])

		full_tensor = full_tensor.reshape(self.ns)

		return full_tensor


	def norm(self, eps: np.float64 = 1e-5) -> np.float64:
		'''
		Returns Frobenius norm of tensor train.

		Args:
			eps: np.float64, optional, default = 1e-5
				The permissible arithmetic error in order to prevent square root of negative.

		Returns:
			norm: float
		'''
		sqnorm = self@self
		if sqnorm < 0 and sqnorm > -eps:
			sqnorm = 0.0
		return np.sqrt(sqnorm)


	def normalized(self) -> TensorTrain:
		'''
		Returns the tensor proportional to given with Frobenius norm 1.

		Returns:
			normalized_tensor_train: TensorTrain
				The normalized tensor train.
		'''

		norm = self.norm()

		return self*(1/norm)


	def __mul__(self, q: np.float64) -> TensorTrain:
		d = len(self.cores)
		absq = np.abs(q)

		new_cores = []
		for i in range(d):
			new_cores += [self.cores[i].copy() * (absq**(1/d))]

		if q <0:
			new_cores[0] *= -1

		return TensorTrain(new_cores)


	def __rmul__(self, q: np.float64) -> TensorTrain:
		return self * q


	def __matmul__(self, tensor_train: TensorTrain) -> np.float64:
		'''
		Contracts tensor train and other tensor train with identical shapes (performs full Einstein sum). On the contrary to O(n^d) operations in full representation, it takes only O(n x d x r^3) operations.
		
		Args: 
			tensor_train: TensorTrain
				The tensor train of shape (n_1, ... n_d), coinciding with self.ns.

		Returns:
			scalar_product: np.float64
				The result of full scalar multiplication of tensors by corresponding indices. 
				In (invariant) Einstein notation the result for tensors T1 and T2 is T1_{i_1, i_2, ... i_d}T2_{i_1, i_2, ... i_d} 

		'''

		if self.ns != tensor_train.ns:
			raise ValueError(f"The shapes of tensors must coincide!")
		T = np.einsum('ijk, ijl', self.cores[0], tensor_train.cores[0])
		

		for i in range(1, len(self.cores)):
			T = np.einsum('ij, ikl', T, self.cores[i])
			T = np.einsum('ijk, ijl', T, tensor_train.cores[i])

		scalar_product = np.einsum('ii', T)
		return scalar_product

	def tensordot(self, tensor_train: TensorTrain) -> TensorTrain:
		''' 
		Performs tensor multiplication on tensor trains with shapes (n_1, ... n_d) and (m_1, ... m_e). 
		The result is tensor train representing tensor of shape (n_1, ... n_d, m_1, ... m_e), that is a tensor product of given tensors.
		This method requires O(d+e) operations.

		Args: 
			tensor_train TensorTrain
				The second tensor train of shape (m_1, ... m_e).

		Returns:
			tensor_prod: TensorTrain
				The tensor product of tensors.
		'''

		new_cores = self.cores + tensor_train.cores
		return TensorTrain(new_cores)

	def __getitem__(self, idx) -> np.float64:
		'''
		Returns an element of tensor encoded by tensor train with indices idx. The method is performed by scalar multiplication with delta function.
		'''	

		delta_func = TensorTrain.delta_function(self.ns, list(idx))
		return self @ delta_func

	def __add__(self, tensor_train: TensorTrain) -> TensorTrain:
		'''
		Adds tensor trains with identical ns lists. On the contrary to O(n^d) operations in full representation, it takes only O(n x d x r^3) operations.
				

		Args: 
			tensor_train TensorTrain
				The second tensor train of shape (n_1, ... n_d), coinciding with self.ns.

		Returns:
			tensor_sum: TensorTrain
				The result of addition of tensors.
		'''

		new_cores = []

		if tensor_train.ns != self.ns:
			raise ValueError(f"The shapes of tensors must coincide!")

		d = len(tensor_train.cores)

		for i in range(d):
			core1 = self.cores[i]
			core2 = tensor_train.cores[i]

			r1 = core1.shape[0]
			r2 = core1.shape[2]
			R1 = core2.shape[0]
			R2 = core2.shape[2]

			new_r_1 = r1 + R1 if i != 0 else 1
			new_r_2 = r2 + R2 if i != d-1 else 1

			new_core = np.zeros((new_r_1, core1.shape[1], new_r_2))

			if i == 0:
				new_core[:, :, :r2] = core1
				new_core[:, :, r2:] = core2

			elif i == d -1:
				new_core[:r1, :, :] = core1
				new_core[r1:, :, :] = core2

			else:
				new_core[:r1, :, :r2] = core1
				new_core[r1:, :, r2:] = core2

			new_cores += [new_core]

		tensor_sum = TensorTrain(new_cores)

		return tensor_sum


	def __sub__(self, tensor_train: TensorTrain) -> TensorTrain:

		return self + tensor_train * (-1)

	@staticmethod
	def svd_splitting_step(tensor: np.ndarray, eps: float, max_rank: int, split_index: int = 1):
		'''
		Auxiliary method to perform TT-SVD splitting of tensor to approximate (or to round). 
		Inputs a tensor of shape (n_1 ,... n_d) and outputs two tensors of shape (n_1, ..., n_{split_index}, r) and (r, n_{split_index+1}, ... n_d), 
		performing TT-SVD step and leaving only singular values that correspond to eps and max_rank defined.
		
		Args: 
			tensor: np.ndarray
				The tensor to split.
			eps: float
				The rounding error admissible in this TT-SVD step.
			max_rank: int
				The maximal rank admissible for r.
			split_index: int, optional, default=1
				The index to split at.

		Returns:
			left_tensor: np.ndarray
				The left tensor of shape (n_1, ..., n_{split_index}, r).
			right_tensor: np.ndarray
				The right tensor of shape (r, n_{split_index+1}, ... n_d).
		'''
		shape = tensor.shape
		M = tensor.reshape(np.prod(tensor.shape[:split_index]), np.prod(tensor.shape[split_index:]))

		U, s, Vh = np.linalg.svd(M, full_matrices = False)

		cum_error_singvalues = np.sqrt(np.cumsum(s[::-1]**2)[::-1])
		s[cum_error_singvalues < eps] = 0
		s[max_rank:] = 0
		r = np.count_nonzero(s)

		left_tensor = (U[:, :r]).reshape(*tensor.shape[:split_index], r)
		right_tensor = np.einsum('ij, jk', np.diag(s[:r]), Vh[:r]).reshape(r, *tensor.shape[split_index:])

		return left_tensor, right_tensor

	def rounded(self, max_rel_eps: float = 1e-5, max_rank: int = 1024) -> TensorTrain:
		'''
		Returns optimal Frobenius norm approximation with accuracy eps or maximal rank r for given tensor train using  using TT-rounding algorithm by Oseledets.

		Args: 
			max_rel_eps: float, optional, default=1e-5
				The error relative Frobenius norm admissible for the tensor train
			max_rank: int, optional, default=1024
				The maximal rank admissible for the tensor train.

		Returns:
			approximation: TensorTrain
				The TT approximation with maximal rank max_rank and error relative Frobenius norm less or equal than max_rel_eps.
				Rank condition is prioritized.
		'''

		d = len(self.ns)

		if d == 1:
			cores =  copy.deepcopy(self.cores)
			return TensorTrain(cores)

		eps = self.norm() * max_rel_eps / np.sqrt(d-1)

		cores =  copy.deepcopy(self.cores)

		# QR orthogonalization
		for i in range(1, d)[::-1]:
			shape = cores[i].shape
			M = cores[i].reshape(shape[0], shape[1] * shape[2])

			Q, R = np.linalg.qr(M.T)
			cores[i] = (Q.T).reshape(shape)
			cores[i-1] = np.einsum('ijk, kl', cores[i-1], R.T)

		# SVD truncation
		for i in range(d-1):
			left, right = TensorTrain.svd_splitting_step(cores[i], eps, max_rank, split_index = 2)
			cores[i] = left
			cores[i+1] = np.einsum('ij, jkl', right, cores[i+1])

		return TensorTrain(cores)


	@staticmethod
	def tt_approximation(full_tensor: np.ndarray, max_rel_eps: float = 1e-5, max_rank: int = 1024) -> TensorTrain:
		'''
		Returns optimal Frobenius norm approximation with accuracy eps or maximal rank r for given full tensor using TT-SVD algorithm by Oseledets.

		Args: 
			full_tensor: np.ndarray
				The tensor to be approximated in full form of shape (n_1, ... n_d).
			max_rel_eps: float, optional, default=1e-5
				The error relative Frobenius norm admissible for the tensor train.
			max_rank: int, optional, default=1024
				The maximal rank admissible for the tensor train.

		Returns:
			approximation: TensorTrain
				The TT approximation with maximal rank max_rank and error relative Frobenius norm less or equal than max_rel_eps.
				Rank condition is prioritized.
		'''

		d = len(full_tensor.shape)

		if d == 1:
			full_tensor = full_tensor.reshape([1] + list(full_tensor.shape) + [1])
			return TensorTrain([full_tensor])

		eps = np.linalg.norm(full_tensor) * max_rel_eps / np.sqrt(d-1)
		cores = []
		
		full_tensor = full_tensor.reshape([1] + list(full_tensor.shape) + [1])
		
		for i in range(d - 1):
			left, right = TensorTrain.svd_splitting_step(full_tensor, eps, max_rank, split_index = 2)
			cores += [left]
			full_tensor = right
			
		cores += [full_tensor]
		return TensorTrain(cores)

	@staticmethod
	def delta_function(shape: list[int], idx: list[int]) -> TensorTrain:
		'''
		Returns tensor train with all ranks 1, representing tensor T with dimensions given by shape, 
		such that T = delta(idx), i.e. element with indices idx is 1 and all other elements are 0.

		Args: 
			shape: list[int]
				The shape of target tensor.
			idx: list[int]
				The index of only nonzero value.

		Returns:
			delta_function: TensorTrain
				The described tensor train.
		'''
		cores = []

		for i in range(len(shape)):
			if idx[i] < 0 or idx[i] >= shape[i]:
				raise ValueError(f"The target index by number {i+1} for delta function is out of range.")
			core = np.zeros(shape[i])
			core[idx[i]] = 1
			core = core.reshape(1, shape[i], 1)

			cores += [core]

		return TensorTrain(cores)

	@staticmethod
	def random_tensor(ns: list[int], ranks: list[int]) -> TensorTrain:
		'''
		Generates tensor train with randomly generated cores.

		Args: 
			ns: list[int]
				The shape of target tensor.
			ranks: list[int]
				The ranks pf target tensor.
		Returns:
			random_tensor: TensorTrain
				The described tensor train.
		''' 
		cores = []
		d = len(ns)

		for i in range(d):
			r1 = 1 if i == 0   else ranks[i-1]
			r2 = 1 if i == d-1 else ranks[i]
			core = np.random.rand(r1, ns[i], r2)
			cores += [core]

		return TensorTrain(cores)