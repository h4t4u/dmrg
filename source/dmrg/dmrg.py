import numpy as np
import scipy.linalg
from source.core.tensor_train import TensorTrain
from source.core.matrix_product_operator import MatrixProductOperator

def generate_A_b(H: MatrixProductOperator, state: TensorTrain, index: int):
	'''
	Generates matrices A and B for DMRG algorithm sweep iteration.

	Args:
		H: MatrixProductOperator
			The quadratic form that needs to be minimized in MPO format.
		state: TensorTrain
			The current candidate for solution in Tensor Train format.
		index: int
			The index of state core that needs to be optimized at current step, starts from 0.

	Returns:
		A, B: np.ndarray
			The matrices A and B for equation Ax = lambda B x.


	'''
	if state.ns != H.ns or state.ns != H.ms :
		raise ValueError(f"The shape of tensor train must be compatible with shape of MPO!")

	# Starting to build tensors from the ends to optimize computational time.

	d = len(state.cores)
	n_ind = state.ns[index]
	I = np.eye(n_ind)

	A_head = None if index == 0   else np.einsum('aqd, bqpe, cpf -> abcdef', state.cores[0],  H.cores[0],  state.cores[0]) # indices abc are rudimentary, have size 1
	A_tail = None if index == d-1 else np.einsum('aqd, bqpe, cpf -> abcdef', state.cores[-1], H.cores[-1], state.cores[-1]) # indices def are rudimentary, have size 1

	b_head = None if index == 0   else np.einsum('aqc, bqd -> abcd', state.cores[0],  state.cores[0]) # indices ab are rudimentary, have size 1
	b_tail = None if index == d-1 else np.einsum('apc, bpd -> abcd', state.cores[-1], state.cores[-1]) # indices cd are rudimentary, have size 1


	# Gradually approaching index of core that we are optimizing on this iteration from both sides.

	if index != 0:
		for i in range(1, index):
			A_head = np.einsum('abcdef, dqg, eqph, fpi -> abcghi', A_head, state.cores[i], H.cores[i], state.cores[i]) # indices abc are rudimentary, have size 1
			b_head = np.einsum('abcd, cqe, dqf -> abef', b_head, state.cores[i], state.cores[i]) # indices ab are rudimentary, have size 1

	if index != d-1:
		for i in range(index+1, d-1)[::-1]:
			A_tail = np.einsum('abcdef, kqa, lqpb, mpc -> klmdef', A_tail, state.cores[i], H.cores[i], state.cores[i]) # indices def are rudimentary, have size 1
			b_tail = np.einsum('abcd, lqa, mqb -> lmcd', b_tail, state.cores[i], state.cores[i]) # indices cd are rudimentary, have size 1

	# Attaching the core for functional corresponding to the state core we are optimizing. 
	# Treating separately head (index = 0), tail (index = d-1) and body (other) due to different shapes of tensors participating.

	if index == 0: # No A_head, everything is attached to A_tail.
		A_tail = np.einsum('abcdef, xlmb -> lamcxdef', A_tail, H.cores[0]) # indices xdef are rudimentary, have size 
		A_shape = (A_tail.shape[0]*A_tail.shape[1], A_tail.shape[2]*A_tail.shape[3])
		A = A_tail.reshape(A_shape) # constructed a proper matrix

		b_tail = np.einsum('abcd, lm-> lambcd', b_tail, I) # indices cd are rudimentary, have size 1
		b_shape = (b_tail.shape[0]*b_tail.shape[1], b_tail.shape[2]*b_tail.shape[3])
		b = b_tail.reshape(b_shape) # constructed a proper matrix
	
	elif index == d-1: # No A_tail, everything is attached to A_head.
		A_head = np.einsum('abcdef, elmx -> dlfmxabc', A_head, H.cores[-1]) # indices xabc are rudimentary, have size 1
		A_shape = (A_head.shape[0]*A_head.shape[1], A_head.shape[2]*A_head.shape[3])
		A = A_head.reshape(A_shape) # constructed a proper matrix

		b_head = np.einsum('abcd, lm-> cldmab', b_head, I) # indices ab are rudimentary, have size 1
		b_shape = (b_head.shape[0]*b_head.shape[1], b_head.shape[2]*b_head.shape[3])
		b = b_head.reshape(b_shape) # constructed a proper matrix

	else: # Both A_tail and A_head, need to attach everything properly.

		A_tail = np.einsum('abcdef, zlmb -> zlamcdef', A_tail, H.cores[index]) # indices def are rudimentary, have size 1. 
		#Now we need to attach A_head to A_tail via index z.
		A_tail = np.einsum('zlamcdef, opqrzs -> rlasmcopqdef', A_tail, A_head) # indices opqdef are rudimentary, have size 1. 
		A_shape = (A_tail.shape[0]*A_tail.shape[1]*A_tail.shape[2], A_tail.shape[3]*A_tail.shape[4]*A_tail.shape[5])
		A = A_tail.reshape(A_shape) # constructed a proper matrix

		b_tail = np.einsum('abcd, lm -> lambcd', b_tail, I) # indices cd are rudimentary, have size 1
		#Now we need to attach b_head to b_tail.
		b_tail = np.einsum('lambcd, opqr -> qlarmbopcd', b_tail, b_head) # indices opcd are rudimentary, have size 1. 
		b_shape = (b_tail.shape[0]*b_tail.shape[1]*b_tail.shape[2], b_tail.shape[3]*b_tail.shape[4]*b_tail.shape[5])
		b = b_tail.reshape(b_shape) # constructed a proper matrix

	return A, b


def dmrg_sweep(H: MatrixProductOperator, state: TensorTrain, is_positive: bool, verbose: bool = True):
	'''
	One sweep of DMRG algorithm. Positive sweep goes from head (index 0) to tail (index d-1) and negative sweep goes from tail to head.

	Args:
		H: MatrixProductOperator
			The quadratic form that needs to be minimized in MPO format.
		state: TensorTrain
			The current candidate for solution in Tensor Train format.
		is_positive: bool
			Boolean variable encoding the direction of the sweep.
	Returns:
		state: TensorTrain
			The state after the sweep.
	'''
	d = len(state.cores)
	indices = list(range(d))
	if not is_positive:
		indices = indices[::-1]

	for index in indices:
		if verbose:
			print(f'index \t {index}, energy_before: {(H@state)@state}')
		A, B = generate_A_b(H, state, index)

		eps = 1e-5
		B = B + eps * np.eye(B.shape[0]) #Tikhonov hack

		w, v = scipy.linalg.eigh(A, b=B)
		state.cores[index] = v[:,0].reshape(state.cores[index].shape)
		state = state.normalized()
		if verbose:
			print(f'index \t {index}, energy_after: {(H@state)@state}')

	return state


def dmrg_optimize(H: MatrixProductOperator, initial_state: TensorTrain = None, n_sweeps = 10, rank = 10, verbose: bool = False):
	'''
	Run DMRG algorithm. 

	Args:
		H: MatrixProductOperator
			The quadratic form that needs to be minimized in MPO format.
		initial_state: TensorTrain, optional, default = None
			The initial candidate for solution in Tensor Train format. If None, a random is chosen.
		n_sweeps: int, optional, default = 10
			Number of DMRG two-way sweeps (positive + negative).
		rank: int, optional, default = 10
			The rank for random initial state, if one needs to be generated.
	Returns:
		state: TensorTrain
			The candidate for optimal state found by algorithm.
	'''

	if H.ns != H.ms:
		raise ValueError(f"The quadratic form must have same shape transposed!")

	if initial_state is None:
		initial_state = TensorTrain.random_tensor(H.ns, [rank]*(len(H.ns)-1))
		initial_state = initial_state.normalized()

	state = initial_state
	for sweep in range(n_sweeps):
		if verbose:
			print(f'Starting sweep {sweep}, direction positive')
		state = dmrg_sweep(H, state, True, verbose=verbose)
		if verbose:
			print(f'Starting sweep {sweep}, direction negative')
		state = dmrg_sweep(H, state, False, verbose=verbose)

	return state
