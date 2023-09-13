from qiskit import quantum_info
from qiskit import execute, Aer
from qiskit.circuit.library import TwoLocal, RealAmplitudes, EfficientSU2

import scipy as sp

import numpy as np


def random_unitary(N):
    """
        Return a Haar distributed random unitary from U(N)
    """

    Z = np.random.randn(N, N) + 1.0j * np.random.randn(N, N)
    [Q, R] = sp.linalg.qr(Z)
    D = np.diag(np.diagonal(R) / np.abs(np.diagonal(R)))
    return np.dot(Q, D)


def haar_integral(num_qubits, samples):
    """
        Return calculation of Haar Integral for a specified number of samples.
    """

    N = 2**num_qubits
    randunit_density = np.zeros((N, N), dtype=complex)
    zero_state = np.zeros(N, dtype=complex)
    zero_state[0] = 1

    for _ in range(samples):
      A = np.matmul(zero_state, random_unitary(N)).reshape(-1,1)
      randunit_density += np.kron(A, A.conj().T)

    randunit_density/=samples

    return randunit_density


def pqc_integral(num_qubits, ansatze, size, samples):
    """
        Return calculation of Integral for a PQC over the uniformly sampled
        the parameters θ for the specified number of samples.
    """

    N = num_qubits
    randunit_density = np.zeros((2**N, 2**N), dtype=complex)

    for _ in range(samples):
      params = np.random.uniform(-np.pi, np.pi, size)
      ansatz = ansatze(params, N)
      result = execute(ansatz, backend=Aer.get_backend('statevector_simulator')).result()
      U = np.reshape(result.get_statevector(ansatz, decimals=5), (-1,1))
      randunit_density += np.kron(U, U.conj().T)

    return randunit_density/samples


def meyer_wallach(circuit, num_qubits, size, sample=1024):
    """
        Returns the meyer-wallach entanglement measure for the given circuit.
    """

    res = np.zeros(sample, dtype=complex)
    N = num_qubits

    for i in range(sample):
        params = np.random.uniform(-np.pi, np.pi, size)
        ansatz = circuit(params, N)
        result = execute(ansatz, backend=Aer.get_backend('statevector_simulator')).result()
        U = np.reshape(result.get_statevector(ansatz, decimals=5), (-1, 1))
        entropy = 0
        qb = list(range(N))

        for j in range(N):
            dens = quantum_info.partial_trace(U, qb[:j]+qb[j+1:]).data
            trace = np.trace(dens**2)
            entropy += trace

        entropy /= N
        res[i] = 1 - entropy

    return 2*np.sum(res).real/sample


def two_local(params, num_qubits):
    ansatz = TwoLocal(num_qubits, ['ry', 'rz'], 'cz', reps=3)
    ansatz = ansatz.bind_parameters(params)

    return ansatz


def real_amplitudes(params, num_qubits):
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=3)
    ansatz = ansatz.bind_parameters(params)

    return ansatz


def efficient_su2(params, num_qubits):
    ansatz = EfficientSU2(num_qubits, su2_gates=['rx', 'y'], entanglement='circular', reps=1)
    ansatz = ansatz.bind_parameters(params)

    return ansatz


num_qubits = 2
shots = 8192

num_params = 16
expr = np.linalg.norm(haar_integral(num_qubits, shots) - pqc_integral(num_qubits, two_local, num_params, shots))
print('TwoLocal expressibility: {}'.format(expr))
ent = meyer_wallach(two_local, num_qubits, num_params)
print('TwoLocal entangling capability: {}'.format(ent))

num_params = 8
expr = np.linalg.norm(haar_integral(num_qubits, shots) - pqc_integral(num_qubits, real_amplitudes, num_params, shots))
print('RealAmplitudes expressibility: {}'.format(expr))
ent = meyer_wallach(real_amplitudes, num_qubits, num_params)
print('RealAmplitudes entangling capability: {}'.format(ent))

num_params = 4
expr = np.linalg.norm(haar_integral(num_qubits, shots) - pqc_integral(num_qubits, efficient_su2, num_params, shots))
print('EfficientSU2 expressibility: {}'.format(expr))
ent = meyer_wallach(efficient_su2, num_qubits, num_params)
print('EfficientSU2 entangling capability: {}'.format(ent))