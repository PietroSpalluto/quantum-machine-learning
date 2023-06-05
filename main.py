import matplotlib.pyplot as plt
from qiskit import Aer
from qiskit.circuit.library import PauliFeatureMap
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit import QuantumCircuit
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms import QSVC
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_state_city, plot_bloch_multivector
from qiskit.visualization import plot_state_paulivec, plot_state_hinton
from qiskit.visualization import plot_state_qsphere

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

features, labels = make_blobs(
    n_samples=20,
    centers=2,
    center_box=(-1, 1),
    cluster_std=0.1,
)

# plt.scatter(features[:, 0], features[:, 1])

# empty quantum circuit
circ = QuantumCircuit(2)
# make a feature map and decompose to add it to the quantum circuit
# decomposition is done because job.result() does not recognize ZZFeatureMap, but it recognizes single gates
feature_map = ZZFeatureMap(2, reps=1)
circ.compose(feature_map.decompose(), range(2), inplace=True)

circ = circ.bind_parameters(features[0].tolist())

print(features[0].tolist())
circ.draw(output='mpl')
plt.show()

backend = Aer.get_backend('statevector_simulator')
job = backend.run(circ)
result = job.result()
outputstate = result.get_statevector(circ, decimals=3)
print(outputstate)

meas = QuantumCircuit(2, 2)
meas.barrier(range(2))
meas.measure(range(2), range(2))

circ.compose(meas, range(2), inplace=True)
circ.draw('mpl')
plt.show()

job = backend.run(circ)
result = job.result()
outputstate = result.get_statevector(circ, decimals=3)
print(outputstate)

backend = AerSimulator()
qc_compiled = transpile(circ, backend)
job_sim = backend.run(qc_compiled, shots=1024)
result_sim = job_sim.result()
counts = result_sim.get_counts(qc_compiled)
print(counts)

fidelity = ComputeUncompute(sampler=Sampler())
new_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

qsvc = QSVC(quantum_kernel=new_kernel)
qsvc.fit(features, labels)
print(qsvc.score(features, labels))
