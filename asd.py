from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit import QuantumCircuit
from qiskit.visualization.bloch import Bloch
from qiskit.visualization import plot_bloch_multivector

import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import make_blobs


# Converts state vectors to points on the Bloch sphere
def state_to_bloch(state_vec, x, y):
    phi = np.angle(state_vec.data[y])-np.angle(state_vec.data[x])
    theta = 2*np.arccos(np.abs(state_vec.data[x]))
    return [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]


features, labels = make_blobs(
    n_samples=200,
    centers=2,
    center_box=(-1, 1),
    cluster_std=0.1,
)

plt.scatter(features[:, 0], features[:, 1])

# empty quantum circuit
circ = QuantumCircuit(2)
feature_map = ZZFeatureMap(2, reps=3)
circ.compose(feature_map.decompose(), range(2), inplace=True)

# Bloch sphere plot formatting
width, height = plt.figaspect(1/2)
fig = plt.figure(figsize=(width, height))
ax1, ax2 = fig.add_subplot(1, 2, 1, projection='3d'), fig.add_subplot(1, 2, 2, projection='3d')
b1, b2 = Bloch(axes=ax1), Bloch(axes=ax2)
b1.point_color, b2.point_color = ['tab:blue'], ['tab:blue']
b1.point_marker, b2.point_marker = ['o'], ['o']
b1.point_size, b2.point_size = [2], [2]

for i in range(len(features)):
    # print(features[i].tolist())
    encode = circ.bind_parameters(features[i].tolist())

    backend = Aer.get_backend('statevector_simulator')
    job = backend.run(encode)
    result = job.result()
    outputstate = result.get_statevector(encode, decimals=3)
    # print(outputstate)
    b1.add_points(state_to_bloch(outputstate, 0, 1))
    b2.add_points(state_to_bloch(outputstate, 2, 3))

b1.show()
b2.show()
plt.show()

print('end')
