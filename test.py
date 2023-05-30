from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit import QuantumCircuit
from qiskit.visualization.bloch import Bloch
from qiskit.visualization import plot_bloch_multivector

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs


# Converts state vectors to points on the Bloch sphere
def state_to_bloch(state_vec, x, y):
    phi = np.angle(state_vec.data[y])-np.angle(state_vec.data[x])
    theta = 2*np.arccos(np.abs(state_vec.data[x]))
    return [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]


def make_bloch_spheres(n_features):
    width, height = plt.figaspect(1)
    spheres= []
    figures = []

    for i in range(n_features):
        figures.append(plt.figure(figsize=(width, height)))
        ax = figures[i].add_subplot(projection='3d')
        b = Bloch(axes=ax)
        b.point_color = ['tab:blue']
        b.point_marker = ['o']
        b.point_size = [2]
        spheres.append(b)

    return spheres, figures


def update_bloch_spheres(spheres):
    i = 0
    for b in spheres:
        b.add_points(state_to_bloch(outputstate, i, i + 1))
        i += 2

    return spheres


n_features = 3
n_centers = 2

features, labels = make_blobs(
    n_samples=200,
    n_features=n_features,
    centers=n_centers,
    center_box=(-1, 1),
    cluster_std=0.1,
)

df = pd.DataFrame(features, columns=range(n_features))
df['labels'] = labels
sns.pairplot(df, hue='labels', corner=True)
plt.savefig('img/data_points')

# empty quantum circuit
circ = QuantumCircuit(n_features)
feature_map = ZZFeatureMap(n_features, reps=3)
circ.compose(feature_map.decompose(), range(n_features), inplace=True)

spheres, figures = make_bloch_spheres(n_features)

backend = Aer.get_backend('statevector_simulator')

for i in range(len(features)):
    # print(features[i].tolist())
    encode = circ.bind_parameters(features[i].tolist())

    job = backend.run(encode)
    result = job.result()
    outputstate = result.get_statevector(encode, decimals=3)
    # print(outputstate)
    spheres = update_bloch_spheres(spheres)

for i, (b, fig) in enumerate(zip(spheres, figures)):
    b.show()
    # fig.axes[0].set_title('qubit {}'.format(i))
    fig.savefig('img/qubit_{}'.format(i))
    # plt.close()

meas = QuantumCircuit(2, 2)
meas.barrier(range(2))
meas.measure(range(2), range(2))
circ.compose(meas, range(2), inplace=True)

spheres, figures = make_bloch_spheres(n_features)

for i in range(len(features)):
    # print(features[i].tolist())
    encode = circ.bind_parameters(features[i].tolist())

    job = backend.run(encode)
    result = job.result()
    outputstate = result.get_statevector(encode, decimals=3)
    # print(outputstate)
    spheres = update_bloch_spheres(spheres)

for i, (b, fig) in enumerate(zip(spheres, figures)):
    b.show()
    # fig.axes[0].set_title('qubit {}'.format(i))
    fig.savefig('img/meas_qubit_{}'.format(i))
    # plt.close()

print('end')
