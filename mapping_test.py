import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from QuantumEncoder import QuantumEncoder

from qiskit.visualization import state_visualization
from qiskit.visualization.bloch import Bloch

import matplotlib.pyplot as plt

data = pd.read_csv('data/Iris.csv')
features = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].to_numpy()
labels = data['Species'].to_numpy()
feature_map = ZZFeatureMap(4, reps=3)
qe = QuantumEncoder(features, feature_map)
qe.make_circuit()
state_vectors = qe.encode()

vectors_data = []
for i in range(len(state_vectors)):
    vectors_data.append(state_visualization._bloch_multivector_data(state_vectors[i]))

vectors_data = pd.DataFrame(vectors_data)

# Bloch sphere plot formatting
width, height = plt.figaspect(1 / 1)
figs = []
for i in range(len(vectors_data.columns)):
    fig = plt.figure(figsize=(width, height))
    figs.append(fig)

axes = []
for i in range(len(vectors_data.columns)):
    axis = figs[i].add_subplot(projection='3d')
    axes.append(axis)

spheres = []
colors = [['tab:blue'], ['tab:orange'], ['tab:green']]
markers = [['o'], ['s'], ['d']]
for i in range(len(vectors_data.columns)):
    class_spheres = []
    sphere_alpha = 0.2
    frame_alpha = 0.2
    xlabel = ['$x$', '']
    ylabel = ['$y$', '']
    zlabel = ['$\\left|0\\right>$', '$\\left|1\\right>$']
    for j, c, m in zip(range(3), colors, markers):
        b = Bloch(axes=axes[i])
        b.sphere_alpha = sphere_alpha
        b.frame_alpha = frame_alpha
        b.xlabel = xlabel
        b.ylabel = ylabel
        b.zlabel = zlabel
        sphere_alpha = 0
        frame_alpha = 0
        xlabel = ['', '']
        ylabel = ['', '']
        zlabel = ['', '']
        b.point_marker = m
        b.point_size = [20]
        b.point_color = c
        class_spheres.append(b)

    spheres.append(class_spheres)

sphere_alpha = 0
frame_alpha = 0
xlabel = ['', '']
ylabel = ['', '']
zlabel = ['', '']
for i in range(len(vectors_data.columns)):
    b = Bloch(axes=axes[i])
    b.sphere_alpha = sphere_alpha
    b.frame_alpha = frame_alpha
    b.xlabel = xlabel
    b.ylabel = ylabel
    b.zlabel = zlabel
    b.point_marker = ['o']
    b.point_size = [30]
    b.point_color = 'r'
    spheres[i].append(b)

random_point = np.random.randint(150)

for i in range(len(spheres)):
    for j in range(len(vectors_data)):
        if j != random_point:
            if labels[j] == 'Iris-setosa':
                spheres[i][0].add_points(vectors_data[i].iloc[j])
            elif labels[j] == 'Iris-versicolor':
                spheres[i][1].add_points(vectors_data[i].iloc[j])
            elif labels[j] == 'Iris-virginica':
                spheres[i][2].add_points(vectors_data[i].iloc[j])

    # the random point is added at the end, so it is above other points
    spheres[i][3].add_points(vectors_data[i].iloc[random_point])

i = 0
for c_spheres, fig in zip(spheres, figs):
    for s in c_spheres:
        s.show()
    fig.savefig('img/mapping/fig {}'.format(i))
    i += 1

print('end')
