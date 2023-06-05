import matplotlib.pyplot as plt
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import ZZFeatureMap, TwoLocal, PauliFeatureMap
from qiskit.visualization import plot_histogram
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit import Aer, transpile
from sklearn.model_selection import train_test_split

from QuantumEncoder import QuantumEncoder
from QuantumClassifier import QuantumClassifier
from OptimizerLog import OptimizerLog

import numpy as np

from sklearn.datasets import make_blobs


def make_data(data_type, n_features, n_classes, n_train, n_test):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    if data_type == 'blobs':
        features, labels = make_blobs(
            n_samples=n_train + n_test,
            n_features=n_features,
            centers=n_classes,
            center_box=(-1, 1),
            cluster_std=0.1)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=n_train, test_size=n_test)

    elif data_type == 'ad-hoc':
        X_train, y_train, X_test, y_test = (ad_hoc_data(training_size=n_train,
                                                        test_size=n_test,
                                                        n=n_features,
                                                        gap=0.3,
                                                        one_hot=False))

    return X_train, X_test, y_train, y_test


n_features = 3  # number of features
n_classes = 2  # number of classes (clusters)
n_train = 20  # number of samples in the training set
n_test = 10  # number of samples in the test set
data_type = 'blobs'

# Feature map Selection
feature_map = ZZFeatureMap(n_features, reps=3)
# feature_map = PauliFeatureMap(2, reps=1, paulis=['ZZ'])  # same as ZZFeatureMap

# Ansatz selection
var_circ = TwoLocal(n_features, ['ry', 'rz'], 'cz', reps=3)

# Optimizer selection
optimizer = SPSA(maxiter=100, blocking=True)

X_train, X_test, y_train, y_test = make_data(data_type, n_features, n_classes, n_train, n_test)

# a random point to be represented as classical and quantum data
random_point = np.random.randint(n_test + n_train)

qe = QuantumEncoder(np.concatenate((X_train, X_test)), feature_map)
qe.plot_data_points(random_point)  # saves data points plot
qe.make_circuit()  # makes the circuit with the feature map
qe.get_statevectors(random_point)  # gets the state vectors of a random point
qe.add_measurement()  # add a measurement to the circuit
qe.get_statevectors(random_point, meas=True)  # gets the measured state vectors of a random point

n_samples = n_train + n_test
# log = OptimizerLog()
vc = QuantumClassifier(X_train, X_test, y_train, y_test, feature_map, var_circ, optimizer, n_samples, n_features)

vc.make_circuit()  # makes the circuit with the feature map, the ansatz and the measurement
vc.train()  # trains the algorithm
score = vc.test()  # gets an accuracy score
print('Testing score: {}'.format(score))

# run simulations on the trained circuit and plots results
data = np.concatenate((X_train, X_test))
labels = np.concatenate((y_train, y_test))
backend_sim = Aer.get_backend('qasm_simulator')
width, height = plt.figaspect(1 / 1)
res = []
for i, (d, l) in enumerate(zip(data, labels)):
    param_feature_map = feature_map.bind_parameters(d)
    param_circuit = vc.make_param_circuit(feature_map)
    job_sim = backend_sim.run(transpile(param_circuit, backend_sim), shots=2048)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(param_circuit)
    if i == random_point:
        # saves the trained circuit with the parameters of the random point
        param_circuit.decompose().draw(output='mpl')
        plt.savefig('img/feature_map/param_trained_variational_circuit')

        # saves the histogram containing the results of the simulation
        plot_histogram(counts, figsize=(width, height))
        plt.savefig('img/variational_circuit/point_sim_result')
        print('Random point label: {}'.format(l))
        print('Random point predicted label: {}'.format(vc.vqc.predict(d)))
    res.append(counts)

# plots the total number of outcomes
sim_outcomes = {}
for r in res:
    for key in r.keys():
        if key not in sim_outcomes.keys():
            sim_outcomes[key] = 0
        sim_outcomes[key] = sim_outcomes[key] + r[key]
print(sim_outcomes)
plot_histogram(sim_outcomes, figsize=(width, height))
plt.savefig('img/variational_circuit/sim_result')

print('end')
