import matplotlib.pyplot as plt
import seaborn as sns
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit

from QuantumEncoder import QuantumEncoder

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import joblib

feature_names = ['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']
label_name = 'species'
n_features = len(feature_names)  # number of features
n_train = 0.8  # number of samples in the training set
n_test = 0.2  # number of samples in the test set

data = sns.load_dataset('penguins')
data = data.dropna()
print(data.isnull().sum().sum())

features = data[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
features['island'] = features['island'].copy().map({'Torgersen': 0, 'Biscoe': 1, 'Dream': 2})
features['sex'] = features['sex'].copy().map({'Male': 0, 'Female': 1})
mapping_dict = {class_name: id for id, class_name in enumerate(data[label_name].unique())}
inverse_dict = {id: class_name for id, class_name in enumerate(data[label_name].unique())}
labels = data['species'].map(mapping_dict)

n_classes = len(labels.unique())  # number of classes (clusters)

# numpy array conversion
features = features.to_numpy()
labels = labels.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    labels,
                                                    train_size=n_train,
                                                    test_size=n_test,
                                                    stratify=labels)

# a random point to be represented as classical and quantum data
random_point = np.random.randint(len(data))

# load the configurations
conf = joblib.load('models/results_penguins')

# select the best configuration according to the score
best_clf_idx = np.argmax(conf['score'])
best_clf = conf['clf'][best_clf_idx]

# plot classical data and encoded quantum data
circuit = QuantumCircuit(n_features)
circuit.compose(best_clf.feature_map.decompose(), range(n_features), inplace=True)

qe = QuantumEncoder(features, labels)
qe.encode(circuit, [])
qe.init_plots()
qe.add_data_points(random_point)
qe.save_bloch_spheres('fig')
qe.plot_data_points(random_point, feature_names, inverse_dict)

# plot the same data after training a variational circuit
qe.encode(best_clf.circuit.decompose(), best_clf.clf.weights, measured=True)
qe.init_plots()
qe.add_data_points(random_point)
qe.save_bloch_spheres('fig_trained')

# plot the used feature map
best_clf.feature_map.decompose().draw(output='mpl')
plt.savefig('img/variational_circuit/feature_map')

# plot the used ansatz
best_clf.ansatz.decompose().draw(output='mpl')
plt.savefig('img/variational_circuit/ansatz')

# plot the complete circuit
best_clf.circuit.decompose().draw(output='mpl', fold=15)
plt.savefig('img/variational_circuit/variational_circuit')

# plot a reduced version of the complete circuit
best_clf.circuit.draw(output='mpl')
plt.savefig('img/variational_circuit/variational_circuit_reduced')

# plot the trained variational circuit
best_clf.feature_map.compose(best_clf.ansatz.bind_parameters(best_clf.clf.weights))\
    .decompose().draw(output='mpl', fold=15)
# self.feature_map.compose(self.var_circ.bind_parameters(self.clf.weights)).draw(output='mpl', fold=15)
plt.savefig('img/variational_circuit/trained_variational_circuit')

# plot the losses
plt.clf()
plt.rcParams["figure.figsize"] = (6, 4)
for i in range(len(conf['log_loss'])):
    plt.plot([*range(len(conf['log_loss'][i]))], conf['log_loss'][i], label='configuration {}'.format(i+1))
plt.legend()
plt.savefig('img/variational_circuit/losses')

# plot training times
times = conf['training_time(s)']
names = ['configuration {}'.format(i+1) for i in range(len(times))]
plt.clf()
plt.barh(names, times)
plt.title('Training times')
plt.savefig('img/variational_circuit/training_times')

# plot scores
scores = conf['score']
names = ['configuration {}'.format(i+1) for i in range(len(scores))]
plt.clf()
plt.barh(names, scores)
plt.title('Accuracy scores')
plt.savefig('img/variational_circuit/accuracy scores')

params = np.concatenate((features[random_point], best_clf.clf.weights))
best_clf.circuit.decompose().bind_parameters(params).draw(output='mpl', fold=15)
plt.savefig('img/variational_circuit/variational_circuit_random_point')

# # run simulations on the trained circuit and plots results
# data = np.concatenate((X_train, X_test))
# labels = np.concatenate((y_train, y_test))
# backend_sim = Aer.get_backend('qasm_simulator')
# width, height = plt.figaspect(1 / 1)
# res = []
# for i, (d, l) in enumerate(zip(data, labels)):
#     param_feature_map = best_clf.feature_map.bind_parameters(d)
#     param_circuit = best_clf.make_param_circuit(best_clf.feature_map)
#     job_sim = backend_sim.run(transpile(param_circuit, backend_sim), shots=2048)
#     result_sim = job_sim.result()
#     counts = result_sim.get_counts(param_circuit)
#     if i == random_point:
#         # saves the trained circuit with the parameters of the random point
#         param_circuit.decompose().draw(output='mpl')
#         plt.savefig('img/feature_map/param_trained_variational_circuit')
#
#         # saves the histogram containing the results of the simulation
#         plot_histogram(counts, figsize=(width, height))
#         plt.savefig('img/variational_circuit/point_sim_result')
#         print('Random point label: {}'.format(l))
#         print('Random point predicted label: {}'.format(best_clf.clf.predict(d)))
#     res.append(counts)
#
# # plots the total number of outcomes
# sim_outcomes = {}
# for r in res:
#     for key in r.keys():
#         if key not in sim_outcomes.keys():
#             sim_outcomes[key] = 0
#         sim_outcomes[key] = sim_outcomes[key] + r[key]
# print(sim_outcomes)
# plot_histogram(sim_outcomes, figsize=(width, height))
# plt.savefig('img/variational_circuit/sim_result')

print('end')
