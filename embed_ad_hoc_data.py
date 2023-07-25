from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from sklearn.preprocessing import OneHotEncoder
from qiskit_machine_learning.algorithms.classifiers import VQC

from ClassifierLog import ClassifierLog
from QuantumEncoder import QuantumEncoder

import numpy as np

algorithm_globals.random_seed = 3142
np.random.seed(algorithm_globals.random_seed)

X_train, y_train, X_test, y_test = (
    ad_hoc_data(training_size=20,
                test_size=5,
                n=2,
                gap=0.3,
                one_hot=False)
)

feature_map = ZZFeatureMap(feature_dimension=2, reps=2)
ansatz = TwoLocal(2, ['ry', 'rz'], 'cz', reps=2)

circuit = feature_map.compose(ansatz)
circuit.measure_all()
circuit.decompose().draw()

encoder = OneHotEncoder()
train_labels_oh = encoder.fit_transform(y_train.reshape(-1, 1)
                                        ).toarray()
test_labels_oh = encoder.fit_transform(y_test.reshape(-1, 1)
                                       ).toarray()

# initial_point = np.random.random(ansatz.num_parameters)
initial_point = np.array([0.3200227, 0.6503638, 0.55995053,
                          0.96566328, 0.38243769, 0.90403094,
                          0.82271449, 0.26810137, 0.61076489,
                          0.82301609, 0.11789148, 0.29667125])

clf_log = ClassifierLog()
vqc = VQC(feature_map=feature_map,
          ansatz=ansatz,
          loss='cross_entropy',
          optimizer=COBYLA(maxiter=100),
          callback=clf_log.update,
          initial_point=initial_point)

vqc.fit(X_train, train_labels_oh)

print(vqc.score(X_test, test_labels_oh))

data = np.concatenate((X_train, X_test))
labels = np.concatenate((y_train, y_test))

random_point = np.random.randint(len(data))
qe = QuantumEncoder(data, labels)
qe.encode(circuit.decompose(), vqc.weights, measured=True)
qe.init_plots()
qe.add_data_points(random_point)
qe.save_bloch_spheres('ad_hoc')

print('end')
