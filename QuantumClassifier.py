import matplotlib.pyplot as plt

from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.utils import algorithm_globals

import numpy as np

from sklearn.preprocessing import OneHotEncoder

from OptimizerLog import OptimizerLog

algorithm_globals.random_seed = 3142
np.random.seed(algorithm_globals.random_seed)


class QuantumClassifier:
    def __init__(self, x_train, x_test, y_train, y_test, feature_map, var_circ, optimizer, n_samples, n_features):
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_map = feature_map
        self.var_circ = var_circ
        self.optimizer = optimizer
        self.n_samples = n_samples
        self.n_features = n_features

        self.circuit = None

        self.train_labels_oh = None
        self.test_labels_oh = None

        self.one_hot_encoding()
        initial_point = np.random.random(self.var_circ.num_parameters)

        self.vqc = VQC(feature_map=self.feature_map,
                       ansatz=self.var_circ,
                       loss='cross_entropy',
                       optimizer=self.optimizer,
                       initial_point=initial_point)

    def make_circuit(self):
        self.circuit = self.feature_map.compose(self.var_circ)
        self.circuit.measure_all()
        self.circuit.decompose().draw(output='mpl', fold=15)
        plt.savefig('img/variational_circuit/variational_circuit')

    def one_hot_encoding(self):
        one_hot_encoder = OneHotEncoder()
        self.train_labels_oh = one_hot_encoder.fit_transform(self.y_train.reshape(-1, 1)).toarray()
        self.test_labels_oh = one_hot_encoder.fit_transform(self.y_test.reshape(-1, 1)).toarray()

    def train(self):
        print('Training...')
        self.vqc.fit(self.X_train, self.train_labels_oh)

        self.feature_map.compose(self.var_circ.bind_parameters(self.vqc.weights))\
            .decompose().draw(output='mpl', fold=15)
        plt.savefig('img/variational_circuit/trained_variational_circuit')

    def test(self):
        print('Testing...')
        score = self.vqc.score(self.X_test, self.test_labels_oh)

        return score

    # binds the VQC parameters to the variational circuit and adds the feature map
    def make_param_circuit(self, param_feature_map):
        var_form_param = self.var_circ.bind_parameters(self.vqc.weights)
        param_circuit = param_feature_map.compose(var_form_param)
        param_circuit.measure_all()

        return param_circuit
