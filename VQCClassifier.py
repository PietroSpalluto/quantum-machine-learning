from qiskit.utils import algorithm_globals

import numpy as np

from sklearn.preprocessing import OneHotEncoder

algorithm_globals.random_seed = 3142
np.random.seed(algorithm_globals.random_seed)


# class to train and test the VQC
class VQCClassifier:
    def __init__(self, x_train, x_test, y_train, y_test, feature_map, ansatz, optimizer, clf, n_features):
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.clf = clf
        self.n_features = n_features

        self.circuit = None

        self.train_labels_oh = None
        self.test_labels_oh = None

        self.one_hot_encoding()

    def make_circuit(self):
        """
        Feature map and ansatz are merged and the measurement is added to the circuit
        """
        self.circuit = self.feature_map.compose(self.ansatz)
        self.circuit.measure_all()

    def one_hot_encoding(self):
        """
        One hot encoding of the features
        """
        one_hot_encoder = OneHotEncoder()
        self.train_labels_oh = one_hot_encoder.fit_transform(self.y_train.reshape(-1, 1)).toarray()
        self.test_labels_oh = one_hot_encoder.fit_transform(self.y_test.reshape(-1, 1)).toarray()

    def train(self):
        """
        Circuit parameters training
        """
        print('Training...')
        self.clf.fit(self.X_train, self.train_labels_oh)

    def test(self):
        """
        Model testing
        """
        print('Testing...')
        score = self.clf.score(self.X_test, self.test_labels_oh)

        return score

    def make_param_circuit(self, param_feature_map):
        """
        Binds the VQC parameters to the variational circuit and adds the feature map
        to obtain a parameterized circuit
        :param param_feature_map: parameterized feature map (feature map encoding a data point)
        :return: the complete parameterized circuit
        """
        var_form_param = self.ansatz.bind_parameters(self.clf.weights)
        param_circuit = param_feature_map.compose(var_form_param)
        param_circuit.measure_all()

        return param_circuit
