import matplotlib.pyplot as plt
import seaborn as sns
from qiskit.primitives import Sampler
from qiskit.algorithms.optimizers import SPSA, QNSPSA, GradientDescent, ADAM, COBYLA, NELDER_MEAD
from qiskit.circuit.library import ZZFeatureMap, TwoLocal, PauliFeatureMap, NLocal, RealAmplitudes, EfficientSU2
from qiskit.visualization import plot_histogram
from qiskit import Aer, transpile, QuantumCircuit
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.algorithms.classifiers import VQC, QSVC, PegasosQSVC, NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN

from sklearn.decomposition import PCA

from VQCClassifier import VQCClassifier
from OptimizerLog import OptimizerLog
from ClassifierLog import ClassifierLog
from QuantumEncoder import QuantumEncoder

import numpy as np
import pandas as pd

import os
import time
import joblib

# feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
# label_name = 'Species'
# n_features = len(feature_names)  # number of features
# n_train = 0.8  # number of samples in the training set
# n_test = 0.2  # number of samples in the test set
#
# data = pd.read_csv('data/Iris.csv')
# features = data[feature_names]
# # mapping of string to number
# mapping_dict = {class_name: id for id, class_name in enumerate(data[label_name].unique())}
# inverse_dict = {id: class_name for id, class_name in enumerate(data[label_name].unique())}
# labels = data[label_name].map(mapping_dict)
#
# n_classes = len(labels.unique())  # number of classes (clusters)

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
labels = data['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})

n_classes = len(labels.unique())  # number of classes (clusters)

# numpy array conversion
features = features.to_numpy()
labels = labels.to_numpy()

# # dimensionality reduction to two components
# pca = PCA(n_components=2)
# pca.fit(features)
# print(pca.explained_variance_ratio_)
# features = pca.transform(features)
# n_features = 2

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    labels,
                                                    train_size=n_train,
                                                    test_size=n_test,
                                                    stratify=labels)

# a random point to be represented as classical and quantum data
random_point = np.random.randint(len(data))

# Sampler
sampler = Sampler()

# Feature maps
feature_map_zz_1 = ZZFeatureMap(n_features, reps=1)
feature_map_zz_3 = ZZFeatureMap(n_features, reps=3)
feature_map_pauli = PauliFeatureMap(n_features, reps=1, paulis=['ZZ', 'ZX', 'ZY'])  # same as ZZFeatureMap
feature_map_pauli2 = PauliFeatureMap(n_features, reps=1, paulis=['ZZ', 'ZX', 'ZY', 'XY'])
feature_map_zzz = PauliFeatureMap(n_features, reps=1, paulis=['Z', 'ZZ', 'ZZZ'])

# Ansatzes
ansatz_tl_3 = TwoLocal(n_features, ['ry', 'rz'], 'cz', reps=3)
ansatz_ra_3 = RealAmplitudes(num_qubits=n_features, reps=3)
ansatz_esu2 = EfficientSU2(n_features, su2_gates=['rx', 'y'], entanglement='circular', reps=1)
ansatz_nl = NLocal(n_features, reps=3)

feature_maps = [feature_map_zz_1, feature_map_zz_3, feature_map_pauli2]
feature_maps = [feature_map_pauli2]
ansatze = [ansatz_ra_3, ansatz_tl_3, ansatz_esu2]

# make configurations
conf = {'feature_map': [], 'ansatz': [], 'optimizer': [],
        'log_loss': [], 'clf': [], 'score': [], 'training_time(s)': []}
for feature_map in feature_maps:
    for ansatz in ansatze:
        # Optimizers, some of them need an ansatz first
        optimizer_spsa = SPSA(maxiter=100, blocking=True)
        optimizer_qnspsa = QNSPSA(QNSPSA.get_fidelity(ansatz), maxiter=100, blocking=True)
        optimizer_cobyla = COBYLA(maxiter=100)
        optimizer_nm = NELDER_MEAD(maxiter=100, maxfev=1000, adaptive=True)

        optimizers = [optimizer_spsa, optimizer_qnspsa, optimizer_cobyla, optimizer_nm]
        for optimizer in optimizers:
            conf['feature_map'].append(feature_map)
            conf['ansatz'].append(ansatz)
            conf['optimizer'].append(optimizer)

# load the results of present and restart the testing from the last configuration saved
# the configuration combination must be the same
starting_conf = 0
if os.path.exists('models/results'):
    conf = joblib.load('models/results')
    print('loaded existing models')
    starting_conf = len(conf['clf'])

# test various configurations
for c in range(starting_conf, len(conf['feature_map'])):
    feature_map = conf['feature_map'][c]
    ansatz = conf['ansatz'][c]
    optimizer = conf['optimizer'][c]

    print('testing configuration {}/{}'.format(c+1, len(conf['feature_map'])))
    print('FEATURE MAP')
    print('name: {}'.format(feature_map.name))
    print('#parameters: {}'.format(feature_map.num_parameters))
    print('#repetitions: {}'.format(feature_map.reps))
    print('gates: {}'.format(feature_map.paulis))
    print('entanglement: {}'.format(feature_map.entanglement))
    print('ANSATZ')
    print('name: {}'.format(ansatz.name))
    print('#parameters: {}'.format(ansatz.num_parameters))
    print('#repetitions: {}'.format(ansatz.reps))
    print('OPTIMIZER')
    print(optimizer)

    # classifier selection
    clf_log = ClassifierLog()
    initial_point = np.random.random(ansatz.num_parameters)
    # initial_point = None
    clf = VQC(sampler=sampler,
              feature_map=feature_map,
              ansatz=ansatz,
              loss='cross_entropy',
              optimizer=optimizer,
              initial_point=initial_point,
              callback=clf_log.update)

    q_clf = VQCClassifier(X_train, X_test, y_train, y_test, feature_map, ansatz, optimizer, clf, n_features)

    q_clf.make_circuit()  # makes the circuit with the feature map, the ansatz and the measurement

    start = time.time()
    q_clf.train()  # trains the algorithm
    elapsed = time.time() - start
    print('Training time: {}'.format(elapsed))
    score = q_clf.test()  # gets an accuracy score
    print('Testing score: {}'.format(score))

    conf['log_loss'].append(clf_log.values)
    conf['clf'].append(q_clf)
    conf['score'].append(score)
    conf['training_time(s)'].append(elapsed)

    # incrementally save results
    joblib.dump(conf, 'models/results')

print('end')
