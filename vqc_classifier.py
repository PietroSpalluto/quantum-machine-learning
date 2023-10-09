import seaborn as sns
from qiskit.primitives import Sampler
from qiskit.algorithms.optimizers import SPSA, QNSPSA, GradientDescent, ADAM, COBYLA, NELDER_MEAD
from qiskit.circuit.library import ZZFeatureMap, TwoLocal, PauliFeatureMap, NLocal, RealAmplitudes, EfficientSU2
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.algorithms.classifiers import VQC

from sklearn.decomposition import PCA

from VQCClassifier import VQCClassifier
from ClassifierLog import ClassifierLog

from generate_circuit import generate_circuit_2local

import numpy as np

import os
import time
import joblib

np.random.seed(42)

# dataset definition
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

# train and test splitting
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    labels,
                                                    train_size=n_train,
                                                    test_size=n_test,
                                                    stratify=labels)

# dimensionality reduction
pca = PCA(n_components=2)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
n_features = 2

# a random point is selected to be represented as classical and quantum data
random_point = np.random.randint(len(data))

# Sampler
sampler = Sampler()

# Feature maps are created
feature_map_zz_1 = ZZFeatureMap(n_features, reps=1)
feature_map_zz_3 = ZZFeatureMap(n_features, reps=3)
# feature_map_pauli = PauliFeatureMap(n_features, reps=1, paulis=['ZZ', 'ZX', 'ZY'])  # same as ZZFeatureMap
feature_map_pauli = PauliFeatureMap(n_features, reps=1, paulis=['ZZ', 'ZX', 'ZY', 'XY'])
feature_map_zzz = PauliFeatureMap(n_features, reps=1, paulis=['Z', 'ZZ', 'ZZZ'])
pop = np.array([[0, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 0, 1],
                [1, 0, 1, 1, 1, 1],
                [0, 1, 1, 0, 0, 1],
                [0, 1, 0, 0, 1, 1],
                [1, 0, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 1],
                [1, 0, 1, 0, 0, 0]])
# generating feature map from genes obtained with the genetic algorithm
custom_feature_map, _, _, _ = generate_circuit_2local(pop, n_features, n_features)

# Ansatze are created
ansatz_tl_3 = TwoLocal(n_features, ['ry', 'rz'], 'cz', reps=3)
ansatz_ra_3 = RealAmplitudes(num_qubits=n_features, reps=3)
ansatz_esu2 = EfficientSU2(n_features, su2_gates=['rx', 'y'], entanglement='circular', reps=1)
ansatz_nl = NLocal(n_features, reps=3)

# lists of feature maps and ansatze to be used
feature_maps = [feature_map_zz_1, feature_map_zz_3, feature_map_pauli, custom_feature_map]
ansatze = [ansatz_ra_3, ansatz_tl_3, ansatz_esu2]

# make configurations
conf = {'feature_map': [], 'ansatz': [], 'optimizer': [],
        'log_loss': [], 'clf': [], 'score': [], 'training_time(s)': []}

for feature_map in feature_maps:
    for ansatz in ansatze:
        # Optimizers, some of them need an ansatz first
        optimizer_spsa = SPSA(maxiter=100, blocking=True)
        optimizer_qnspsa = QNSPSA(QNSPSA.get_fidelity(ansatz), maxiter=100, blocking=True)
        optimizer_cobyla = COBYLA(maxiter=377)
        optimizer_nm = NELDER_MEAD(maxiter=377, maxfev=1000, adaptive=True)
        optimizer_gd = GradientDescent()
        optimizer_adam = ADAM(maxiter=100)

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

    # print configuration info
    print('testing configuration {}/{}'.format(c+1, len(conf['feature_map'])))
    print('FEATURE MAP')
    print('name: {}'.format(feature_map.name))
    print('#parameters: {}'.format(feature_map.num_parameters))
    # print('#repetitions: {}'.format(feature_map.reps))
    # print('gates: {}'.format(feature_map.paulis))
    # print('entanglement: {}'.format(feature_map.entanglement))
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
    # the classifier it the Variational Quantum Classifier
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
