import matplotlib.pyplot as plt
from qiskit.primitives import Sampler
from qiskit.algorithms.optimizers import SPSA, QNSPSA, GradientDescent, ADAM, COBYLA
from qiskit.circuit.library import ZZFeatureMap, TwoLocal, PauliFeatureMap, NLocal, RealAmplitudes, EfficientSU2
from qiskit.visualization import plot_histogram
from qiskit import Aer, transpile, QuantumCircuit
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.algorithms.classifiers import VQC, QSVC, PegasosQSVC, NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.kernels import FidelityQuantumKernel, TrainableFidelityQuantumKernel
from qiskit import QuantumCircuit
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit.circuit import ParameterVector, Parameter
from qiskit_machine_learning.utils.loss_functions import SVCLoss
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer

from VQCClassifier import VQCClassifier
from OptimizerLog import OptimizerLog
from ClassifierLog import ClassifierLog
from OptimizerLog import OptimizerLog
from QuantumEncoder import QuantumEncoder

import numpy as np
import pandas as pd

feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
label_name = 'Species'
n_features = len(feature_names)  # number of features
n_train = 0.8  # number of samples in the training set
n_test = 0.2  # number of samples in the test set

data = pd.read_csv('data/Iris.csv')
features = data[feature_names]
# mapping of string to number
mapping_dict = {class_name: id for id, class_name in enumerate(data[label_name].unique())}
inverse_dict = {id: class_name for id, class_name in enumerate(data[label_name].unique())}
labels = data[label_name].map(mapping_dict)

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

# make a feature map
feature_map = ZZFeatureMap(n_features, reps=1)

# add trainable gate at the beginning of the circuit
# training_params = [ParameterVector('θ', 1)]
training_params = [Parameter('θ[{}]'.format(i)) for i in range(4)]
# training_params = [Parameter('θ')] * 4
circ = QuantumCircuit(n_features)
circ.ry(training_params[0], 0)
circ.ry(training_params[1], 1)
circ.ry(training_params[2], 2)
circ.ry(training_params[3], 3)

# make trainable feature map
feature_map = circ.compose(feature_map)
feature_map.decompose().draw(output='mpl')
plt.savefig('img/qsvm/trainable_feature_map')

# instantiate a trainable kernel
fidelity = ComputeUncompute(sampler=Sampler())
# kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
kernel = TrainableFidelityQuantumKernel(feature_map=feature_map,
                                        fidelity=fidelity,
                                        training_parameters=training_params)

opt_log = OptimizerLog()
optimizer = SPSA(maxiter=50, callback=opt_log.update)
loss = SVCLoss(C=1.0)
trainer = QuantumKernelTrainer(quantum_kernel=kernel, loss=loss, optimizer=optimizer)

# optimize the kernel
print('optimizing quantum kernel...')
results = trainer.fit(X_train, y_train)
kernel = results.quantum_kernel

# save the optimized kernel
# ...

# save kernel matrices
kernel_matrix_train = kernel.evaluate(x_vec=X_train)
plt.imshow(np.asmatrix(kernel_matrix_train), interpolation='nearest', origin='upper', cmap='Blues')
plt.title('Training kernel matrix')
plt.savefig('img/qsvm/kernel_matrix_train')

kernel_matrix_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)
plt.imshow(np.asmatrix(kernel_matrix_test), interpolation='nearest', origin='upper', cmap='Reds')
plt.title('Testing kernel matrix')
plt.savefig('img/qsvc/kernel_matrix_test')

qsvc = QSVC(quantum_kernel=kernel)
print('training qsvc...')
qsvc.fit(X_train, y_train)
score = qsvc.score(X_test, y_test)
print('testing score: {}'.format(score))

print('end')
