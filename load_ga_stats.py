import os
import pickle
import time

import numpy as np
import pandas as pd
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms.classifiers import QSVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from QuantumEncoder import QuantumEncoder
from generate_circuit import generate_circuit_2local

np.random.seed(42)

feature_names = ['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']
label_name = 'species'
n_features = len(feature_names)  # number of features
n_train = 0.8  # number of samples in the training set
n_test = 0.2  # number of samples in the test set

# dictionaries to convert categorical variables to numbers
island_map = {'Dream': 0, 'Biscoe': 1, 'Torgersen': 2}
sex_map = {'Male': 0, 'Female': 1}
species_map = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}

data = sns.load_dataset('penguins')
data.dropna(inplace=True)
# mapping of string to number
mapping_dict = {class_name: id for id, class_name in enumerate(data[label_name].unique())}
inverse_dict = {id: class_name for id, class_name in enumerate(data[label_name].unique())}
data['island'] = data['island'].map(island_map)
data['sex'] = data['sex'].map(sex_map)
data['species'] = data['species'].map(species_map)
features = data[feature_names]

labels = data[label_name]

n_classes = len(labels.unique())  # number of classes (clusters)

# numpy array conversion
features = features.to_numpy()
labels = labels.to_numpy()

train_x, test_x, train_y, test_y = train_test_split(features,
                                                    labels,
                                                    train_size=n_train,
                                                    test_size=n_test,
                                                    stratify=labels)

pca = PCA(n_components=2)
pca.fit(train_x)
train_x = pca.transform(train_x)
test_x = pca.transform(test_x)
n_features = 2
features = np.concatenate((train_x, test_x))
labels = np.concatenate((train_y, test_y))

with open('genetic_algorithm/population.pkl', 'rb') as file:
    pop = pickle.load(file)

with open('genetic_algorithm/statistics.pkl', 'rb') as file:
    stats = pickle.load(file)
    gen_save = stats['Generation']
    plt_acc_save = stats['Plot Accuracy']
    plt_gate_save = stats['Plot Gate Complexity']
    fitness_save = stats['Fitness Values']
    score_save = stats['Scores']
    qc_r_save = stats['R Gates']
    qc_h_save = stats['H Gates']
    qc_cnot_save = stats['CNOT Gates']
    qc_swap_save = stats['SWAP Gates']
    cost_pool_save = stats['Cost']
    obj_gate_save = stats['Gate Complexity']
    obj_acc_save = stats['Accuracy']
    population_save = stats['Population']
    parents_save = stats['Parents']
    cost = stats['Mean Pool Fitness']
    obj_gate_list = stats['Mean Pool Gate Complexity']
    obj_acc_list = stats['Mean Pool Accuracy']
    pareto = stats['Pareto']

# saves pareto front plot
for g, plt_acc, plt_gate in zip(gen_save, plt_acc_save, plt_gate_save):
    plt.scatter(plt_acc, plt_gate, s=10, c="#4863A0", alpha=(g + 100 / 2) / (1.5 * 100))
plt.savefig('img/genetic_algorithm/plot')
plt.clf()

obj_x, obj_y1, obj_y2 = np.arange(len(obj_acc_list)), np.array(obj_gate_list), np.array(obj_acc_list)
cost_x, cost_y = np.arange(len(obj_gate_list)), np.array(cost)

# plots gate cost, accuracy and total fitness score
plt.xlabel('Generation')
plt.ylabel('Gate cost')
plt.plot(obj_x+1, obj_y1, label="Gate cost", color="#4863A0")
plt.tick_params(axis='y')
plt.title('Gate Cost')
plt.savefig('img/genetic_algorithm/gate_cost')
plt.clf()

plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.plot(obj_x+1, obj_y2, label="Accuracy", color="#EE9A4D")
plt.tick_params(axis='y')
plt.title('Mean Accuracy Score')
plt.savefig('img/genetic_algorithm/accuracy')
plt.clf()

plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.plot(cost_x+1, cost_y, label="Fitness", color="#3B9C9C")
plt.tick_params(axis='y')
plt.title('Fitness Value')
plt.savefig('img/genetic_algorithm/fitness')
plt.clf()

stats_df = pd.DataFrame(stats)
# the best feature map is the first individual of the last population generated
best_individual = pop[0]

# the best feature map is saved
best_feature_map, _, _, _ = generate_circuit_2local(best_individual, 2, 2)
best_feature_map.draw(output='mpl')
plt.savefig('img/genetic_algorithm/feature_map')

# a random point is selected to be highlighted in the scatter plot and in the Bloch spheres
random_point = np.random.randint(len(data))
qe = QuantumEncoder(features, labels)
qe.encode(best_feature_map, [])
qe.init_plots()
qe.add_data_points(random_point)
qe.save_bloch_spheres('fig_qsvm')
qe.plot_data_points(random_point, feature_names, inverse_dict)

# a QSVC model is fitted to data using the kernel obtained by the genetic algorithm
kernel = FidelityQuantumKernel(feature_map=best_feature_map)
# model_svc = SVC(kernel=kernel.evaluate)
model_svc = QSVC(quantum_kernel=kernel)
start = time.time()
svc_l = model_svc.fit(train_x, train_y)
elapsed = time.time() - start
print('Training time: {}'.format(elapsed))
score = model_svc.score(test_x, test_y)
print(score)
y_pred = model_svc.predict(test_x)
y_score = model_svc.decision_function(test_x)

# classification metrics are obtained
print(classification_report(test_y, y_pred))

ConfusionMatrixDisplay.from_estimator(model_svc, test_x, test_y)
plt.savefig('img/genetic_algorithm/confusion_matrix_qsvm')

fig, ax = plt.subplots()
RocCurveDisplay.from_predictions(test_y, y_score[:, 0], pos_label=0, ax=ax, name=f'{inverse_dict[0]} vs the rest')
RocCurveDisplay.from_predictions(test_y, y_score[:, 1], pos_label=1, ax=ax, name=f'{inverse_dict[1]} vs the rest')
RocCurveDisplay.from_predictions(test_y, y_score[:, 2], pos_label=2, ax=ax, name=f'{inverse_dict[2]} vs the rest')
plt.savefig('img/genetic_algorithm/roc_curves_qsvm')

# decision boundary plot (for quantum algorithms is very time consuming)
X = np.concatenate((train_x, test_x))
y = np.concatenate((train_y, test_y))
# define bounds of the domain
min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
# define the x and y scale
x1grid = np.arange(min1, max1, 1)
x2grid = np.arange(min2, max2, 1)
# create all the lines and rows of the grid
xx, yy = np.meshgrid(x1grid, x2grid)
# flatten each grid to a vector
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
# horizontal stack vectors to create x1,x2 input for the model
grid = np.hstack((r1, r2))
# make predictions for the grid
if os.path.exists('genetic_algorithm/grid_prediction.pkl'):
    with open('genetic_algorithm/grid_prediction.pkl', 'rb') as file:
        yhat = pickle.load(file)
else:
    yhat = []
for i in range(len(yhat), len(grid)):
    point = grid[i]
    yhat.append(model_svc.predict(point)[0])
    if i % 1000 == 0:
        print(i)
        with open('genetic_algorithm/grid_prediction.pkl', 'wb') as file:
            pickle.dump(yhat, file)
yhat = np.array(yhat)
# yhat = model_svc.predict(grid)
# reshape the predictions back into a grid
zz = yhat.reshape(xx.shape)
# plot the grid of x, y and z values as a surface
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(name='',
                                                           colors=['tab:blue', 'tab:orange', 'tab:green'])
plt.contourf(xx, yy, zz, cmap=cmap, alpha=0.5)
for class_value in range(3):
    # get row indexes for samples with this class
    row_ix = np.where(y == class_value)
    # create scatter of these samples
    plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap=cmap)
plt.show()

print('end')
