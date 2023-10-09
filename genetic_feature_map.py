from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from GeneticAlgorithm import GeneticAlgorithm

import seaborn as sns

import numpy as np

np.random.seed(42)

# dataset definition
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
data['island'] = data['island'].map(island_map)
data['sex'] = data['sex'].map(sex_map)
data['species'] = data['species'].map(species_map)
data.dropna(inplace=True)
features = data[feature_names]

# mapping of string to number
mapping_dict = {class_name: id for id, class_name in enumerate(data[label_name].unique())}
inverse_dict = {id: class_name for id, class_name in enumerate(data[label_name].unique())}
labels = data[label_name].map(mapping_dict)

n_classes = len(labels.unique())  # number of classes (clusters)

# numpy array conversion
features = features.to_numpy()
labels = labels.to_numpy()

# train and test splitting
train_x, test_x, train_y, test_y = train_test_split(features,
                                                    labels,
                                                    train_size=n_train,
                                                    test_size=n_test,
                                                    stratify=labels)

# dimensionality reduction
pca = PCA(n_components=2)
pca.fit(train_x)
train_x = pca.transform(train_x)
test_x = pca.transform(test_x)

# parameters initialization for the genetic algorithm
W = 10
MODE = '2local'
QUBIT = 2
FEATURE = 2

BIT = 6
GENE = 8
POPULATION = 12

POOL_SIZE = 2
OFFSPRING_SIZE = POPULATION - POOL_SIZE
PROB = 8
MUTATION_PROB = PROB / GENE

GENERATION = 100
EARLY_STOP = 10

ga = GeneticAlgorithm(W, MODE, QUBIT, FEATURE, GENE, BIT, POPULATION, POOL_SIZE, OFFSPRING_SIZE, PROB,
                      MUTATION_PROB, GENERATION, EARLY_STOP)

ga.execute(0, 100, train_x, train_y, test_x, test_y)
# ga.execute_pareto_front(0, [0.1, 100], train_x, train_y, test_x, test_y)
