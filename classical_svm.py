import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, classification_report
from sklearn.svm import SVC

from sklearn.decomposition import PCA

import numpy as np

import time

np.random.seed(42)

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
mapping_dict = {class_name: id for id, class_name in enumerate(data[label_name].unique())}
inverse_dict = {id: class_name for id, class_name in enumerate(data[label_name].unique())}

n_classes = len(labels.unique())  # number of classes (clusters)

# numpy array conversion
features = features.to_numpy()
labels = labels.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    labels,
                                                    train_size=n_train,
                                                    test_size=n_test,
                                                    stratify=labels)

# for visualization purposes only
pca = PCA(n_components=6)
pca.fit(X_train)
X_train_plot = pca.transform(X_train)
X_test_plot = pca.transform(X_test)
print('Explained variance: {}'.format(pca.explained_variance_ratio_))
plt.plot([*range(1, 7)], np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.yticks(ticks=np.linspace(0.9999, 1, 6),
           labels=np.linspace(0.9999, 1, 6))
plt.title('Variance retained after PCA for each component')
plt.xlabel('Principal Components')
plt.ylabel('Variance Explained Ratio')
plt.gcf().subplots_adjust(left=0.15)
plt.savefig('img/svm/variance_retained')
plt.clf()

# dimensionality reduction
pca = PCA(n_components=2)
pca.fit(X_train)
print('Explained variance: {}'.format(pca.explained_variance_ratio_))
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# classical SVM model training
model = SVC(kernel='linear')
print('training SVC...')
start = time.time()
model.fit(X_train, y_train)
elapsed = time.time() - start
print('Training time: {}'.format(elapsed))
score = model.score(X_test, y_test)
print('testing score: {}'.format(score))

# decision boundary plot
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))
# define bounds of the domain
min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
# define the x and y scale
x1grid = np.arange(min1, max1, 0.1)
x2grid = np.arange(min2, max2, 0.1)
# create all the lines and rows of the grid
xx, yy = np.meshgrid(x1grid, x2grid)
# flatten each grid to a vector
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
# horizontal stack vectors to create x1,x2 input for the model
grid = np.hstack((r1, r2))
# make predictions for the grid
yhat = model.predict(grid)
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
    plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap=cmap, label=inverse_dict[class_value])
plt.title('Decision Boundaries')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.savefig('img/svm/decision_boundaries')

plt.clf()

y_pred = model.predict(X_test)
y_score = model.decision_function(X_test)

# classification metrics
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title('Confusion Matrix')
plt.savefig('img/svm/confusion_matrix')
plt.clf()

fig, ax = plt.subplots()
RocCurveDisplay.from_predictions(y_test, y_score[:, 0], pos_label=0, ax=ax,
                                 name=f'{inverse_dict[0]} vs the rest')
RocCurveDisplay.from_predictions(y_test, y_score[:, 1], pos_label=1, ax=ax,
                                 name=f'{inverse_dict[1]} vs the rest')
RocCurveDisplay.from_predictions(y_test, y_score[:, 2], pos_label=2, ax=ax,
                                 name=f'{inverse_dict[2]} vs the rest')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.savefig('img/svm/roc_curves')
plt.clf()
