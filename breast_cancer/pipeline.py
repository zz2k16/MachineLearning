# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
# load Wisconsin Breast Cancer dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
df = pd.read_csv('Data/wdbc.data.txt', header=None)

# split data into features and class labels
# feature columns
X = df.loc[:, 2:].values
# split class label column and encode into integer values using LabelEncoder
Y = df.loc[:, 1].values
labels = LabelEncoder()
# fit labels into new y array
y = labels.fit_transform(Y)

# create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# create pipeline of transformers and estimators
# pipeline standardises data, reduces dimensionality using PCA, and estimates using LogisticRegression
pipe_logr = Pipeline([
    ('scl', StandardScaler()),  # standardize data
    ('pca', PCA(n_components=2)),  # reduce data from 30 to 2 dimensions
    ('logr', LogisticRegression(random_state=1))  # classify using logistic regression
])

pipe_logr.fit_transform(X_train, y_train)
print("Accuracy on Test set {:3.3f}".format(pipe_logr.score(X_test, y_test)))

# use Stratified Kfold to evaluate model performance using k=10 folds
scores = cross_val_score(estimator=pipe_logr, X=X_train, y=y_train, cv=10, n_jobs=1)
print('Cross Validation accuracy {}'.format(scores))
print('CV Accuracy {:3.3f} +/- {:3.3f}'. format(np.mean(scores), np.std(scores)))


# Learning Curve
# create and plot learning curve to evaluate model performance over 10 fold cross validation
pipe_lr = Pipeline([
    ('scl', StandardScaler()),  # standardize data
    ('logr', LogisticRegression(random_state=0))  # classify using logistic regression
])

train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
train_mean = np.mean(train_scores, axis=1); train_std = np.std(train_scores, axis=1); test_mean = np.mean(test_scores, axis=1); test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color = 'blue', marker= 'o', markersize = 5, label = 'training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha = 0.15, color = 'blue')

plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
# plt.show()


# Validation Curve
# create and plot a validation curve for the inverse regularization of logistic regression over 10 fold cross validation
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
trainScores, testScores, = validation_curve(estimator=pipe_lr, X=X_train, y= y_train, param_name='logr__C', param_range=param_range, cv=10)

trainMean = np.mean(trainScores, axis=1)
trainStd = np.std(trainScores, axis=1)
testMean = np.mean(testScores, axis=1)
testStd = np.std(testScores, axis=1)

plt.plot(param_range, trainMean, color = 'blue', marker= 'o', markersize = 5, label = 'training accuracy')
plt.fill_between(param_range, trainMean + trainStd, trainMean - trainStd, alpha = 0.15, color = 'blue')

plt.plot(param_range, testMean, color='green', linestyle='--', marker='s', markersize=5,label='validation accuracy')
plt.fill_between(param_range, testMean + testStd, testMean - testStd, alpha=0.15, color='green')
plt.grid()
plt.xscale('log')
plt.legend(loc = 'lower right')
plt.xlabel('Paramater C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
#plt.show()

# Tune model parameters using Grid Search
# create pipeline with standard scaler and SVM classifier
pipe_scv = Pipeline([
    ('scl', StandardScaler()), ('clf', SVC(random_state=1))
])
# define parameter range of C values
params_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# define parameter grid using dictionary of hyper-parameter values
param_grid = [
    {'clf__C': params_range, 'clf__kernel': ['linear']},  # cycle through different C values for a linear kernel
    {'clf__C': params_range, 'clf__gamma': params_range, 'clf__kernel': ['rbf']}  # cycle through different C and gamma values for radial bias function kernel
]
# construct Grid Search and use all system cores for parallel processing using n_jobs = -1
gs = GridSearchCV(
    estimator=pipe_scv, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1
)
gs.fit(X_train, y_train)

# print best model scores and parameter options
print(gs.best_score_)
print(gs.best_params_)
# estimate performance on test dataset using the best selected model of the grid search
clf = gs.best_estimator_  # get the best model
clf.fit(X_train, y_train)  # fit on training data
# test model on test data set
print("Test accuracy {:3.3f}%".format(clf.score(X_test, y_test)*100))

# create nested cross validation on Grid search
gn = GridSearchCV(
    estimator=pipe_scv, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1
)
gn_scores = cross_val_score(gn, X, y, scoring='accuracy', cv=5)
print('Nested Cross Validation accuracy {:3.3f} +/- {:3.3f}'.format(np.mean(gn_scores), np.std(gn_scores)))

