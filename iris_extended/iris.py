# import libraries
import pandas
import matplotlib
import numpy as np
from pandas import scatter_matrix
matplotlib.use('TkAgg')  # fix for Mac OS
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# - Load Data
# load data set from UCI Repo
#url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# or load from local directory
url = 'Data/iris.data.txt'
# list of feature names
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
# verify data has loaded correctly by printing first 5 samples
# print(dataset.head(5))


# - Summarise Data set
# Statistical summary of all features, dimension of data, breakdown of data by class variable
# shape of data
print('Data is of size {}:\n'.format(dataset.shape))

# statistical summary of data features
print(dataset.describe()); print()

# class distribution of data
print(dataset.groupby('class').size()); print()


# - Data Visualisation

# uni-variate : create box and whisker plots, histograms
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# # scatter plot of sepal length x sepal width
# dataset.plot(kind='scatter', x='petal-length', y='petal-width')
# # histogram of data
# dataset.hist()
#
# # multi-variate : create scatter matrix
# scatter_matrix(dataset)
# plt.show()



# - Evaluate Algorithms

# create test and validation sets of 80:20 test:train
# split data set into feature array and class label array
iris = dataset.values
X = iris[:, 0:4]
Y = iris[:, 4]
# use sklearn to split data into test and validation sets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7)
# verify shape of split data
print('Size of training data: {}'.format(X_train.shape))
print('Size of test data: {}'.format(X_test.shape))

# print values of test data for verification
print('\nTest data')
for x, y in zip(X_test[:5], Y_test[:5]):
        print(x, y)

# verify training data
print('\nTraining data')
for x, y in zip(X_train[:5], Y_train[:5]):
        print(x, y)

print()

# create variables for use in cross validation models
seed = 7
# evaluation metrics calculate by ratio of (correctly predicted instance / total number of instances in data) x 100
scoring = 'accuracy'

# Build classification models
# test using different algorithms:
# Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbour,
# Classification and Regression Trees, Gaussian Naive Bayes, Support Vector Machines

# spot check algorithms
models = [('LR', LogisticRegression()), ('LDA', LinearDiscriminantAnalysis()), ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()), ('SVM', (SVC()))]

# evaluate each of the models
results = []
names = []

for name, model in models:
    # use 10 fold cross validation
    kFold = model_selection.KFold(n_splits=10, random_state=seed)
    cross_val_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kFold, scoring=scoring)
    # print results of each model
    results.append(cross_val_results); names.append(name)
    evaluation = 'Model name: {}\nMean: {:2.6f}\nStd: {:2.6f}\n'.format(name, cross_val_results.mean(), cross_val_results.std())
    print(evaluation)

# compare algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()


# - Classify data on validation set

# use KNN
print("KNN")
knn = KNeighborsClassifier()
# fit training data
knn.fit(X_train, Y_train)
# predict on test data
prediction = knn.predict(X_test)

# print results and confusion matrix
print('Model Accuracy: ', accuracy_score(Y_test, prediction))
print('\nConfusion Matrix:\n', confusion_matrix(Y_test, prediction))
print(classification_report(Y_test, prediction))

# use SVM
print("Support Vector Machine")
svm = SVC(kernel='rbf', C=1.0, gamma=0.2, random_state=0)
# kernel='linear', C=1.0, random_state=0
svm.fit(X_train, Y_train)
predictsvm = svm.predict(X_test)

# print results and confusion matrix
print('Model Accuracy: ', accuracy_score(Y_test, predictsvm))
print('\nConfusion Matrix:\n', confusion_matrix(Y_test, predictsvm))
print(classification_report(Y_test, predictsvm))

print("Test SVM on unseen data")
print(svm.predict([[5.0,  3.6,  1.3,  0.25]]))
print(svm.predict([[2.0, 3.2, 2.0, 3.2]]))



# create standardised data set variables
print('\n\nData frame of Iris\n\n ')

#load data and split class labels into numpy arrays
df = pandas.read_csv(url, header=None)
print(df.head())
x = df.iloc[:, 0:4]
y = df.iloc[0:150, 4].values
print(df[4].unique())

# convert class labels into integers
df.loc[df[4] == 'Iris-virginica', 4] = 1
df.loc[df[4] == 'Iris-versicolor', 4] = 0
df.loc[df[4] == 'Iris-setosa', 4] = -1

# verify reshaped data
print(df[4].unique())
print(len(y))
print(x.head())
print(y[0:5])

# create new test and train splits
irisX_train, irisX_test, irisY_train, irisY_test, = model_selection.train_test_split(X, Y, test_size=0.30, random_state=0)

# verify values of split data
print(type(irisX_train))
print(len(irisX_train))
print(irisX_train[0].astype(np.float64))

print(len(irisX_test))
print(len(irisY_train))
print(len(irisY_test))

print(irisX_train[:1])
print(irisY_train[:1])

# standardise data for optimal feature scaling
sc = StandardScaler()
sc.fit(irisX_train)
X_train_std = sc.transform(irisX_train)
X_test_std = sc.transform(irisX_test)
# verify standardised feature data
print(X_test_std[1])
print(irisY_test[1])

sVm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
sVm.fit(X_train_std, irisY_train)
predictSVC = sVm.predict(X_test_std)

print(accuracy_score(irisY_test, predictSVC))
print(classification_report(irisY_test,predictSVC))



