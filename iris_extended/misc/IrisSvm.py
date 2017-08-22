# import libraries
import pandas
import matplotlib
import numpy as np
from pandas import scatter_matrix
matplotlib.use('TkAgg')  # fix for Mac OS
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC



# - Load Data

# load data set from UCI Repo
#url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# or load from local directory
url = 'Data/iris.data.txt'
# list of feature names
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pandas.read_csv(url, names=names)

# verify data has loaded correctly by printing first 5 samples
print(df.head(5))

# - Summarize Data set

# Statistical summary of all features, dimension of data, breakdown of data by class variable
# shape of data

print('Data is of size {}:\n'.format(df.shape))
print(df.describe()); print()
# class distribution of data
print(df.groupby('class').size()); print()
#print(df.head(5))

# convert features into floating point values and split data set
dataset = df.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]
# print(X[0])
# print(Y[0])

# - encode class labels into one-hot and standardise data set
# encode labels into one hot
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
#print(encoded_Y)

# split data set into train and test samples
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.30, random_state=1337)

# - Standardise features
# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)
# verify standardised data
#print(X_test_std[0])
#print(X_test_std[0])


# - Classify using SVM with Radial Basis Function Kernel, Regularisation, and Cut Off
# for best compromise between miniziming overfitting / underfitting and accuracy
svm = SVC(kernel='rbf', random_state= 0, gamma= 0.2, C=1.0)
svm.fit(X_train, Y_train)
predictSVM = svm.predict(X_test)

# print classification accuracy
print('SVM Test accuracy {:3.3f}%\n'.format(accuracy_score(Y_test, predictSVM)*100))
print('SVM Classification Report \n', classification_report(Y_test, predictSVM))
print('Confusion Matrix : \n', confusion_matrix(Y_test, predictSVM))


# print("\nTest SVM on unseen data\n")

# define function for predicting new iris samples
# function params as list to convert as 1D numpy array to pass into svm classifier
# optionally could be created as a class with its own predict method. function chosen for simplicity

# def newIris(*args):
#     args = np.asarray(args[0])
#     # argsX = args.reshape(1, -1)
#     X = np.array(args).reshape(-1, 4)  # reshape necessary to pass into svm as 1d feature array
#     return X

# test svm predict function to determine behaviour before defining an Iris class
# print("This should be iris virginica" , svm.predict(np.array([[6.7, 3.0, 5.2, 2.3]])))
# print("This should be iris setosa", svm.predict([[5.0,3.5,1.3,0.3]]))
# print("This should be iris verisicolor", svm.predict(np.array([[5.8,2.7,4.1,1.0]])))

# creating function to return 2d numpy array for classification
# def Iris(*args):
#     X = np.array([args]).reshape(-1,4)
#     return X
#
# tst = Iris([5.8, 2.7, 4.1, 1.0])
# tst2 = Iris([[5.0,3.5,1.3,0.3]])
# print(tst)
# print(svm.predict(tst))
# print(tst2)
# print(svm.predict(tst2))


# taking methods learn from the Iris function, create a class to define behaviour for new iris samples
# takes in a list input, reshapes to a 1x4 numpy array and operates on it
# find k nearest neighbours to new instances of iris sampels
class Iris:

    def __init__(self, *args):
        self.array = np.array([args]).reshape(-1, 4)

    def getFeatures(self):
        return self.array

    def classify(self):
        return svm.predict(self.array)

    def findNeighbours(self, k):
        # fit K nearest neighbours classifier on data to find k similar neighbours
        neighb = NearestNeighbors(n_neighbors=k)
        neighb.fit(X)
        return neighb.kneighbors_graph(self.array)


# end class


# create new iris samples and test methods

setosa1 = Iris([[5.0, 3.4, 1.4, 0.5]])
print("\nThis flower is {} with features: {}".format(setosa1.classify(), setosa1.getFeatures()))
print("The closest neighbours to this flower are in rows :", setosa1.findNeighbours(10))

virginica1 = Iris([6.2, 3.2, 5.1, 1.8])
print("\nThis flower is {} with features: {}".format(virginica1.classify(), virginica1.getFeatures()))
print("The closest neighbours to this flower are in rows :", virginica1.findNeighbours(10))

versicolor1 = Iris([[5.3, 2.7, 4.1, 1.0]])
print("\nThis flower is {} with features: {} ".format(versicolor1.classify(), versicolor1.getFeatures()))
print("The closest neighbours to this flower are in rows :", versicolor1.findNeighbours(10))


# - Data visualisation

#box and whisker plots
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

# scatter plot of sepal length x sepal width
df.plot(kind='scatter', x='sepal-length', y='sepal-width')

# scatter plot of petal length x petal width
df.plot(kind='scatter', x='petal-length', y='petal-width', title="Petal Length | Petal Width")
# histogram of data
df.hist()
# multivariate plots
scatter_matrix(df)
plt.show()

