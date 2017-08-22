from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pylab as pl
from itertools import cycle
# using principal component analysis on the iris data set
# understanding the practical applications of the theory behind pca and its operation

# load iris data set and verify shape and class labels
iris = load_iris()
n_samples, n_features = iris.data.shape
print("Samples:{}, features:{}".format(n_samples, n_features))
print("Class labels:", iris.target_names)

# split data into features
X = iris.data
# verify data structure
# print(X[0:3])

# load PCA object and transform data down to 2 dimensions
pca = PCA(n_components=2, whiten=True)
pca.fit(X)
x_pca = pca.transform(X)

# print pca eigenvectors and variance
print("PCA Eigenvectors:\n", pca.components_)
print("PCA Explained Variance:{}".format(pca.explained_variance_ratio_))
print("Sum of preserved variance ration: {:.3f}%".format(sum(pca.explained_variance_ratio_)*100))


# visualise pca transformation
colours = cycle('rbg')
target_labels = range(len(iris.target_names))
pl.figure()

# iterate over data set and colours to visualise 4d -> 2d transformed vectors of all class labels
for x, c, label in zip(target_labels, colours, iris.target_names):
    pl.scatter(x_pca[iris.target == x, 0], x_pca[iris.target == x, 1], c=c, label=label)
pl.legend()
pl.show()


