# calculating covariance and correlation the long way just for fun and to develop an intuition for and understand the underlying math
import matplotlib
from pylab import *
import numpy as np

# covariance is the measure of the dot product between two high dimensional vectors
# the dot product is the angle between the given vectors from the deviations from the mean

# define function to compute deviation from the mean of a given sample
def dev_mean(x):
    # caluclate mean of data set
    xmean = mean(x)
    # use list comprehension to compute difference of each sample from the mean
    return [xi - xmean for xi in x]
    # could be optimised into a single line
    # return [xi - np.mean(x) for xi in x ]

# define covariance function to return dot product of two vectors divided by the sample size (n-1)
# function:
# vector x dot vector y / sample size - 1
def covar(x,y):
    n = len(x)
    return dot(dev_mean(x), dev_mean(y)) / (n-1)


# generate random data sets to compute covariance
# measure the relationship of two data sets: page speeds and purchase amounts
# page speeds measure time taken to load the webpage
# purchase amounts measure how much customers spend
# goal of covariance is to find relationship between these two features. is there a correlation between
# time taken to load page and how much customer spends?

pageSpeed = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50, 10.0, 1000) / pageSpeed # divide by page speed to force a strong correlation

web_covar = covar(pageSpeed, purchaseAmount)
print(web_covar)

# define covariance function to normalise data set
# take standard deviation of both attributes and compute covariance and divide by standard dev of both data sets
def corr(x,y):
    std_x = x.std()
    std_y = y.std()
    return covar(x,y) /std_x / std_y
    # improve by adding check for divide by zero


scatter(purchaseAmount, pageSpeed)
plt.show()

web_corr = corr(pageSpeed, purchaseAmount)
print(web_corr)

# use numpy to calculate correlation and covariance and compare to our own functions
np_corr = np.corrcoef(pageSpeed, purchaseAmount)
print(np_corr)

np_cov = np.cov(pageSpeed, purchaseAmount)
print(np_cov)
