# learning more in depth about how to compute linear regression and using r-squared as a performance measure
# generating random but correlated data to explore the model and its features. for fun :)
import numpy as np
from pylab import *
from scipy import stats
import matplotlib.pyplot as plt

# create two sets of random data, one with normal distribution and the other a linear function of it
pagespeed = np.random.normal(3.0,1.0,1000)
# force a linear relationship using page speed
checkoutcart = 100 - (pagespeed + np.random.normal(0, 0.1, 1000) * 3)

# visualise dummy data
scatter(pagespeed, checkoutcart)
# plt.show()

# use ordinary least squares to measure the linear relationship with the scipy linear regression method
slope, intercept, r_val, p_val, std_err = stats.linregress(pagespeed, checkoutcart)
r_squared = r_val ** 2

print(r_squared)
# returns 0.916

# define function to compute slope using stored values of the linear regression model
# or simply the line of best fit, y = mx + b
def predict_slope(x):
    return slope * x + intercept

# compute slope fitted to pagespeed
fit_line = predict_slope(pagespeed)

# visualise linear regression line
plt.scatter(pagespeed,checkoutcart, c='orange')
# super impose line over scatter plot in blue for contrast
plt.plot(pagespeed, fit_line, c='b')
plt.title('Linear regression Least ordinary squares')
plt.show()

