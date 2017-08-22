import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

# some simple matplotlib visualisation for fun and exploring plotting features
# adjust graph axes limits and unit labels
axes = plt.axes()
axes.set_xlim([-5,5])
axes.set_ylim([0, 1.0])
axes.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# add grid
axes.grid()


# simple line graph using randomly generated values
x = np.arange(-3, 3, 0.001)
# plot values using normal distribution probability density function
# add b- param for blue solid line
plt.plot(x, norm.pdf(x), 'b-')
# stack plot of normal distribution at mean = 1, and std.dev of 0.5
# add r: for red dashed line or r-- for double dash or r-. for dash period
plt.plot(x, norm.pdf(x, 1, 0.5), 'r-.')
# adding legends and keys
plt.xlabel('animals'); plt.ylabel('probability')
plt.legend(['cats', 'dogs'], loc=1)
# add title
plt.title('Chances of neighbourhood pets ruining your lawn')
plt.show()

