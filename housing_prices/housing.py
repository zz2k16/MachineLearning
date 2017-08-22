import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, KFold
from sklearn.preprocessing import Imputer
# random seed to reproduce results per run
np.random.seed(42)

# import data as csv and load into DataFrame
data = "datasets/housing.csv"
housing = pd.read_csv(data)

# view basic information about dataset
#print(housing.head())
#print(housing.info())

# visualise data attributes as histograms
# housing.hist(bins=50, figsize=(20,15))
#plt.show()

# create income cat attribute to categorise income ranges
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# split data set into train and test sets and use stratified sampling
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# drop income cat column
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


# verify data shape is retained
# print(strat_train_set.shape)
# print(strat_test_set.shape)

# split training data into features and label
x_train = strat_train_set.drop('median_house_value', axis=1)
y_train = strat_train_set['median_house_value']


# further data visualisation
housing_data = strat_train_set.copy()

# visualising geographical data
#housing_data.plot(kind='scatter', x='longitude', y='latitude', title='geographical data', figsize=(10, 10), alpha=0.4, s=housing['population']/100, label='population', c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)

# look for correlations between attributes using Pearson's r (standard correlation coeff)
corr_matrix = housing_data.corr()
# sort by house value
print(corr_matrix['median_house_value'].sort_values(ascending=False))
# create list of top most positive correlating features and plot scatter matrix
corr_features = ['median_house_value','median_income', 'total_rooms', 'housing_median_age']
#scatter_matrix(housing_data[corr_features], alpha=0.3, figsize=(12,8))
# view scatter of median income x house value
# housing_data.plot(kind='scatter', x='median_income', y='median_house_value',alpha=0.1)

# attribute combinations
housing_data['rooms_per_household'] = housing_data['total_rooms']/housing_data['households']
housing_data['bedrooms_per_household'] = housing_data['total_bedrooms']/housing_data['total_rooms']
housing_data['pop_per_household'] = housing_data['population']/housing_data['households']
# run correlation matrix again to include new attributes
house_corr = housing_data.corr()
print(house_corr['median_house_value'].sort_values(ascending=False))

# prepare data for machine learning
# split training data into features and labels , xtrain and ytrain

# data cleaning

# clean missing values from data
# clean total_bedrooms attribute missing values with median
imputer = Imputer(strategy='median')
# imputer needs numerical values, so create dataset variable without ocean_proximity feature
housing_num = housing_data.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)

print(housing_num.median().values)
# replace imputed median values into housing dataset
median_X = imputer.transform(housing_num)

