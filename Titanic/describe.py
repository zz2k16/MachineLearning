# import libraries
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# load data
titanic_train = pd.read_csv('Data/train.csv')

# describe data
# print(titanic_train.head(5))
# print(titanic_train.describe())

print("The median of Ages are: {}".format(titanic_train['Age'].median()))

# fill null values in Age
titanic_train['Age'] = titanic_train['Age'].fillna(titanic_train['Age'].median())

# print(titanic_train.describe())

# convert values of Sex column into integers

print("The unique values in the Sex column are", titanic_train['Sex'].unique())
# convert males to zero
titanic_train.loc[titanic_train['Sex'] == 'male', 'Sex'] = 0
# convert females to 1
titanic_train.loc[titanic_train['Sex'] == 'female', 'Sex'] = 1
print(titanic_train['Sex'].unique())

# convert Embarked into integers

print('The unique Embarked values are : ', titanic_train['Embarked'].unique())
# fill null values
titanic_train['Embarked'] = titanic_train['Embarked'].fillna('S')
# verify fill na
print(titanic_train['Embarked'].unique())
# assign integers to each value
titanic_train.loc[titanic_train['Embarked'] == 'S', 'Embarked'] = 0
titanic_train.loc[titanic_train['Embarked'] == 'C', 'Embarked'] = 1
titanic_train.loc[titanic_train['Embarked'] == 'Q', 'Embarked'] = 2
# verify values are now integers
print(titanic_train['Embarked'].unique())

# use logistic regression and cross val to compute accuracy

# labels to use for prediction
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# initialise algorithm
alg = LogisticRegression(random_state=1)
forests = RandomForestClassifier(n_estimators=50, min_samples_split=2, min_samples_leaf=3)

# compute accuracy using cross validation
scores = cross_val_score(alg, titanic_train[predictors],titanic_train["Survived"], cv=3)
print('Average accuracy of logistic regression: ', scores.mean())

# compute accuracy using random forests
forestScores = cross_val_score(forests, titanic_train[predictors], titanic_train['Survived'], cv=3)
print('Random Forest accuracy ',forestScores)
print(forestScores.mean())



