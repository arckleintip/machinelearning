# Data wrangling
import pandas as pd
import numpy as np
import missingno
from collections import Counter

# Data visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning models
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier

# Model evaluation
from sklearn.model_selection import cross_val_score

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Remove warnings
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("data/titanic/train.csv")
test = pd.read_csv("data/titanic/test.csv")
ss = pd.read_csv("data/titanic/gender_submission.csv")

train.head()
test.head()

print("Training set shape: ", train.shape)
print("Test set shape: ", test.shape)

ss.head()
ss.shape

train.info()
print('-'*40)
test.info()

train.isnull().sum().sort_values(ascending = False)
test.isnull().sum().sort_values(ascending = False)

missingno.matrix(train)
plt.show()