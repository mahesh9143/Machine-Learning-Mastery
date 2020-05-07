# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

# Importing the dataset
dataset = pd.read_csv('Data.csv')

dataset['Country'].value_counts()

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1:]
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
enc = LabelEncoder()
Oenc = OrdinalEncoder()

b = X.iloc[:,0:1]


dataset['Country'].value_counts()
X['Country'].value_counts()
c = pd.DataFrame(dataset['Country'].value_counts())
c['Index'] = c.index

c = c.reset_index(drop = True)

c = c.rename(columns =  {'Index':'Country','Country':'enc' })
n = dataset

new = pd.merge(c,n)

new = new.drop(['Country'],axis = 1).rename(columns = {'enc':'Country'})
X.iloc[:,0:1] = enc.fit_transform(X.iloc[:,0:1])

X.iloc[:,0:1] = Oenc.fit_transform(X.iloc[:,0:1])

Oenc.categories_
Oenc.get_params
sns.heatmap(dataset.isnull(),yticklabels= False, cbar = False)

dataset['Age'].fillna(dataset['Age'].mean(),inplace = True)
dataset['Salary'].fillna(dataset['Salary'].mean(),inplace = True)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder,StandardScaler
imputer = SimpleImputer(strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.pipeline import Pipeline

num_pipline = Pipeline([
        ('impute',SimpleImputer()),
        ('std_scaling',StandardScaler())])
cat_pipline = dataset['Country']

num_d = list(dataset.drop(['Country','Purchased'],axis = 1))

dataset['Country'] = pd.get_dummies(dataset['Country'])

df = np.c_[pd.get_dummies(dataset['Country']),dataset.drop('Country',axis = 1)]

df = pd.DataFrame(df)

from sklearn.compose import ColumnTransformer
num_attr = list(dataset.drop(['Country','Purchased'],axis = 1))
cat_attr = ['Country']
full_pipline = ColumnTransformer([
        ('num',num_pipline,num_attr),
        ('cat',OneHotEncoder(),cat_attr)])

df_transformed2 = full_pipline.fit_transform(dataset)
df_transformed2 = pd.DataFrame(df_transformed2)

df_transformed2.describe()

df_transformed2.describe()
dataset.describe()

