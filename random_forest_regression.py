# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
df = dataset.iloc[:, [1,2,-1]].values
df = pd.DataFrame(df)
df.columns = ['week','center_id','orders']

#grouping the week wise total orders
res = df.groupby(['week','center_id']).orders.sum().reset_index()

X = res.iloc[:,0:2].values
y = res.iloc[:,2].values

centres = np.unique(X[:,1])

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)


#test file of unique week from 146 to 155
X_forecast = pd.read_csv('forecast 145 to 155.csv')
X_forecast = X_forecast.iloc[:,0:2].values

# Predicting the week 146 to 155 orders

y_forecast = regressor.predict(X_forecast)

#saving the output of week 146 to 155 orders

final_output = pd.DataFrame(X_forecast)
final_output['orders'] = pd.DataFrame(y_forecast)

final_output.columns = ['week','center_id','num_orders']

df2 = dataset.iloc[:, [0,2]].values
df2 = pd.DataFrame(df2)
df2.columns = ['id','center_id']


final_output.to_csv('week146_155_orders.csv')
df3 =  final_output[['center_id','num_orders']]

output = pd.merge(df2,df3,left_on = 'center_id',right_on = 'center_id',how = 'left')
output2 = output.groupby(['id']).num_orders.mean().reset_index()
# Visualising the Random Forest Regression results (higher resolution)

df4 = pd.read_csv('sample_submission_hSlSoT6.csv')
df5 = df4['id']
df5 = pd.DataFrame(df5)


output2.to_csv('final_submission.csv')
plt.title('training set')
plt.scatter(X[:,0],y,color = 'red',label = 'Actual Orders',marker = '*')
plt.plot(X_forecast[:,0],regressor.predict(X_forecast),color = 'blue',marker = '*',label = 'Orders Predicted')
plt.xlabel('weeks')
plt.ylabel('orders')
plt.legend()
plt.show()



