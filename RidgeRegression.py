import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error

# load data from csv and convert to array
df_test = pd.read_csv("test_brain.csv")
df_train = pd.read_csv("train_brain.csv")
df_test = np.asarray(df_test)
df_train = np.asarray(df_train)

# get number of missing values for age
nan_count_train = np.count_nonzero(pd.isnull(df_train[:,-2]))
nan_count_test = np.count_nonzero(pd.isnull(df_test[:,-2]))

# delete rows that have a missing value for Age
for i in range(len(df_train)-nan_count_train):
   if np.isnan(df_train[i,-2]):
       df_train = np.delete(df_train,i,axis=0)

for i in range(len(df_test)-nan_count_test):
   if np.isnan(df_test[i,-2]):
       df_test = np.delete(df_test,i,axis=0)

# columns for regression - brain measures that will be used for the prediction
col = [14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,1273]

df_train_s = df_train[:,col]
df_test_s = df_test[:,col]

# delete rows with nan values
df_train_s = pd.DataFrame(df_train_s)
new_train = df_train_s.dropna(axis=0)

df_test_s = pd.DataFrame(df_train_s)
new_test = df_test_s.dropna(axis=0)

new_test = np.asarray(new_test)
new_train = np.asarray(new_train)

# impute nan values in mri data
# imp_mean = IterativeImputer(random_state=0)
# imp_mean.fit_transform(df[:,:-1])

# declare training and testing data - X: brain variables, Y: actual age
X_train = new_train[:,:-1]
Y_train = new_train[:,-1]
X_test = new_test[:,:-1]
Y_test = new_test[:,-1]

# randomly split data into training and testing set
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=0)

# ridge regression initialization
ridge = Ridge(alpha= 0.5)

# train model
ridge.fit(X_train,Y_train)

# get predicted values for test set
y_pred = ridge.predict(X_test)
print(y_pred)

# get mean absolute error (MAE)
print(mean_absolute_error(Y_test,y_pred))

# plot figure with x: actual age Y: predicted age, and a line with slope 1 for reference
plt.figure()
plt.scatter(Y_test, y_pred, alpha=0.5)
plt.axline((60,60), slope=1, color='r')
plt.xticks(range(40,90,5))
plt.yticks(range(40,110,5))
plt.ylabel('Predicted Age')
plt.xlabel('Age')
plt.show()
