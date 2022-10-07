import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error

# load data from csv and convert to array
brain_test = pd.read_pickle("test_brain.pkl")
brain_train = pd.read_pickle("train_brain.pkl")
# main_test = pd.read_csv("test_main.csv")
# main_train = pd.read_csv("train_main.csv")
brain_test = pd.DataFrame(brain_test)
brain_train = pd.DataFrame(brain_train)
# main_test = pd.DataFrame(main_test)
# main_train = pd.DataFrame(main_train)

# delete rows that have a missing value for Age
brain_train = brain_train.drop(brain_train[np.isnan(brain_train.iloc[:,-1])].index)
brain_test = brain_test.drop(brain_test[np.isnan(brain_test.iloc[:,-1])].index)


# Freesurfer ASEG
whole_brain = brain_train.iloc[:,27:70:2]
left_hemi = brain_train.iloc[:,103:134:2]
right_hemi = brain_train.iloc[:,165:196:2]
age = brain_train["Age"]
brain_train_s = pd.concat([whole_brain,left_hemi, right_hemi,age],axis = 1)

whole_brain = brain_test.iloc[:,27:70:2]
left_hemi = brain_test.iloc[:,103:134:2]
right_hemi = brain_test.iloc[:,165:196:2]
age = brain_test["Age"]
brain_test_s = pd.concat([whole_brain,left_hemi, right_hemi,age],axis = 1)


# delete rows with nan values
brain_train_s = pd.DataFrame(brain_train_s)
brain_training = brain_train_s.dropna(axis=0)
brain_test_s = pd.DataFrame(brain_test_s)
brain_testing = brain_test_s.dropna(axis=0)

# df_test_s = pd.DataFrame(df_train_s)
# new_test = df_test_s.dropna(axis=0)

# # impute nan values in mri data
# # imp_mean = IterativeImputer(random_state=0)
# # imp_mean.fit_transform(df[:,:-1])

# declare training and testing data - X: brain variables, Y: actual age
X_train = brain_training.iloc[:,:-1]
Y_train = brain_training.iloc[:,-1]
X_test = brain_testing.iloc[:,:-1]
Y_test = brain_testing.iloc[:,-1]

# X = brain_training.iloc[:,:-1]
# Y = brain_training.iloc[:,-1]

# # # randomly split data into training and testing set
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
