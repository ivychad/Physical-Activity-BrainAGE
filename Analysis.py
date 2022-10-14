import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.svm import SVR
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import *
from random import seed
from scipy import stats

seed(888)

# load data from csv and convert to array
brain_test = pd.read_pickle("test_brain.pkl")
brain_train = pd.read_pickle("train_brain.pkl")
main_test = pd.read_pickle("test_main.pkl")
main_train = pd.read_pickle("train_main.pkl")
brain_test = pd.DataFrame(brain_test)
brain_train = pd.DataFrame(brain_train)
main_test = pd.DataFrame(main_test)
main_train = pd.DataFrame(main_train)


# delete rows that have a missing value for Age
index_age_train = brain_train[np.isnan(brain_train.iloc[:,-1])].index
index_age_test = brain_test[np.isnan(brain_test.iloc[:,-1])].index
brain_train = brain_train.drop(index_age_train)
brain_test = brain_test.drop(index_age_test)
main_test = main_test.drop(index_age_test)
main_train = main_train.drop(index_age_train)

# exclude people with long standing illness diability or infirmity
index_illness_train = main_train[main_train.iloc[:,967]!= "No"].index
index_illness_test = main_test[main_test.iloc[:,967]!= "No"].index
brain_train = brain_train.drop(index_illness_train)
brain_test = brain_test.drop(index_illness_test)
main_test = main_test.drop(index_illness_test)
main_train = main_train.drop(index_illness_train)

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

scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)

# X = brain_training.iloc[:,:-1]
# Y = brain_training.iloc[:,-1]

# # # randomly split data into training and testing set
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=0)

alpha_parameters = np.power(10,np.linspace(start=-3, stop=5, num=100))

# model initialization - options to use Lasso, Ridge or SVM
#model = LassoCV(alphas = alpha_parameters, max_iter=100000)
#model = RidgeCV(alphas = alpha_parameters)
model = SVR(kernel = 'rbf')

# train model
model.fit(X_train_std,Y_train)

# get predicted values for test set
y_pred = model.predict(X_test_std)

# calculate brain age gap
brain_age_delta = y_pred-Y_test

# check whether brain age gap and age are correlated
correlation = stats.pearsonr(brain_age_delta, Y_test)
#print(correlation)

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

plt.figure()
plt.scatter(Y_test,brain_age_delta)
plt.axline((60,5), slope=-0.69, color="r")
plt.ylabel("Brain Age Delta")
plt.xlabel("Age")
plt.show()
