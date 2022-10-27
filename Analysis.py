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
index_age_train = brain_train[np.isnan(brain_train["Age"])].index
index_age_test = brain_test[np.isnan(brain_test["Age"])].index
brain_train = brain_train.drop(index_age_train)
brain_test = brain_test.drop(index_age_test)
main_test = main_test.drop(index_age_test)
main_train = main_train.drop(index_age_train)

# for later comparison: all data -> compute things for excluded people
all_main_test = main_test
all_main_train = main_train
all_brain_test = brain_test
all_brain_train = brain_train

# exclude people with long standing illness diability or infirmity
index_illness_train = main_train[main_train["longstanding_illness_disability_or_infirmity_f2188_0_0"]!= "No"].index
index_illness_test = main_test[main_test["longstanding_illness_disability_or_infirmity_f2188_0_0"]!= "No"].index
brain_train = brain_train.drop(index_illness_train)
brain_test = brain_test.drop(index_illness_test)
main_test = main_test.drop(index_illness_test)
main_train = main_train.drop(index_illness_train)

# exclude people with disorders of the nervous system
nervous_train = main_train[main_train["diagnoses_icd10_f41270_0_0"].str.contains("G")==True].index
nervous_test = main_test[main_test["diagnoses_icd10_f41270_0_0"].str.contains("G")==True].index
brain_train = brain_train.drop(nervous_train)
brain_test = brain_test.drop(nervous_test)
main_test = main_test.drop(nervous_test)
main_train = main_train.drop(nervous_train)

# exclude people with malicious neoplasm in the brain
neoplasm_train = main_train[main_train["diagnoses_icd10_f41270_0_0"].str.contains("C71")==True].index
neoplasm_test = main_test[main_test["diagnoses_icd10_f41270_0_0"].str.contains("C71")==True].index
brain_train = brain_train.drop(neoplasm_train)
brain_test = brain_test.drop(neoplasm_test)
main_test = main_test.drop(neoplasm_test)
main_train = main_train.drop(neoplasm_train)

# exclude people with circulatory disease 
circulatory_train = main_train[main_train["diagnoses_icd10_f41270_0_0"].str.contains("I")==True].index
circulatory_test = main_test[main_test["diagnoses_icd10_f41270_0_0"].str.contains("I")==True].index
brain_train = brain_train.drop(circulatory_train)
brain_test = brain_test.drop(circulatory_test)
main_test = main_test.drop(circulatory_test)
main_train = main_train.drop(circulatory_train)

# exclude people who don't have good or excellent self-rated health
health_train = main_train[(main_train["overall_health_rating_f2178_0_0"]!= "Good") & (main_train["overall_health_rating_f2178_0_0"]!= "Excellent")].index
health_test = main_test[(main_test["overall_health_rating_f2178_0_0"]!= "Good") & (main_test["overall_health_rating_f2178_0_0"]!= "Excellent")].index
brain_train = brain_train.drop(health_train)
brain_test = brain_test.drop(health_test)
main_test = main_test.drop(health_test)
main_train = main_train.drop(health_train)

# # plot self-reported moderate intensity physical activity
# plt.figure()
# plt.hist(main_train.iloc[:,526])
# plt.hist(main_test.iloc[:,526])
# plt.title("Days per week of self-rated moderate physical activity")
# plt.show()

# # MET of moderate intensity physical activity
# plt.figure()
# plt.hist(main_train.iloc[:,9990])
# plt.hist(main_test.iloc[:,9990])
# plt.title("MET Minutes per week of self-rated moderate physical activity")
# plt.show()

# # plot self-reported vigorous intensity physical activity
# plt.figure()
# plt.hist(main_train.iloc[:,534])
# plt.hist(main_test.iloc[:,534])
# plt.title("Days per week of self-rated vigorous physical activity")
# plt.show()

# # MET of vigorous intensity physical activity
# plt.figure()
# vPA_train = main_train.drop(main_train[main_train.iloc[:,9991] > 5000].index)
# vPA_test = main_test.drop(main_test[main_test.iloc[:,9991] > 5000].index)
# plt.hist(vPA_train.iloc[:,9991])
# plt.hist(vPA_test.iloc[:,9991])
# plt.xticks(range(0,6000,1000))
# plt.title("MET Minutes per week of self-rated vigorous physical activity")
# plt.show()

# # plot age
# plt.figure
# plt.hist(main_train["Age"])
# plt.hist(main_test["Age"])
# plt.title("Age")
# plt.show()

# # plot gender
# train_female = (main_train.iloc[:,22] == "Female").sum()/14852
# train_male = (main_train.iloc[:,22] == "Male").sum()/14582
# test_female = (main_test.iloc[:,22] == "Female").sum()/12018
# test_male = (main_test.iloc[:,22] == "Male").sum()/12018
# plt.figure()
# plt.pie([train_female,train_male],labels=["Female","Male"], textprops={'fontsize': 15}, colors= ["b","navy"])
# plt.show()
# plt.figure()
# plt.pie([test_female,test_male],labels=["Female","Male"], textprops={'fontsize': 15}, colors = ["orange","orangered"])
# plt.show()


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
brain_train_s = pd.DataFrame(brain_train_s)
brain_test_s = pd.DataFrame(brain_test_s)

# delete rows with nan values
nan_test = brain_test_s[brain_test_s.isnull().any(axis=1)].index
nan_train = brain_train_s[brain_train_s.isnull().any(axis=1)].index
brain_testing = brain_test_s.dropna(axis=0)
brain_training = brain_train_s.dropna(axis=0)
main_test = main_test.drop(nan_test)
main_train = main_train.drop(nan_train)

# delete same rows for main data -> for later correlation analysis

# here also make a dataset with all the varaibles that we will later need -> new dataset combining all varaibles of interest in one dataframe

# # impute nan values in mri data
# # imp_mean = IterativeImputer(random_state=0)
# # imp_mean.fit_transform(df[:,:-1])

# declare training and testing data - X: brain variables, Y: actual age
X_train = brain_training.iloc[:,:-1]
Y_train = brain_training.iloc[:,-1]
X_test = brain_testing.iloc[:,:-1]
Y_test = brain_testing.iloc[:,-1]

# set of alphas to try
alpha_parameters = np.power(10,np.linspace(start=-3, stop=5, num=100))

# standardize x data
scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)

# cross validation on training set only
# X_CV = X_train_std
# Y_CV = Y_train
# # randomly split data into training and testing set
# X_train_cv, X_test_cv, Y_train_cv, Y_test_cv = train_test_split(X_CV,Y_CV, random_state=8)
# model = RidgeCV(alphas = alpha_parameters)
# model.fit(X_train_cv,Y_train_cv)
# y_pred_cv = model.predict(X_test_cv)
# print("The MAE for cross-validation:", mean_absolute_error(Y_test_cv,y_pred_cv))


# model on trained on training tested on test
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
main_test["BrainAge Delta"] = brain_age_delta

# check whether brain age gap and age are correlated
correlation = stats.pearsonr(brain_age_delta, Y_test)
#print(correlation)

# get mean absolute error (MAE)
print("The MAE for testing set:", mean_absolute_error(Y_test,y_pred))

# #plot figure with x: actual age Y: predicted age, and a line with slope 1 for reference
# plt.figure()
# plt.scatter(Y_test, y_pred, alpha=0.5)
# plt.axline((60,60), slope=1, color='r')
# plt.xticks(range(40,90,5))
# plt.yticks(range(40,90,5))
# plt.ylabel('Predicted Age', fontsize = 15)
# plt.xlabel('Age', fontsize = 15)
# plt.show()

# plt.figure()
# plt.scatter(Y_test,brain_age_delta, alpha=0.5)
# plt.axline((60,5), slope=-0.69, color="r")
# plt.axline((60,0),slope=0, color = "g")
# plt.ylabel("Brain Age Delta", fontsize = 15)
# plt.xlabel("Age", fontsize = 15)
# plt.show()



# physical activity accelerometer processing

# get weartime duration in minutes (it's in days at default)
main_test["Wear time in minutes"] = main_test.iloc[:,16686]*1440
plt.figure
plt.hist(main_test["Wear time in minutes"])
plt.title("Wear time in minutes")
plt.show()

# exclude people with insufficient weartime weartime < 8000 minutes
wear_time_test = main_test[main_test["Wear time in minutes"]<8000].index
brain_test = brain_test.drop(wear_time_test)
main_test = main_test.drop(wear_time_test)

# distributions are cumulative -> substract ditributions from each other
# determine fraction of weartime spent doing light PA (between 30 and 125 milligravites)
light_PA = np.asarray(main_test.iloc[:,16763]-main_test.iloc[:,16748])
# get it in minutes / week
light_PA = light_PA * main_test["Wear time in minutes"]
main_test["Light PA"] = light_PA

# determine fraction of weartime spent doing moderate PA (between 125 and 400 milligravites)
moderate_PA = np.asarray(main_test.iloc[:,16774]-main_test.iloc[:,16763])
moderate_PA = moderate_PA * main_test["Wear time in minutes"]
main_test["Moderate PA"] = moderate_PA

# determine fraction of weartime spent doing vigorous PA (above 400 milligravites)
vigorous_PA = np.asarray(1-main_test.iloc[:,16774])
vigorous_PA = vigorous_PA * main_test["Wear time in minutes"]
main_test["Vigorous PA"] = vigorous_PA

# fraction of weartime for above 30 mg -> total summed light, moderate, and vigorous PA
total_PA = np.asarray(1-main_test.iloc[:,16748])
total_PA = total_PA * main_test["Wear time in minutes"]
main_test["Total PA"] = total_PA


# get correlation between BrainAge Delta and light PA
correlation = stats.pearsonr(main_test["BrainAge Delta"], main_test["Light PA"])
print(correlation)
correlation = stats.pearsonr(main_test["BrainAge Delta"], main_test["Moderate PA"])
print(correlation)
correlation = stats.pearsonr(main_test["BrainAge Delta"], main_test["Vigorous PA"])
print(correlation)
correlation = stats.pearsonr(main_test["BrainAge Delta"], main_test["Total PA"])
print(correlation)

plt.figure()
plt.scatter(main_test["BrainAge Delta"], main_test["Light PA"], alpha=0.5)
plt.ylabel("Light PA", fontsize = 15)
plt.xlabel("Brain Age Delta", fontsize = 15)
plt.show()

plt.figure()
plt.scatter(main_test["BrainAge Delta"], main_test["Moderate PA"], alpha=0.5)
plt.ylabel("Moderate PA", fontsize = 15)
plt.xlabel("Brain Age Delta", fontsize = 15)
plt.show()

plt.figure()
plt.scatter(main_test["BrainAge Delta"], main_test["Vigorous PA"], alpha=0.5)
plt.ylabel("Vigorous PA", fontsize = 15)
plt.xlabel("Brain Age Delta", fontsize = 15)
plt.show()

plt.figure()
plt.scatter(main_test["BrainAge Delta"], main_test["Total PA"], alpha=0.5)
plt.ylabel("Total PA", fontsize = 15)
plt.xlabel("Brain Age Delta", fontsize = 15)
plt.show()