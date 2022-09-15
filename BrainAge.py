
import pandas as pd
import numpy as np
from datetime import datetime

# load feather data and convert to dataframe
# general data
general_data = pd.read_feather("/mnt/UKBiobank/main_reduced.ftr", columns=None, use_threads=True, storage_options=None)
general_data = pd.DataFrame(general_data)
# mri data
brain_data = pd.read_feather("/mnt/UKBiobank/fs_derivatives_reduced.ftr", columns=None, use_threads=True, storage_options=None)
brain_data = pd.DataFrame(brain_data)


# get year the brain MRI scan was taken
for i in range(43167):
    if type(general_data.iloc[i,9912]) == str:
        date_format = datetime.strptime(general_data.iloc[i,9912], '%Y-%m-%d %H:%M:%S')
        only_year = date_format.year
        general_data.iloc[i,9912] = only_year

age = np.empty(43167)
age.fill(np.nan)

# compute age at time of MRI scan by (year of MRI scan - year of birth) and add as column to dataframe
for i in range(43167):
    if type(general_data.iloc[i,23]) == np.int32 and type(general_data.iloc[i,9912]) == int:
        age[i] = general_data.iloc[i,9912] - general_data.iloc[i,23]

# add at the end of general dataframe
general_data["Age"] = age
# add at the end of mri data (for brain age prediction)
brain_data["Age"] = age 

# split data into people with completed physical activity accelerometer assessment (PA_yes) and without (PA_no)
PA_yes = []
PA_no = []
PA_yes_brain = []
PA_no_brain = []
comp = "Completed"
main_data = np.asmatrix(general_data)
# add Physical activity completed? field to MRI dataframe (at the end) for splitting the mri data
brain_data["PA"] = main_data[:,18220]
brain_data = np.asarray(brain_data)

# split data based on whether the participant completed PA measurement for general and mri data
for i in range(43167):
    if main_data[i,18220] == comp:
        PA_yes.append(main_data[i])
    else:
        PA_no.append(main_data[i])

for i in range(43167):
    if brain_data[i,2546] == comp:
        PA_yes_brain.append(brain_data[i])
    else:
        PA_no_brain.append(brain_data[i])


# save preprocessed general and mri data as .csv file 
       
# test = np.asarray(PA_yes)
# train = np.asarray(PA_no)
# rtest = np.reshape(test, (19122,18223))
# rtrain = np.reshape(train, (24045,18223))
# np.savetxt("test_main.csv", rtest, delimiter=",", fmt='%s')
# np.savetxt("train_main.csv", rtrain, delimiter=",", fmt='%s')

# test_brain = np.asarray(PA_yes_brain)
# train_brain = np.asarray(PA_no_brain)
# train_brain = np.delete(train_brain, np.s_[2:2545:2], axis=1)
# test_brain = np.delete(test_brain, np.s_[2:2545:2], axis=1)
# np.savetxt("test_brain.csv", test_brain, delimiter=",", fmt='%s')
# np.savetxt("train_brain.csv", train_brain, delimiter=",", fmt='%s')

