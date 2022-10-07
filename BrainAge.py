
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

# get year and month the brain MRI scan was taken -> format year + month/12
for i in range(43167):
    if type(general_data.iloc[i,9912]) == str:
        date_format = datetime.strptime(general_data.iloc[i,9912], '%Y-%m-%d %H:%M:%S')
        month = int(date_format.month)
        year = int(date_format.year)
        date = year + month/12
        general_data.iloc[i,9912] = date

# create new array that will contain the age
age = np.empty(43167)
age.fill(np.nan)

# compute age at time of MRI scan by (year.month of MRI scan - year.month of birth) and add at proper index to age array
# if either MRI date or date of birth are missing, the value will stay nan
for i in range(43167):
    if type(general_data.iloc[i,23]) == np.int32 and type(general_data.iloc[i,88]) == str and type(general_data.iloc[i,9912]) == float:
        year = general_data.iloc[i,23]
        datetime_object = datetime.strptime(general_data.iloc[i,88], "%B")
        month = datetime_object.month
        birth_date = year + month/12 
        age[i] = general_data.iloc[i,9912] - birth_date

# add at the end of general dataframe
general_data["Age"] = age
# add at the end of mri data (for brain age prediction)
brain_data["Age"] = age 


# split data into people with completed physical activity accelerometer assessment (PA_yes) and without (PA_no)
comp = "Completed"
PA_yes = general_data[general_data["invitation_to_physical_activity_study_acceptance_f110005_0_0"] == comp]
PA_no = general_data[general_data["invitation_to_physical_activity_study_acceptance_f110005_0_0"] != comp]
PA_yes_brain = brain_data[general_data["invitation_to_physical_activity_study_acceptance_f110005_0_0"] == comp]
PA_no_brain = brain_data[general_data["invitation_to_physical_activity_study_acceptance_f110005_0_0"] != comp]

# np.savetxt("test_main.csv", PA_yes, delimiter=",", fmt='%s')
# np.savetxt("train_main.csv", PA_no, delimiter=",", fmt='%s')
# np.savetxt("test_brain.csv", PA_yes_brain, delimiter=",", fmt='%s')
# np.savetxt("train_brain.csv", PA_no_brain, delimiter=",", fmt='%s')

PA_yes.to_pickle("test_main.pkl")
PA_no.to_pickle("train_main.pkl")
PA_yes_brain.to_pickle("test_brain.pkl")
PA_no_brain.to_pickle("train_brain.pkl")


# old way of getting only the 2nd visit (all the data)
# train_brain = np.delete(train_brain, np.s_[2:2545:2], axis=1)
# test_brain = np.delete(test_brain, np.s_[2:2545:2], axis=1)


