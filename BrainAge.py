
import pandas as pd
import numpy as np
from datetime import datetime

# load data
general_data = pd.read_feather("/mnt/UKBiobank/main_reduced.ftr", columns=None, use_threads=True, storage_options=None)
general_data = pd.DataFrame(general_data)

# get year the brain MRI scan was taken
for i in range(43167):
    if type(general_data.iloc[i,9912]) == str:
        date_format = datetime.strptime(general_data.iloc[i,9912], '%Y-%m-%d %H:%M:%S')
        only_year = date_format.year
        general_data.iloc[i,9912] = only_year

# #print(ukb_data.iloc[:,9912])
age = np.empty(43167)
age.fill(np.nan)

# compute age at time of MRI scan by (year of MRI scan - year of birth) and add as column to dataframe
for i in range(43167):
    if type(general_data.iloc[i,23]) == np.int32 and type(general_data.iloc[i,9912]) == int:
        age[i] = general_data.iloc[i,9912] - general_data.iloc[i,23]

general_data["Age"] = age


# split data into people with completed physical activity accelerometer assessment (PA_yes) and without (PA_no)
main_data = np.asmatrix(general_data)
PA_yes = []
PA_no = []
comp = "Completed"

for i in range(43167):
    if main_data[i,18220] == comp:
        PA_yes.append(main_data[i])
    else:
        PA_no.append(main_data[i])
        
test = np.asarray(PA_yes)
rtest = np.reshape(test, (19122,18223))
np.savetxt("test.csv", rtest, delimiter=",", fmt='%s')


train = np.asarray(PA_no)
rtrain = np.reshape(train, (24045,18223))
np.savetxt("train.csv", rtrain, delimiter=",", fmt='%s')
