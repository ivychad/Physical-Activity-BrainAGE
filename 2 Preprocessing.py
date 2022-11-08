"""
- Exclude subjects based on:
    - Missing Age
    - Illness, Disability, Infirmity
    - Disorders of Nervous system
    - Brain malicious neoplasms
    - Circulatory disease
    - Not good or excellent self-rated health
    - For test set: Insufficient wear time <8000 minutes
- Physical activity accelerometry preprocessing (light, moderate, vigorous, total PA)
- Crop main data to only variables used for later analysis
    xyz
- Pickle set with all the excluded data for later comparison

ToDo
- excluded data

Variables to include from main
- ID
- Age at MRI
- Gender
- Self reported MET for light, moderate, vigorous
- For test: Accelerometery light, moderate, vigorous, total
- Grip stength left and right
- ECG (maximum workload, maximum heartrate)
- BMI

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import seed


seed(888)

# load data from pickle and convert to dataframe
brain_test = pd.read_pickle("1_brain_test.pkl")
brain_train = pd.read_pickle("1_brain_train.pkl")
main_test = pd.read_pickle("1_main_test.pkl")
main_train = pd.read_pickle("1_main_train.pkl")
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

# select data for later analysis
main_test_crop = []
main_test_crop = pd.DataFrame(main_test_crop)
main_train_crop = []
main_train_crop = pd.DataFrame(main_train_crop)

main_test_crop["Age"] = main_test["Age"]
main_test_crop["Gender"] = main_test.iloc[:,22]
main_test_crop["Total PA"] = main_test["Total PA"]
main_test_crop["Vigorous PA"] = main_test["Vigorous PA"]
main_test_crop["Moderate PA"] = main_test["Moderate PA"]
main_test_crop["Light PA"] = main_test["Light PA"]
main_test_crop["BMI"] = main_test.iloc[:,11580]
main_test_crop["ECG max workload"] = main_test.iloc[:, 5773]
main_test_crop["ECG max heart rate"] = main_test.iloc[:, 5775]
main_test_crop["SR Walking PA"] = main_test.iloc[:,9989]
main_test_crop["SR Moderate PA"] = main_test.iloc[:,9990]
main_test_crop["SR Vigorous PA"] = main_test.iloc[:,9991]
#main_test_crop["Job PA"] = main_test.iloc[:,507]
main_test_crop["Hand grip L"] = main_test.iloc[:,64]
main_test_crop["Hand grip R"] = main_test.iloc[:,68]

# select excluded data for train and test -> cross reference IDs


# pickle end results 
main_test_crop.to_pickle("2_main_test.pkl")
main_train.to_pickle("2_main_train.pkl")
brain_test.to_pickle("2_brain_test.pkl")
brain_train.to_pickle("2_brain_train.pkl")
