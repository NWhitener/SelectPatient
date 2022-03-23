import pandas as pd
import random


def renumber_did(label):
   did_list_raw = label['DID'].value_counts().sort_index().index.values.tolist()
   did_list_renumbered = list(range(0,len(did_list_raw)))
   label_did_list = label['DID'].tolist()
   for num in range(0,len(label_did_list)):
      index = did_list_raw.index(label_did_list[num])
      label_did_list[num] = did_list_renumbered[index]
   label['DID'] = label_did_list
   return label



def read_data(countfile, labelfile):
   data = pd.read_csv(countfile, index_col=None, low_memory=False)
   data.rename(columns={'Unnamed: 0': 'cell_id'}, inplace=True)
   label = pd.read_csv(labelfile, index_col=None, low_memory=False)
   label.rename(columns={'X': 'cell_id'}, inplace=True)
   return data, label

def select_data(data, col2select, ids):
   sample = data[data[col2select].isin(ids)]
   return sample

def prepareData(data, labels):
   data.rename(columns={'Unnamed: 0':'CELLID'}, inplace=True)  
   labels.rename(columns={'Unnamed: 0':'CELLID'}, inplace = True)
   return data, labels






# This method is designed to select the values of a patient based on their label
# and return it to the user so that it can be used in whatever test they want 
# this mehtod will only work on male female, patients first 
def selectPatient(labelsSet,data, option = False,  whatLabels = "STATUS", valueWanted = "normal", balance = 10):
        data, labels = prepareData(data,labelsSet)
        labels_men = labels.loc[labels["SEX"] == "male"]
        labels_women = labels.loc[labels["SEX"] == "female"]
        #Gives the user the option to input the labels they want from the command line         
        if(option == True):
            print(labelsSet.columns)
            whatLabels, valueWanted = input("Enter the Label you want, then the discriminating value you want \n").split()
        labels_men = labels_men.loc[labels_men[whatLabels] == valueWanted]
        labels_women = labels_women.loc[labels_women[whatLabels] == valueWanted]
        labels_women_ordered = renumber_did(labels_women)
        uni_num_women = random.randint(0, len(pd.unique(labels_women_ordered["DID"]))-1)
        labels_men_ordered = renumber_did(labels_men)
        uni_num_men = random.randint(0, len(pd.unique(labels_men_ordered["DID"]))-1)
        labels_patient_women = labels_women_ordered.loc[labels_women_ordered["DID"] == uni_num_women]
        labels_patient_men = labels_men_ordered.loc[labels_men_ordered["DID"] == uni_num_men]
        percent_male = len(labels_patient_men)
        percent_women = len(labels_patient_women)
        if balance != 1:
            if(percent_male != percent_women):
                print("Warning: The number of male samples is ", percent_male, " and the number of female patients is ",
                    percent_women, ". This is a", round((percent_male/(percent_male+percent_women))*100,2), "% male population and a ",
                    round((percent_women/(percent_male+percent_women))*100,2), "% female population. This is an imbalance.\n",
                    "This may effect performance, consider adding the balance parameter")
        elif balance == 1:
            if percent_women > percent_male:
                labels_patient_women = labels_patient_women.sample(n=percent_male)
            else:
                labels_patient_men = labels_patient_men.sample(n=percent_women)
        data_male = select_data(data, 'CELLID',labels_patient_men['CELLID'].to_list())
        data_female = select_data(data, 'CELLID',labels_patient_women['CELLID'].to_list())
        patient_ids = [labels_patient_women["DID"].values[0], labels_patient_men["DID"].values[0]]
        frames = [labels_patient_women,labels_patient_men]
        labels_patients = pd.concat(frames)
        frames2 = [data_female,data_male]
        data_patients = pd.concat(frames2)
        print("Patient 
        return labels_patients, data_patients, patient_ids


