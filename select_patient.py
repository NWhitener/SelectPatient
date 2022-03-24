import pandas as pd
import random

def renumberDid(label):
    '''
  Help for the renumberDid() function. 
  
  Purpose: renumberDid() takes the labels passed in through the sole required argument.
           This labels set need to be in the form of a Pandas DataFrame, this method will not 
           automatically convert the label set. 
  Return: renumberDid() will return the label set with the Patient Donor ID "DID" numbered from 0 
           to the number of unique Patient Donor ID's
  Example: labels_renumbered = renumberDid(labels)
    '''
    did_list_raw = label['DID'].value_counts().sort_index().index.values.tolist()
    did_list_renumbered = list(range(0,len(did_list_raw)))
    label_did_list = label['DID'].tolist()
    for num in range(0,len(label_did_list)):
      index = did_list_raw.index(label_did_list[num])
      label_did_list[num] = did_list_renumbered[index]
    label['DID'] = label_did_list
    return label

def readData(countfile, labelfile):
    data = pd.read_csv(countfile, index_col=None, low_memory=False)
    data.rename(columns={'Unnamed: 0': 'cell_id'}, inplace=True)
    label = pd.read_csv(labelfile, index_col=None, low_memory=False)
    label.rename(columns={'X': 'cell_id'}, inplace=True)
    return data, label

def selectData(data, col2select, ids):
    '''
    Help for the selectData() function. 

    Purpose: selectData() takes a data set, a column from the label set, and the match annotations to the dataset.
             With this information, selectData(), matches the data in the data file with the labels in the label 
             file based on the column. 
    Return:  selectData() will return data that matches the labels in based on the column selected 
           to the number of unique Patient Donor ID's
    Example: data = selectData(data, 'CELLID',labels['CELLID'].to_list())
    '''
    sample = data[data[col2select].isin(ids)]
    return sample

def prepareData(data, labels):
    '''
    Help for the prepareData() function. 

    Purpose: prepareData() takes a dataset and a lables set, and imposes the cell-id coulmn as the defining feature. Do not use this 
             function if the first column in the dataset is not the Cell-ID. Also, note that this function will not work after the 
             readData function is used
    Return:  prepareData() will return the dataset and labels with the first column renamed to CELLID, for use of selectData,
             and other functions in the library
    Example: data, labels = prepareData(data,labelsSet)
    '''
    data.rename(columns={'Unnamed: 0':'CELLID'}, inplace=True)  
    labels.rename(columns={'Unnamed: 0':'CELLID'}, inplace = True)
    return data, labels

def selectPatient(labelsSet,data, option = False,  whatLabels = "STATUS", valueWanted = "normal", balance = True, PrepareData = True):
    '''
    Help for the selectPatient() function. 

    Purpose: selectPatient() takes a label set and a dataset and returns a dataset composed of a randomly selected male patient and a 
             randomly selected female patient.
    Arguements: option: This is a boolean parameter that can take either True or False. 
                        to be used in conjuction with the whatLabels and valueWanted parameter
                        if oprtion = False, the default values of whatLabels and valueWanted will be used. 
                        Otherwise the user will be promted to input the whatLabels and valueWanted manually. 
                        Defaulted to False. 
                whatLabels: This arguement is the determining factor, that selectes which label will be 
                            selected upon. For example, STATUS will select patients based only on the STATUS label. 
                            If set before execution then option should be set to False or be left out 
                valueWanted: This arguement is the discrinimating value for the patients to be selected on. For example, if whatLabels is 
                             STATUS then valueWanted = "normal", then only patients with a normal STATUS sample 
                             will be considered. 
                             If set before execution then option should be set to False or be left out 
                balance: This arguement allows for the balancing of the number of samples in the  male patient and the number of 
                         samples in the female patient. If the arguement is set to true, then the function will ramdomly limit the number of 
                         samples to the value of the smaller sample, "balancing" the data. IF set to false, then the samples 
                         will return all of the sampled data. Default True. 
                PrepareData: This arguement uses the prepareData() function to ensure that the data is in the proper format for 
                             the execution
                             of the rest of the function. If False, then the data is assumed to be already perpared and the function is run 
                             Defaulted to True.
    Return:  selectPatient() will return a dataset and annotation set that contain 1 male patient, and 1 female patient. 
    Example:labels_patient, data_patient = sp.selectPatient(labels,data,option=True, balance = True, PrepareData = True)
    '''
    if(PrepareData):
         data, labels = prepareData(data,labelsSet)
    labels_men = labels.loc[labels["SEX"] == "male"]
    labels_women = labels.loc[labels["SEX"] == "female"]
        #Gives the user the option to input the labels they want from the command line         
    if(option == True):
        print(labelsSet.columns)
        whatLabels, valueWanted = input("Enter the Label you want, then the discriminating value you want \n").split()
    labels_men = labels_men.loc[labels_men[whatLabels] == valueWanted]
    labels_women = labels_women.loc[labels_women[whatLabels] == valueWanted]
    labels_women_ordered = renumberDid(labels_women)
    uni_num_women = random.randint(0, len(pd.unique(labels_women_ordered["DID"]))-1)
    labels_men_ordered = renumberDid(labels_men)
    uni_num_men = random.randint(0, len(pd.unique(labels_men_ordered["DID"]))-1)
    labels_patient_women = labels_women_ordered.loc[labels_women_ordered["DID"] == uni_num_women]
    labels_patient_men = labels_men_ordered.loc[labels_men_ordered["DID"] == uni_num_men]
    percent_male = len(labels_patient_men)
    percent_women = len(labels_patient_women)
    if (not balance):
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
    data_male = selectData(data, 'CELLID',labels_patient_men['CELLID'].to_list())
    data_female = selectData(data, 'CELLID',labels_patient_women['CELLID'].to_list())
    labels = labels.drop(labels["DID"]!=uni_num_men)
    labels = labels.drop(labels["DID"]!=uni_num_women)
    data = selectData(data, "CELLID",labels["CELLID"].to_list())
    frames = [labels_patient_women,labels_patient_men]
    labels_patients = pd.concat(frames)
    frames2 = [data_female,data_male]
    data_patients = pd.concat(frames2)
    return labels_patients, data_patients, labels, data 


