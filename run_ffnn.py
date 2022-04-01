
import preprocess as ps 
import ffnn as ffnn 

def full_model(data, labels):
    labels_patient, data_patient, num_labels = ps.selectPatient(labels, data, option = False, balance=True, PrepareData= True)
    ffnn.run_ffnn_model(data_patient,labels_patient, num_labels)

