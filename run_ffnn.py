
import preprocess as ps 
import ffnn as ffnn 
from sklearn.model_selection import StratifiedKFold
import evaluate as ev 
import ffnn as ffnn 
import pandas as pd 
import numpy as np 

def run_ffnn_model(data, labels):
    labels_patient, data_patient, num_labels = ps.selectPatient(labels, data, option = False, balance=True, PrepareData= True)
    
    model = ffnn.create_ffnn_model
    model = ffnn.train_ffnn(model,)
    results = pd.DataFrame(columns=['FFNN_NMI_AVE', 'FFNN_NMI_STD', 'FFNN_ROC_AVE','FFNN_ROC_STD','FFNN_KAPPA_AVE', 'FFNN_KAPPA_STD'])
    fold_number = 1
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    nmi_scores_ffnn = []
    auc_scores_ffnn = []
    kappa_scores_ffnn = []
    f1_scores_ffnn = []
    for train, test in kfold.split(data, np.argmax(labels, axis= -1)):
        fold_result = pd.DataFrame()
        fold_result['class'] = np.argmax(labels[test], axis= -1)
        fold_result['FFNN'] =  ffnn.test_ffnn(model)
        nmi_scores_ffnn.append(ev.evaluate_nmi(fold_result['class'], fold_result['FFNN']))
        auc_scores_ffnn.append(ev.evaluate_roc(fold_result['class'], fold_result['FFNN']))
        kappa_scores_ffnn.append(ev.evaluate_kappa(fold_result['class'], fold_result['FFNN']))
        f1_scores_ffnn.append(ev.evaluate_f1(fold_result['class'], fold_result['FFNN']))
        fold_number = fold_number + 1
    results = results.append({
              'FFNN_NMI_AVE': np.mean(nmi_scores_ffnn),
              'FFNN_NMI_STD': np.std(nmi_scores_ffnn),
              'FFNN_ROC_AVE': np.mean(auc_scores_ffnn),
              'FFNN_ROC_STD': np.std(auc_scores_ffnn),
              'FFNN_KAPPA_AVE': np.mean(kappa_scores_ffnn),
              'FFNN_KAPPA_STD': np.std(kappa_scores_ffnn),
              },
              ignore_index=True)
    results.to_csv('Metrics_3layer_binary_acc_sig.csv')