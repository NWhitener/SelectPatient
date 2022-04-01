import pandas as pd
from keras.utils import to_categorical
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from keras import optimizers
import time
import sys
from keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
import tracemalloc
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, accuracy_score
from sklearn.metrics import roc_curve,roc_auc_score

def evaluate_nmi(observed, predicted):
   nmi = normalized_mutual_info_score(observed, predicted)
   return(nmi)

def evaluate_roc(observed, predicted):
   roc = roc_auc_score(observed, predicted)
   return(roc)

def evaluate_kappa(observed, predicted):
   kap = cohen_kappa_score(observed, predicted)
   return(kap)

def evaluate_f1(observed, predicted):
   f1 = precision_recall_fscore_support(observed, predicted, average='weighted')
   return(f1)

def train_ffnn_model(train_x, train_y, test_x, num_labels):
    model = Sequential()
    model.add(Dense(256, input_dim=train_x.shape[1], kernel_regularizer=l2(0.01), activation="relu"))
    model.add(Dropout(0.7))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(numlabels,activation = "sigmoid"))

    model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['binary_accuracy'])

    model.fit(train_x, train_y, epochs=50, batch_size=128,verbose=1)
    #(loss, accuracy) = model.evaluate(train_x, train_y, batch_size=128, verbose=1)
    predictions = np.argmax(model.predict(test_x), axis=-1)
    return(predictions)


def create_labs(labels):

    #if row["tissue"] == "TUMOR":
    #if row["sex"] == "M":
    if labels["SEX"] == "male":
        val = 0
    else:
        val = 1
    return val

def run_ffnn_model(data, labels, num_labels):
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
        fold_result['FFNN'] = train_ffnn_model(data[train], labels[train], data[test], numlabels)
        nmi_scores_ffnn.append(evaluate_nmi(fold_result['class'], fold_result['FFNN']))
        auc_scores_ffnn.append(evaluate_roc(fold_result['class'], fold_result['FFNN']))
        kappa_scores_ffnn.append(evaluate_kappa(fold_result['class'], fold_result['FFNN']))
        f1_scores_ffnn.append(evaluate_f1(fold_result['class'], fold_result['FFNN']))
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

    print(results)
    results.to_csv('Metrics_3layer_binary_acc_sig.csv')


