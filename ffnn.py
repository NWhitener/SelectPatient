import preprocess as prepro 
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


def evaluate_performance(observed, predicted):
   nmi = normalized_mutual_info_score(observed, predicted)
   return(nmi)

def train_ffnn_model(train_x, train_y, test_x, num_labels):
    model = Sequential()
    model.add(Dense(256, input_dim=train_x.shape[1], kernel_regularizer=l2(0.01), activation="relu"))
#    model.add(Dropout(0.7))
#    model.add(Dense(128, activation="relu"))
#    model.add(Dense(64, activation="relu"))
    model.add(Dense(num_labels, activation="softmax"))

    model.compile(loss='categorical_crossentropy', 
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['categorical_accuracy'])

    model.fit(train_x, train_y, epochs=50, batch_size=128,verbose=1)
    #(loss, accuracy) = model.evaluate(train_x, train_y, batch_size=128, verbose=1)
    predictions = np.argmax(model.predict(test_x), axis=-1)
    return(predictions)

def prepare_data(dir_path, j):
   data = pd.read_csv(dir_path+j+"/benchmark_"+j+".txt", sep="\t",index_col=0)
   data = np.array(data)
   from sklearn.preprocessing import OrdinalEncoder
   labels = pd.read_csv(dir_path+j+"/labels.txt", sep="\t")
   ord_enc = OrdinalEncoder()
   labels['encoding'] = ord_enc.fit_transform(labels[["labels"]])
   numlabels = len(set(labels['encoding']))
   labels = to_categorical(labels['encoding'], num_classes=len(pd.unique(labels['encoding'])), dtype='float32')
   return(data, labels, numlabels)

results = pd.DataFrame(columns=['ID', 'FFNN_AVE', 'FFNN_STD'])

for j in range(1, 16): 
    j = str(j)
    (data, labels, numlabels) = prepare_data(data, j)
    fold_number = 1
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    nmi_scores_ffnn = []
    nmi_scores_xgboost = []
    for train, test in kfold.split(data, np.argmax(labels, axis= -1)):
        fold_result = pd.DataFrame()
        fold_result['class'] = np.argmax(labels[test], axis= -1)
        fold_result['FFNN'] = train_ffnn_model(data[train], labels[train], data[test], numlabels)
        nmi_scores_ffnn.append(evaluate_performance(fold_result['class'], fold_result['FFNN']))
        fold_number = fold_number + 1
    results = results.append({'ID': j, 
              'FFNN_AVE': np.mean(nmi_scores_ffnn), 
              'FFNN_STD': np.std(nmi_scores_ffnn)},
              ignore_index=True)
print(results)


results = pd.DataFrame(columns=['ID', 'FFNN_AVE', 'FFNN_STD'])



