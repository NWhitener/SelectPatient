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
