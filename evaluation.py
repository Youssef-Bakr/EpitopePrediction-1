import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def getAUC(y_true, y_pred):
    fp, tp, thresholds = roc_curve(y_pred, y_true)
    roc_auc = auc(fp, tp)
    return roc_auc
 
def plotROC(y_true, y_pred):
    fp, tp, thresholds = roc_curve(y_pred, y_true)
    roc_auc = auc(fp, tp)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fp, tp, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    print("ROC-Curve plotted")
    plt.show()

def cros_val_training_accuracy(clf, x_train, y_train):
    # 10-fold cross validation training scores
    cv_scores = cross_val_score(clf, x_train, y_train, cv=10)
    avg_cv_score = sum(cv_scores) / len(cv_scores)

    return avg_cv_score   


def mcc_training_accuracy(y_true, y_pred):
    mcc = matthews_corrcoef(y_true, y_pred)
    return mcc


def f1_score_result(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average="binary", pos_label="Positive")
    return f1


def precision_score_value(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average="binary", pos_label="Positive")
    return precision 

def confusion(y_true, y_pred):
    df_confusion = pd.crosstab(y_test_np, y_pred, rownames=['Actual'], colnames=['Predicted'])
    df_conf_norm = df_confusion / df_confusion.sum(axis=0)

    sn.heatmap(df_confusion, cmap="YlOrRd", annot=False)
    plt.show()
    sn.heatmap(df_conf_norm, cmap="YlOrRd", annot=False)
    plt.show()