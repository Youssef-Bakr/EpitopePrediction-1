import classification
import evaluation
import hyperparameter_tuning
import data_cleaning
import prep
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

#get features, labels from trainngsdata
features, labels = prep.prepareTrainingData()

# stratified k fold
skf = StratifiedKFold(labels, n_folds=10)
for train, test in skf:
        X_train, X_test, y_train, y_test = features[train], features[test], labels[train], \
                                                                   labels[test]

#parameter tuning of classifier
#MLP:
classifier, tuned_parameters = hyperparameter_tuning.hyperparams_mlp(random_state=3)
#SVM:
#classifier, tuned_parameters = hyperparameter_tuning.hyperparams_svc()
#random forest:
#classifier, tuned_parameters = hyperparameter_tuning.hyperparams_rf(random_state=3)

y_true, y_pred = hyperparameter_tuning.hyperparams(classifier, tuned_parameters, X_train, y_train, X_test, y_test)

evaluation.plotROC(y_true, y_pred)
evaluation.confusion(y_true, y_pred)
                    
if __name__ == "__main__":
    main()                                                          