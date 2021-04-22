import pandas as pd
import numpy as np
import evaluation
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from sklearn.ensemble import RandomForestClassifier



def hyperparams_svc():
    # parameters to optimize
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    classifier = SVC()
    return classifier, tuned_parameters


def hyperparams_mlp(random_state):
    #tuned_parameters = [{'hidden_layer_sizes': [(randint.rvs(20, 100, 1), randint.rvs(20, 100, 1),), (randint.rvs(20, 100, 1),)], 'activation': ['tanh', 'relu', 'logistic'],
                     #'solver': ['sgd', 'adam'],
                    #'alpha': [.001, .005, .01, .05, .1, .2, .4, .8], 'learning_rate': ['constant', 'adaptive']}]

    tuned_parameters = {
        'hidden_layer_sizes': [(randint.rvs(20, 100, 1), randint.rvs(20, 100, 1),), (randint.rvs(20, 100, 1),)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [.001, .005, .01, .05, .1, .2, .4, .8],
        'learning_rate': ['constant','adaptive'],
    }   
    classifier = MLPClassifier(random_state=random_state, max_iter=100000)
    
    return classifier, tuned_parameters

def hyperparams_rf(random_state):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)

    tuned_parameters = {'n_estimators': n_estimators,
                'max_features': ['auto', 'sqrt'],
                'max_depth': max_depth,
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]}

    classifier = RandomForestClassifier(random_state=random_state)
    
    return classifier, tuned_parameters

def hyperparams(classifier, tuned_parameters, X_train, y_train, X_test, y_test):

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)

        clf = GridSearchCV(
            classifier, tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
 
        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))

