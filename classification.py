from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


def build_classifier(classifier, x_train, y_train, random_state):
    # Choose classifier
    if classifier == "RF":
        clf = RandomForestClassifier(random_state=random_state)
    elif classifier == "SVM":
        clf = svm.SVC(random_state=random_state, probability=True, cache_size = 5000)
    elif classifier == "KNN_3":
        clf = KNeighborsClassifier(n_neighbors=3)
    elif classifier == "KNN_5":
        clf = KNeighborsClassifier(n_neighbors=5)
    elif classifier == "MLP":
        clf = MLPClassifier(random_state=random_state, max_iter=100000)
    elif classifier == "xgboost":
        clf = xgb.XGBClassifier(random_state=random_state)
    else:
        clf = RandomForestClassifier(random_state=random_state)

    clf.fit(x_train, y_train)

    return clf



