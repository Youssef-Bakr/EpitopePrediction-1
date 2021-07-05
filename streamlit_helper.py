import base64
#import streamlit.components.v1 as components
import streamlit as st

import prep
import hyperparameter_tuning

import pandas as pd
import numpy as np
import seaborn as sn

from sklearn.model_selection import StratifiedKFold
import hyperparameter_tuning

def SVM_prediction():
    # Lädt Trainingsdaten von training_data.txt
    #trainingData, ligands, labels = prep.parseProjectTraining()
    features, labels = prep.prepareTrainingData()
    classifier, tuned_parameters = hyperparameter_tuning.hyperparams_svc()

    classifier.fit(features, labels)
    classifier.predict(input)




