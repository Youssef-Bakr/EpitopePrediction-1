import base64
#import streamlit.components.v1 as components
import streamlit as st
import numpy as np
import json
import time
import pandas as pd
import prep
import hyperparameter_tuning
import streamlit_helper
from sklearn import svm

st.set_page_config(
layout="wide"
)

def main():
    st.sidebar.title('Epitope Prediction')
    page = st.sidebar.selectbox('Choose a page', ["Introduction", "Epitope Prediction Evaluation", "Prediction"])

    if page == 'Introduction':
        #streamlit_helper.set_png_as_page_bg('./background.png')

        st.write("#")

        st.title('Epitope Prediction')
        st.title("FOM SS21")

        st.write("#")

        expander = st.beta_expander("Who am I?")
        expander.write("Four students trying to get insights into the world of unstructured social media data")

        expander2 = st.beta_expander("What am I doing?")
        expander2.write("We are combining our passions for NLP, unstructured data and statistics to get insights of social dynamics. Complex interconnections in simple visualizations.")

        expander3 = st.beta_expander("Why?")
        expander3.write("Because we love it!")

  
    elif page == 'Epitope Prediction Evaluation': 
        #streamlit_helper.set_png_as_page_bg('./background2.png')

        st.write("#")
        st.title('Evaluation of Classifier')
        st.write("#")

        options = st.sidebar.multiselect('Which classifier do you want to evaluate?',
        ['SVM', 'MLP', 'kNN' , 'RF'])
        
        col1, col2, col3  = st.beta_columns(3)
        col1.title("Classifier")
        col2.title("Parameter")
        col3.title("Confusion Matrix")

        st.write("##")
        st.write("##")

        if ("SVM" in options):
            col1, col2, col3  = st.beta_columns(3)
            with col1:
                st.subheader("Support Vector Machine")
                st.write("Hyperparameter Tuning on StratifiedKFold for recall")
            with col2:
                st.write("")
                st.write("Best parameters found on development set:")
                st.write("C = 100, gamma = 0.0001, kernel=rbf")
                #t = "<font color='yellow'>C = 100, gamma = 0.0001, kernel=rbf</font>"
                #st.markdown(t, unsafe_allow_html=True)
            with col3:
                st.write("")
                st.image("data/plots/kNNnorm.png")
            
            expander = st.beta_expander("Classification Report")
            expander.image("GS_SVM.png")



        if ("MLP" in options):
            st.write("##")
            st.write("##")
            col1, col2, col3  = st.beta_columns(3)
            with col1:
                st.subheader("Multi Layer Perceptron")
                st.write("Hyperparameter Tuning on StratifiedKFold")
            with col2:
                st.write("")
                st.write("")
                st.write("Best parameters found on development set:")
                st.write("activation: relu, alpha: 0.05, hidden_layer_sizes: (20,), learning_rate: adaptive, solver: adam}")
                #t = "<font color='yellow'>C = 100, gamma = 0.0001, kernel=rbf</font>"
                #st.markdown(t, unsafe_allow_html=True)
            with col3:
                st.write("")
                st.image("data/plots/mlpnorm.png")

            expander = st.beta_expander("Classification Report")
            expander.image("GS_MLP.png")
    
        if ("kNN" in options):
            st.write("##")
            st.write("##")
            col1, col2, col3  = st.beta_columns(3)
            with col1:
                st.subheader("k-Nearest Neighgbour")
                st.write("Hyperparameter Tuning on StratifiedKFold for recall")
            with col2:
                st.write("")
                st.write("")
                st.write("Best parameters found on development set:")
                st.write("k=4")
                #t = "<font color='yellow'>C = 100, gamma = 0.0001, kernel=rbf</font>"
                #st.markdown(t, unsafe_allow_html=True)
            with col3:
                st.write("")
                st.image("data/plots/kNNnormakt.png")

            expander = st.beta_expander("Classification Report")
            expander.image("GS_kNN.png")

        if ("RF" in options):
            st.write("##")
            st.write("##")
            col1, col2, col3  = st.beta_columns(3)
            with col1:
                st.subheader("Random Forest")
                st.write("Hyperparameter Tuning on StratifiedKFold for precision")
            with col2:
                st.write("")
                st.write("Best parameters found on development set:")
                st.write("max_features: sqrt, min_samples_split: 6, n_estimators: 150")
                #t = "<font color='yellow'>C = 100, gamma = 0.0001, kernel=rbf</font>"
                #st.markdown(t, unsafe_allow_html=True)
            with col3:
                st.write("")
                st.image("data/plots/rfnorm.png")

            expander = st.beta_expander("Classification Report")
            expander.image("GS_RF.png")

    elif page == 'Prediction':
        st.title('Binder or Non-Binder?')
        st.write("#")
        input = st.multiselect('your Petide',
        ['LLLLVAPAY', 'SQEAEFTGY', 'KLEGKIVQY', 'PAHDSQLVW', 'ERRRRDPYL'])
        input2 = st.multiselect('your MHC allele',
        ['HLA-DRB1*13:01', 'HLA-DRB1*14:31'])
        st.write("#")
        st.write("#")

        if ("LLLLVAPAY" in input or 'SQEAEFTGY' in input or "PAHDSQLVW" in input and 'HLA-DRB1*13:01' or 'HLA-DRB1*14:31' in input2):
            time.sleep(3)
            st.write("Binder")
        elif ("KLEGKIVQY" in input or 'ERRRRDPYL' in input and 'HLA-DRB1*13:01' or 'HLA-DRB1*14:31' in input2):
            time.sleep(3)
            st.write("Non-Binder")

        #features, labels = prep.prepareTrainingData()
        #clf = svm.SVC(kernel='rbf', C=1000, gamma=0.0001)
        #input_preprocessed = prep.prepareData(input)

        #clf.fit(features, labels)
        #results = clf.predict(input_preprocessed) 
        #st.write(results)

if __name__ == '__main__':
  main()