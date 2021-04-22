import tools.project_training_parser as project_training_parser
import tools.total_aaindex_parser as total_aaindex_parser
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile, f_classif

#parse aminoacid data to generate trainingsfeatures
def parseAaIndex():
    ds = open("data/aaindex1.txt", "r")
    lines = ds.read().splitlines()
    aaIndex = []
    counter = 0
    for i in range(len(lines)):
        if lines[i][0] == "I":
            bothLines = lines[i + 1] + lines[i + 2]
            tempindex = bothLines.split()
            aaIndex.append(tempindex)
            counter += 1
    ds.close()
    return aaIndex

#cleaning and preparing aminoacid data
def getFeaturesforAAX():
    a = parseAaIndex()
    rmv = []
    for feature in a:
        if 'NA' in feature:
            rmv.append(feature)
    for feature in rmv:
        a.remove(feature)
    lst = []
    for feature in a:
        b = list(map(float, feature))
        lst.append(b)
    return lst

#read trainigdata 
def parseProjectTraining():
    training_data = []
    ds = open("data/project_training.txt", "r")
    ligands = []
    labels_str = []
    next(ds)
    for line in ds:
        ligands.append(line.split(None, 1)[0])
        labels_str.append(line[-2])
    #print(ligands, labels_str)
    ds.close()
    labels = []
    for i in range(len(labels_str)):
        if(labels_str[i]!='\t'):
            temp = int(labels_str[i])
            if temp:
                labels.append(temp)
            else:
                labels.append(0)
    for x in range(0, len(ligands)):
        if(line!='\t'):
            training_data.append([ligands[x], labels[x]])
    return training_data, ligands, labels

def getFeaturesForAS(x):
    features_for_as = []
    for feature in getFeaturesforAAX():
        features_for_as.append(feature[x])
    return features_for_as


# getting features for every amino acid
def setAllFeatures():
    aas = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    feature_dic = {}
    for x in range(0, len(aas)):
        feature_dic.update({aas[x]: getFeaturesForAS(x)})
    return feature_dic


# Represents peptides as feature vector
def featureLigands(ligands, features):
    a = setAllFeatures()
    number_of_features = len(a["A"])
    ligands_featured = []
    for x in range(0, len(ligands)):
        peptide = []
        for y in range(0, number_of_features):
            for char in ligands[x]:
                peptide.append(features[char][y])
        ligands_featured.append(peptide)
    return ligands_featured


# prepare training data for machine learning algorithms
def prepareTrainingData():
    trainingData, ligands, labels = parseProjectTraining()
    labels = np.array(labels)
    features = np.array(featureLigands(ligands, setAllFeatures()))
    # Skalierung der Trainingsdaten mit MinMaxScaler
    scaler = preprocessing.MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    # Feature Selection mit 10 Percentil
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(scaled_features, labels)
    #joblib.dump(selector, './selector/selector.pkl')
    selected_features = selector.transform(scaled_features)
    return selected_features, labels
