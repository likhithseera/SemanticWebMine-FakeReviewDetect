from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import nltk
#from feature_extraction import *
import pandas as pd
import numpy as np
import string
import re
import collections
from statistics import mean
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import entropy
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import Normalizer
import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import plotly.express as px
import os.path
from feature_extraction import *


# perform undersampling

def train_rf(train_features, test_features, train_labels, test_labels,feature_names, type=""):
    

    #labels = np.array(table['label'])
    #table = table.drop('label', axis = 1)
    #scale= StandardScaler()
    #scaled_train = scale.fit_transform(train_features) 
    # Split the data into training and testing sets
    #train_features, test_features, train_labels, test_labels = train_test_split(scaled_train, labels, test_size = 0.25, random_state = 42)
    root = os.getcwd()
    filepath = os.path.join(root, 'CODE', 'models', 'rf_v1.joblib')
    if not os.path.exists(filepath):
        clf = RandomForestClassifier(n_estimators=64,max_depth=10)
        clf.fit(train_features, train_labels)
        joblib.dump(clf, filepath)
    clf = joblib.load(filepath)    
    y_pred=clf.predict(test_features)
    evaluate(test_labels, y_pred, type)
    #print(classification_report(test_labels, y_pred))

    return feature_importance(clf,feature_names)

def feature_importance(clf,feature_names):
    

    feats = {}
    for feature, importance in zip(feature_names, clf.feature_importances_):
        feats[feature] = importance
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
    importances = importances.sort_values(by='Gini-Importance', ascending=False)
    importances = importances.reset_index()
    importances = importances.rename(columns={'index': 'Features'})
    sns.set(font_scale = 5)
    sns.set(style="whitegrid", color_codes=True, font_scale = 1.7)
    fig, ax = plt.subplots()
    fig.set_size_inches(30,15)
    sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
    plt.xlabel('Importance', fontsize=25, weight = 'bold')
    plt.ylabel('Features', fontsize=25, weight = 'bold')
    plt.title('Feature Importance', fontsize=25, weight = 'bold')
    plt.savefig(f'{os.getcwd()}/EVALUATIONS/RF_feature_importance.png')
    plt.show()
    
    return [importances['Features'][i] for i in range(10)]
    
def pca_visualization(train_features,labels):
    pca = PCA(n_components=2)
    components = pca.fit_transform(train_features)
    fig = px.scatter(components, x=0, y=1, color=labels)
    # fig.write_image("./EVALUATIONS/RF_PCA_Visualization.png")
    fig.show()
    

def compare_rf(x_train,y_train,x_test,y_test):
    n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
    max_depths = [1,2,3,4,8,9,10,16,32]
    train_results = []
    test_results = []
    i = 0
    for estimator in n_estimators:
        rf = RandomForestClassifier(n_estimators=estimator, max_depth = max_depths[i])
        i += 1
        rf.fit(x_train, y_train)
        train_pred = rf.predict(x_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = rf.predict(x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    line1, = plt.plot(n_estimators, train_results, 'b', label= "Train AUC")
    line2, = plt.plot(n_estimators, test_results, 'r', label= "Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('n_estimators')
    plt.savefig(f'{os.getcwd()}/EVALUATIONS/RF_accuracy_scores.png')
    plt.show()
    
    
