import sklearn
import numpy as np
import os.path
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from feature_extraction import *



def svm_feature_selection(train_features, train_labels, test_features, test_labels, feature_names, type=""):
    """
    Trains svm model with inputted features. Visualizes and returns top n coeff feature names
    """
    # check if trained model already exists
    root = os.getcwd()
    filepath = os.path.join(root, 'CODE', 'models', 'lin_svm_v2.joblib')
    if not os.path.exists(filepath):
        train_lin_model(train_features, train_labels, filepath)

    # uncomment this line to predict classification for test set
    # y_pred = clf.predict(X_test)
    clf = load(filepath)

    test_pred = clf.predict(test_features)
    evaluate(test_labels, test_pred, type)

    return feature_selection(clf, feature_names)


def train_lin_model(features, labels, filepath):
    clf = svm.SVC(kernel='linear')
    clf.fit(features, labels)

    # save svm model
    dump(clf, filepath)

def feature_selection(classifier, feature_names, n=10):
    coef = classifier.coef_.ravel()

    print(coef)
    
    #get top 5 positive and negative coefficients
    top_positive = np.argsort(coef)[-int(n/2):]
    top_negative = np.argsort(coef)[:int(n/2)]
    top_coefficients = np.hstack([top_negative, top_positive])

    print(top_coefficients)

    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(n), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0, n), feature_names[top_coefficients], rotation=60, ha='right')
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(f'{os.getcwd()}/EVALUATIONS/SVM_feature_importance.png')
    plt.show()


    

    return [feature_names[i] for i in top_coefficients], [feature_names[top_negative[0]], feature_names[top_positive[-1]]]


