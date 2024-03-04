import pandas as pd
import numpy as np
import nltk
from pathlib import Path 
from svm_model import *
from feature_extraction import *
from Random_Forrest import *
from FFDL_model import *

def main():
    # check if sampled dataset already in repo

    rootdir = os.getcwd()
    correct_dir = False
    for file in os.listdir(rootdir):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            if 'CODE' in d.split('\\')[-1]:
                correct_dir = True
    if not correct_dir:
        print("Error: Please run main.py from general repo directory. Ex. python ./CODE/main.py")
        return
    


    sampled_filepath = Path(f'{os.getcwd()}/DATA/dataset_v1.csv')
    if not os.path.exists(sampled_filepath):
        # check if feature engineering has already been completed and consolidated into table
        filepath = Path(f'{os.getcwd()}/DATA/review_feature_table.csv')
        if not os.path.exists(filepath):
            # download punkt package
            print("shouldn't be here")
            nltk.download('punkt')

            data_path = f'{os.getcwd()}/DATA/YelpCSV'
            cols_meta = ["user_id", "prod_id", "rating", "label", "date"]
            meta_data = pd.read_csv(data_path+"/metadata.csv", names=cols_meta)
            cols_reviewContent = ["user_id", "prod_id", "date", "review"]
            reviewContent = pd.read_csv(
            data_path+"/reviewContent.csv", names=cols_reviewContent)
            table = pd.concat([meta_data, reviewContent["review"]], axis=1).dropna()

            # consolidates all features into one table
            table=pd.concat([table, review_metadata(table), review_textual(table), reviewer_burst(table), behavioral_features(table), rating_features(table), temporal(table)], axis=1)  

            # writes dataframe containing all features to csv file
            os.makedirs(filepath, exist_ok=True)  
            table.to_csv(filepath, index=False)
    
        table = pd.read_csv(filepath)

        os.makedirs(sampled_filepath, exist_ok=True)
        #undersample
        undersampled_table = undersample(table)
        undersampled_table.to_csv(sampled_filepath, index=False)

    sample = pd.read_csv(sampled_filepath)
    features, labels, feature_names = pre_process(sample)
    features_df = pre_process_df(sample)
    # remove
     

    # split data set into 80% training data and 20% test data
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,random_state=573)
    
    # random forest and feature selection
    
    # will return accuracy score plot for different parameters
    compare_rf(train_features,train_labels,test_features,test_labels)
    # # will return names of top 10 features and return accuracy of the model
    rf_top_features = train_rf(train_features, test_features, train_labels, test_labels,feature_names, type="rf")
    print(rf_top_features)
    pca_visualization(train_features,train_labels)

    # SVM and feature selection
    # will return names of top 10 features
    svm_top_features, top2_svm = svm_feature_selection(train_features, train_labels, test_features, test_labels, feature_names, type="lin_svm")
    print(svm_top_features)
    

    # deep learning approach with selected features
    # rf_top_features = ['activity_time', 'rating_entropy', 'date_var', 'RL', 'rating_variance', 'reviewer_dev', 'MRD', 'date_entropy', 'MNR', 'DFTLM']
    # svm_top_features = ['MRD', 'rating_variance', 'density', 'MNR', 'date_var', 'activity_time', 'rating_entropy', 'date_entropy', 'singleton', 'RL']
    rf_features = features_df[rf_top_features]
    svm_features = features_df[svm_top_features]
    feature_id = range(len(svm_top_features))
    union_features = list(set(rf_top_features).union(set(svm_top_features)))
    intersect_features = list(set(rf_top_features).intersection(set(svm_top_features)))
    top2_rf = rf_top_features[:2]
    # top2_svm = svm_top_features[:2]
    print("trainin DL model with all features")
    run_DL(features_df, labels, feature_id, top2_rf , n_epoches = 100, batch_size = 256, type="DL_all")

    print("trainin DL model with selected features by RF features")
    run_DL(rf_features, labels, feature_id, top2_rf , n_epoches = 100, batch_size = 256, type="DL_rf")

    print("trainin DL model with selected features by SVM features")
    run_DL(svm_features, labels, feature_id, top2_svm , n_epoches = 100, batch_size = 256, type="DL_svm")

    print("trainin DL model with union of features")
    feature_id = range(len(union_features))
    run_DL(svm_features, labels, feature_id, top2_svm , n_epoches = 100, batch_size = 256, type="union")

    print("trainin DL model with intersection of features")
    feature_id = range(len(intersect_features))
    run_DL(svm_features, labels, feature_id, top2_svm , n_epoches = 100, batch_size = 256, type="intersect")


if __name__ == "__main__":
    main()
