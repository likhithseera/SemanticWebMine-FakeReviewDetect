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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def undersample_v2(table):
    labels = table["label"]
    features = table.drop(['user_id', 'prod_id', "label", "date", "review"], axis=1)
    feature_names = list(features.columns)
    scaler = Normalizer().fit(features)
    normalized_features = scaler.transform(features)
    undersample = NearMiss(version=3, n_neighbors_ver3=3)
    features, labels = undersample.fit_resample(normalized_features, labels)
    labels = np.array([1 if label == -1 else 0 for label in labels ])
    return features, labels, feature_names


def undersample(table):
    """
    Performs undersampling on data by keeping fake reviews and obtaining random sample
    of real reviews such that # of each class (real and fake) are equal
    """
    fake_reviews = table[table['label'] == -1]
    real_reviews = table[table['label'] == 1].sample(n=fake_reviews.shape[0], random_state=573)
    sample = pd.concat([fake_reviews, real_reviews], ignore_index=True)
    return sample

def pre_process(table):
    labels = table["label"]
    features = table.drop(['user_id', 'prod_id', "label", "date", "review"], axis=1)
    # normalize
    for column in features.columns:
        features[column] = features[column]  / features[column].abs().max()
    feature_names = list(features.columns)
    features = features.to_numpy()
    labels = np.array([1 if label == -1 else 0 for label in labels ])
    return features, labels, feature_names
# return dataframe for DL
def pre_process_df(table):
    features = table.drop(['user_id', 'prod_id', "label", "date", "review"], axis=1)
    for column in features.columns:
        features[column] = features[column]  / features[column].abs().max()
    return features

def evaluate(test_labels,y_pred, type):
    tn, fp, fn, tp = confusion_matrix(test_labels, y_pred).ravel()
    acc = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred)
    recall = recall_score(test_labels, y_pred)
    specificity = tn/(tn+fp)
    metrics_val = [acc, precision, recall, specificity]
    metrics = pd.DataFrame()
    metrics["metric_type"] = ['Accuracy', 'Precision', 'Recall', 'Specificity']
    metrics["value"] = metrics_val
    metrics.to_csv(f"./EVALUATIONS/Metrics_{type}.csv", header=False, index=False)

def review_metadata(table):
    """
    Metadata features: 
    Rating: rating(1-5) given in review (no calculation needed)
    Singleton: 1 if review is only one written by user on date, 0 otherwise
    """
    # singleton
    date_counts = table.groupby(['user_id', 'date']).size().to_frame('size')
    table = pd.merge(table, date_counts, on=['user_id', 'date'], how='left')
    table['singleton'] = table['size'] == 1
    table['singleton'] = table['singleton'].astype('int')

    return table[['singleton']]


def review_textual(table):
    """
    Text statistics: 
    Number of words, i.e., the length of the review in terms of words;
    Ratio of capital letters, i.e., the number of words containing capital letters with respect to the total number of words in the review;
    Ratio of capital words, i.e., considering the words where all the letters are uppercase;
    Ratio of first person pronouns,e.g.,‘I’,‘mine’,‘my’, etc.;
    Ratio of ‘exclamation’ sentences, i.e., ending with the symbol ‘!’. 
    """
    statistics_table = {"RationOfCapL": [], "RatioOfCapW": [
    ], "RatioOfFirstPerson": [], "RatioOfExclamation": []}  # , "sentiment":[]
    first_person_pronouns = set(
        ["i", "mine", "my", "me", "we", "our", "us", "ourselves", "ours"])

    for i, row in table.iterrows():
        sentences = sent_tokenize(row["review"])
        countExclamation = 0
        wordCountinAReview = 0
        countCapL = 0
        countCapW = 0
        countFirstP = 0
        for sentence in sentences:
            if sentence[-1] == "!":
                countExclamation += 1
            sentence = sentence.translate(
                str.maketrans('', '', string.punctuation))
            sentence = sentence.split(" ")

            wordCountinAReview += len(sentence)

            for word in sentence:
                if word.isupper():
                    countCapW += 1
                if word.lower() in first_person_pronouns:
                    countFirstP += 1
                for w in word:
                    if w.isupper():
                        countCapL += 1
                        break

        RatioOfExclamation = countExclamation/len(sentences)
        RationOfCapL = countCapL/wordCountinAReview
        RatioOfCapW = countCapW/wordCountinAReview
        RatioOfFirstPerson = countFirstP/wordCountinAReview
        statistics_table["RatioOfExclamation"].append(RatioOfExclamation)
        statistics_table["RationOfCapL"].append(RationOfCapL)
        statistics_table["RatioOfCapW"].append(RatioOfCapW)
        statistics_table["RatioOfFirstPerson"].append(RatioOfFirstPerson)

    text_statistics = pd.DataFrame.from_dict(statistics_table)
    return text_statistics


def reviewer_burst(table):
    """
    Burst features: 
    Density: # reviews for entity on given day
    Mean Rating Deviation(MRD): |avg_prod_rating_on_date - avg_prod_rating|
    Deviation From Local Mean(DFTLM):  |prod_rating - avg_prod_rating_on_date|
    """
    # Density
    df1 = table.groupby(['prod_id', 'date'], as_index=False)[
        'review'].agg('count')
    df1.rename(columns={'review': 'density'}, inplace=True)
    table = pd.merge(table, df1, left_on=['prod_id', 'date'], right_on=[
                     'prod_id', 'date'], validate='m:1')

    # Mean Rating Deviation
    df4 = table.groupby(['prod_id', 'date'], as_index=False).agg(avg_date=pd.NamedAgg(column='rating', aggfunc='mean'))
    table = pd.merge(table, df4, left_on=['prod_id', 'date'], right_on=[
                     'prod_id', 'date'], validate='m:1')

    # Deviation From The Local Mean
    df3 = table.groupby(['prod_id'], as_index=False).agg(
        avg=pd.NamedAgg(column='rating', aggfunc=np.mean))
    table = pd.merge(table, df3, left_on=['prod_id'], right_on=[
                     'prod_id'], validate='m:1')
    table['DFTLM'] = abs(table['rating'] - table['avg_date'])
    table['MRD'] = abs(table['avg_date'] - table['avg'])

    return table[['density', 'MRD', 'DFTLM']]

def behavioral_features(table):
    """
    General behavioral features: 
    Maximum Number of Reviews (MNR): max number of reviews written by user on any given day
    Percentage of Positive Reviews (PPR): % of positive reviews(4-5 stars) / total reviews by user
    Percentage of Negative Reviews (PNR): % of positive reviews(1-2 stars) / total reviews by user
    Review Length (RL): Avg length of reviews (in words) written by user
    Rating Deviation: Deviation of review from other reviews on same business (rating - avg_prod_rating)
    Reviewer Deviation: Avg of rating deviation across all user's reviews
    """
    # MNR calculation
    count_table = table[['user_id', 'date', 'rating']].groupby(['user_id', 'date']).agg(count=pd.NamedAgg(column='rating', aggfunc='count'))
    res = count_table.groupby(['user_id']).agg(MNR=pd.NamedAgg(column='count', aggfunc='max'))
    table = pd.merge(table, res, on='user_id', how='left')

    # PPR calculation
    totals = table[['user_id', 'rating']].groupby(['user_id']).agg(total=pd.NamedAgg(column='rating', aggfunc='count'))
    pos = table[table['rating'] >= 4].groupby(['user_id']).agg(pos=pd.NamedAgg(column='rating', aggfunc='count'))
    neg = table[table['rating'] <= 2].groupby(['user_id']).agg(neg=pd.NamedAgg(column='rating', aggfunc='count'))
    table = pd.merge(table, pd.merge(totals,pos,on='user_id', how='left').fillna(0), on="user_id", how='left')
    table = pd.merge(table, neg, on="user_id", how='left').fillna(0)
    table['PPR'] = table['pos'] / table['total']
    table['PNR'] = table['neg'] / table['total']

    # RL calculation
    len_table = table[['user_id', 'review']]
    len_table['length'] = len_table['review'].str.split(" ").str.len()
    temp = len_table[['user_id', 'length']].groupby(['user_id']).agg(RL=pd.NamedAgg(column='length', aggfunc='mean'))
    table = pd.merge(table, temp, on="user_id", how='left')

    # Rating Deviation calculation
    avg_rating = table[['prod_id', 'rating']].groupby(['prod_id']).agg(avg=pd.NamedAgg(column='rating', aggfunc='mean'))
    table = pd.merge(table, avg_rating, on='prod_id', how='inner')
    table['rating_dev'] = abs(table['rating'] - table['avg'])

    # Reviewer Deviation calculation
    temp = table[['user_id', 'rating_dev']].groupby(['user_id']).agg(reviewer_dev=pd.NamedAgg(column='rating_dev', aggfunc='mean'))
    table = pd.merge(table, temp, on='user_id', how='left')

    return table[['MNR', 'PPR','PNR', 'RL', 'rating_dev', 'reviewer_dev']]

def rating_features(table):
    """
    Rating features:
    """
    rating_features = table[["user_id", "rating"]]
    """
    Average deviation from entity's average, 
    i.e., the evaluation if a user's ratings assigned in her/his reviews are 
    often very different from the mean of an entity's rating(far lower for instance);
    """
    avg_rating_of_prods = table[["prod_id", "rating"]].groupby('prod_id').mean()
    # simply using the rating minus the avegerage in the original table
    """
    Rating entropy , 
    i.e., the entropy of rating distribution of user's reviews;
    """
    grouped_users = rating_features.groupby('user_id')
    user_rating_entropy = collections.defaultdict(int)

    for name, group in grouped_users:
        rating_peruser = list(group["rating"])
        user_rating_entropy[name] = entropy(rating_peruser)
    # search for user id and add the entropy value to the original table
    """
    Rating variance , i.e., the squared deviation of the rating assigned by a user with respect to the ratings mean. 
    The variance as a rating feature has been added to further describe how the ratings for a particular user are distributed.
    """
    avg_rating_of_users = rating_features.groupby('user_id').mean()
    # simply subtract the rating for each row in the original table
    rating_features_output = collections.defaultdict(list)
    for index, row in table.iterrows():
        # ratio calculation 
        user_id = row["user_id"]

        # rating variance 
        rating_features_output["rating_variance"].append((row["rating"] - avg_rating_of_users.loc[user_id]["rating"])**2)

        # rating_entropy
        rating_features_output["rating_entropy"].append(user_rating_entropy[user_id])

        # Average deviation from entity's average
        rating_features_output["avg_dev_from_entity_avg"].append(avg_rating_of_prods.loc[row["prod_id"]]["rating"])
    rating_features_df = pd.DataFrame.from_dict(rating_features_output)
    return rating_features_df

def temporal(table):
    """
    Temporal features:
    Activity time: Number of days between first and last review of user.
    Maximum rating per day: Maximum rating provided by user in considered day.
    Date entropy: Number of days between current review and next review of user.
    Date variance: |date_of_review - avg_review_date_of_user|^2

    """
   ## activity time
    table['date'] = pd.to_datetime(table['date'])
    temp = table.loc[:, ['user_id', 'date', 'rating']]
    temp.sort_values(by=['date'], inplace=True)
    act_time_table = temp[['user_id', 'date']].groupby(['user_id']).agg(first=pd.NamedAgg(column='date', aggfunc='min'),
                                            last = pd.NamedAgg(column='date', aggfunc='max'))
    act_time_table['activity_time'] = ((act_time_table['last'] - act_time_table['first']) / np.timedelta64(1, 'D')).astype(int)
    
    ## maxium rating per day
    temp2 = table
    temp2['date'] = pd.to_datetime(temp2['date'])
    temp2 = temp2[['user_id', 'date', 'rating']].groupby(['user_id', 'date']).agg(MRPD=pd.NamedAgg(column='rating', aggfunc='max'))

    ## date entropy
    temp['prev_date'] = temp.groupby('user_id')['date'].shift()
    temp['date_entropy'] = temp['date'] - temp['prev_date']
    temp.replace({pd.NaT: '0 day'}, inplace=True)
    temp['date_entropy'] = (temp['date_entropy'] / np.timedelta64(1, 'D')).astype(int)

    ## date var
    temp['original_index'] = temp.index
    temp3 = temp[['user_id', 'date']].groupby(['user_id']).agg(date_mean=pd.NamedAgg(column='date', aggfunc='mean'))
    temp = pd.merge(temp, temp3, on='user_id', how='left')
    temp['date_var'] = abs(((temp['date'] - temp['date_mean']) / np.timedelta64(1, 'D')))**2
    temp.set_index('original_index')

    ## join with original table
    table = table.loc[:, ['user_id', 'date']]
    table['date'] = pd.to_datetime(table['date'])
    table = pd.merge(table, act_time_table, on='user_id', how='left')
    table = pd.merge(table, temp2, on=['user_id', 'date'], how='left')
    table = pd.merge(table, temp, left_on=table.index, right_on='original_index')
    return table[['activity_time', 'MRPD', 'date_entropy', 'date_var']]

   

    
