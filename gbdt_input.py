#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 21:48:44 2018

@author: heavens
"""
import pandas as pd
import numpy as np
import string
import re
import time
from sklearn import preprocessing

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 


CAT_LABELS = ['user_id','image_top_1','region','city','parent_category_name','category_name','param_1','param_2','param_3','user_type']
VAL_LABELS = ['price']
DROP_LABELS = ['title','description','activation_date','image','item_seq_number']
RANDOM_SEED = 2018
def prepare_train_dataset(df):
    df = preprocess(df)
    x = df.drop(labels = DROP_LABELS, axis = 1)
    x = retyping(x,cat_labels = CAT_LABELS, val_labels = VAL_LABELS)
    return(x)
def cleanName(text):
    try:
        textProc = text.lower()
        # textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
        #regex = re.compile(u'[^[:alpha:]]')
        #textProc = regex.sub(" ", textProc)
        textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except: 
        return "name error"
    
def preprocess(df):
    ###Preprocess data feature(according to Samrat P "avitolightGBM with Ridge feature v3.0" kernal)
    df["price"] = np.log(df["price"]+0.001)
    df["price"].fillna(df.price.mean(),inplace=True)
    df["image_top_1"].fillna(-999,inplace=True)
    print("\nCreate Time Variables")
    df["Weekday"] = df['activation_date'].dt.weekday
    # Meta Text Features
    print("Preprocess text data")
    textfeats = ["description", "title"]
    df['desc_punc'] = df['description'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    
    df['title'] = df['title'].apply(lambda x: cleanName(x))
    df["description"]   = df["description"].apply(lambda x: cleanName(x))
    
    for cols in textfeats:
        df[cols] = df[cols].astype(str) 
        df[cols] = df[cols].astype(str).fillna('missing') # FILL NA
        df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
        df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
        df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
        df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words
        df[cols + '_num_letters'] = df[cols].apply(lambda comment: len(comment)) # Count number of Letters
        df[cols + '_num_alphabets'] = df[cols].apply(lambda comment: (comment.count(r'[a-zA-Z]'))) # Count number of Alphabets
        df[cols + '_num_alphanumeric'] = df[cols].apply(lambda comment: (comment.count(r'[A-Za-z0-9]'))) # Count number of AlphaNumeric
        df[cols + '_num_digits'] = df[cols].apply(lambda comment: (comment.count('[0-9]'))) # Count number of Digits
        
    # Extra Feature Engineering
    df['title_desc_len_ratio'] = df['title_num_letters']/df['description_num_letters']
    
    return df

def balance_datasets(x,y,amp = None, shuffle = True):
    """Rebalacne the dataset by over/sub sample the 0 data"""
    x['deal_probability'] = y
    x_0 = x[x['deal_probability']==0]
    x_1 = x[x['deal_probability']!=0]
    if shuffle:
        x_0 = x_0.sample(frac = 1)
        x_1 = x_1.sample(frac = 1)
    if amp is None:
        hist_y = np.histogram(y)[0]
        amp = int(x_0.size/np.mean(hist_y[1:]))
    sub_size = int(x_0.size / amp)+1
    for i in range(0,x_0.size,sub_size):
        sub_x = pd.concat([x_0[i:i+sub_size],x_1])
        sub_x = sub_x.sample(frac = 1)
        yield (sub_x.drop(labels = 'deal_probability',axis = 1),sub_x['deal_probability'])
        
def retyping(dataset,cat_labels,val_labels):
    le = preprocessing.LabelEncoder()
    for i in range(0,dataset.shape[1]):
        col = dataset.columns[i]
        if dataset.dtypes[i]=='object':
            if col in cat_labels:
                dataset[col].fillna('Unknown')
                print(col)
                dataset[col] = le.fit_transform(dataset[col].astype(str))
            if col in val_labels:
                dataset[col] = dataset[col].astype('float')
    return dataset

if __name__ == "__main__":
    train_f = "/home/havens_teng/avito/avito-demand-prediction/train.csv"
    test_f = "/home/havens_teng/avito/avito-demand-prediction/test.csv"
    x_train,y_train,x_valid,y_valid = prepare_train_dataset(train_f,valid_size = 1000)
    x_test = prepare_train_dataset(test_f, for_eval = True)
