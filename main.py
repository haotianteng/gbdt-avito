# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn import preprocessing

print('load data.')
df_train = pd.read_csv("/home/heavens/cqt-avito/data/train.csv",sep = ',')
df_test = pd.read_csv("/home/heavens/cqt-avito/data/test.csv",sep = ',')

print('preprocess data')
#Preprocess data
df_train = df_train.fillna('NaN')
df_test = df_test.fillna('NaN')
cat_labels = ['region','city','parent_category_name','category_name','param_1','param_2','param_3','user_type']
val_labels = ['price','image_top_1']
drop_labels = ['deal_probability','item_id','user_id','title','description','activation_date','image','item_seq_number']
y_train = df_train['deal_probability'].values
x_train = df_train.drop(labels=drop_labels,axis = 1)
x_test = df_test.drop(labels = drop_labels[1:],axis = 1)

le = preprocessing.LabelEncoder()
nan_x_train = x_train == 'NaN'
nan_x_test = x_test == 'NaN'
#Label encoding the categorical data
def retype(dataset,cat_labels,val_labels):
    for i in range(0,dataset.shape[1]):
        if dataset.dtypes[i]=='object':
            if dataset.columns[i] in cat_labels:
                print(dataset.columns[i])
                dataset[dataset.columns[i]] = le.fit_transform(dataset[dataset.columns[i]])
            if dataset.columns[i] in val_labels:
                dataset[dataset.columns[i]] = dataset[dataset.columns[i]].astype('float')
    return dataset

x_train = retype(x_train,cat_labels = cat_labels,val_labels = val_labels)
x_test = retype(x_test,cat_labels = cat_labels,val_labels = val_labels)
x_test[nan_x_test] = np.nan
x_train[nan_x_train] = np.nan
#x_train = x_train.values

print('Construct lgb dataset')
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.3,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
}
lgb_train = lgb.Dataset(x_train, 
                        label = y_train,
                       free_raw_data=False)

print('begin training')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=lgb_train,
                feature_name = cat_labels+val_labels,
                categorical_feature = cat_labels,
                early_stopping_rounds=10)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)

