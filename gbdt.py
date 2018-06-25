# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
import csv
from gbdt_input import prepare_train_dataset as data
from gbdt_input import balance_datasets

CAT_LABELS = ['region','city','parent_category_name','category_name','param_1','param_2','param_3','user_type']
VAL_LABELS = ['price','image_top_1']

print('load data.')
train_f = '/home/heavens/cqt-avito/data/train.csv'
test_f = '/home/heavens/cqt-avito/data/test.csv'
pred_file = '/home/heavens/cqt-avito/data/test_summary'

x_train,y_train,x_valid,y_valid = data(train_f,valid_size = 1000)
x_test = data(test_f,for_eval = True)
datasets_iter = balance_datasets(x_train,y_train)

print('Train data size:' + str(x_train.size))
print('Valid data size:' + str(x_valid.size))

print('Construct lgb dataset')
params = {
    'boosting_type': 'gbdt',
    'objective':'regression',
    'metric': {'rmse'},
    'num_leaves': 270,
    'lambda_l1': 0.01,
    'learning_rate': 0.0175,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.75,
    'bagging_freq': 2,
    'verbose': 0,
}
#for x_sub in datasets_iter:
#    x_train,y_train = x_sub
#    break
lgb_train = lgb.Dataset(x_train, 
                        label = y_train,
                       free_raw_data=False)
lgb_valid = lgb.Dataset(x_valid, label = y_valid, free_raw_data = False)

print('begin training')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20000,
                valid_sets=lgb_valid,
                feature_name = x_train.columns.tolist(),
                categorical_feature = CAT_LABELS,
                early_stopping_rounds = 50)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)

with open(pred_file,'w+') as f:
    f.write('item_id,deal_probability')
    f.write('\n')
    for idx,item in enumerate(y_pred):
        if y_pred[idx]<0:
            y_pred[idx] = 0
        if y_pred[idx]>1:
            y_pred[idx] =1
        f.write(str(x_test['item_id'][idx]))
        f.write(',')
        f.write(str(y_pred[idx]))
        f.write('\n')
