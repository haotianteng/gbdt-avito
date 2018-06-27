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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

CAT_LABELS = ['region','city','image_top_1','parent_category_name','category_name','param_1','param_2','param_3','user_type']
VAL_LABELS = ['price']

print('load data.')
train_f = '/home/heavens/cqt-avito/data/train.csv'
test_f = '/home/heavens/cqt-avito/data/test.csv'
pred_file = '/home/heavens/cqt-avito/data/test_summary'

train_df = pd.read_csv(train_f,sep = ',',header = 0,index_col = "item_id", parse_dates = ["activation_date"])
test_df = pd.read_csv(test_f,sep = ',',header = 0,index_col = "item_id", parse_dates = ["activation_date"])
ntrain = train_df.shape[0]
traindex = train_df.index
ntest = test_df.shape[0]
testindex = test_df.index
y = train_df['deal_probability']
train_df.drop('deal_probability',axis = 1,inplace = True)
df = pd.concat([train_df,test_df],axis=0)
del train_df,test_df
df = data(df)
x = df.loc[traindex,:]
x_test = df.loc[testindex,:]
x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.10, random_state=2018)
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
                        categorical_feature = CAT_LABELS)
lgb_valid = lgb.Dataset(x_valid, label = y_valid,categorical_feature = CAT_LABELS)

print('begin training')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20000,
                valid_sets=[lgb_train,lgb_valid],
                feature_name = x_train.columns.tolist(),
                early_stopping_rounds = 50,
                verbose_eval=10)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(x_test)
print('RMSE:', np.sqrt(mean_squared_error(y_valid, gbm.predict(x_valid))))
with open(pred_file,'w+') as f:
    f.write('item_id,deal_probability')
    f.write('\n')
    for idx,item in enumerate(y_pred):
        if y_pred[idx]<0:
            y_pred[idx] = 0
        if y_pred[idx]>1:
            y_pred[idx] =1
        f.write(x_test.index[idx])
        f.write(',')
        f.write(str(y_pred[idx]))
        f.write('\n')
