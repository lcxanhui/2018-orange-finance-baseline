import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import datetime
import os
import gc
import warnings
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

#读数据
path='/Users/pc/Desktop/orange'
op_train = pd.read_csv(path + '/operation_train_new.csv')
trans_train = pd.read_csv(path + '/transaction_train_new.csv')

op_test = pd.read_csv(path + '/operation_round1_new.csv')
trans_test = pd.read_csv(path + '/transaction_round1_new.csv')
y = pd.read_csv(path + '/tag_train_new.csv')
sub = pd.read_csv(path + '/sub.csv')
print('load data ok...')

#训练集和测试集特征处理
def get_feature(op,trans,label):
    for feature in op.columns[2:]:
        label = label.merge(op.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
        label =label.merge(op.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
    
    for feature in trans.columns[2:]:
        if trans_train[feature].dtype == 'object':
            label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
        else:
            label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].max().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].min().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].sum().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].mean().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].median().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].std().reset_index(),on='UID',how='left')
    return label



train = get_feature(op_train,trans_train,y).fillna(-1)
test = get_feature(op_test,trans_test,sub).fillna(-1)

print('merge ok...')
train = train.drop(['UID','Tag'],axis = 1).fillna(-1)
label = y['Tag']

test_id = test['UID']
test = test.drop(['UID','Tag'],axis = 1).fillna(-1)

#官方的评价指标
def tpr_weight_funtion(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 'TC_AUC',0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3,True

#lgb模型
lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=64, reg_alpha=0, reg_lambda=0, max_depth=-1,
    n_estimators=1000, objective='binary', subsample=0.9, colsample_bytree=0.8, subsample_freq=1, learning_rate=0.05,
    random_state=1024, n_jobs=10, min_child_weight=4, min_child_samples=5, min_split_gain=0, silent=True)

print('Fiting...')
dev_X, val_X, dev_y, val_y = train_test_split(train, label, test_size = 0.2, random_state = 2018)
lgb_model.fit(dev_X, dev_y,
                  eval_set=[(dev_X, dev_y),
                            (val_X, val_y)], early_stopping_rounds=100,verbose=50)
baseloss = lgb_model.best_score_['valid_1']['binary_logloss']
#特征重要性
se = pd.Series(lgb_model.feature_importances_)
col =list(se.sort_values(ascending=False).index)
pd.Series(col).to_csv(path + '/col_sort_one.csv',index=False)
n = lgb_model.best_iteration_
print('best_iteration:',n)


lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=64, reg_alpha=0, reg_lambda=0, max_depth=-1,
    n_estimators=1000, objective='binary', subsample=0.9, colsample_bytree=0.8, subsample_freq=1, learning_rate=0.05,
    random_state=1024, n_jobs=10, min_child_weight=4, min_child_samples=5, min_split_gain=0, silent=True)


skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
best_score = []

oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test_id.shape[0])


for index, (train_index, test_index) in enumerate(skf.split(train, label)):
    lgb_model.fit(train.iloc[train_index], label.iloc[train_index], verbose=50,
                  eval_set=[(train.iloc[train_index], label.iloc[train_index]),
                            (train.iloc[test_index], label.iloc[test_index])], early_stopping_rounds=100)
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(best_score)
    oof_preds[test_index] = lgb_model.predict_proba(train.iloc[test_index], num_iteration=lgb_model.best_iteration_)[:,1]

    test_pred = lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
    sub_preds += test_pred / 5

score = tpr_weight_funtion(y_predict=oof_preds,y_true=label)
print('score:',score[1])
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
sub = pd.read_csv(path + '/sub.csv')
sub['Tag'] = sub_preds
sub.to_csv(path + '/lgb_baseline_%s.csv'% now,index=False)
