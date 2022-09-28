import pandas as pd
import numpy as np


from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, auc,roc_auc_score
import scorecardpy as sc
import joblib
dat = sc.germancredit()
dt_s = sc.var_filter(dat, y="creditability")

print(dt_s['creditability'].value_counts)
print(dt_s['creditability'].value_counts)

unum_var=['other.installment.plans','present.employment.since','housing',\
          'status.of.existing.checking.account','property','savings.account.and.bonds',\
          'purpose','other.debtors.or.guarantors','credit.history']
num_var=['age.in.years','credit.amount','duration.in.month','installment.rate.in.percentage.of.disposable.income']
N=0
for i in unum_var:
    vars=list(set(dt_s[i]))
    for varss in vars:
        N+=1
        dt_s["is_"+str(N)]=dt_s[i].apply(lambda x:1 if x==varss else 0)
    del dt_s[i]


dt_s=dt_s.rename(columns={'creditability':'label'})
import xgboost as xgb


def train_model(df_train,learning_rate,n_estimators,max_depth,min_child_weight,alpha,modelname):
    df_train=df_train.fillna(-99)
    y=df_train['label']
    varuse=[i for i in list(df_train.columns) if i not in ['phone_MD5','date_credit','label','EXCLUSION']]
    data_matrix=xgb.DMatrix(df_train[varuse],y,missing=-99)
    params={
        "learning_rate":learning_rate,
        "objective":'binary:logistic',
        "n_estimators":n_estimators,
        "max_depth":max_depth,
        "min_child_weight":min_child_weight,
        "metric":['auc'],
        "subsample":0.8,
        "colsample_butree":0.8,
        "early_stopping_rounds":5,
        "alpha":alpha,
        "silent":0,
        "random_state":12 #random_state
    }
    xgb_risk_model=xgb.train(dtrain=data_matrix,params=params,verbose_eval=15)
    data_matrix=xgb.DMatrix(df_train[list(xgb_risk_model.get_fscore().keys())],y)
    xgb_risk_model = xgb.train(dtrain=data_matrix, params=params, verbose_eval=15, num_boost_round=n_estimators)
    y_train_pred=xgb_risk_model.predict(xgb.DMatrix(df_train[xgb_risk_model.feature_names]))
    fpr,tpr,thresholds=roc_curve(y,y_train_pred)
    ks=round(max(tpr-fpr),4)
    auc_score=round(roc_auc_score(y,y_train_pred),4)
    print(ks,auc_score)
    joblib.dump(xgb_risk_model,modelname)
    return xgb_risk_model

xgbmodel=train_model(dt_s,0.01,50,3,2,2,'xgbddmode.pmml')


