import pandas as pd
import numpy as np

import lightgbm as lgb


df_train=pd.read_json('/Users/weiliangli/Downloads/phase1_train.json')
df_test=pd.read_json('/Users/weiliangli/Downloads/phase1_test.json')
sheet_name=['运政业户信息', '运政车辆信息', '车辆违法违规信息（道路交通安全，来源抄告）',
 '车辆违法违规信息（交通运输违法，来源抄告）', '动态监控报警信息（车辆，超速行驶）',
 '动态监控报警信息（车辆，疲劳驾驶）', '动态监控上线率（企业，%）',
 '运政车辆年审记录信息', '运政质量信誉考核记录']

df_ori1=pd.read_excel('/Users/weiliangli/Downloads/data.xlsx',sheet_name='运政业户信息')
df_ori2=pd.read_excel('/Users/weiliangli/Downloads/data.xlsx',sheet_name='运政车辆信息')
df_ori3=pd.read_excel('/Users/weiliangli/Downloads/data.xlsx',sheet_name='车辆违法违规信息（道路交通安全，来源抄告）')
df_ori4=pd.read_excel('/Users/weiliangli/Downloads/data.xlsx',sheet_name='车辆违法违规信息（交通运输违法，来源抄告）')
df_ori5=pd.read_excel('/Users/weiliangli/Downloads/data.xlsx',sheet_name='动态监控报警信息（车辆，超速行驶）')
df_ori6=pd.read_excel('/Users/weiliangli/Downloads/data.xlsx',sheet_name='动态监控报警信息（车辆，疲劳驾驶）')
df_ori7=pd.read_excel('/Users/weiliangli/Downloads/data.xlsx',sheet_name='动态监控上线率（企业，%）')
df_ori8=pd.read_excel('/Users/weiliangli/Downloads/data.xlsx',sheet_name='运政车辆年审记录信息')
df_ori9=pd.read_excel('/Users/weiliangli/Downloads/data.xlsx',sheet_name='运政质量信誉考核记录')
del df_ori5['Unnamed: 0'],df_ori5['Unnamed: 1']
df_ori2=df_ori2.rename(columns={'车辆牌照号':'car_id'})
df_ori3=df_ori3.rename(columns={'车牌号':'car_id'})
df_ori4=df_ori4.rename(columns={'车牌号':'car_id'})
df_ori5=df_ori5.rename(columns={'车牌号码':'car_id'})
df_ori6=df_ori6.rename(columns={'车牌号码':'car_id'})
df_ori8=df_ori8.rename(columns={'车辆牌照号':'car_id'})

def get_onehot(df,var,list):
  for value in list:
      df['is_'+str(value)]=df[var].apply(lambda x:1 if x==value else 0)
  del df[var]
  return df
df_ori2=get_onehot(df_ori2,'行业类别',list(set(df_ori2['行业类别'])))
df_ori2=get_onehot(df_ori2,'车牌颜色',list(set(df_ori2['车牌颜色'])))
df_ori3=df_ori3['car_id'].value_counts().reset_index()
df_ori3.columns=['car_id','sum']
df_ori3['cnt']=1

df_ori4=df_ori4['car_id'].value_counts().reset_index()
df_ori4.columns=['car_id','sum2']
df_ori4['cnt2']=1

df_ori51=df_ori5['car_id'].value_counts().reset_index()
df_ori51.columns=['car_id','sum51']
df_ori51['cnt51']=1

df_ori52=df_ori5['最高时速(Km/h)'].groupby(df_ori5['car_id']).agg({'sum','max','min','mean','median','std','count'}).reset_index()
df_ori53=df_ori5['持续点数'].groupby(df_ori5['car_id']).agg({'sum','max','min','mean','median','std','count'}).reset_index()
df_ori54=df_ori5['持续时长(秒)'].groupby(df_ori5['car_id']).agg({'sum','max','min','mean','median','std','count'}).reset_index()
df_ori52.columns=["car_id",'sum52','max52','min52','mean52','median52','std52','count52']
df_ori53.columns=["car_id",'sum53','max53','min53','mean53','median53','std53','count53']
df_ori54.columns=["car_id",'sum54','max54','min54','mean54','median54','std54','count54']

import  datetime
from datetime import timedelta

df_ori6['结束时间']=df_ori6['结束时间'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
df_ori6['开始时间']=df_ori6['开始时间'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
df_ori6['diff']=df_ori6['结束时间']-df_ori6['开始时间']
df_ori6['diff']=df_ori6['diff'].apply(lambda x:timedelta.total_seconds(x))
df_ori6=df_ori6['diff'].groupby(df_ori6['car_id']).agg({'sum','max','min','mean','median','std','count'}).reset_index()
df_ori6.columns=["car_id",'sum6','max6','min6','mean6','median6','std6','count6']
df_ori6['cnt6']=1

df_ori8['审批结果']=df_ori8['审批结果'].apply(lambda x:1 if x=='年审不合格' else 0)
df_ori8=df_ori8['审批结果'].groupby(df_ori6['car_id']).agg({'sum','max','min','mean','median','std','count'}).reset_index()
df_ori8.columns=["car_id",'sum8','max8','min8','mean8','median8','std8','count8']
df_ori8['cnt8']=1

df_ori9['质量信誉考核结果']=df_ori9['质量信誉考核结果'].apply(lambda x:4 if x.find('AAA')>=0 else
                                                                 3 if x.find('AA')>=0 else
                                                                 2 if x.find('A')>=0 else
                                                                1)

df_train=pd.merge(df_train,df_ori2,how='left')
df_train=pd.merge(df_train,df_ori3,how='left')
df_train=pd.merge(df_train,df_ori4,how='left')
df_train=pd.merge(df_train,df_ori51,how='left')
df_train=pd.merge(df_train,df_ori52,how='left')
df_train=pd.merge(df_train,df_ori53,how='left')
df_train=pd.merge(df_train,df_ori54,how='left')
df_train=pd.merge(df_train,df_ori6,how='left')
df_train=pd.merge(df_train,df_ori8,how='left')
# df_train=pd.merge(df_train,df_ori1,how='left')
# df_train=pd.merge(df_train,df_ori7,how='left')
# df_train=pd.merge(df_train,df_ori9,how='left')
df_train=df_train.drop_duplicates()

df_test=pd.merge(df_test,df_ori2,how='left')
df_test=pd.merge(df_test,df_ori3,how='left')
df_test=pd.merge(df_test,df_ori4,how='left')
df_test=pd.merge(df_test,df_ori51,how='left')
df_test=pd.merge(df_test,df_ori52,how='left')
df_test=pd.merge(df_test,df_ori53,how='left')
df_test=pd.merge(df_test,df_ori54,how='left')
df_test=pd.merge(df_test,df_ori6,how='left')
df_test=pd.merge(df_test,df_ori8,how='left')
# df_test=pd.merge(df_test,df_ori1,how='left')
# df_test=pd.merge(df_test,df_ori7,how='left')
# df_test=pd.merge(df_test,df_ori9,how='left')
df_test=df_test.drop_duplicates()
# df_train=pd.merge(df_train,df_ori1,how='left')
# df_train=pd.merge(df_train,df_ori9,how='left')

###模型训练

from sklearn.model_selection import train_test_split
random_state=10
modeldata=df_train.copy(deep=True)

modeldata1s=modeldata[modeldata['score']<85]
modeldata1t=modeldata[modeldata['score']==85]
modeldata2=modeldata[modeldata['score']>85]
modeldata1s=modeldata1s.sample(n=4500,replace=True)
modeldata1t=modeldata1t.sample(n=1000,replace=True)
modeldata1=modeldata1s.append(modeldata1t)
modeldata=modeldata2.append(modeldata1)
modeldata2=modeldata2.sample(n=2000)
del modeldata['car_id']
del modeldata['业户ID']
X=modeldata[[i for i in list(modeldata.columns) if i not in ['score']]]
y=modeldata[['score']]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=random_state)

import lightgbm as lgb
from sklearn.metrics import accuracy_score
import numpy as np

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval  = lgb.Dataset(X_test, y_test, reference=lgb_train)

# 将参数写成字典下形式
params = {
     'task': 'train',
    'boosting_type': 'gbdt',    # 设置提升类型
    'objective': 'regression',  # 目标函数
    'num_leaves': 42,           # 一棵树的叶子节点数
    'learning_rate': 0.1,      # 学习速率
    'feature_fraction': 0.9,    # 建树的特征采样比例
    'bagging_fraction': 0.8,    # 建树的样本采样比例
    'bagging_freq': 5,          # 每k次迭代执行bagging
    'max_depth':7,
    'metric': {'mse'}
}

# 训练 cv and train
gbm = lgb.train(params, lgb_train, num_boost_round=50, valid_sets=lgb_eval, early_stopping_rounds=5)

y_train_pred= gbm.predict(X_train, num_iteration=gbm.best_iteration)

ys=gbm.predict(df_train[gbm.feature_name()], num_iteration=gbm.best_iteration)

df_train['predict']=ys

df_train['rt']=df_train.apply(lambda row:(row['score']-row['predict'])**2,axis=1)

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
dtg=pd.DataFrame({'score':list(y_test['score']),'predict':list(y_pred)})
dtg['rt']=dtg.apply(lambda row:(row['score']-row['predict'])**2,axis=1)



dt=pd.DataFrame({'fact':list(y_test['score']),'label':list(y_pred)})

y_oot = gbm.predict(df_test[gbm.feature_name()], num_iteration=gbm.best_iteration)
df_test['score']=list(y_oot)

df_test_result=df_test[['car_id','score']]
df_test_result['score']=df_test_result['score'].apply(lambda x:1 if x<=0 else 100 if x>=100 else round(x,2))

alist=[]
for i in range(df_test_result.shape[0]):
    df_test_results=df_test_result.iloc[i,:].to_dict()
    alist.append(df_test_results)



import json
filename='/Users/weiliangli/Downloads/result.json'
with open(filename,'w',encoding='utf-8') as file_obj:
  json.dump(alist,file_obj)



df_train_label=df_train_label.rename(columns={'业户ID':'car_id'})

df_trains=df_train[['car_id']]
df_tests=df_test[['car_id']]


alist=[]
alist.append(df_tests.set_index(['car_id'])['score'].to_dict())

import json
filename='/Users/weiliangli/Downloads/result.json'
with open(filename,'w') as file_obj:
  json.dump(alist,file_obj)