#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Python.
# @File         : baseline
# @Time         : 2022/11/22 上午9:02
# @Author       : yuanjie
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :


from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ME
from meutils.pipe import *
from meutils.hash_utils import md5

from aizoo.tab.models import LGBMOOF
from aizoo.tab.eda import EDA
from aizoo.tab.feature_engineer import FE


@decorator
def disk_cache(func, location='cachedir', *args, **kwargs):
    k = md5(f"cache_{func.__name__}_{args}_{kwargs}")
    output = Path(location) / Path(k) / '__output.pkl'

    if output.is_file():
        return joblib.load(output)

    else:
        logger.info(f"CacheKey: {k}")
        output.parent.mkdir(parents=True, exist_ok=True)
        _ = func(*args, **kwargs)
        joblib.dump(_, output)
        return _


def submit(scores=100.00, filename='result.json'):
    print(pd.Series(scores).value_counts())
    df_label[['car_id']][7117:].assign(score=scores).to_json(filename, 'records')


@disk_cache
def load_data():
    dfs = pd.read_excel('data.xlsx', sheet_name=None)
    df_label = pd.read_json('phase1_train.json').append(pd.read_json('phase1_test.json'), ignore_index=True)
    return dfs, df_label


dfs, df_label = load_data()
df_label = df_label.assign(
    label_100=lambda df: np.where(df.score >= 100, 1, 0),
    label_99=lambda df: np.where(df.score >= 99, 1, 0),
    label_98=lambda df: np.where(df.score >= 98, 1, 0),
    label_97=lambda df: np.where(df.score >= 97, 1, 0),
    label_96=lambda df: np.where(df.score >= 96, 1, 0),
)
car_id_set = set(df_label.car_id)

columns = {
    '车辆牌照号': 'car_id',
    '车牌号': 'car_id',
    '车牌号码': 'car_id',
    '单位名称': '企业名称'
}
for sheet in dfs:
    dfs[sheet] = dfs[sheet].rename(columns=columns).dropna(axis=1, how='all').drop_duplicates()

# df_label[df_label.car_id.isin(dfs['运政车辆年审记录信息'][lambda df: df['审批结果']=='年审不合格'].car_id)] # 有车牌颜色会变 脏数据
df_运政车辆年审记录信息 = (
    dfs['运政车辆年审记录信息'].replace({'年审合格': 1, '年审不合格': 0})
        .pivot_table(values='审批结果', columns='年审年度', aggfunc=max, index='car_id')
        .add_prefix('年审')
        .assign(年审sum=lambda df: df.sum(1))
        .reset_index()
)

df_车辆违法违规信息 = (
    dfs['车辆违法违规信息（道路交通安全，来源抄告）'].assign(道路交通安全=1)
    .append(dfs['车辆违法违规信息（交通运输违法，来源抄告）'].assign(道路交通安全=0)).sort_values(['car_id', '违规时间'], ignore_index=True)
)
df_车辆违法违规信息['违规时间'] = pd.to_datetime(df_车辆违法违规信息['违规时间'])
df_车辆违法违规信息['违规时间间隔'] = df_车辆违法违规信息.groupby('car_id')['违规时间'].diff().dt.days
df_车辆违法违规信息 = df_车辆违法违规信息.groupby('car_id').agg({'道路交通安全': ['nunique', 'count'], '违规时间间隔': ['mean', 'std']})
df_车辆违法违规信息.columns = ['_'.join(i) for i in df_车辆违法违规信息.columns]
df_车辆违法违规信息.reset_index(inplace=True)


df_动态监控报警信息 = dfs['动态监控报警信息（车辆，超速行驶）'].append(dfs['动态监控报警信息（车辆，疲劳驾驶）']).drop_duplicates().replace({'超速报警': 0, '疲劳报警': 1})


def kurt(x):
    return x.kurt()

funcs = ['sum','max','min','mean','median','std'] + ['skew', kurt]
agg_func = {
    '报警类型': 'count',
    '最高时速(Km/h)': funcs,
    '持续点数': funcs,
    '持续时长(秒)': funcs,
    # '时间差': funcs,
}
df_报警类型 = df_动态监控报警信息.groupby('报警类型').agg(agg_func)
df_报警类型.columns = ['_'.join(i) for i in df_报警类型.columns]

df_动态监控报警信息 = df_动态监控报警信息.merge(df_报警类型.reset_index())


agg_func = dict(zip(df_报警类型.columns, [['mean', 'std']]*len(df_报警类型.columns)))
agg_func['car_id'] = 'count'
agg_func['报警类型'] = agg_func['最高时速(Km/h)'] = agg_func['持续点数'] = agg_func['持续时长(秒)'] = funcs


df_动态监控报警信息 = df_动态监控报警信息.groupby('car_id').agg(agg_func)
df_动态监控报警信息.columns = ['_'.join(i) for i in df_动态监控报警信息.columns]
df_动态监控报警信息.reset_index(inplace=True)


df = (
    df_label
    .merge(dfs['运政车辆信息'], 'left')
    .merge(dfs['运政业户信息'][['企业名称', '业户ID']], 'left')
    .merge(dfs['运政质量信誉考核记录'].sort_values('考核日期').drop_duplicates('业户ID', 'last').replace({'优良(AAA)': 0, '合格(AA)': 1, '基本合格(A)': 2, '不合格(B)': 3}), 'left')
    .merge(dfs['动态监控上线率（企业，%）'], 'left')
    .merge(df_运政车辆年审记录信息, 'left')
    .merge(df_动态监控报警信息, 'left')
    .merge(df_车辆违法违规信息, 'left')
)

for feat in df.columns:
    if feat in ['label', 'car_id']:
        continue
    if df[feat].dtypes!='float':
        df[feat] = LabelEncoder().fit_transform(df[feat])   
