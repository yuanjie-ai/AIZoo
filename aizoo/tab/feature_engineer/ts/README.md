构造时序特征时一定要算好时间窗口，特别是在工作的时候，需要自己去设计训练集和测试集，千万不要出现数据泄露的情况（比如说预测明天的数据时，是拿不到今天的特征的）；
针对上面的情况，可以尝试将今天的数据进行补齐；
有些特征加上去效果会变差，大概率是因为过拟合了；
有些特征加上去效果出奇好，第一时间要想到是不是数据泄露了；
拟合不好的时间（比如说双休日）可以分开建模；


```
# 构造过去 n 天的统计数据，此外，还可以对这些统计值进行分桶，增强模型的鲁棒性。

def get_statis_n_days_num(data, col, n):
  temp = pd.DataFrame()
  for i in range(n):
    temp = pd.concat([temp, data[col].shift((i+1)*24)], axis=1)
    data['avg_'+ str(n) +'_days_' + col] = temp.mean(axis=1)
    data['median_'+ str(n) +'_days_' + col] = temp.median(axis=1)
    data['max_'+ str(n) +'_days_' + col] = temp.max(axis=1)
    data['min_'+ str(n) +'_days_' + col] = temp.min(axis=1)
    data['std_'+ str(n) +'_days_' + col] = temp.std(axis=1)
    data['mad_'+ str(n) +'_days_' + col] = temp.mad(axis=1)
    data['skew_'+ str(n) +'_days_' + col] = temp.skew(axis=1)
    data['kurt_'+ str(n) +'_days_' + col] = temp.kurt(axis=1)
    data['q1_'+ str(n) +'_days_' + col] = temp.quantile(q=0.25, axis=1)
    data['q3_'+ str(n) +'_days_' + col] = temp.quantile(q=0.75, axis=1)
    data['var_'+ str(n) +'_days_' + col] = data['std_'+ str(n) +'_days_' + col]/data['avg_'+ str(n) +'_days_' + col]  # 离散系数
    return data

data_df = get_statis_n_days_num(data_df, 'num_events', n=7)
data_df = get_statis_n_days_num(data_df, 'num_events', n=14)
data_df = get_statis_n_days_num(data_df, 'num_events', n=21)
data_df = get_statis_n_days_num(data_df, 'num_events', n=28)
```
2.2 同期值

```python
# n 个星期前同期特征
data_df['ago_7_day_num_events'] = data_df['num_events'].shift(7*24)
data_df['ago_14_day_num_events'] = data_df['num_events'].shift(14*24)
data_df['ago_21_day_num_events'] = data_df['num_events'].shift(21*24)
data_df['ago_28_day_num_events'] = data_df['num_events'].shift(28*24)

# 昨天的同期特征
data_df['ago_7_day_num_events'] = data_df['num_events'].shift(1*24)
```

# 6. 是否是特殊节假日等
special_days = ['2021.5.1','2021.10.1',...]
df['isholiday'] = df['day'].isin(special_days)

# 1. 距离年初的时间差
df['day_to_year_start']  =  df['day'] - df['year_start']
# 2. 距离周末的时间差
df['day_to_weekend']     =  df['weekday'].apply(lambda x: 6-x if x<=5 else 0)
# 3. 距离发薪日的时间差
df['day_to_paymentday']  =  df['paymentday'] - df['day']
# ...

相邻时间差
相邻时间差频率编码

# 窗口函数 https://www.jianshu.com/p/b8c795345e93
# df.ewm 指数平滑 df[‘a’].ewm(span=2).mean() # https://blog.csdn.net/weixin_43055882/article/details/86736510

df.rolling(len(df)  == df.expanding # 窗口累计cumsum


# tsfresh https://blog.csdn.net/qq_42658739/article/details/122358303
# https://zhuanlan.zhihu.com/p/548010190
# 时间序列 特征选择 https://mp.weixin.qq.com/s/GhpJhbs9LZr7tmsb43SPng
from tsfresh import extract_relevant_features, feature_selection
from tsfresh.examples import robot_execution_failures

feature_selection.relevance.calculate_relevance_table(X, y)

robot_execution_failures.download_robot_execution_failures()
df, y = robot_execution_failures.load_robot_execution_failures()

features_filtered_direct = extract_relevant_features(df, y, column_id='id', column_sort='time')

# https://mp.weixin.qq.com/s/_hZLklBSFH-kXmBEfxdq_g
