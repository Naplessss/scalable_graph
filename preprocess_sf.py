import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Pool
import warnings
warnings.filterwarnings(action='ignore')
import datetime
import os


def read_raw_data(DIR = '../data/0529'):
    edge_h = pd.read_csv(os.path.join(DIR,'edge_h.csv'))
    edge_w = pd.read_csv(os.path.join(DIR,'edge_w.csv'))
    node = pd.read_csv(os.path.join(DIR,'node.csv'))
    node.columns = [item.split('.')[-1] for item in node.columns]
    edge_h.columns = [item.split('.')[-1] for item in edge_h.columns]
    edge_w.columns = [item.split('.')[-1] for item in edge_w.columns]
    edge_h = edge_h.sort_values(['source_zonecode','dest_zonecode']).reset_index(drop=True)
    edge_w = edge_w.sort_values(['source_zonecode','dest_zonecode']).reset_index(drop=True)

    return edge_h,edge_w,node


def parallelize_dataframe(df, func, n_cores=12):
    idx = np.array_split(df[['source_zonecode','dest_zonecode']].drop_duplicates(keep='last').index.values,n_cores)
    start_idx,end_idx = 0,0
    df_split = []
    for item in idx:
        end_idx = item[-1]
        df_split.append(df.iloc[start_idx:end_idx+1])
        start_idx = item[-1]
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()

    return df

def flatten_hour(tmp=None):
    tmp = pd.merge(pd.Series(index=range(24),name='dest_hour'),
         tmp.set_index('dest_hour',drop=True),
         left_index=True,
         right_index=True,
         how='left')
    select_cols = ['inflow_volume_hour',
                  'inflow_avg_tm_hour',
                  'dest_flow_weight_hour',
                  'src_flow_weight_hour']
    _index = ['{}_{}'.format(col,i) for col in select_cols for i in range(24)]
    _value = []
    for col in select_cols:
        _value = _value + tmp[col].tolist()
    stats = pd.Series(_value,index=_index)
    return stats

def flatten_week(tmp=None):
    tmp = pd.merge(pd.Series(index=range(1,8),name='dest_week'),
         tmp.set_index('dest_week',drop=True),
         left_index=True,
         right_index=True,
         how='left')
    select_cols = ['inflow_volume_week',
                  'inflow_avg_tm_week',
                  'dest_flow_weight_week',
                  'src_flow_weight_week']
    _index = ['{}_{}'.format(col,i) for col in select_cols for i in range(1,8)]
    _value = []
    for col in select_cols:
        _value = _value + tmp[col].tolist()
    stats = pd.Series(_value,index=_index)
    return stats   

def agg_hour(df):
    return df.groupby(['source_zonecode','dest_zonecode']).apply(flatten_hour)

def agg_week(df):
    return df.groupby(['source_zonecode','dest_zonecode']).apply(flatten_week)


def write_normalized_edge(DIR,edge_h,edge_w,n_cores=12):
    edge_flatten_h = parallelize_dataframe(edge_h,agg_hour,n_cores=n_cores)
    edge_flatten_w = parallelize_dataframe(edge_w,agg_week,n_cores=n_cores)
    edge = pd.concat([edge_flatten_h,edge_flatten_w],axis=1).reset_index(drop=False)
    edge.to_csv(os.path.join(DIR,'edge.csv'),index=False,header=True)
    return edge

def normalized_egde_node(DIR,node,edge,threshold=0.05):
    edge['source_is_shenzhen'] = edge.source_zonecode.map(lambda x:x.startswith('755'))
    edge['dest_is_shenzhen'] = edge.dest_zonecode.map(lambda x:x.startswith('755'))
    edge.fillna(0.0,inplace=True)

    dest_flow_cols = ['dest_flow_weight_week_{}'.format(i) for i in range(1,8)]
    edge['dest_flow_avg'] = edge[dest_flow_cols].mean(axis=1)
    edge = edge[edge['dest_flow_avg']>threshold] 

    order_1st_zonecode = set(edge[(edge['source_is_shenzhen']+edge['dest_is_shenzhen'])!=0]['source_zonecode'].unique().tolist() + \
    edge[(edge['source_is_shenzhen']+edge['dest_is_shenzhen'])!=0]['dest_zonecode'].unique().tolist())
    order_2nd_zonecode = set(edge[edge.source_zonecode.isin(order_1st_zonecode)]['dest_zonecode'].unique().tolist() + \
    edge[edge.dest_zonecode.isin(order_1st_zonecode)]['source_zonecode'].unique().tolist())
    edge = edge[edge.source_zonecode.isin(order_2nd_zonecode) & edge.dest_zonecode.isin(order_2nd_zonecode)].reset_index(drop=True)

    node = node[node.zonecode.isin(order_2nd_zonecode)]
    node_zonecode = node.zonecode.unique()
    edge  = edge[edge.source_zonecode.isin(node_zonecode) & edge.dest_zonecode.isin(node_zonecode)].reset_index(drop=True)

    print(len(order_1st_zonecode),len(order_2nd_zonecode),len(node_zonecode))

    index = pd.MultiIndex.from_product([sorted(node.zonecode.unique()),
                                pd.date_range(start='2020-04-01',end='2020-04-08',freq='H',
                                closed='left').map(lambda x: datetime.datetime.strftime(x,'%Y-%m-%d-%H'))],
                                    names=['zonecode','tm'])
    node_flatten = pd.DataFrame(index=index).reset_index(drop=False)   
    node_flatten = pd.merge(node_flatten,node.replace('N',np.nan),how='left',on=['zonecode','tm']).fillna(0.0)
    node_features = [
        'outflow_delivery_volume',   # LABEL
        'outflow_transfer_volume', 
        'inflow_user_volume',
       'inflow_transfer_volume', 
       'inflow_dest_volume',
       'outflow_delivery_success',
       'outflow_delivery_fail', 
       'inflow_lastzone_volume',
       'inflow_firstzone_volume',
       'inflow_total_volume',
       'outflow_total_volume']
    node_flatten[node_features] = node_flatten[node_features].astype(float)
    edge.to_csv(os.path.join(DIR,'edge.3258.csv'))
    node_flatten.to_csv(os.path.join(DIR,'node.3258.csv'))  

def dup_daily_node(DIR, periods=7*12):
    node_features = [
        'outflow_delivery_volume',   # LABEL
        'outflow_transfer_volume', 
        'inflow_user_volume',
       'inflow_transfer_volume', 
       'inflow_dest_volume',
       'outflow_delivery_success',
       'outflow_delivery_fail', 
       'inflow_lastzone_volume',
       'inflow_firstzone_volume',
       'inflow_total_volume',
       'outflow_total_volume']
    node_flatten = pd.read_csv(os.path.join(DIR,'node.3258.csv'))
    node_flatten['tm_day'] = node_flatten['tm'].map(lambda x:x[:-3])
    node_flatten_day = node_flatten.groupby(['tm_day','zonecode'])[node_features].sum()
    node_label = pd.read_csv(os.path.join(DIR,'zone_history_sent.csv'))
    node_label.columns = ['zonecode','LABEL','tm_day','TYPE']
    node_label.tm_day = node_label.tm_day.map(lambda x:'-'.join([str(x)[:4],str(x)[4:6],str(x)[6:8]]))
    node_label = node_label[['zonecode','tm_day','LABEL']]
    node_flatten_day = pd.merge(node_flatten_day,node_label,on=['zonecode','tm_day'],how='left')
    node_flatten_day = node_flatten_day.set_index(['tm_day','zonecode'])

    node_flatten_test = node_flatten_day.loc['2020-04-07']
    tm_index = ['2020-04-01','2020-04-02','2020-04-03','2020-04-04','2020-04-05','2020-04-06','2020-04-06']
    tm_day = pd.date_range(end='2020-04-07',freq='D',periods=periods,closed='left').map(lambda x: datetime.datetime.strftime(x,'%Y-%m-%d'))

    node_flatten_list = []
    for i in range(periods-1):
        np.random.seed(i)
        tm_1,tm_2,tm_3 = (i-2)%7,(i-1)%7,i%7
        w_1,w_2,w_3 = np.random.rand(3)
        tmp_node = (node_flatten_day.loc[tm_index[tm_1]]*w_1 + \
                    node_flatten_day.loc[tm_index[tm_2]]*3*w_2 + \
                    node_flatten_day.loc[tm_index[tm_3]]*w_3) / (w_1 + w_2*3 + w_3)
        tmp_node['tm'] = tm_day[i]
        node_flatten_list.append(tmp_node)
    node_flatten_test['tm'] = '2020-04-07'
    node_flatten_list.append(node_flatten_test)
    node_flatten_sample = pd.concat(node_flatten_list,axis=0).reset_index(drop=False)
    node_features_selected = node_features
    node_flatten_sample = node_flatten_sample[['tm','zonecode'] + node_features_selected]
    node_flatten_sample[node_features_selected] = node_flatten_sample[node_features_selected].astype(int)

    node_flatten_sample.to_csv(os.path.join(DIR,'node.oversample.3258.csv'))
    


def collect_edge_node(DIR, add_self_loop=True):
    node_flatten_sample = pd.read_csv(os.path.join(DIR,'node.oversample.3258.csv'))
    edge = pd.read_csv(os.path.join(DIR,'edge.3258.csv'))
    edge_features = [item for item in edge.columns if '_hour_' in item or '_week_' in item]
    node_features = [
        # 'LABEL',
        'outflow_delivery_volume',   # LABEL
        'outflow_transfer_volume', 
        'inflow_user_volume',
       'inflow_transfer_volume', 
       'inflow_dest_volume',
       'outflow_delivery_success',
       'outflow_delivery_fail', 
       'inflow_lastzone_volume',
       'inflow_firstzone_volume',
       'inflow_total_volume',
       'outflow_total_volume']

    node_zonecode = sorted(node_flatten_sample.zonecode.unique())
    zonecode2idx = {item:i for i,item in enumerate(node_zonecode)}
    idx2zonecode = {i:item for i,item in enumerate(node_zonecode)}
    shenzhen_mask = [item.startswith('755') for item in node_zonecode]
    edge['source_idx'] = edge.source_zonecode.map(zonecode2idx)
    edge['dest_idx'] = edge.dest_zonecode.map(zonecode2idx)   

    A = edge[['source_idx','dest_idx']]
    A_self_loop = pd.DataFrame({'source_idx':range(len(node_zonecode)),'dest_idx':range(len(node_zonecode))})
    if add_self_loop:
        A = pd.concat([A,A_self_loop],ignore_index=True,axis=0)
    edge_index = np.transpose(A[['source_idx','dest_idx']].values, [1,0]).astype(int)
    edge_feature = edge[edge_features].values.astype(np.float32)
    if add_self_loop:
        edge_feature_self_loop = np.zeros(shape=(len(node_zonecode), len(edge_features)),dtype=np.float32)
        edge_feature = np.vstack((edge_feature,edge_feature_self_loop)).astype(np.float32)

    node_flatten_sample['idx'] = node_flatten_sample.zonecode.map(zonecode2idx)
    node_feature = node_flatten_sample.sort_values(['idx','tm'])[node_features].values
    num_nodes, num_timesteps, num_features = len(zonecode2idx),-1, len(node_features)
    node_feature = node_feature.reshape(num_nodes,num_timesteps,num_features)
    node_feature = np.transpose(node_feature, [0,2,1]).astype(np.float32)  # num_nodes, num_timesteps, num_features -> num_nodes, num_features, num_timesteps

    np.save(os.path.join(DIR,'zonecode2idx.npy'),zonecode2idx)
    np.save(os.path.join(DIR,'idx2zonecode.npy'),idx2zonecode)
    np.save(os.path.join(DIR, 'shenzhenidxmask.npy'), shenzhen_mask)
    np.save(os.path.join(DIR,'edge.attr.npy'),[edge_index,edge_feature,edge_features])
    np.save(os.path.join(DIR,'node.attr.npy'),[node_feature,node_features])

if __name__ == '__main__':
    DIR = 'data/SF-Example'
    # edge_h,edge_w,node = read_raw_data(DIR = DIR)
    # edge = write_normalized_edge(DIR,edge_h,edge_w,24)
    # normalized_egde_node(DIR, node, edge, threshold=0.05)
    dup_daily_node(DIR)
    collect_edge_node(DIR)