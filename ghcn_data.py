import pandas as pd
import numpy as np
import calendar
import glob
import os
from math import sin, asin, cos, radians, sqrt
from tqdm import tqdm as tqdm
from joblib import Parallel,delayed
from collections import defaultdict

DIR = 'data/GHCN'

def get_candidates_dly(threshold = 3000):
    cnt = 0
    candidates = []
    for dly in tqdm(glob.glob(os.path.join(DIR,'ghcnd_all/*.dly'))):
        with open(dly,'r') as f:
            if len(f.readlines())>threshold:
                cnt +=1
                dly_name = dly.split('/')[-1].split('.')[0]
                candidates.append(dly_name)
    print(cnt)
    return candidates

def get_features(line):
    line = line.strip()
    _id = line[:11]
    _year = line[11:15]
    _month = line[15:17]
    _ele = line[17:21]
    _values = []
    start=21
    sep=8
    for i in range(31):
        tmp = int(line[start+i*sep:start+i*sep+5])
        _values.append(tmp)
    return _id, _year, _month, _ele, len(_values), _values

def get_ts(dly,start_date='1990-01-01',end_date='1999-12-31'):
    with open(os.path.join(DIR,'ghcnd_all',dly+'.dly'),'r') as f:
        ts = defaultdict(list)    
        for line in f:
            _id,_year,_month,_ele,_length,_values = get_features(line)
            if _ele not in ['PRCP','SNOW','SNWD','TMAX','TMIN']:
                continue
            _days = calendar.monthrange(int(_year), int(_month))[1]
            assert _days <= _length
            for i in range(_days):
                _day = i + 1
                date = '-'.join([_year,_month,str(_day).zfill(2)])
                _v = _values[i] if _values[i]!=-9999 else np.nan
                ts[_ele].append([date,_values[i]])
    
    df_ts = pd.DataFrame(index=pd.date_range(start=start_date,end=end_date,name='DATE').astype(str))
    for name in ['PRCP','SNOW','SNWD','TMAX','TMIN']:
        tmp = pd.DataFrame(ts[name],columns=['DATE',name]).set_index('DATE')
        df_ts = pd.merge(df_ts,tmp,left_index=True,right_index=True,how='left')
    os.makedirs(os.path.join(DIR,'node_features/{}/'.format(dly)), exist_ok=True)
    df_ts.to_csv(os.path.join(DIR,'node_features/{}/{}_{}.csv'.format(dly,start_date,end_date)))
    return (dly,df_ts[['PRCP','SNOW','SNWD','TMAX','TMIN']].isnull().mean().mean())


def get_lon_lat(candidates_dly):
    dly_list,lon_list,lat_list = [],[],[]
    with open(os.path.join(DIR, 'ghcnd-stations.txt'),'r') as f:
        for line in f:
            items = line.split()
            dly, lon, lat = items[:3]
            dly_list.append(dly)
            lon_list.append(lon)
            lat_list.append(lat)
    
    df = pd.DataFrame({'dly':dly_list,'lon':lon_list,'lat':lat_list})
    df_cand = pd.DataFrame({'dly':candidates_dly})
    
    return pd.merge(df_cand,df,on='dly',how='left')

def calculate_distance(lon1, lat1, lon2, lat2):
    '''
    Use haversine formula to calculate distance between two points by longtude and latitude.
    The output is in kilometers
    '''
    EARTH_RADIUS=6371           # radius of the earth is 6371km
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)
 
    dlon = abs(lon1 - lon2)
    dlat = abs(lat1 - lat2)
    h = hav(dlat) + cos(lat1) * cos(lat2) * hav(dlon)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))
 
    return max(distance, 0.1)

def hav(theta):
    s = sin(theta / 2)
    return s * s

def cal_adj(df):
    dly_list, lat_list, lon_list, = df['dly'].values,df['lon'].astype(float).values,df['lat'].astype(float).values
    A = np.zeros((df.shape[0],df.shape[0]))
    # calculate adj matrix
    for i in tqdm(range(df.shape[0])):
        for j in  range(i,df.shape[0]):
            i_lat,i_lon = lat_list[i],lon_list[i]
            j_lat,j_lon = lat_list[j],lon_list[j]
            A[i][j] = 1 / calculate_distance(i_lon,i_lat,
                                             j_lon,j_lat)
    return A


if __name__ == '__main__':
    candidates_dly = get_candidates_dly(threshold=1000)
    stats = Parallel(n_jobs=-1, backend='multiprocessing',verbose=2)(delayed(get_ts)(dly,'2006-01-01','2010-12-31')
                                                            for dly in candidates_dly)

    miss_df = pd.DataFrame(stats,columns=['dly','miss']) 

    miss_threshold = 0.2
    print((miss_df.miss < 0.2).sum())
    candidates_dly_used = miss_df.loc[miss_df.miss< 0.2,'dly'].tolist()   
    df = get_lon_lat(candidates_dly=candidates_dly_used)
    A = cal_adj(df)
    A_sym = A + A.T
    for i in range(A.shape[0]):
        A_sym[i,i] = 1.0
    threshold = 0.004 
    A_adj = A_sym.copy()
    A_adj[A_adj<threshold] = 0

    print((A_adj!=0).mean() * A_adj.shape[0])

    os.makedirs(os.path.join(DIR,'meta'),exist_ok=True)
    np.save(os.path.join(DIR, 'meta', 'A_adj.npy'), A_adj)
    np.save(os.path.join(DIR, 'meta', 'nodes_list.npy'), candidates_dly_used)