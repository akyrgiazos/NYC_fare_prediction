import time
notebookstart= time.time()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.spatial.distance import pdist
from tqdm import tqdm
import matplotlib.pyplot as plt
#import seaborn as sns
plt.rcParams['figure.figsize'] = 20,9
#import datetime
#from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
#from math import radians, cos, sin, asin, sqrt
# Modeling
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import gc
#print(os.listdir("./input"))


def cityblock(a,dist='cityblock'):
    return pdist(a.reshape(2,2),dist)[0]

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))

def prepare_distance_features(df):
    # Distance is expected to have an impact on the fare
    df['longitude_distance'] = abs(df['pickup_longitude'] - df['dropoff_longitude'])
    df['latitude_distance'] = abs(df['pickup_latitude'] - df['dropoff_latitude'])

    # Straight distance
    df['distance_travelled'] = (df['longitude_distance'] ** 2 + df['latitude_distance'] ** 2) ** .5
    df['distance_travelled_sin'] = np.sin((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5)
    df['distance_travelled_cos'] = np.cos((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5)
    df['distance_travelled_sin_sqrd'] = np.sin((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5) ** 2
    df['distance_travelled_cos_sqrd'] = np.cos((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5) ** 2

    # Haversine formula for distance
    # Haversine formula:	a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    #R = 6371e3 # Metres

    df['haversine'] = distance(df['pickup_latitude'],df['pickup_longitude'],df['dropoff_latitude'],df['dropoff_longitude'])
    df['dist_pas'] = df['haversine']*df['passenger_count']
    # Bearing
    # Formula:	θ = atan2( sin Δλ ⋅ cos φ2 , cos φ1 ⋅ sin φ2 − sin φ1 ⋅ cos φ2 ⋅ cos Δλ )
    #y = np.sin(delta_chg * np.cos(phi2))
    #x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    #df['bearing'] = np.arctan2(y, x)

    return df


def cluster_routes (train_df,test_df,num_cluster=200):
    coords = ['pickup_latitude',  'pickup_longitude']
    coordsoff=['dropoff_latitude', 'dropoff_longitude']
    concat = np.vstack([train_df.loc[:,coords+coordsoff].values,test_df[coords+coordsoff].values])
    db =  MiniBatchKMeans(init='k-means++', n_clusters=num_cluster, batch_size=10*6,n_init=10, 
                            init_size=2*num_cluster,max_no_improvement=400, verbose=0,random_state=0).fit(concat)
    labels = db.labels_
    train_df['cluster'] = labels[:train_df.shape[0]]
    test_df['cluster'] = labels[train_df.shape[0]:]
    return train_df,test_df


def read_csv_sampled(path, frac, chunksize=10**5, random_state=None,train=True):
    samples = []

    for df_chunk in tqdm(pd.read_csv(path, chunksize=chunksize,nrows=20_000_000)):
        if train:
            df_chunk = df_chunk.sample(frac=frac, random_state=random_state)
        df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].apply(pd.Timestamp).dt.tz_convert(None)
        df_chunk['weekday'] = df_chunk['pickup_datetime'].apply(lambda x: x.weekday())
        df_chunk['request_year'] = df_chunk['pickup_datetime'].apply(lambda x: x.year)
        df_chunk['request_month'] = df_chunk['pickup_datetime'].apply(lambda x: x.month)
        df_chunk['request_day']= df_chunk['pickup_datetime'].apply(lambda x: x.day)
        df_chunk['dayofyear']= df_chunk['pickup_datetime'].apply(lambda x: x.dayofyear)
        df_chunk['request_hour'] = df_chunk['pickup_datetime'].apply(lambda x: x.hour)
        df_chunk['weekofyear'] = df_chunk['pickup_datetime'].apply(lambda x: x.weekofyear)
        df_chunk['quarter'] = df_chunk['pickup_datetime'].apply(lambda x: x.quarter)
        if train:
            df_chunk['fare_amount'] = df_chunk['fare_amount'].apply(lambda x : np.log1p(x))

        samples.append(df_chunk)

    df = pd.concat(samples, ignore_index=True)
    
    return df


def checkdata(train,test):
    return not (train.isnull().values.any() or test.isnull().values.any())


def clean_data(df):
    df = df.dropna()
    is_weird = (df['fare_amount'] < 0)
    is_weird |= ~df['pickup_latitude'].between(35, 45)
    is_weird |= ~df['pickup_longitude'].between(-80, -70)
    is_weird |= ~df['dropoff_latitude'].between(35, 45)
    is_weird |= ~df['dropoff_longitude'].between(-80, -70)
    is_weird |= (df['passenger_count'] == 0)
    print(is_weird.sum())
    df = df[~is_weird]
    return df

def add_distances(df,dist_types):
    for dist_type in dist_types:
        print(dist_type)
        coords1 = df[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']]
        df[dist_type] = [cityblock(x,dist_type) for x in coords1.values]
    return df

def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_



# Build ime Aggregate Features
def time_agg(train, test_df, vars_to_agg, vars_be_agg):
    for var in vars_to_agg:
        print(var)
        agg = train.groupby(var)[vars_be_agg].agg(["sum","mean","std","skew",percentile(90),percentile(10)])
        for coli in agg.columns:
            agg[coli] = agg[coli].astype('float32')
        if isinstance(var, list):
            agg.columns = pd.Index(["fare_by_" + "_".join(var) + "_" + str(e) for e in agg.columns.tolist()])
        else:
            agg.columns = pd.Index(["fare_by_" + var + "_" + str(e) for e in agg.columns.tolist()]) 
        try:
            train = train.join(agg, on=var, how= "left")
            test_df = test_df.join(agg, on=var, how= "left")
            
            
        except ValueError as err:
            print(agg.dtypes)
            print(test_df.dtypes)
            print(err)
            break
    
    return train, test_df

def dist(pickup_lat, pickup_long, dropoff_lat, dropoff_long):  
    distance = np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)
    
    return distance

def airport_feats(train,test_df):
    for data in [train,test_df]:
        nyc = (-74.0063889, 40.7141667)
        jfk = (-73.7822222222, 40.6441666667)
        ewr = (-74.175, 40.69)
        lgr = (-73.87, 40.77)
        data['distance_to_center'] = dist(nyc[1], nyc[0],
                                          data['pickup_latitude'], data['pickup_longitude'])
        
        #pickup jfk
        data['pickup_distance_to_jfk'] = dist(jfk[1], jfk[0],
                                             data['pickup_latitude'], data['pickup_longitude'])
        #data['pickup_to_jfk'] = 0
        #data.loc[data['pickup_distance_to_jfk']<0.1,'pickup_to_jfk'] = 1
        
        # dropoff jfk
        data['dropoff_distance_to_jfk'] = dist(jfk[1], jfk[0],
                                               data['dropoff_latitude'], data['dropoff_longitude'])
        #data['dropoff_to_jfk'] = 0
        #data.loc[data['dropoff_distance_to_jfk']<0.1,'dropoff_to_jfk'] = 1

        #pickup ewr
        #data['pickup_to_ewr'] = 0
        data['pickup_distance_to_ewr'] = dist(ewr[1], ewr[0], 
                                              data['pickup_latitude'], data['pickup_longitude'])
        #data.loc[data['pickup_distance_to_ewr']<0.1,'pickup_to_ewr'] = 1

        #dropoff ewr
        data['dropoff_distance_to_ewr'] = dist(ewr[1], ewr[0],
                                               data['dropoff_latitude'], data['dropoff_longitude'])
        #data['dropoff_to_ewr'] = 0
        #data.loc[data['dropoff_distance_to_ewr']<0.1,'dropoff_to_ewr'] = 1

        #pickup lgr
        data['pickup_distance_to_lgr'] = dist(lgr[1], lgr[0],
                                              data['pickup_latitude'], data['pickup_longitude'])
        #data['pickup_to_lgr'] = 0
        #data.loc[data['pickup_distance_to_lgr']<0.1,'pickup_to_lgr'] = 1


        #dropoff lgr
        data['dropoff_distance_to_lgr'] = dist(lgr[1], lgr[0],
                                               data['dropoff_latitude'], data['dropoff_longitude'])
        #data['dropoff_to_lgr'] = 0
        #data.loc[data['dropoff_distance_to_lgr']<0.1,'dropoff_to_lgr'] = 1
        #data.drop(['pickup_distance_to_jfk','dropoff_distance_to_jfk','pickup_distance_to_ewr','dropoff_distance_to_ewr','pickup_distance_to_lgr',
        #	'dropoff_distance_to_lgr'],axis=1,inplace=True)

    return train, test_df

def memory_reduce(df):
    df['weekday'] = df['weekday'].astype('category').cat.codes
    df['request_month'] = df['request_month'].astype('category').cat.codes
    df['request_day'] = df['request_day'].astype('category').cat.codes
    df['dayofyear'] = df['dayofyear'].astype('category').cat.codes
    df['request_hour'] = df['request_hour'].astype('category').cat.codes
    df['request_year'] = df['request_year'].astype('category').cat.codes
    df['cluster'] = df['cluster'].astype("category").cat.codes
    df['weekofyear'] = df['weekofyear'].astype('category').cat.codes
    df['quarter'] = df['quarter'].astype('category').cat.codes
    
    return df

def modelling(train,y,num_splits=5):
    trainshape = train.shape
    #testshape = test.shape
    categorical = ['weekday','request_month','request_day','dayofyear','request_hour','request_year','cluster','weekofyear','quarter']
    # LGBM Dataset Formating
    predictors = train.columns.get_values().tolist()
    dtrain = lgb.Dataset(train, label=y, feature_name=predictors, categorical_feature=categorical,free_raw_data=False)
    #train_data_v1 = lightgbm.Dataset(train[predictors],label=train['is_attributed'],feature_name=predictors, categorical_feature=categorical)
    dtrain.save_binary('train_v1.bin')
    dtrain = lightgbm.Dataset('train_v1.bin', feature_name=predictors, categorical_feature=categorical)
    print("Light Gradient Boosting Regressor: ")
    lgbm_params =  {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate' : 0.03,
        'num_leaves' : 31,
        'max_depth' : -1,
        'subsample' : .8,
		'colsample_bytree' : 0.6,
		'min_split_gain' : 0.5,
		'min_child_weight' : 1
                    }

    folds = KFold(n_splits=num_splits, shuffle=True, random_state=1)
    #fold_preds = np.zeros(testshape[0])
    #fold_preds_exp = np.zeros(testshape[0])
    oof_preds = np.zeros(trainshape[0])
    # format 
   
    dtrain.construct()

    models = []
    # Fit 5 Folds
    modelstart = time.time()
    for trn_idx, val_idx in folds.split(train):
        clf = lgb.train(
            params=lgbm_params,
            train_set=dtrain.subset(trn_idx),
            valid_sets=dtrain.subset(val_idx),
            num_boost_round=3500, 
            early_stopping_rounds=125,
            verbose_eval=500
        )
        models.append(clf)
        oof_preds[val_idx] = clf.predict(dtrain.data.iloc[val_idx])
        #fold_preds += clf.predict(test) / folds.n_splits
        #fold_preds_exp += np.expm1(clf.predict(test)) / folds.n_splits
        print(mean_squared_error(y.iloc[val_idx], oof_preds[val_idx]) ** .5)
    print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))

    return models
    

def main():
    filename="./input/train.csv"
    filenametest="./input/test.csv"
    train=read_csv_sampled(filename,0.5, random_state=1989)
    print(train.shape)
    gc.collect()

    test=read_csv_sampled(filenametest,1.0, random_state=1989,train=False)
    test.dropna(axis=0,inplace=True)
    print(test.shape)
    KID = test.pop('key')
    gc.collect()

    train = clean_data(train)
    train = prepare_distance_features(train)
    test = prepare_distance_features(test)

    print(train.isnull().values.any())
    print(test.isnull().values.any())
    assert checkdata(train,test), "Houston we've got a problem"

    train,test = cluster_routes(train,test,num_cluster=200)
    print(train.groupby('cluster').size())
    dist_types = ['euclidean','canberra','cityblock']

    train = add_distances(train,dist_types)
    test = add_distances(test,dist_types)
    train, test = time_agg(train, test,
                          vars_to_agg  = ["cluster","passenger_count", "weekday","quarter", "request_month", "request_year","request_hour",
                                          ['cluster',"weekday", "request_month","request_year"], ["cluster","request_hour", "weekday", "request_month","request_year"]],
                          vars_be_agg = "fare_amount")
    
    train,test = airport_feats(train,test)
    train = memory_reduce(train)
    test = memory_reduce(test)

    y = train.pop('fare_amount')
    train.drop(['key','pickup_datetime'],axis=1,inplace=True)
    test.drop('pickup_datetime',axis=1,inplace=True)
    models = modelling(train,y)
    fold_preds_exp = np.zeros(test.shape[0])
    for model in models:
        fold_preds_exp += np.expm1(model.predict(test)) / 5
        print(np.sum(np.expm1(model.predict(test))))
   
    # Write the predictions to a CSV file which we can submit to the competition.
    submission = pd.DataFrame(
        {'key': KID, 'fare_amount': fold_preds_exp},
        columns = ['key', 'fare_amount'])
    submission.to_csv('submission_local.csv', index = False)

    print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))
    d2 = {'Feature':train.columns,'LGB_Importance' : models[-1].booster_.feature_importance(importance_type='split')}
    features_import = pd.DataFrame(data=d2)
    features_import = features_import.sort_values(['LGB_Importance'], ascending = False)
    features_import.to_csv('lgb_features.csv', index = False)
    #ax = lgb.plot_importance(models[-1], max_num_features=50)
    #plt.show()

if __name__== "__main__":
    main() 