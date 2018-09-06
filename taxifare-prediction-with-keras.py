# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

## Script is a combination of cleaning techniques I picked up from other kernels and applied in this before running my own version of a keras MLP
from __future__ import print_function
import os
import geohash
import gc
import time
from tqdm import tqdm
#from functools import partial
#from itertools import repeat
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import keras.backend as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras import losses
from keras import regularizers
from keras.utils import np_utils
from keras.utils import generic_utils, Sequence
import multiprocessing
import os
import threading
# show tensorflow version
#print(tf.__version__)


def get_model(inputdim):
    model = Sequential()
    model.add(layers.Dense(512, activation='relu',input_dim=inputdim))#,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, activation='relu'))#,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation='linear'))

    sgd = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(
        loss='mean_squared_error', 
        optimizer=sgd,
        metrics=[losses.mean_absolute_error]
    )

    return model


# options for models
pd.set_option('display.width', 900)
cal = calendar()
holidays = cal.holidays(start="2009-01-01", end="2016-01-01")
def geohash_codes(train_df,precision=6):
    start_time = time.time()
    train_df['geohash_pickup'] = [geohash.encode(y,x, precision=precision) for y,x in train_df[['pickup_latitude','pickup_longitude']].values]
    train_df['geohash_dropoff'] = [geohash.encode(y,x, precision=precision) for y,x in train_df[['dropoff_latitude','dropoff_longitude']].values]
    print("--- %s seconds ---" % (time.time() - start_time))
    train_df['geohash_pickup'] = train_df['geohash_pickup'].astype('category')
    train_df['geohash_dropoff'] = train_df['geohash_dropoff'].astype('category')
    train_df['geohash_pickup'] = train_df['geohash_pickup'].cat.codes
    train_df['geohash_dropoff'] = train_df['geohash_dropoff'].cat.codes
    
    return train_df
    
def prepare_time_features(df):
    df['hour_class'] = 0#"overnight"
    #df.loc[(df['request_hour']<7) & (df['request_hour']>23),'hour_class'] = 'overnight'
    df.loc[(df['hour_of_day']<11) & (df['hour_of_day']>7),'hour_class'] = 1#'morning'
    df.loc[(df['hour_of_day']<16) & (df['hour_of_day']>11),'hour_class'] = 2#'noon'
    df.loc[(df['hour_of_day']<23) & (df['hour_of_day']>16),'hour_class'] = 3#'evening'
    df['hour_of_day']=df['hour_of_day'].astype('category')
    df['hour_class'] = df['hour_class'].astype('category')

    return df    
def clean_data(df):
    df = df.dropna()
    is_weird = (df['fare_amount'] <= 1)
    is_weird |= ~df['pickup_latitude'].between(35, 45)
    is_weird |= ~df['pickup_longitude'].between(-80, -70)
    is_weird |= ~df['dropoff_latitude'].between(35, 45)
    is_weird |= ~df['dropoff_longitude'].between(-80, -70)
    is_weird |= (df['passenger_count'] == 0)
    print(is_weird.sum())
    df = df[~is_weird]
    return df
# start working on model
# Set columns to most suitable type to optimize for memory usage
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}


# compute distances
def degree_to_radion(degree):
    return degree * (np.pi / 180)

def calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)
    
    radius = 6371.01
    
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = (np.sin(lat_diff / 2) ** 2 
         + np.cos(degree_to_radion(from_lat)) 
         * np.cos(degree_to_radion(to_lat)) 
         * np.sin(long_diff / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c

def add_time_features(data):
    start_time = time.time()
    #format="%Y/%m/%d %H:%M:%S UTC"
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'], infer_datetime_format=True).dt.tz_localize(None)#data['pickup_datetime'].apply(pd.Timestamp).dt.tz_convert(None)
    print("--- %s seconds ---" % (time.time() - start_time))
    data['hour_of_day'] = data.pickup_datetime.dt.hour.astype('category').cat.codes
    data['day_of_week'] = data.pickup_datetime.dt.dayofweek.astype('category').cat.codes
    data['day_of_month'] = data.pickup_datetime.dt.day.astype('category').cat.codes
    data['week_of_year'] = data.pickup_datetime.dt.weekofyear.astype('category').cat.codes
    data['month_of_year'] = data.pickup_datetime.dt.month.astype('category').cat.codes
    data['quarter_of_year'] = data.pickup_datetime.dt.quarter.astype('category').cat.codes
    data['year'] = data.pickup_datetime.dt.year.astype('category').cat.codes
    print("--- %s seconds ---" % (time.time() - start_time))
    data['holiday'] = data['pickup_datetime'].dt.normalize().isin(holidays)
    #data['holiday'] = data['holiday'].astype('category')
    print("--- %s seconds ---" % (time.time() - start_time))

    return data.drop('pickup_datetime',axis=1)

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
    
def prepare_distance_features(df):
    # Distance is expected to have an impact on the fare
    df['longitude_distance'] = abs(df['pickup_longitude'] - df['dropoff_longitude']).astype('float32')
    df['latitude_distance'] = abs(df['pickup_latitude'] - df['dropoff_latitude']).astype('float32')

    # Straight distance
    df['distance_travelled'] = ((df['longitude_distance'] ** 2 + df['latitude_distance'] ** 2) ** .5).astype('float32')
    df['distance_travelled_sin'] = np.sin((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5).astype('float32')
    df['distance_travelled_cos'] = np.cos((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5).astype('float32')
    df['distance_travelled_sin_sqrd'] = (np.sin((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5) ** 2).astype('float32')
    df['distance_travelled_cos_sqrd'] = (np.cos((df['longitude_distance'] ** 2 * df['latitude_distance'] ** 2) ** .5) ** 2).astype('float32')

    # Haversine formula for distance
    # Haversine formula:	a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    #R = 6371e3 # Metres

    df['haversine'] = distance(df['pickup_latitude'],df['pickup_longitude'],df['dropoff_latitude'],df['dropoff_longitude']).astype('float32')
    df['dist_pas'] = (df['haversine']*df['passenger_count']).astype('float32')
    # Bearing
    # Formula:	θ = atan2( sin Δλ ⋅ cos φ2 , cos φ1 ⋅ sin φ2 − sin φ1 ⋅ cos φ2 ⋅ cos Δλ )
    #y = np.sin(delta_chg * np.cos(phi2))
    #x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    #df['bearing'] = np.arctan2(y, x)

    return df   

def add_distance_features(data):    
    lgr = (-73.8733, 40.7746)
    jfk = (-73.7900, 40.6437)
    ewr = (-74.1843, 40.6924)
    nyc = (-74.0063889, 40.7141667)
    
    data['trip_distance_km'] = calculate_distance(data.pickup_latitude, data.pickup_longitude, data.dropoff_latitude, data.dropoff_longitude).astype('float32')
    data['pickup_distance_nyc'] = calculate_distance(data['pickup_latitude'], data['pickup_longitude'], nyc[1], nyc[0]).astype('float32')
    data['dropoff_distance_nyc'] = calculate_distance(data['dropoff_latitude'], data['dropoff_longitude'], nyc[1], nyc[0]).astype('float32')
    data['pickup_distance_jfk'] = calculate_distance(data['pickup_latitude'], data['pickup_longitude'], jfk[1], jfk[0]).astype('float32')
    data['dropoff_distance_jfk'] = calculate_distance(data['dropoff_latitude'], data['dropoff_longitude'], jfk[1], jfk[0]).astype('float32')
    data['pickup_distance_ewr'] = calculate_distance(data['pickup_latitude'], data['pickup_longitude'], ewr[1], ewr[0]).astype('float32')
    data['dropoff_distance_ewr'] = calculate_distance(data['dropoff_latitude'], data['dropoff_longitude'], ewr[1], ewr[0]).astype('float32')
    data['pickup_distance_laguardia'] = calculate_distance(data['pickup_latitude'], data['pickup_longitude'], lgr[1], lgr[0]).astype('float32')
    data['dropoff_distance_laguardia'] = calculate_distance(data['dropoff_latitude'], data['dropoff_longitude'], lgr[1], lgr[0]).astype('float32')
    return data

    
from sklearn.preprocessing import StandardScaler
numerics = [
    'passenger_count',                         
    'trip_distance_km',
    'longitude_distance',
    'latitude_distance',
    'distance_travelled',
    'distance_travelled_sin',
    'distance_travelled_cos',
    'distance_travelled_sin_sqrd',
    'distance_travelled_cos_sqrd',
    'haversine',
    'dist_pas',
    'pickup_distance_nyc',
    'dropoff_distance_nyc',
    'pickup_distance_jfk',
    'dropoff_distance_jfk',
    'pickup_distance_ewr',
    'dropoff_distance_ewr',
    'pickup_distance_laguardia',
    'dropoff_distance_laguardia'
]
#standard_scaler = StandardScaler().fit(df[numerics])
#numeric_scale = standard_scaler.transform(df[numerics])

cat_cols = [
    'hour_of_day',
    'day_of_week',
    'day_of_month',
    'week_of_year',
    'month_of_year',
    'quarter_of_year',
    'year',
    #'geohash_pickup',
    #'geohash_dropoff',
    'hour_class',
    'holiday'
]
# This is what you need




def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def gen(x_train,y_train,batch_size):
    print('generator initiated')
    idx = 0
    l = x_train.shape[0]
    while True:
        batch_index = np.random.randint(0, l - batch_size)
        start = batch_index
        end = start + batch_size
        #for ndx in range(0, l, n):
        #yield iterable[ndx:min(ndx + n, l),:], y[ndx:min(ndx + n, l)]
        yield x_train[start: end,:], y_train[start: end]
        #print('generator yielded a batch %d' % idx)
        #idx += 1

class MY_Generator(Sequence):

    def __init__(self, data, labels, batch_size):
        self.data, self.labels = data, labels
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.data) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y


import multiprocessing as mp
class threaded_batch_iter(object):
    '''
    Batch iterator to make transformations on the data. 
    Uses multiprocessing so that batches can be created on CPU while GPU runs previous batch
    '''
    def __init__(self, batchsize):
        self.batchsize = batchsize

    def __call__(self, X, y):
        self.X, self.y = X, y
        return self

    def __iter__(self):
        '''
        multi thread the iter so that the GPU does not have to wait for the CPU to process data
        runs the _gen_batches function in a seperate process so that it can be run while the GPU is running previous batch
        '''
        q = mp.Queue(maxsize=128)

        def _gen_batches():
            num_samples = len(self.X)
            idx = np.random.permutation(num_samples)
            batches = range(0, num_samples - self.batchsize + 1, self.batchsize)
            for batch in batches:
                X_batch = self.X[idx[batch:batch + self.batchsize]]
                y_batch = self.y[idx[batch:batch + self.batchsize]]
          
                # do some stuff to the batches like augment images or load from folders
                
                yield [X_batch, y_batch]

        def _producer(_gen_batches):
            # load the batch generator as a python generator
            batch_gen = _gen_batches()
            # loop over generator and put each batch into the queue
            for data in batch_gen:
                q.put(data, block=True)
            # once the generator gets through all data issue the terminating command and close it
            q.put(None)
            q.close()
        
        # start the producer in a seperate process and set the process as a daemon so it can quit easily if you ctrl-c
        thread = mp.Process(target=_producer, args=[_gen_batches])
        thread.daemon = True
        thread.start()
        
        # grab each successive list containing X_batch and y_batch which were added to the queue by the generator
        for data in iter(q.get, None):
            yield data[0], data[1]

    
if __name__== "__main__":
    cols = list(traintypes.keys())
    filename="./input/train.csv"
    filenametest="./input/test.csv"
    df = pd.read_csv(filename, usecols=cols, dtype=traintypes,nrows=500000)
    df = clean_data(df)
    print(df.shape)
    gc.collect()
    print("Adding time features...") 
    df = add_time_features(df)
    print("Preparing distance features...")
    df = prepare_distance_features(df)
    print("Adding distance features...")
    df = add_distance_features(df)
    print("Adding geohash...")
    #df = geohash_codes(df)
    print("Adding time more features...")
    df = prepare_time_features(df)
    print(df.info())
    gc.collect()
    gc.collect()
    enc = OneHotEncoder(sparse=False).fit(df[cat_cols])
    categorical_scale = enc.transform(df[cat_cols])
    df.drop(cat_cols,axis=1,inplace=True)
    y = df.fare_amount.values
    
    X = np.column_stack((df[numerics].values,categorical_scale)).astype('float32')
    del categorical_scale
    del df
    gc.collect()

    _BATCH_SIZE = 512
    _EPOCHS = 50
    model = get_model(X.shape[1])
    train_index, test_index = train_test_split(range(X.shape[0]),test_size=0.01)
    my_training_batch_generator = MY_Generator(X[train_index,:], y[train_index], _BATCH_SIZE)
    my_validation_batch_generator = MY_Generator(X[test_index], y[test_index], _BATCH_SIZE)

    model.fit_generator(generator=my_training_batch_generator,
                                      steps_per_epoch=(len(train_index) // _BATCH_SIZE),
                                      epochs=_EPOCHS,
                                      verbose=1,
                                      validation_data=my_validation_batch_generator,
                                      validation_steps=(len(test_index) // _BATCH_SIZE),
                                      use_multiprocessing=False,
                                      workers=4,
                                      max_queue_size=32)
 
    try:
        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")
    except Exception as e:
        print("Couldn't save model to file.")



    print('generating predictions for test data')
    test = pd.read_csv('./input/test.csv')
    print(test.shape)

    print('adding features to test data')

    test = add_time_features(test)
    test = prepare_distance_features(test)
    test = add_distance_features(test)
    #test = geohash_codes(test)
    test = prepare_time_features(test)

    print('calling model for predictions')
    #enc = OneHotEncoder(sparse=False).fit(df[cat_cols])
    test_categorical_scale = enc.transform(test[cat_cols])

    predictions = model.predict(
        np.hstack((test_categorical_scale, test[numerics].values))
    )
    submission = pd.DataFrame(predictions, columns=['fare_amount'])
    submission['key'] = test['key']

    submission[['key', 'fare_amount']].to_csv('code_kernel_submission_withoutgeohash.csv', index=False)
    print(submission.head())





















