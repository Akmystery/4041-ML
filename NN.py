from datetime import date, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib as plp
import tensorflow as tf
from tensorflow import keras

import warnings
warnings.filterwarnings('ignore')

#Uncommnet this line if you want to read in Colab
# Code to read csv file into Colaboratory:
#!pip install -U -q PyDrive
#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive
#from google.colab import auth
#from oauth2client.client import GoogleCredentials
## Authenticate and create the PyDrive client.
#auth.authenticate_user()
#gauth = GoogleAuth()
#gauth.credentials = GoogleCredentials.get_application_default()
#drive = GoogleDrive(gauth)

#link = 'https://drive.google.com/open?id=1ekktdLVfdd3opLMZEWeJvS67uOyV7OEE'
#fluff, id = link.split('=')
#downloaded = drive.CreateFile({'id':id})
#downloaded.GetContentFile('train_set.csv')

df_train = pd.read_csv(
    'train_set.csv', usecols=[1, 2, 3, 4, 5],
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],dtype={'onpromotion': bool}
)

#link = 'https://drive.google.com/open?id=1MaM5OXakSUglf4qoe1vovsIk_sXUUhQ9'
#fluff, id = link.split('=')
#downloaded = drive.CreateFile({'id':id})
#downloaded.GetContentFile('test.csv')

df_test = pd.read_csv(
    "test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)

items = pd.read_csv(
    "items.csv",
).set_index("item_nbr") #In order to give weight to item perishable

promo_train = df_train.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)

promo_train.columns = promo_train.columns.get_level_values(1)

promo_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_test.columns = promo_test.columns.get_level_values(1)

promo_test = promo_test.reindex(promo_train.index).fillna(False)
promo_2017 = pd.concat([promo_train, promo_test], axis=1)
del promo_test, promo_train

df_train = df_train.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
df_train.columns = df_train.columns.get_level_values(1)

items = items.reindex(df_train.index.get_level_values(1))

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
          "day_1_2017": get_timespan(df_train, t2017, 1, 1).values.ravel()
        })
    for i in [3,7,14,30,60,140]:
            X["mean_"+str(i)] = get_timespan(df_train, t2017, i, i).mean(axis=1).values
            X["median_"+str(i)] = get_timespan(df_train, t2017, i, i).median(axis=1).values
            X["std_"+str(i)] = get_timespan(df_train, t2017, i, i).std(axis=1).values
            X["promo_"+str(i)] = get_timespan(promo_2017, t2017, i, i).sum(axis=1).values
            X["max_"+str(i)] = get_timespan(promo_2017, t2017, i, i).max(axis=1).values
            X["min_"+str(i)] = get_timespan(promo_2017, t2017, i, i).min(axis=1).values
    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_train, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_train, t2017, 140-i, 20, freq='7D').mean(axis=1).values
    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[t2017 + timedelta(days=i)].values.astype(np.uint8)
    if is_train:
        y = df_train[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X

print("Preparing dataset...")
t2017 = date(2017, 6, 21)
X_l, y_l = [], []
for i in range(4):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(t2017 + delta)
    X_l.append(X_tmp)
    y_l.append(y_tmp)
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l

X_val, y_val = prepare_dataset(date(2017, 7, 26))

X_test = prepare_dataset(date(2017, 8, 16), is_train=False)

scaler = StandardScaler()
scaler.fit(pd.concat([X_train, X_val, X_test]))
X_train[:] = scaler.transform(X_train)
X_val[:] = scaler.transform(X_val)
X_test[:] = scaler.transform(X_test)

X_train = X_train.as_matrix()
X_test = X_test.as_matrix()
X_val = X_val.as_matrix()
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

def build_model():
    model = keras.Sequential([
     keras.layers.Flatten(input_shape=(X_train.shape[1],X_train.shape[2])),
     keras.layers.Dense(512, activation=tf.nn.relu),
     keras.layers.Dense(256, activation=tf.nn.relu),
     keras.layers.Dense(128, activation=tf.nn.relu),
     keras.layers.Dense(64, activation=tf.nn.relu),
     keras.layers.Dense(32, activation=tf.nn.relu),
     keras.layers.Dense(16, activation=tf.nn.relu),
     keras.layers.Dense(1)
     ])
    return model

sample_weights=np.array( pd.concat([items["perishable"]] * 4) * 0.25 + 1 )

N_EPOCHS = 1000

val_pred = []
test_pred = []
for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    y = y_train[:, i]
    y_mean = y.mean()
    xv = X_val
    yv = y_val[:, i]
    model = build_model()
    model.compile(optimizer= tf.train.AdamOptimizer(0.001), loss='mse', metrics=['mse'])
    callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss')
    ]
    model.fit(X_train, y - y_mean, batch_size = 65536,epochs = N_EPOCHS, verbose=2,callbacks=callbacks, validation_data=(xv,yv-y_mean))
    val_pred.append(model.predict(X_val)+y_mean)
    test_pred.append(model.predict(X_test)+y_mean)

print("Making submission...")
y_test = np.array(test_pred).squeeze(axis=2).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_train.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv('NN_prediction.csv', float_format='%.4f', index=None)
