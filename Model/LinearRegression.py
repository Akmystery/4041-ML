import pandas as pd
import numpy as np
from sklearn import linear_model

train = pd.read_csv('../Data/train_set_short.csv')

regr = linear_model.LinearRegression()

dummy = np.array([train.item_nbr])
shop_number = np.reshape(dummy,(-1,1))
dum = np.array([train.unit_sales])
units = np.reshape(dum,(-1,1))
regr.fit(shop_number,units)
regr.predict(np.array([[96995]]))

dum_pro = np.array([train.onpromotion])
promotion = np.reshape(dum_pro,(-1,1))
number_promotion = np.concatenate((shop_number,promotion),axis=1)
regr.fit(number_promotion,units)
regr.predict(np.array([[96995,1]]))
regr.predict(np.array([[96995,0]]))
