# importing csv module
import csv
import pandas as pd


original_train_data = pd.read_csv('Data/train.csv',dtype={'onpromotion': bool})

train_set = original_train_data.loc[(original_train_data['date'] >= '2017-04-01') & (original_train_data['date'] <= '2017-08-15')]

print(train_set.isna().sum())

train_set_csv = train_set.to_csv(r'train_set.csv',index = None, header=True)
