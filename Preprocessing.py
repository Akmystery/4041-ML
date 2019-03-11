# importing csv module
import csv
import pandas as pd


original_train_data = pd.read_csv('Data/train.csv')

short_period_set = original_train_data.loc[(original_train_data['date'] >= '2017-06-01') & (original_train_data['date'] <= '2017-06-28')]

short_period_set.isna().sum() #making sure that there is nan value

long_period_set = original_train_data.loc[(original_train_data['date'] >= '2017-06-13') & (original_train_data['date'] <= '2017-08-08')]

long_period_set.isna().sum()

valid_set = original_train_data.loc[(original_train_data['date'] >= '2017-08-09') & (original_train_data['date'] <= '2017-08-15')]

valid_set.isna().sum()

one_year_set = original_train_data.loc[(original_train_data['date'] >= '2017-01-01') & (original_train_data['date'] <= '2017-12-31')]

one_year_set.isna().sum()

short_period_set_csv = short_period_set.to_csv(r'train_set_short.csv',index = None, header=True)

long_period_set_csv = long_period_set.to_csv(r'train_set_long.csv',index = None, header=True)

valid_set_csv = valid_set.to_csv(r'valid_set.csv',index=None, header=True)

one_year_set_csv = one_year_set.to_csv(r'train_set_one_year.csv',index=None,header=True)
