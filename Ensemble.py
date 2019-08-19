import pandas as pd
import numpy as np

nn_model = pd.read_csv('NN_prediction.csv')
lgm_model = pd.read_csv('LGBM_prediction.csv')


nn_lgm = 0.40*nn_model['unit_sales']+0.60*lgm_model['unit_sales']
lgm_model['unit_sales'] = nn_lgm
lgm_model.to_csv('Final.csv', float_format='%.4f', index=None)
