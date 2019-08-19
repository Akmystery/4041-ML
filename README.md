#Time-series Forecasting for Corporación Favorita Grocery Sales Forecasting

This is the Kaggle Competition project, which is to predict the sales of items in every shop of Corporación Favorita Grocery.
We used the ensemble of LightGBM and NN models to make predictions.

Please find the detailed information about project in Project_Report.pdf.

Youtube link - https://youtu.be/DiWotiZURsw


You need to install LightGBM and Tensor Flow packages first. If you have installed Anaconda, you might not need to install numpy, panda and other packages. Otherwise, install Anaconda.

https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html
https://www.tensorflow.org/install

--------------------------------
How to run Preprocessing
--------------------------------

-> There are preprocessing.py and preprocessing_two.py

-> You have to download the data files from Kaggle before running python files.

#cmd to run python

- python "FileName"

- This steps is needed to be done before running ML models.

--------------------------
Forecasting
--------------------------

There are LGBM.py and NN.py. 

LGBM.py is for LightGBM model NN.py is for Neural Network. Run them first. 

NN.py is recommended to run at Google Colab unless you have strong GPU.

After they produce prediction files and run ensemble.py to produce the final.csv.

Final.csv is the final submission file!!!!!!
