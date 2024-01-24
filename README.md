# ML-Project-Enefit
This is a machine learning project (and a kaggle competition) that aims to build an energy prediction model; for reducing the imbalance costs due to the increasing number of prosumers.

The detail description can be found in:
https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers

Some of the datasets are too large to upload, but they can be downloaded here:
https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/data

In general, there are 3 sections in the code:

**1. Data preparation:**
   Data cleaning, sorting and feature engineering, followed by data splitting for training and validation.
   
**2. XGBoost approach:**
   Applying XGBoost algorithm for training and evaluation of the model.
   
**3. LightGBM approach:**
   Applying LightGBM algorithm for training and evaluation of the model.

## 1. Data preparation and feature engineering
This section aims to combine all datasets into one that can be trained by the model.

First of all, let's load up all datasets:
![alt text](images/load_data.png)

For each dataset, its shape and first observation are printed out:
![alt text](images/load_data2.png)

Data cleaning, sorting and grouping are then performed (The code is too long, please look at the code).

Feature "data_block_id" is used for data splitting. Observations with data_block_id = 0 to 499 are for training while the rest are for validation. 78.25% of the data was used for training.

![alt text](images/tr_val_data_splitting.png)

From feature engineering, prediction can also be based on the energy usage ("target") certain days ago:
![alt text](images/features.png)

At the end, "Target" (energy comsumption/production amount) is the response variable, while 59 features are used as explanatory varibles:
![alt text](images/features2.png)

## 2. XGBoost
Applying XGBoost algorithm to train on the training set and generate a fit for prediction. Categorical data are allowed and 1500 gradient boosted trees are used. Note that mean absolute error (MAE) is calculated for evaluation on the model performance. The algorithm will stop earlier if there is no obvious decrease in loss value for 100 iterations (early_stopping_round = 100) instead of going through all iterations.
![alt text](images/xgboost.png)
![alt text](images/xgboost2.png)

A plot is produced to see both training and validation error. The best iteration is round #278 and the corresponding MAE is 
![alt text](images/xgboost3.png)

A horizontal bar plot is used to see the importance of feature (top 20). 
![alt text](images/xgboost4.png)

## 3. LightBGM
Similar structure as XGBoost section, but with LightBGM algorithm.
![alt text](images/lightgbm.png)
![alt text](images/lightgbm2.png)
![alt text](images/lightgbm3.png)
![alt text](images/lightgbm4.png)
