import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from warnings import filterwarnings
filterwarnings('ignore')

# Data
features = pd.read_csv('dengue_features_train.csv', index_col=[0, 1, 2])
label = pd.read_csv('dengue_labels_train.csv', index_col=[0, 1, 2])

# SJ and IQ
sj_features, iq_features = features.loc['sj'], features.loc['iq']
sj_label, iq_label = label.loc['sj'], label.loc['iq']

# NaN fill
sj_features.fillna(method='ffill', inplace=True)
iq_features.fillna(method='ffill', inplace=True)

# feature engineering
sj_features['station_range_temp_c'] = sj_features['station_max_temp_c'] - \
    sj_features['station_min_temp_c']
iq_features['station_range_temp_c'] = iq_features['station_max_temp_c'] - \
    iq_features['station_min_temp_c']
sj_features.drop(columns=['station_max_temp_c',
                          'station_min_temp_c'], axis=1, inplace=True)
iq_features.drop(columns=['station_max_temp_c',
                          'station_min_temp_c'], axis=1, inplace=True)
print(sj_features.columns)

# features used
sj = ['reanalysis_specific_humidity_g_per_kg',
      'reanalysis_dew_point_temp_k', 'ndvi_se', 'reanalysis_tdtr_k', 'ndvi_ne', 'station_range_temp_c']

iq = ['reanalysis_specific_humidity_g_per_kg',
      'reanalysis_dew_point_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_tdtr_k', 'ndvi_sw', 'station_range_temp_c']

sj_features = sj_features[sj]
iq_features = iq_features[iq]

# Merge
sj_data = sj_features.merge(sj_label, on=['year', 'weekofyear'])
#sj_data.drop(columns=['week_start_date'], axis=1, inplace=True)
iq_data = iq_features.merge(iq_label, on=['year', 'weekofyear'])
#iq_data.drop(columns=['week_start_date'], axis=1, inplace=True)

# Train test split
data_list = [sj_data, iq_data]
for idx in range(len(data_list)):
    if idx == 0:
        X_sj = sj_data.drop(columns='total_cases', axis=1)
        y_sj = sj_data['total_cases']
    elif idx == 1:
        X_iq = iq_data.drop(columns='total_cases', axis=1)
        y_iq = iq_data['total_cases']


sj_x_train, sj_x_test, sj_y_train, sj_y_test = train_test_split(
    X_sj, y_sj, test_size=0.2, random_state=42)
iq_x_train, iq_x_test, iq_y_train, iq_y_test = train_test_split(
    X_iq, y_iq, test_size=0.2, random_state=42)

print(f'\nSJ X train shape:{sj_x_train.shape}  SJ y train shape:{sj_y_train.shape}  SJ X test shape:{sj_x_test.shape}  SJ y test shape:{sj_y_test.shape}')
print(f'\nIQ X train shape:{iq_x_train.shape}  IQ y train shape:{iq_y_train.shape}  IQ X test shape:{iq_x_test.shape}  IQ y test shape:{iq_y_test.shape}')


# Scaled
x_train = [sj_x_train, iq_x_train]
x_test = [sj_x_test, iq_x_test]

scaler = MinMaxScaler()
for idx in range(len(x_train)):
    if idx == 0:
        sj_scaled = scaler.fit_transform(sj_x_train)
    elif idx == 1:
        iq_scaled = scaler.fit_transform(iq_x_train)

scaled_data = [sj_scaled, iq_scaled]

# XGBoost


class Model:
    def training(self, X, y, estimator):
        estimator.fit(X, y)
        X_predict = estimator.predict(X)
        mae = mean_absolute_error(y, X_predict)
        print(f'Training set Mean Absolute Error :{mae}')

    def validation(self, X, y, estimator):
        scores = cross_val_score(
            estimator, X, y, cv=10, scoring='neg_mean_absolute_error')
        negative_score = -scores
        print(f'10 fold validation scores :{negative_score}')
        print(f'\n mean :{negative_score.mean()}')


# Models
xgb = XGBRegressor(max_depth=3, learning_rate=0.01,
                   n_estimators=150, subsample=0.9, min_child_weight=2, reg_alpha=0.7, verbosity=0)
xg = Model()

for idx in range(len(scaled_data)):
    if idx == 0:
        print('\nSJ')
        sj_train = xg.training(sj_scaled, sj_y_train, xgb)
    elif idx == 1:
        print('\nIQ')
        iq_train = xg.training(iq_scaled, iq_y_train, xgb)


for idx in range(len(scaled_data)):
    if idx == 0:
        print('\nSJ')
        sj_val = xg.validation(sj_scaled, sj_y_train, xgb)
    elif idx == 1:
        print('\nIQ')
        iq_val = xg.validation(iq_scaled, iq_y_train, xgb)
