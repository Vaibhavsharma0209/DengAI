import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from warnings import filterwarnings
import tensorflow as tf
from tensorflow import keras
filterwarnings('ignore')

# Data
features = pd.read_csv('dengue_features_train.csv', index_col=[0, 1, ])
label = pd.read_csv('dengue_labels_train.csv', index_col=[0, 1, ])

# SJ and IQ
sj_features, iq_features = features.loc['sj'], features.loc['iq']
sj_label, iq_label = label.loc['sj'], label.loc['iq']
sj_features.drop(columns=['week_start_date'], axis=1, inplace=True)
iq_features.drop(columns=['week_start_date'], axis=1, inplace=True)


# NaN fill
sj_features.fillna(method='ffill', inplace=True)
iq_features.fillna(method='ffill', inplace=True)


# feature engineering
sj_features['station_range_temp_c'] = sj_features['station_max_temp_c'] - \
    sj_features['station_min_temp_c']
iq_features['station_range_temp_c'] = iq_features['station_max_temp_c'] - \
    iq_features['station_min_temp_c']
sj_features.drop(columns=['station_max_temp_c',
                          'station_min_temp_c', 'station_avg_temp_c'], axis=1, inplace=True)
iq_features.drop(columns=['station_max_temp_c',
                          'station_min_temp_c', 'station_avg_temp_c'], axis=1, inplace=True)

sj_features['reanalysis_range_air_temp_k'] = sj_features['reanalysis_max_air_temp_k'] - \
    sj_features['reanalysis_min_air_temp_k']
iq_features['reanalysis_range_air_temp_k'] = iq_features['reanalysis_max_air_temp_k'] - \
    iq_features['reanalysis_min_air_temp_k']
sj_features.drop(columns=['reanalysis_max_air_temp_k',
                          'reanalysis_min_air_temp_k', 'reanalysis_tdtr_k', 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k'], axis=1, inplace=True)
iq_features.drop(columns=['reanalysis_max_air_temp_k',
                          'reanalysis_min_air_temp_k', 'reanalysis_tdtr_k', 'reanalysis_avg_temp_k', 'reanalysis_air_temp_k', 'reanalysis_dew_point_temp_k'], axis=1, inplace=True)


ndvi = ['ndvi_sw', 'ndvi_se', 'ndvi_nw', 'ndvi_ne']
precip = ['station_precip_mm', 'precipitation_amt_mm',
          'reanalysis_sat_precip_amt_mm']
sj_ndvi_data, sj_precip_data = sj_features[ndvi], sj_features[precip]
sj_ndvi_data['minimum_ndvi'], sj_precip_data['precip'] = sj_ndvi_data.min(
    axis=1), sj_precip_data.max(axis=1)

iq_ndvi_data, iq_precip_data = iq_features[ndvi], iq_features[precip]
iq_ndvi_data['minimum_ndvi'], iq_precip_data['precip'] = iq_ndvi_data.min(
    axis=1), iq_precip_data.max(axis=1)

sj_features['minimum_ndvi'], sj_features['precip'] = sj_ndvi_data['minimum_ndvi'].copy(
), sj_precip_data['precip'].copy()
iq_features['minimum_ndvi'], iq_features['precip'] = iq_ndvi_data['minimum_ndvi'].copy(
), iq_precip_data['precip'].copy()

sj_features.drop(columns=ndvi, axis=1, inplace=True)
sj_features.drop(columns=precip, axis=1, inplace=True)

iq_features.drop(columns=ndvi, axis=1, inplace=True)
iq_features.drop(columns=precip, axis=1, inplace=True)

print(sj_features.columns)


# Merge
sj_data = sj_features.merge(sj_label, on=['year', 'weekofyear'])
iq_data = iq_features.merge(iq_label, on=['year', 'weekofyear'])


# iq_data['total_cases'].hist()
# plt.show()

'''

# Seaborn
# print(plt.colormaps())
corr_sj = sj_data.corr()
sns.heatmap(corr_sj, cmap='gist_heat_r', )
plt.xticks(size=7)
plt.yticks(size=7)
plt.show()

corr_iq = iq_data.corr()
sns.heatmap(corr_iq, cmap='gist_heat_r', )
plt.xticks(size=7)
plt.yticks(size=7)
plt.show()
'''
sj_index = np.all(stats.zscore(sj_data) < 3.5, axis=1)
sj_data = sj_data.loc[sj_index]

iq_index = np.all(stats.zscore(iq_data) < 3.5, axis=1)
iq_data = iq_data.loc[iq_index]

# iq_data['total_cases'].hist()
# plt.show()

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

scaler = StandardScaler()
for idx in range(len(x_train)):
    if idx == 0:
        sj_scaled = scaler.fit_transform(sj_x_train)
    elif idx == 1:
        iq_scaled = scaler.fit_transform(iq_x_train)

scaled_data = [sj_scaled, iq_scaled]


# class

'''
class Model:
    def training(self, X, y, estimator):
        estimator.fit(X, y)
        print(estimator)
        X_predict = estimator.predict(X)
        mae = mean_absolute_error(y, X_predict)
        print(f'Training set mean Absolute Error :{mae}')

    def validation(self, X, y, estimator):
        scores = cross_val_score(
            estimator, X, y, cv=10, scoring='neg_mean_absolute_error')
        negative_score = -scores
        print(f'10 fold validation scores :{negative_score}')
        print(f'\n mean :{negative_score.mean()}')

    def prediction_test(self, test_X, y_test, estimator):
        X_test_predict = estimator.predict(test_X)
        mae_test = mean_absolute_error(y_test, X_test_predict)
        print(mae_test)


# Models
xg = Model()
xgbsj = XGBRegressor(seed=42, verbosity=0, colsample_bytree=0.9, learning_rate=0.01,
                     max_depth=7, min_child_weight=1, n_estimators=170, reg_alpha=0.7, subsample=0.9, objective='reg:tweedie')
xgbiq = XGBRegressor(seed=42, verbosity=0, colsample_bytree=0.5, learning_rate=0.01,
                     max_depth=7, min_child_weight=1, n_estimators=170, reg_alpha=0.1, subsample=0.7, objective='reg:tweedie')
mlp_sj = MLPRegressor(hidden_layer_sizes=(3, 2, 1,),
                      learning_rate='adaptive', solver='adam', random_state=42, nesterovs_momentum=False,)
mlp_iq = MLPRegressor(hidden_layer_sizes=(9, 5, 1,),
                      learning_rate='adaptive', solver='adam', random_state=42, nesterovs_momentum=False,)

'''

# NN
model = keras.models.Sequential([keras.layers.Dense(64, activation='softplus', input_shape=sj_scaled.shape[1:]),
                                 keras.layers.Dense(32, activation='softplus'), keras.layers.Dense(1)])

model.compile(loss='mean_absolute_error', optimizer='sgd')
history = model.fit(sj_scaled, sj_y_train, validation_split=0.1, epochs=200)

print(history.history)

histo = pd.DataFrame(history.history)
histo.plot()
plt.grid(True)
plt.show()

for idx in range(len(scaled_data)):
    if idx == 0:
        print('\nSJ')
        sj_train = xg.training(sj_scaled, sj_y_train, mlp_sj)
    elif idx == 1:
        print('\nIQ')
        iq_train = xg.training(iq_scaled, iq_y_train, xgbiq)

for idx in range(len(scaled_data)):
    if idx == 0:
        print('\nSJ')
        sj_val = xg.validation(sj_scaled, sj_y_train, mlp_sj)
    elif idx == 1:
        print('\nIQ')
        iq_val = xg.validation(iq_scaled, iq_y_train, xgbiq)

'''
# GridSearchCV
param = {'max_depth': [3, 5, 7], 'learning_rate': [0.01], 'n_estimators': [170], 'subsample': [
    0.7, 0.5, 0.9], 'min_child_weight': [1, 2, 3], 'reg_alpha': [0.7, 0.9, 0.5], 'colsample_bytree': [0.9, 0.5, 0.7], }

grid_search = GridSearchCV(xgb, param_grid=param, cv=10,
                           scoring='neg_mean_absolute_error', return_train_score=True)
grid_search.fit(iq_scaled, iq_y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
'''

# Scaling
x_test = [sj_x_test, iq_x_test]

for idx in range(len(x_test)):
    if idx == 0:
        sj_scaled_test = scaler.fit_transform(sj_x_test)
    elif idx == 1:
        iq_scaled_test = scaler.fit_transform(iq_x_test)


test_scaled_data = [sj_scaled_test, iq_scaled_test]

for idx in range(len(test_scaled_data)):
    if idx == 0:
        sj_test_predict = xg.prediction_test(sj_scaled_test, sj_y_test, mlp_sj)
    elif idx == 1:
        iq_test_predict = xg.prediction_test(iq_scaled_test, iq_y_test, xgbiq)


mlp_sj.fit(sj_scaled, sj_y_train)
mlp_iq.fit(iq_scaled, iq_y_train)

# Dengue feature test

test_features = pd.read_csv('dengue_features_test.csv', index_col=[0, 1, ])
sj_test_features, iq_test_features = test_features.loc['sj'], test_features.loc['iq']
sj_test_features.drop(columns=['week_start_date'], axis=1, inplace=True)
iq_test_features.drop(columns=['week_start_date'], axis=1, inplace=True)

# NaN fill
sj_test_features.fillna(method='ffill', inplace=True)
iq_test_features.fillna(method='ffill', inplace=True)

# feature engineering
sj_test_features['station_range_temp_c'] = sj_test_features['station_max_temp_c'] - \
    sj_test_features['station_min_temp_c']
iq_test_features['station_range_temp_c'] = iq_test_features['station_max_temp_c'] - \
    iq_test_features['station_min_temp_c']
sj_test_features.drop(columns=['station_max_temp_c',
                               'station_min_temp_c', 'station_avg_temp_c'], axis=1, inplace=True)
iq_test_features.drop(columns=['station_max_temp_c',
                               'station_min_temp_c', 'station_avg_temp_c'], axis=1, inplace=True)

sj_test_features['reanalysis_range_air_temp_k'] = sj_test_features['reanalysis_max_air_temp_k'] - \
    sj_test_features['reanalysis_min_air_temp_k']
iq_test_features['reanalysis_range_air_temp_k'] = iq_test_features['reanalysis_max_air_temp_k'] - \
    iq_test_features['reanalysis_min_air_temp_k']
sj_test_features.drop(columns=['reanalysis_max_air_temp_k',
                               'reanalysis_min_air_temp_k', 'reanalysis_tdtr_k', 'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k'], axis=1, inplace=True)
iq_test_features.drop(columns=['reanalysis_max_air_temp_k',
                               'reanalysis_min_air_temp_k', 'reanalysis_tdtr_k', 'reanalysis_avg_temp_k', 'reanalysis_air_temp_k', 'reanalysis_dew_point_temp_k'], axis=1, inplace=True)


sj_test_ndvi_data, sj_test_precip_data = sj_test_features[ndvi], sj_test_features[precip]
sj_test_ndvi_data['minimum_ndvi'], sj_test_precip_data['precip'] = sj_test_ndvi_data.min(
    axis=1), sj_test_precip_data.max(axis=1)

iq_test_ndvi_data, iq_test_precip_data = iq_test_features[ndvi], iq_test_features[precip]
iq_test_ndvi_data['minimum_ndvi'], iq_test_precip_data['precip'] = iq_test_ndvi_data.min(
    axis=1), iq_test_precip_data.max(axis=1)

sj_test_features['minimum_ndvi'], sj_test_features['precip'] = sj_test_ndvi_data['minimum_ndvi'].copy(
), sj_test_precip_data['precip'].copy()
iq_test_features['minimum_ndvi'], iq_test_features['precip'] = iq_test_ndvi_data['minimum_ndvi'].copy(
), iq_test_precip_data['precip'].copy()

sj_test_features.drop(columns=ndvi, axis=1, inplace=True)
sj_test_features.drop(columns=precip, axis=1, inplace=True)

iq_test_features.drop(columns=ndvi, axis=1, inplace=True)
iq_test_features.drop(columns=precip, axis=1, inplace=True)


# Scaled
x_test_features = [sj_test_features, iq_test_features]

for idx in range(len(x_test_features)):
    if idx == 0:
        sj_test_actual = scaler.fit_transform(sj_test_features)
    elif idx == 1:
        iq_test_actual = scaler.fit_transform(iq_test_features)


sj_prediction = mlp_sj.predict(sj_test_actual)
iq_prediction = mlp_iq.predict(iq_test_actual)
print(f'\n{sj_prediction.shape} {iq_prediction.shape}')

sj_final = pd.DataFrame(sj_prediction)
iq_final = pd.DataFrame(iq_prediction)

final = pd.concat([sj_final, iq_final], axis=0)
print(f'\n{final.shape}')

final.to_csv('mlp_recent.csv')

