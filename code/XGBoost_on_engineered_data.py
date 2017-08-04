import numpy as np
import pandas as pd
import xgboost as xgb
import gc
import math

print('Loading data ...')

train = pd.read_csv('train_2016_v2.csv')
prop = pd.read_csv('properties_2016.csv')
sample = pd.read_csv('sample_submission.csv')

print('Binding to float32')

for c, dtype in zip(prop.columns, prop.dtypes):
	if dtype == np.float64:
		prop[c] = prop[c].astype(np.float32)

print('Creating training set ...')

df_train = train.merge(prop, how='left', on='parcelid')

train_df = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)

# I added this
train_df['actypenan'] = [True if math.isnan(x) else False for x in train_df['airconditioningtypeid']]
train_df['actype1'] = [True if x == 1 else False for x in train_df['airconditioningtypeid']]
train_df = pd.concat([train_df, pd.get_dummies(train_df['fips'], drop_first=True, prefix='fips')], axis=1)
del train_df['fips']
train_df['fireplacecnt'].fillna(0, inplace=True)
train_df['fullbathcnt'].fillna(0, inplace=True)
for i, size in enumerate(train_df['garagetotalsqft']):
    if size == 0:
        if train_df['garagecarcnt'][i] != 0:
            train_df['garagetotalsqft'][i] = np.nan
train_df['hashottuborspa'].fillna(False, inplace=True)
train_df['poolcnt'] = train_df['poolcnt'].map(lambda x: False if math.isnan(x) else True)
train_df['pooltypeid10'] = train_df['pooltypeid10'].map(lambda x: False if math.isnan(x) else True)
train_df['pooltypeid2'] = train_df['pooltypeid2'].map(lambda x: False if math.isnan(x) else True)
train_df['pooltypeid7'] = train_df['pooltypeid7'].map(lambda x: False if math.isnan(x) else True)
train_df = pd.concat([train_df, pd.get_dummies(train_df['regionidcounty'], drop_first=True, prefix='regionid')], axis=1)
del train_df['regionidcounty']
train_df['storytypeid'] = train_df['storytypeid'].map(lambda x: True if math.isnan(x) else False)
train_df['decktypeid'] = [True if x == 1 else False for x in train_df['decktypeid']]
train_df['unitcnt'].fillna(0, inplace=True)
train_df['numberofstories'].fillna(0, inplace=True)
train_df['fireplaceflag'] = train_df['fireplaceflag'].map(lambda x: False if math.isnan(x) else True)
train_df['taxvaluedollarcnt'].fillna(train_df['taxvaluedollarcnt'].mean(), inplace=True)
train_df['landtaxvaluedollarcnt'].fillna(train_df['landtaxvaluedollarcnt'].mean(), inplace=True)
train_df['taxamount'].fillna(train_df['taxamount'].mean(), inplace=True)
train_df['taxdelinquencyflag'] = train_df['taxdelinquencyflag'].map(lambda x: True if x == 'Y' else False)

prediction_df = train_df.dropna(axis=1,how='any')

from sklearn.ensemble import RandomForestRegressor
X = prediction_df
predict_cols = ['buildingqualitytypeid', 'calculatedbathnbr', 'calculatedfinishedsquarefeet', 'finishedsquarefeet12',
               'garagecarcnt', 'garagetotalsqft', 'lotsizesquarefeet', 'yearbuilt', 'structuretaxvaluedollarcnt',
               'censustractandblock']
for col in predict_cols:
    print(col)
    new_col = train_df[col].values
    missing = np.isnan(new_col)

    mod = RandomForestRegressor()
    mod.fit(X[~missing], new_col[~missing])

    mother_hs_pred = mod.predict(X[missing])
    mother_hs_pred

    new_col[missing] = mother_hs_pred
    train_df[col] = new_col

x_train = train_df

y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

from sklearn.model_selection import train_test_split
x_train, y_train, x_valid, y_valid = train_test_split(x_train, y_train, test_size=0.15, random_state=42)

print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

del x_train, x_valid; gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

del d_train, d_valid

print('Building test set ...')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')

del prop; gc.collect()

x_test = df_test[train_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

del df_test, sample; gc.collect()

d_test = xgb.DMatrix(x_test)

del x_test; gc.collect()

print('Predicting on test ...')

p_test = clf.predict(d_test)

del d_test; gc.collect()

sub = pd.read_csv('sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

print('Writing csv ...')
sub.to_csv('xgb_starter.csv', index=False, float_format='%.4f') # Thanks to @inversion
