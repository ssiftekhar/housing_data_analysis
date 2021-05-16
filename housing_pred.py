
#import necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv('kc_house_data.csv')
df_important_columns = df[[
    'price', 'sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms', 'lat', 'long',
    'view', 'waterfront', 'grade']]

#create train and test sets
df_train, df_test = train_test_split(df_important_columns, test_size=0.2)

#create variables for the training data sets
df_train_y = df_train['price']
df_train_features = df_train[['sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms', 'lat', 'long',
    'view', 'waterfront', 'grade']]

#create simpleimputer for potential nulls/nones
house_imputer = SimpleImputer(strategy = 'mean')
house_imputer.fit(df_train_features)
np_train_features = house_imputer.transform(df_train_features) #fill missing values
features_column_names = list(df_train_features)
df_train_features_filled = pd.DataFrame(np_train_features, columns=features_column_names)

#log transformation
df_train_features_filled['log_sqft_living'] = np.log(df_train_features_filled['sqft_living'])
del df_train_features_filled['sqft_living']
df_train_features_filled['log_sqft_lot'] = np.log(df_train_features_filled['sqft_lot'])
del df_train_features_filled['sqft_lot']

df_train_features_filled.hist(bins=50) #just to see how it looks

#splitting our training set into scaling types
df_train_features_filled_no_scl = df_train_features_filled[['view', 'waterfront']]
df_train_features_filled_minmax = df_train_features_filled[['bedrooms', 'grade', 'lat', 'long']]
df_train_features_filled_stand_scl = df_train_features_filled[['bathrooms', 'log_sqft_living', 'log_sqft_lot']]

#numpy representation
train_features_filled_no_scl = df_train_features_filled_no_scl.values

#minmax of some of the chosen columns
house_minmax = MinMaxScaler()
house_minmax.fit(df_train_features_filled_minmax)
train_features_minmax= house_minmax.transform(df_train_features_filled_minmax)

#standard sclaer on chosen columns
house_stand_scl = StandardScaler()
house_stand_scl.fit(df_train_features_filled_stand_scl)
train_features_stand_scl = house_stand_scl.transform(df_train_features_filled_stand_scl)

#concat all the scaling dataframes to X_train so that we can now train the model
X_train = np.concatenate([train_features_filled_no_scl, train_features_minmax, train_features_stand_scl], axis=1)
y_train = np.c_[df_train_y]


#training the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_train_pred = lr_model.predict(X_train)
mean_absolute_error(y_train, y_train_pred) #140513.5465281011
np.sqrt(mean_squared_error(y_train, y_train_pred)) #227811.89159016978

'''test data'''

df_test_y = df_test['price']
df_test_features = df_test[[
    'sqft_living', 
    'sqft_lot', 
    'bedrooms', 
    'bathrooms', 
    'lat',
    'long',
    'view',
    'waterfront',
    'grade']]

np_test_features = house_imputer.transform(df_test_features)

# numpy to dataframe
df_test_features_filled = pd.DataFrame(np_test_features, columns=features_column_names)

df_test_features_filled['sqft_living_log'] = np.log(df_test_features_filled['sqft_living'])
del df_test_features_filled['sqft_living']
df_test_features_filled['sqft_lot_log'] = np.log(df_test_features_filled['sqft_lot'])
del df_test_features_filled['sqft_lot']

df_test_features_filled_no_scl = df_test_features_filled[['view', 'waterfront']]
df_test_features_filled_minmax_scl = df_test_features_filled[['bedrooms', 'grade', 'lat', 'long']]
df_test_features_filled_std_scl = df_test_features_filled[['bathrooms', 'sqft_living_log', 'sqft_lot_log']]


test_features_no_scl = df_test_features_filled_no_scl.values

test_features_minmax = house_minmax.transform(df_test_features_filled_minmax_scl)

test_features_std = house_stand_scl.transform(df_test_features_filled_std_scl)

X_test = np.concatenate([test_features_no_scl, test_features_minmax, test_features_std], axis=1)
y_test = np.c_[df_test_y]

'''test vs train'''

#linear regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
#train
y_train_pred = lr_model.predict(X_train)
mean_absolute_error(y_train, y_train_pred) #140513.5465281011
np.sqrt(mean_squared_error(y_train, y_train_pred)) #227811.89159016978

#test

y_pred_test = lr_model.predict(X_test)
mean_absolute_error(y_test, y_pred_test) #143571.53874968764 fra nmy lr_model
np.sqrt(mean_squared_error(y_test, y_pred_test)) #229605.95471760462 fra ny lr_model


#testing k-neighrest n=5
from sklearn.neighbors import KNeighborsRegressor

#train
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

y_knn_model = knn_model.predict(X_train)
print(mean_absolute_error(y_train, y_knn_model)) #87984.75023713129
print(np.sqrt(mean_squared_error(y_train, y_knn_model))) #160587.23544563883

#test
y_knn_model_test = knn_model.predict(X_test)
print(mean_absolute_error(y_test, y_knn_model_test)) #112442.63451306963
print(np.sqrt(mean_squared_error(y_test, y_knn_model_test))) #205594.56785914666
