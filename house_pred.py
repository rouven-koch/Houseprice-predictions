#------------------------------------------------------------------------------
# House price predictions
#------------------------------------------------------------------------------
# General Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Sklearn libraries
from sklearn.preprocessing   import MinMaxScaler, StandardScaler
from sklearn.preprocessing   import Imputer
from sklearn.preprocessing   import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics         import confusion_matrix

# Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.models as km

#------------------------------------------------------------------------------
# Preprocessing
#------------------------------------------------------------------------------
# Loading data
traindata = pd.read_csv('train.csv')
# !test!
#traindata = traindata[traindata.GrLivArea < 4500]
testdata  = pd.read_csv('test.csv')
X_train   = traindata.iloc[:, [1,3,4,17,18,19,61,62,27,46,49,36,37,38,43,44,45,52]].values
y_train   = traindata.iloc[:, -1].values
#X_test   = testdata.iloc[:, [1,3,4,17,18,19,36,37,38,43,44,45,46,49,50,51,52,54,56,61,62]].values
X_test    = testdata.iloc[:, [1,3,4,17,18,19,61,62,27,46,49,36,37,38,43,44,45,52]].values

# Preprocessing traindata
le = LabelEncoder()
X_train[:,8] = le.fit_transform(X_train[:, 8])
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) 
imputer = imputer.fit(X_train[:, :])
X_train[:, :] = imputer.transform(X_train[:, :])
ohe = OneHotEncoder(categorical_features = [8])
X_train = ohe.fit_transform(X_train).toarray()
X_train = X_train[: , 1:]

# Preprocessing testdata
le_2 = LabelEncoder()
X_test[:,8] = le_2.fit_transform(X_test[:, 8])
imputer_2 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) 
imputer_2 = imputer_2.fit(X_test[:,:])
X_test[:, :] = imputer_2.transform(X_test[:, :])
ohe_2 = OneHotEncoder(categorical_features = [8])
X_test = ohe_2.fit_transform(X_test).toarray()
X_test = X_test[: , 1:]

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

#------------------------------------------------------------------------------
# Building the ANN (regressor)
#------------------------------------------------------------------------------
# Function for ANNs
def build_regressor():
	model = Sequential()
	model.add(Dense(units = 10, kernel_initializer='normal', activation='relu', input_dim = 6))
	model.add(Dense(units =  5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(units =  1, kernel_initializer='normal'))
	model.compile(optimizer='adam', loss='mean_squared_error',  metrics = ['accuracy'] )
	return model

# Training of the ANN
regressor_ann = build_regressor()
regressor_ann.fit(X_train, y_train, batch_size = 16, epochs = 100)
y_pred_ann = regressor_ann.predict(X_test)

#------------------------------------------------------------------------------
# Building Random Forest model
#------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor_rf.fit(X_train, y)
y_pred_rf = regressor_rf.predict(X_test)
accuracies_rf = cross_val_score(estimator = regressor_rf, X = X_train, y = y_train, cv = 10)
accuracies_rf.mean()

#------------------------------------------------------------------------------
# Building XGBoost model
#------------------------------------------------------------------------------
from xgboost import XGBRegressor
regressor_xg = XGBRegressor(gamma=0.01, learning_rate=0.2, max_delta_step=0,  max_depth=3)
regressor_xg.fit(X_train, y)
y_pred_xg = regressor_xg.predict(X_test)
accuracies_xg = cross_val_score(estimator = regressor_xg, X = X_train, y = y_train, cv = 10)
accuracies_xg.mean()
#accuracies_xg.std()

#------------------------------------------------------------------------------
# Building SVM model
#------------------------------------------------------------------------------
sc_y = StandardScaler()
y_svr = sc_y.fit_transform(y_train.reshape(-1,1))
from sklearn.svm import SVR
regressor_svr = SVR(kernel = 'rbf', gamma = 'auto_deprecated')
regressor_svr.fit(X_train, y_svr)
y_pred_svr = regressor_svr.predict(X_test)
y_pred_svr = sc_y.inverse_transform(y_pred_svr)
accuracies_xg = cross_val_score(estimator = regressor_svr, X = X_train, y = y_svr, cv = 10)
accuracies_xg.mean()

#------------------------------------------------------------------------------
# Building multi-linear model
#------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
regressor_lin = LinearRegression()
regressor_lin.fit(X_train, y_train)
y_pred_lin = regressor_lin.predict(X_test)
accuracies_lin = cross_val_score(estimator = regressor_lin, X = X_train, y = y_train, cv = 10)
accuracies_lin.mean()

#------------------------------------------------------------------------------
# "Stacking" of models
#------------------------------------------------------------------------------
y_tot = np.exp((np.log(y_pred_rf) + np.log(y_pred_xg))  / 2 )
#------------------------------------------------------------------------------
# Save and make ready for submission
#------------------------------------------------------------------------------
y_tot = np.expm1(y_tot)
output    = pd.DataFrame({'Id': testdata.Id, 'SalePrice': y_tot})
output_2  = pd.DataFrame({'Id': testdata.Id, 'SalePrice': y_pred_xg})
output.to_csv('my_submission.csv', index=False)
output_2.to_csv('my_submission_2.csv', index=False)

#------------------------------------------------------------------------------
# Visualization Tools
#------------------------------------------------------------------------------
traindata['SalePrice'].hist(bins = 50)

# ! rumprobieren !
train = traindata[traindata.GrLivArea < 4500]
train = traindata
train.reset_index(drop=True, inplace=True)
train["SalePrice"] = np.log1p(train["SalePrice"])
y = train['SalePrice'].reset_index(drop=True)
train['SalePrice'].hist(bins = 40)
    


