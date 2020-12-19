import pandas as pd
import numpy as np
import warnings
import pickle

warnings.filterwarnings("ignore")


dataset_2 = pd.read_csv("Admission_Predict.csv")
dataset_3 = pd.read_csv("Admission_Predict_Ver1.1.csv")
new_data = pd.concat([dataset_2,dataset_3] )



# MODELING
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.linear_model as lm
from sklearn import datasets,linear_model
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
#RMSE Score ---------------------
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#convert dataframe into matrix
dataArray = new_data.values

#splitting input features & o/p vars
X = dataArray[:, 1:8]
y = dataArray[:, 8:9]


#splitting training & testing
validation_size = 0.10
seed = 9 # used to assign random state (If random_state values is 10 then the algo will take the same value every time.)
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=validation_size, random_state=10)

rfr = RandomForestRegressor()
svr = SVR(kernel = 'rbf')

model_LR =linear_model.LinearRegression()

# Fit models
model_SVM = svr.fit(X_train, Y_train)
model_LR.fit(X_train, Y_train)
model_rf = rfr.fit(X_train, Y_train)

# Implimenting and calculating RMSE score of each model
prediction_svr = model_SVM.predict(X_test)
mse = mean_squared_error(Y_test, prediction_svr)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, prediction_svr)

print("Root Mean Squared Error for SVM : ",rmse)
print("R-Squared Error for SVM:", r2)


prediction_svr = model_rf.predict(X_test)
mse = mean_squared_error(Y_test, prediction_svr)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, prediction_svr)

print("Root Mean Squared Error for RF  : ",rmse)
print("R-Squared Error for RF:", r2)


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

prediction_svr = model_LR.predict(X_test)
mse = mean_squared_error(Y_test, prediction_svr)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, prediction_svr)

print("Root Mean Squared Error for LR : ",rmse)
print("R-Squared Error for LR:", r2)

from sklearn.neighbors import  KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=7) 
  
model_knn = knn.fit(X_train, Y_train) 

prediction_svr = knn.predict(X_test)
mse = mean_squared_error(Y_test, prediction_svr)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, prediction_svr)

print("Root Mean Squared Error for KNN : ",rmse)
print("R-Squared Error for KNN:", r2)
#---------------------------------------------------------------------------------------------

# Creating pickel file for each model.
pickle.dump(model_LR, open('model_LR.pkl', 'wb'))
pickle.dump(model_SVM, open('model_SVM.pkl', 'wb'))
pickle.dump(model_rf, open('model_rf.pkl', 'wb'))
pickle.dump(model_knn, open('model_knn.pkl', 'wb'))


