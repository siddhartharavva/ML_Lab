import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import accuracy_score

def get_data_csv(csv):
    d = pd.read_csv(csv)
    return d

#A1

csv = "../fmri_gap_cnn_extracted_features.csv"
dataF = get_data_csv(csv)

X = dataF.iloc[1:790,14:15]
y = dataF.iloc[1:790,129]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


reg = LR().fit(X_train,y_train)


y_train_pred = reg.predict(X_train)

y_test_pred =  reg.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape * 100:.2f}%")
print(f"R²   : {r2:.4f}")



#A1

csv = "../fmri_gap_cnn_extracted_features.csv"
dataF = get_data_csv(csv)

X = dataF.iloc[1:790,1:128]
y = dataF.iloc[1:790,129]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


reg = LR().fit(X_train,y_train)


y_train_pred = reg.predict(X_train)

y_test_pred =  reg.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("\n\nWith almost all attributes")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape * 100:.2f}%")
print(f"R²   : {r2:.4f}")

