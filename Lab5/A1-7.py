import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score 
from sklearn.metrics import calinski_harabasz_score 
from sklearn.metrics import davies_bouldin_score 


def get_data_csv(csv):
    d = pd.read_csv(csv)
    return d

#A1
'''
A1. If your project deals with a regression problem, please use one attribute of your dataset 
(X_train) along with the target values (y_train) for training a linear regression model. Sample code 
suggested below. 

'''

csv = "../fmri_gap_cnn_extracted_features.csv"
dataF = get_data_csv(csv)

X = dataF.iloc[1:790,14:15]
y = dataF.iloc[1:790,129]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


reg = LR().fit(X_train,y_train)



y_train_pred = reg.predict(X_train)

y_test_pred =  reg.predict(X_test)

#A2

'''
A2. Calculate MSE, RMSE,     and R2 scores for prediction made by the trained model in A1.  
Perform prediction on the test data and compare the metric values between train and test set.
'''

mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape * 100:.2f}%")
print(f"R²   : {r2:.4f}")



#A3

'''

A3. Repeat the exercises A1 and A2 with more than one attribute or all attributes. 


'''


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

'''
A4. Perform k-means clustering on your data. Please remove / ignore the target variable for 
performing clustering. Sample code suggested below.

'''


kmeans = KMeans(n_clusters=2, random_state=0,n_init="auto").fit(X_train.iloc[:,14:15])  
y_kmeans = kmeans.predict(X_test.iloc[:,14:15])





plt.figure(figsize=(8,6))
for x_val, y_val, y in zip(X_test.iloc[:,14], X_test.iloc[:,15], y_kmeans):
    color = 'red' if y <= 0.5 else 'blue'
    plt.scatter(x_val, y_val, c=color)
plt.show()


'''
A5. For the clustering done in A4, calculate the: (i) Silhouette Score, (ii) CH Score and (iii) DB Index.

'''


sScore = silhouette_score(X_train, kmeans.labels_) 
CHScore = calinski_harabasz_score(X_train, kmeans.labels_) 
DBScore = davies_bouldin_score(X_train, kmeans.labels_)

print('Silhouette score',sScore,'\ncalinski harabasz score',CHScore,'\ndavies bouldin score',DBScore)

'''
A6. Perform k-means clustering for different values of k. Evaluate the above scores for each k value. 
Make a plot of the values against the k value to determine the optimal cluster count. 
'''



kmeans = [KMeans(n_clusters=i, random_state=0,n_init="auto").fit(X_train.iloc[:,14:15]) for i in range(2,20)]

y_kmeans = [ kmeans[i].predict(X_test.iloc[:,14:15]) for i in range(18) ]


sScorek = [silhouette_score(X_train, kmeans[i].labels_) for i in range(18)]
CHScorek = [calinski_harabasz_score(X_train, kmeans[i].labels_) for i in range(18)]
DBScorek = [davies_bouldin_score(X_train, kmeans[i].labels_) for i in range(18)]
points = [i for i in range(2,20)]


plt.plot(points,sScorek,c='r')
plt.show()
plt.plot(points,CHScorek,c='g')
plt.show()
plt.plot(points,DBScorek,c='b')
plt.show()


distorsions = []
elbow = [distorsions.append(kmeans[i].inertia_) for i in range(18)]
plt.plot(distorsions)
plt.show()
