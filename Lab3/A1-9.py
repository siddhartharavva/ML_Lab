import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


#A1
def get_data(excel):
    d = pd.read_csv(excel)
    return d

def data_mean(data):
    return np.mean(data)

def data_std(data):
    return np.std(data)

def euclidean_d(D1,D2):
    return np.linalg.norm(D1-D2)


csv = "../fmri_gap_cnn_extracted_features.csv"


data = get_data(csv)



D1 = data_mean(data['fmri_feature_14'])
D2 = data_mean(data['fmri_feature_15'])
STD1 = data_std(data['fmri_feature_14'])
STD2 = data_std(data['fmri_feature_15'])

Euc2 = euclidean_d(D1,D2)
print("Mean of fmri_feature_14:",D1)
print("Mean of fmri_feature_15:",D2)
print("Std of fmri_feature_14:",STD1)
print("Std of fmri_feature_15:",STD2)
print("distance between fmri_feature_14,fmri_feature_15:",Euc2)




#A2


d1,d2 = np.histogram(data['fmri_feature_15'],bins=[0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55])

plt.hist(data['fmri_feature_15'],bins=d2)
plt.show()

#A3 

def minkowski_d(v1,v2,r):

    xdy = [abs(v1[i]-v2[i])**r for i in range(len(v1))]

    sxdy = sum(xdy)
    return sxdy**(1/r)

data['fmri_feature_14'] = data['fmri_feature_14'].fillna(0)
data['fmri_feature_15'] = data['fmri_feature_15'].fillna(0)


md = [minkowski_d(data['fmri_feature_14'],data['fmri_feature_15'],i) for i in range(1,11)]
print(md)
plt.scatter(range(len(md)), md, c='r', marker='o')  # red circles
plt.show()


#A4,A5

X = data[['fmri_feature_15']]
y = data['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)  

n =[ KNeighborsClassifier(n_neighbors=i) for i in range(1,11)]
for i in range(0,10):
    n[i].fit(X,y)


#A6
print("k = 3",n[2].score(X_test,y_test))

#A7
print("k = 3",n[2].predict(X_test))

#A8
acc = [n[i].score(X_test,y_test) for i in range(10)]

plt.scatter(range(len(acc)),acc,c='g',marker="*")
plt.show()


#A9

y_pred = n[2].predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (k=3):")
print(cm)


disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for k = 3")
plt.show()



