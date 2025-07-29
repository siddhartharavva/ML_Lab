import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def get_data_csv(csv):
    d = pd.read_csv(csv)
    return d
def get_data_excel(excel,sheet):
    d  = pd.read_excel(excel,sheet_name= sheet)
    return d

#A1

csv = "../fmri_gap_cnn_extracted_features.csv"
data = get_data_csv(csv)
print(data)

X = data[['fmri_feature_15']]
y = data['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)  

n = KNeighborsClassifier(n_neighbors=3) 
n.fit(X_train,y_train)


y_pred_test = n.predict(X_test)
y_pred_train = n.predict(X_train)


cm_test = confusion_matrix(y_test, y_pred_test)
cm_train = confusion_matrix(y_train, y_pred_train)

print("Confusion Matrix (k=3):")

print(cm_test,"\n",cm_train)


def sensitivityRecall(cm):
    return cm[1][1]/(cm[1][1]+cm[1][0])

def specificity(cm):
    return cm[0][0]/(cm[0][0]+cm[0][1])

def precision(cm):
    return cm[1][1]/(cm[1][1]+cm[0][1])

def accuracy(cm):
    return (cm[1][1]+cm[0][0])/(cm[1][1]+cm[0][1]+cm[0][0]+cm[1][0])

def FbScore(cm,b):
    precision = cm[1][1]/(cm[1][1]+cm[0][1])
    sensitivityRecall = cm[1][1]/(cm[1][1]+cm[1][0])

    return  ((1+b**2)*(precision*sensitivityRecall))/((b**2)*precision+sensitivityRecall)     



print("F1Score",FbScore(cm_test,1),"\n","sensitivityRecall",sensitivityRecall(cm_test),"\n","specificity",specificity(cm_test),"\n","precision",precision(cm_test),"\n","accuracy",accuracy(cm_test))



#A2
excel = "../Lab_session_Data.xlsx"
sheet = "Purchase data"

dataP = get_data_excel(excel , sheet)
X = dataP.iloc[:,1:4]
y = dataP.iloc[:,4]
print(X,'\n',y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)  

n = KNeighborsClassifier(n_neighbors=3)
n.fit(X_train,y_train)

