import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


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


#A1


'''
A1. Evaluate the intraclass spread and interclass distances between the classes in your dataset. If 
your data deals with multiple classes, you can take any two classes. Steps below (refer below 
diagram for understanding): 
• Calculate the mean for each class (also called as class centroid) 
(Suggestion: You may use numpy.mean() function for finding the average vector for all 
vectors in a given class. Please define the axis property appropriately to use this function. EX: 
feat_vecs.mean(axis=0)) 
• Calculate spread (standard deviation) for each class 
(Suggestion: You may use numpy.std() function for finding the standard deviation vector 
for all vectors in a given class. Please define the axis property appropriately to use this 
function.) 
• Calculate the distance between mean vectors between classes 
(Suggestion: numpy.linalg.norm(centroid1 – centroid2) gives the Euclidean 
distance between two centroids.) 
 


'''


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
'''

A2. Take any feature from your dataset. Observe the density pattern for that feature by plotting the 
histogram. Use buckets (data in ranges) for histogram generation and study. Calculate the mean and 
variance from the available data.  
(Suggestion: numpy.histogram()gives the histogram data. Plot of histogram may be 
achieved with matplotlib.pyplot.hist())


'''



d1,d2 = np.histogram(data['fmri_feature_15'],bins=[0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55])

plt.hist(data['fmri_feature_15'],bins=d2)
plt.show()

#A3 
'''
A3. Take any two feature vectors from your dataset. Calculate the Minkwoski distance with r from 1 
to 10. Make a plot of the distance and observe the nature of this graph. 

'''

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

'''
A4. Divide dataset in your project into two parts – train & test set. To accomplish this, use the train-
test_split() function available in SciKit. See below sample code for help:

A5. Train a kNN classifier (k =3) using the training set obtained from above exercise. Following code 
for help:

'''


X = data[['fmri_feature_15']]
y = data['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)  

n =[ KNeighborsClassifier(n_neighbors=i) for i in range(1,11)]
for i in range(0,10):
    n[i].fit(X,y)


#A6
'''

A6. Test the accuracy of the kNN using the test set obtained from above exercise. Following code for 
help.

'''

print("k = 3",n[2].score(X_test,y_test))

#A7
'''
A7. Use the predict() function to study the prediction behavior of the classifier for test vectors.
 
Perform classification for a given vector using neigh.predict(<<test_vect>>). This shall produce the 
class of the test vector (test_vect is any feature vector from your test set).


'''


print("k = 3",n[2].predict(X_test))

#A8

'''

A8. Make k = 1 to implement NN classifier and compare the results with kNN (k = 3). Vary k from 1 to 
11 and make an accuracy plot.

'''

acc = [n[i].score(X_test,y_test) for i in range(10)]

plt.scatter(range(len(acc)),acc,c='g',marker="*")
plt.show()


#A9
'''
A9. Please evaluate confusion matrix for your classification problem. From confusion matrix, the 
other performance metrics such as precision, recall and F1-Score measures for both training and test 
data. Based on your observations, infer the models learning outcome (underfit / regularfit / overfit).

'''

y_pred = n[2].predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (k=3):")
print(cm)


disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for k = 3")
plt.show()




print("Train Accuracy for k=3:", n[2].score(X_train, y_train))