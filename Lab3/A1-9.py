import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#A1
def get_data(excel,sheet):
    d = pd.read_excel(excel,sheet_name=sheet)
    return d

def data_mean(data):
    return np.mean(data)

def data_std(data):
    return np.std(data)

def euclidean_d(D1,D2):
    return np.linalg.norm(D1-D2)


excel = "Titanic.xlsx"
sheet = "Sheet1"

data = get_data(excel,sheet)



D1 = data_mean(data['Age'])
D2 = data_mean(data['Fare'])
STD1 = data_std(data['Age'])
STD2 = data_std(data['Fare'])

Euc2 = euclidean_d(D1,D2)
print("Mean of Age:",D1)
print("Mean of Fare:",D2)
print("Std of Age:",STD1)
print("Std of Fare:",STD2)
print("distance between Age,Fare:",Euc2)




#A2

d1,d2 = np.histogram(data['Fare'],bins=[10,20,30,40,50,60,70,80,90])
print(d2)
plt.hist(data['Age'],bins=d2)
plt.show()

#A3

def minkowski_d(v1,v2,r):

    xdy = [abs(v1[i]-v2[i])**r for i in range(len(v1))]

    sxdy = sum(xdy)
    return sxdy**(1/r)

data['Age'] = data['Age'].fillna(0)
data['Fare'] = data['Fare'].fillna(0)


md = [minkowski_d(data['Age'],data['Fare'],i) for i in range(1,11)]
print(md)
plt.scatter(range(len(md)), md, c='r', marker='o')  # red circles
plt.show()

ad
#A4

X = data[['Fare']]
y = data['Survived']
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



