import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


excel = "Vote.xlsx"
sheet = "Sheet1"

data = get_data(excel,sheet)
'''

D1 = data_mean(data['AGE295214'])
D2 = data_mean(data['PST045214'])
STD1 = data_std(data['AGE295214'])
STD2 = data_std(data['PST045214'])

Euc2 = euclidean_d(D1,D2)
print("Mean of AGE295214:",D1)
print("Mean of PST045214:",D2)
print("Std of AGE295214:",STD1)
print("Std of PST045214:",STD2)
print("distance between AGE295214,PST045214:",Euc2)



'''
#A2
'''
d1,d2 = np.histogram(data['AGE295214'],bins=10)
plt.hist(data['AGE295214'],bins=d2)
plt.show()
'''
#A3

def minkowski_d(v1,v2,r):
    xdy = [abs(v1[i]-v2[i])**r for i in range(len(v1))]
    sxdy = sum(xdy)
    return sxdy**(1/r)

md = [minkowski_d(data['AGE295214'],data['AGE135214'],i) for i in range(1,11)]
print(md)


# CHange to an ml data set as this does not have classes
#A4
