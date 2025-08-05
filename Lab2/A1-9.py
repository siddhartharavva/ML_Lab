import pandas as pd
import numpy as np
import statistics as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


#A1

'''
A1. Please refer to the “Purchase Data” worksheet of Lab Session Data.xlsx. Please load the data 
and segregate them into 2 matrices A & C (following the nomenclature of AX = C). Do the following 
activities. 
• What is the dimensionality of the vector space for this data? 
• How many vectors exist in this vector space? 
• What is the rank of Matrix A? 
• Using Pseudo-Inverse find the cost of each product available for sale.  
(Suggestion: If you use Python, you can use numpy.linalg.pinv() function to get a 
pseudo-inverse.)

'''

def get_data(excel,sheet):
    data = pd.read_excel(excel , sheet_name=sheet)
    return data

excel = '../Lab_session_Data.xlsx'
sheet = 'Purchase data'
data = get_data(excel,sheet)

A = data.iloc[:, 1:4]
C  = data.iloc[:,4:5]

print("Dimensionality of Vector Space A")
print(A.shape[1])

print("Number of Vectors in Vector Space A")
print(A.shape[0])

print("This is the rank")
print(np.linalg.matrix_rank(A))

print(f"Cost of each product in the order Candies, Mango, milk packet:{np.dot(np.linalg.pinv(A),C)}")

print(A,C)



#A2

'''
A2. Mark all customers (in “Purchase Data” table) with payments above Rs. 200 as RICH and others 
as POOR. Develop a classifier model to categorize customers into RICH or POOR class based on 
purchase behavior. 

'''



def classify_richness(data):
    for i in range(data.shape[0]):
        if(data['Payment (Rs)'][i] > 200):
            print("Is Rich")        
        else:
            print("Is Poor")

classify_richness(data)
'''
X = data['Payment (Rs)']
y = data['Rich']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)  

n =KNeighborsClassifier(n_neighbors=3)

n.fit(X_train,y_train)

print("k = 3",n.predict(X_test))
'''

#A3 

'''

A3. Please refer to the data present in “IRCTC Stock Price” data sheet of the above excel file. Do the 
following after loading the data to your programming platform. 
• Calculate the mean and variance of the Price data present in column D.  
(Suggestion: if you use Python, you may use statistics.mean() & 
statistics.variance() methods). 
• Select the price data for all Wednesdays and calculate the sample mean. Compare the mean 
with the population mean and note your observations. 
• Select the price data for the month of Apr and calculate the sample mean. Compare the 
mean with the population mean and note your observations. 
• From the Chg% (available in column I) find the probability of making a loss over the stock. 
(Suggestion: use lambda function to find negative values) 
• Calculate the probability of making a profit on Wednesday. 
• Calculate the conditional probability of making profit, given that today is Wednesday. 
• Make a scatter plot of Chg% data against the day of the week

'''
def mean(data):
    return st.mean(data['Price'])
    
def variance(data):
    return st.variance(data['Price'])

def mean_day(data, day='all'):
    if day == 'all':
        return st.mean(data['Price'])
    else:
            return st.mean(data['Price'][data['Day'] == day])
        
def mean_Mon(data, Mon):
    return st.mean(data['Price'][data['Month'] == Mon])

def loss_mean(data):
    loss = data['Chg%'].apply(lambda x: 1 if x < 0  else 0)
    print(loss)
    summation = sum(loss)
    return summation/len(loss)
    
    
excel = '../Lab_session_Data.xlsx'
sheet = 'IRCTC Stock Price'

data = get_data(excel,sheet)
print("Mean of Price: ", mean_day(data))
print("Variance of Price: ", variance(data))
print("Mean on Wednesday of Price: ", mean_day(data,'Wed'))
print("Mean on Month April of Price: ", mean_Mon(data,'Apr'))
print("Loss probability",loss_mean(data))


#A4

'''

A4. Data Exploration: Load the data available in “thyroid0387_UCI” worksheet. Perform the 
following tasks: 
• Study each attribute and associated values present. Identify the datatype (nominal etc.) 
for the attribute. 
• For categorical attributes, identify the encoding scheme to be employed. (Guidance: 
employ label encoding for ordinal variables while One-Hot encoding may be employed 
for nominal variables). 
• Study the data range for numeric variables. 
• Study the presence of missing values in each attribute. 
• Study presence of outliers in data.  
• For numeric variables, calculate the mean and variance (or standard deviation).

'''

excel = '../Lab_session_Data.xlsx'
sheet = 'thyroid0387_UCI'

th_Data = get_data(excel,sheet)

th_Data.info()
print(th_Data.dtypes)
print(th_Data.isnull().sum())

numerical_cols = th_Data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = th_Data.select_dtypes(include=['object', 'bool']).columns.tolist()

#print(numerical_cols)
#print(categorical_cols)


numer_data = pd.to_numeric(th_Data[numerical_cols[1]], errors='coerce')

mean = th_Data[numerical_cols[1]].mean()
std = th_Data[numerical_cols[1]].std()

th_Data['z_score'] = (th_Data[numerical_cols[1]] - mean) / std

outliers = th_Data[abs(th_Data['z_score']) > 3]

print(f"{outliers}: Found {len(outliers)} outlier(s)")


mean = th_Data[numerical_cols[1]].mean()
std = th_Data[numerical_cols[1]].std()
print("Mean",mean,"Standard Deviation",std)


#A5


'''
A5. Similarity Measure: Take the first 2 observation vectors from the dataset. Consider only the 
attributes (direct or derived) with binary values for these vectors (ignore other attributes). Calculate 
the Jaccard Coefficient (JC) and Simple Matching Coefficient (SMC) between the document vectors. 
Use first vector for each document for this. Compare the values for JC and SMC and judge the 
appropriateness of each of them. 
JC = (f11) / (f01+ f10+ f11) 
SMC = (f11 + f00) / (f00 + f01 + f10 + f11) 
f11= number of attributes where the attribute carries value of 1 in both 
the vectors.


'''

B1 = th_Data["on thyroxine"]
B2 = th_Data["query on thyroxine"]
B1 = [1 if x == 't' else 0 for x in B1]
B2 = [1 if x == 't' else 0 for x in B2]
f11 = 0
f00 = 0
f10 = 0
f01 = 0
for i in range(len(B1)):
    if(B1[i]==B2[i]):
        if(B1[i]==0):
            f00+=1
        elif(B1[i]==1):
            f11 +=1
    if(B1[i]==1 and B2[i]==0):
        f10+=1
    elif(B1[i]==0 and B2[i]==1):
        f01+=1

jc = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) > 0 else float('nan')
smc = (f11 + f00) / (f00 + f01 + f10 + f11)
print("jaccard_distance",jc," Simple Matching Coefficient ",smc)


#A6
'''

A6. Cosine Similarity Measure: Now take the complete vectors for these two observations (including 
all the attributes). Calculate the Cosine similarity between the documents by using the second 
feature vector for each document. 


If A and B are two document vectors, then
cos( A, B) = <A, B>/||A|| ||B||

< A, B > =summation (k=1,n) a(k) * bk

||A|| and ||B|| are lengths of vectors A & B

'''


dot_product = sum(a * b for a, b in zip(B1, B2))
magnitude_B1 = sum(a**2 for a in B1) ** 0.5
magnitude_B2 = sum(b**2 for b in B2) ** 0.5

print(dot_product,magnitude_B1,magnitude_B2)

# Avoid division by zero
if magnitude_B1 == 0 or magnitude_B2 == 0:
    cosine_similarity = 0
else:
    cosine_similarity = dot_product / (magnitude_B1 * magnitude_B2)

print("cosine similarity",cosine_similarity)


#A7
'''
A7. Heatmap Plot: Consider the first 20 observation vectors. Calculate the JC, SMC and COS between 
the pairs of vectors for these 20 vectors. Employ similar strategies for coefficient calculation as in A4 
& A5. Employ a heatmap plot to visualize the similarities.  
 Suggestion to Python users → 
 import seaborn as sns 
sns.heatmap(data, annot = True) 


'''

def f_counts(row1, row2):
    f11 = ((row1 == 1) & (row2 == 1)).sum()
    f00 = ((row1 == 0) & (row2 == 0)).sum()
    f10 = ((row1 == 1) & (row2 == 0)).sum()
    f01 = ((row1 == 0) & (row2 == 1)).sum()
    return f11, f00, f10, f01

def jc(v1, v2):
    f11 = sum((a == 1 and b == 1) for a, b in zip(v1, v2))
    f10 = sum((a == 1 and b == 0) for a, b in zip(v1, v2))
    f01 = sum((a == 0 and b == 1) for a, b in zip(v1, v2))
    denom = f11 + f10 + f01
    if denom == 0:
        return np.nan  # or you could return np.nan or -1, depending on your needs
    return f11 / denom

def smc(row1, row2):
    f11, f00, f10, f01 = f_counts(row1, row2)
    denom = f00 + f01 + f10 + f11
    return (f11 + f00) / denom if denom else float('nan')

def cosine(row1, row2):
    dot = (row1 * row2).sum()
    norm1 = (row1 ** 2).sum() ** 0.5
    norm2 = (row2 ** 2).sum() ** 0.5
    return dot / (norm1 * norm2) if norm1 and norm2 else 0
range1 = 15

selected_cols = ["on thyroxine", "query on thyroxine"]

binary_columns = [col for col in selected_cols 
                  if set(th_Data[col].dropna().unique()) <= {'t', 'f', 0, 1}]
binary_df = th_Data[binary_columns].replace({'t': 1, 'f': 0}).astype(int).iloc[:range1]

JC  = pd.DataFrame([[jc(binary_df.iloc[i], binary_df.iloc[j])   for j in range(range1)] for i in range(range1)])
SMC = pd.DataFrame([[smc(binary_df.iloc[i], binary_df.iloc[j])  for j in range(range1)] for i in range(range1)])
COS = pd.DataFrame([[cosine(binary_df.iloc[i], binary_df.iloc[j])  for j in range(range1)] for i in range(range1)])

plt.figure(figsize=(36, 10))

plt.subplot(1, 3, 1)
sns.heatmap(JC, annot=True, cmap='Blues', fmt='.2f')
plt.title('Jaccard Coefficient (JC)')

plt.subplot(1, 3, 2)
sns.heatmap(SMC, annot=True, cmap='Greens', fmt='.2f')
plt.title('Simple Matching Coefficient (SMC)')

plt.subplot(1, 3, 3)
sns.heatmap(COS, annot=True, cmap='Oranges', fmt='.2f')
plt.title('Cosine Similarity')

plt.tight_layout()
plt.show()


import pandas as pd

# A8: Data Imputation
'''
A8. Data Imputation: employ appropriate central tendencies to fill the missing values in the data 
variables. Employ following guidance. 
• Mean may be used when the attribute is numeric with no outliers 
• Median may be employed for attributes which are numeric and contain outliers 
• Mode may be employed for categorical attributes 

'''

th_Data.replace("?", pd.NA, inplace=True)


for col in th_Data.columns:
    th_Data[col] = pd.to_numeric(th_Data[col], errors='coerce')

use_mean = ['T3']             
use_median = ['TSH', 'TT4']   
categorical_cols = th_Data.select_dtypes(include=['object', 'bool']).columns.tolist()
use_mode = categorical_cols   


for col in use_mean:
    th_Data[col] = th_Data[col].fillna(th_Data[col].mean())


for col in use_median:
    th_Data[col] = th_Data[col].fillna(th_Data[col].median())


for col in use_mode:
    mode_val = th_Data[col].mode()
    if not mode_val.empty:
        th_Data[col] = th_Data[col].fillna(mode_val[0])
    else:
        print(f"Warning: No mode value for column '{col}', imputation skipped.")

print("Missing values after imputation per column:")
print(th_Data.isna().sum())

# A9: Data Normalization / Scaling
'''

A9. Data Normalization / Scaling: from the data study, identify the attributes which may need 
normalization. Employ appropriate normalization techniques to create normalized set of data.

'''

numeric_cols = th_Data.select_dtypes(include=['float64', 'int64']).columns.tolist()
print("Numeric columns:", numeric_cols)


def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

th_Data_Scaled = th_Data.copy()
for col in numeric_cols:
    th_Data_Scaled[col] = min_max_normalize(th_Data_Scaled[col])

def z_score_normalize(series):
    return (series - series.mean()) / series.std()

th_Data_standardized = th_Data.copy()
for col in numeric_cols:
    th_Data_standardized[col] = z_score_normalize(th_Data_standardized[col])


print("\nMin-Max Normalized Data (first 5 rows):")
print(th_Data_Scaled[numeric_cols].head())

print("\nZ-Score Standardized Data (first 5 rows):")
print(th_Data_standardized[numeric_cols].head())

