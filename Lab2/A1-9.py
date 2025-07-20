import pandas as pd
import numpy as np
import statistics as st
import seaborn as sns
import matplotlib.pyplot as plt

#A1
def get_data(excel,sheet):
    data = pd.read_excel(excel , sheet_name=sheet)
    return data

excel = 'Lab_session_Data.xlsx'
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

print("Cost of each product in the order Candies, Mango, milk packet")
print(np.dot(np.linalg.pinv(A),C))
print(A,C)


#A2
def classify_richness(data):
    for i in range(data.shape[0]):
        if(data['Payment (Rs)'][i] > 200):
            print("Is Rich")        
        else:
            print("Is Poor")

classify_richness(data)
    

#A3 
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
    
    
excel = 'Lab_session_Data.xlsx'
sheet = 'IRCTC Stock Price'

data = get_data(excel,sheet)
print("Mean of Price: ", mean_day(data))
print("Variance of Price: ", variance(data))
print("Mean on Wednesday of Price: ", mean_day(data,'Wed'))
print("Mean on Month April of Price: ", mean_Mon(data,'Apr'))
print("Loss probability",loss_mean(data))


#A4

excel = 'Lab_session_Data.xlsx'
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

# Replace '?' with NA
th_Data.replace("?", pd.NA, inplace=True)

# Ensure all columns are numeric where possible
for col in th_Data.columns:
    th_Data[col] = pd.to_numeric(th_Data[col], errors='coerce')

# Identify column lists for imputation
use_mean = ['T3']             # Numeric, no outliers
use_median = ['TSH', 'TT4']   # Numeric, possibly with outliers
categorical_cols = th_Data.select_dtypes(include=['object', 'bool']).columns.tolist()
use_mode = categorical_cols   # Categorical columns

# Impute mean for selected columns
for col in use_mean:
    th_Data[col] = th_Data[col].fillna(th_Data[col].mean())

# Impute median for selected columns
for col in use_median:
    th_Data[col] = th_Data[col].fillna(th_Data[col].median())

# Impute mode for categorical columns, check mode exists
for col in use_mode:
    mode_val = th_Data[col].mode()
    if not mode_val.empty:
        th_Data[col] = th_Data[col].fillna(mode_val[0])
    else:
        print(f"Warning: No mode value for column '{col}', imputation skipped.")

# Optional: print count of missing values after imputation
print("Missing values after imputation per column:")
print(th_Data.isna().sum())

# A9: Data Normalization / Scaling

# Refresh numeric columns list
numeric_cols = th_Data.select_dtypes(include=['float64', 'int64']).columns.tolist()
print("Numeric columns:", numeric_cols)

# Min-Max Normalization
def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

th_Data_Scaled = th_Data.copy()
for col in numeric_cols:
    th_Data_Scaled[col] = min_max_normalize(th_Data_Scaled[col])

# Z-Score Standardization
def z_score_normalize(series):
    return (series - series.mean()) / series.std()

th_Data_standardized = th_Data.copy()
for col in numeric_cols:
    th_Data_standardized[col] = z_score_normalize(th_Data_standardized[col])

# Example preview: print first few rows of normalized data
print("\nMin-Max Normalized Data (first 5 rows):")
print(th_Data_Scaled[numeric_cols].head())

print("\nZ-Score Standardized Data (first 5 rows):")
print(th_Data_standardized[numeric_cols].head())
