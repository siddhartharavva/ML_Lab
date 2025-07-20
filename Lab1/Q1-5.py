def vowels_num(sent):
    vowels  = 0
    consonants = 0
    sent = sent.lower().strip(" ").replace(" ", "")
    # makes all the text lowercase removes spaces in between and end
    if(sent.isalpha()!=True):#checks if it is alpahbets only
        return -1
    for i in sent:
        if(i == 'a' or i == 'e' or i == 'i' or i == 'o' or i == 'u'):
            vowels +=1 #vowel check
        else:
            consonants += 1 # consonants checl
    return vowels, consonants


sentence = input("enter the string")
result = vowels_num(sentence)
if(result==-1):
    print("not a Valid sentence")
else:
    print(f"The vowels are: {result[0]} and consonants are: {result[1]}")

import numpy as np


def matrix_mul(A1,A2):
        
    A = np.array(A1)
    B = np.array(A2)
    
    return np.dot(A,B) # Matrix multiplication
    

def Mat_input():
    matrix_S = input("Enter the sizes of the matrix separated by space: ")
    matrix_S = matrix_S.strip().split(" ")
    sizes = [int(x) for x in matrix_S ]
    print(sizes)
    M1D = input("Enter the data of matrix 1 separated by space: ")
    M1D = M1D.strip().split(" ")
    M1D = [int(x) for x in M1D ]

    M1D = [M1D[i*sizes[1]:(i+1)*sizes[1]] for i in range(sizes[0]) ]
    print(M1D)
    return M1D

A1 = Mat_input()
A2 = Mat_input()
result = matrix_mul(A1,A2)
print(result)


def get_matrix():
    s = input("Enter size of the matrix:")
    s = s.strip().split(" ")
    s = [int(x) for x in s]
    data = input("Enter the data")
    data = data.strip().split(" ")
    data =  [int(x) for x in data]
    data  = [data[i*s[1]:(i+1)*s[1]] for i in range(s[0])]
    return data

def Transpose_Mat(A):
    
    T_Mat = [[0 for _ in range(len(A[0]))]for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            T_Mat[j][i] = A[i][j]
    return T_Mat
x = get_matrix()

T = Transpose_Mat(x)
print(x,T)



def get_List():
    s = int(input("Enter size of the list"))
    data = input("Enter the data")
    data = data.strip().split(" ")
    data =  [int(x) for x in data]
    return data
def common_elements(l1,l2):
    c = 0
    for i in l1:
        for j in l2:
            if i==j:
                c+=1
    return c
a = get_List()
b = get_List()
print(common_elements(a,b))