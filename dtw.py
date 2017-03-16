# Implementation of the DTW algorithm as described originally by 
# Sakoe,H. and Chiba, S. Dynamic programming algorithm optimization for spoken word recognition.
# IEEE Trans. on Acoust., Speech, and Signal Process., ASSP 26, 43-49 (1978). 

import numpy as np

#Use this function in your program
def dist_DTW(A, B):
    n = len(A); m = len(B)
    DTW = np.zeros((n,m))
    for i in range(1,n):
        DTW[i][0] = np.inf
    for i in range(1,m):
        DTW[0][i] = np.inf
    DTW[0][0] = 0
    for i in range(1,n):
        for j in range(1,m):
            cost = np.abs(A[i]-B[j])    #L1 norm
            DTW[i][j] = cost+min(DTW[i-1][j], DTW[i][j-1], DTW[i-1][j-1])
    return DTW[n-1,m-1]

#Sample data
if __name__ == "__main__":
    x = np.array([3,4,5,6,7,8,9,0,2])
    y = np.array([5,6,7,8,0,3,4])
    print dist_DTW(x,y)
