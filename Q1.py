import numpy as np
from Library import *


f = open("assign2fit.txt","r")
A = readfile(f)
A = np.transpose(A)
print(A)
X = A[0]
Y = A[1]
print(X)
print(Y)

# Error in Y
sigma = [0.1 for i in range(len(Y))]
parameters = polyfit(X,Y,sigma, order = 3)
print("The values for a0,a1,a2,a3 are: \n",parameters)

"""
The values for a0,a1,a2,a3 are: 
 [[  0.57465867]
 [  4.72586144]
 [-11.12821778]
 [  7.66867762]]
"""