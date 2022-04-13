import numpy as np
from Library import *


# The function for finding
def FindPi(N):
    inside = 0
    outside = 0
    # Finding area by throwing darts sort of
    dart = np.zeros(shape=(N,2))
    # circle of radius 1 (diameter 2) centered at origin is taken
    dart[:,0] = LCG(2,N,35,0,17453345)*2-1
    dart[:,1] = LCG(5,N,43,0,17454543)*2-1
    for i in range(N):
        term = (dart[i][0])**2 + (dart[i][1])**2
        if term <= 1:
            inside += 1
        else:
            outside += 1
    print("The value of pi by throwing darts method is: ",inside/len(dart)*4)

FindPi(10000)
# finding pi using Monte Carlo
def f1(x):
    return np.sqrt(1-(x**2))

pi = 4 * MonteCarlo(f1,10000)
print("\nThe value of pi by Monte Carlo method is: ",pi)