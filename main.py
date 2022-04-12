import numpy as np


def LCG(seed, num, a, c, m):
    # seed - starting number
    # num - Total number of random numbers
    # a,c,m - parameters
    rand = np.zeros(num)
    rand[0] = seed
    for l in range(1, num):
        rand[l] = (a*rand[l-1] + c) % m
    return rand/m
print(LCG(3,100, 356, .9, 867))
def MonteCarlo(f, N):
    # f - function
    # N - number of points for integration
    x = LCG(0.4,N,487,2,18745)
    total = 0
    for i in range(0,N):
        total += f(x[i])
    total = total/N
    return total


