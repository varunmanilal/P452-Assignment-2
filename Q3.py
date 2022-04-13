
from Library import *

# The function to be integrated for finding the volume of Steinmentz Solid
def f1(x):
    return 4 * (1-x**2)

# The function does integration from 0 to 1. But we need to integrate from
# -1 to 1. Since it's symmetric we can just multiply by 2
Volume = MonteCarlo(f1,10000)*2

print("Volume of Steinmets solid calculated using Monte Carlo method is:", Volume)