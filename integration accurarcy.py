import scipy.linalg as la
import matplotlib.pyplot as plt
import math
import time
import leads_self_energy
import parameters
import warnings
from typing import List


def integrate1():
    # in this function, the green functions are 1d arrays in energy. this is becasue we have passed the diagonal component of the green fun( lesser, or retarded).The  
    delta_x = ( parameters.pi ) / parameters.steps
    result = 0    
    x = 0
    for i in range(0, parameters.steps ):
        x += delta_x 
        result = delta_x * math.sin(x)  +result

    return result

def integrate2():
    # in this function, the green functions are 1d arrays in energy. this is becasue we have passed the diagonal component of the green fun( lesser, or retarded).The  
    delta_x = ( parameters.pi ) / parameters.steps
    result = 0    
    x = 0
    for i in range(0, parameters.steps ):
        x += delta_x 
        result = delta_x * math.sqrt( 1 - math.cos(x) * math.cos(x) )  +result

    return result

result1 = integrate1()
result2 = integrate2()
print(result1 , result2)