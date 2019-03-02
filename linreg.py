"""
this is a first try at doing linear regression
it will be improved, especially the cost function minimum finding (I should implement gradient descent)
"""

import numpy as np
from math import ceil, floor
import matplotlib.pyplot as plt

def linreg(xarray, yarray, accuracy):
    """
    Returns in a tuple:
    - a and b in equation y = ax+b such that the sum of the squared error epsilon = (y - input_y)**2 is minimal for
    input_y in yarray associated to a input_x in xarray
    - third element of the tuple is the coefficient of determination ranging from 0 to 1:
        
    """
    
    Ymean = np.mean(yarray)
        
    def h(x, a, b):
        """
        Affine function
        """
        return x*a + b
     
    def j(x, y, a, b):
        """
        Cost function, takes for arguments a and b and computes the average difference
        between our hypothesis and actual input y
        """
        assert len(x) == len(y)
        m = len(x)
        image_of_j = sum([(h(x[i], a, b) - y[i])**2 for i in range(m)])/(2*m)
        return image_of_j
    
    step = 10**(-accuracy)
    
    """
    I first grossly estimate a and b to limit the minimum searching interval :
        it is not the right way to do it
    """
    
    estim_a = (yarray[-1]-yarray[0])/(xarray[-1]-xarray[0])
    
    estim_b = yarray[0] - estim_a*xarray[0]
    
    if estim_a < 0:
        a = np.arange(int(ceil(estim_a*1.5))-1, int(floor(estim_a*0.5))+1, step)
    else:
        a = np.arange(int(floor(estim_a*0.5))-1, int(ceil(estim_a*1.5))+1, step)
                                                                                    # those tests gotta be removed
    if estim_b < 0:
        b = np.arange(int(ceil(estim_b*1.5))-1, int(floor(estim_b*0.5))+1, step)
    else:
        b = np.arange(int(floor(estim_b*0.5))-1, int(ceil(estim_b*1.5))+1, step)
        
    a, b = np.meshgrid(a, b)
    
    Z = j(xarray, yarray, a, b)
        
    i, j = np.where(Z == np.amin(Z))
        
    a = round(j[0]*step + a[0][0], accuracy)
    b = round(i[0] * step + b[0][0], accuracy)
    
    r_squared = 1 - (sum([(yarray[i]-(xarray[i]*a + b))**2 for i in range(len(yarray))])/sum([(yarray[i]-Ymean)**2 for i in range(len(yarray))]))

    result = (a, b, r_squared)
    
    return result

if __name__ == '__main__':
    
    # Exemple plotted using matplotlib
    
    x = [1, 2, 5, 9, 15, 25, 42]
    
    y = [10, 15, 32, 45, 67, 89, 99]
    
    coeffs = linreg(x, y, 1)
        
    Y = [coeffs[0]*xi + coeffs[1] for xi in x]
    
    print(coeffs[2])
        
    fig = plt.figure()   
    
    fig.set_size_inches(14, 9)

    plt.plot(x, y,'o', x, Y)
   
