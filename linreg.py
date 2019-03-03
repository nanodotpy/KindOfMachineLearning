import numpy as np
import matplotlib.pyplot as plt

def linreg(x, y, learning_rate = 0.001, iterations = 100000):
    
    assert len(x) == len(y)
    
    Ymean = np.mean(y)
    m = len(x)
    
    def h(x, t0, t1):
        return t0 + t1*x
         
    a = 0
    b = 0
        
    for e in range(iterations):
        tempb = b - (learning_rate/m)*(sum([h(x[i], b, a) - y[i] for i in range(m)]))
        tempa = a - (learning_rate/m)*(sum([(h(x[i], b, a) - y[i])*x[i] for i in range(m)]))
        b = tempb
        a = tempa
        
    r_squared = 1 - (sum([(y[i]-(x[i]*a + b))**2 for i in range(len(y))])/sum([(x[i]-Ymean)**2 for i in range(len(y))]))

    result = (a, b, r_squared)
    
    return result

if __name__ == '__main__':
    
    # Exemple plotted using matplotlib
    
    x = [1, 2, 5, 9, 15, 25, 42]
    
    y = [2*xi for xi in x]
    
    coeffs = linreg(x, y, 0.001, 100000)
        
    Y = [coeffs[0]*xi + coeffs[1] for xi in x]
    
    print(coeffs)
        
    fig = plt.figure()   
    
    fig.set_size_inches(14, 9)

    plt.plot(x, y,'o', x, Y)
   

   
