"""
Linear regression taking  the array x of x values and their corresponding
y values in y array
"""

def linreg(x, y, iterations=10000, learning_rate=0.001):
    """
    Linear regression using gradient descent and cost function.

    Takes for arguments the y values array corresponding to the
    x values in x array, the number of iterations (the bigger it is the
    longer the algorithm takes, it has a linear complexity) set by default
    to 10000 (it is clearly enough), and the learning rate, set by default
    to 0.001, I think it is the optimum value in most of cases
    (bigger value won't permit to converge to a and b coefficients,
    lower will increase running time).

    Returns in a tuple :
        - the coefficients a and b of the ax + b equation line
        - the r_squared value, the more it is near of 1.0, the more the data
        is 'linear'

    """

    assert len(x) == len(y)

    m = len(x)

    y_mean = sum(y)/m

    def h(x, b, a):
        return b + a*x

    # Arbitrary initial values of a and b
    a = 0
    b = 0

    # Gradient descent
    for JUL in range(iterations):
        tempb = b - (learning_rate/m)*(sum([h(x[i], b, a) - y[i] for i in range(m)]))
        tempa = a - (learning_rate/m)*(sum([(h(x[i], b, a) - y[i])*x[i] for i in range(m)]))
        b = tempb
        a = tempa

    # R_squared value calculation
    r_squared = 1 - (sum([(y[i]-(x[i]*a + b))**2 for i in range(m)])/sum([(x[i]-y_mean)**2 for i in range(m)]))

    result = (a, b, r_squared)

    return result

try:
    import matplotlib.pyplot as plt
    if __name__ == '__main__':

        # Exemple plotted using matplotlib

        x_values = [1, 2, 5, 9, 15, 25, 42]

        y_values = [10, 15, 32, 45, 67, 89, 99]

        coeffs = linreg(x_values, y_values, 100000)

        """
        In this case, using 100000 iterations just change the 3rd significant
        number in both a and b coefficients values.
        """

        images_of_regression = [coeffs[0]*xi + coeffs[1] for xi in x_values]

        print(coeffs)

        fig = plt.figure()

        fig.set_size_inches(14, 9)

        plt.plot(x_values, y_values, 'o', x_values, images_of_regression)
except ImportError:
    pass
