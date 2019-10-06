# Compare affine regression, quadratic regression and Lagrange interpolation on noisy quadratic function P(x) = a*(x-alpha)^2+b
# We aim at identify coefficients w0, w1 and w2 such that P(x) = w0 + w1*x + w2*x^2

from numpy import linspace, identity, array, outer, sum, floor, ceil, min,  max, sqrt
from numpy.random import uniform, randn
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

# Parameters for proper figure export
fig_width_pt = 2*252.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)+1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt      # height in inches
fig_height = fig_width/golden_mean  # width in inches
fig_size =  [fig_width,fig_height]

# RLS function
def RLS(m, x_data, y_data):
    """
        Applies Recursive Least Squares algorithm
        
        :param m: weight vector dimension
        :param x_data: hidden function input
        :param y_data: hidden function output
        :type m: int
        :param x_data: numpy.ndarray
        :param y_data:numpy.ndarray
        :return weights: weights estimated by the RLS algorithm
    """
    
    n = len(x_data)
    weights = randn(m);
    matP = identity(m)
    
    for k in range(n):
        # Prepare data
        xList = [1]
        for i in range(1,m):
            xList.append(xList[-1]*x_data[k])
        x = array(xList)
        y = y_data[k];
        
        # Update gain vector g
        denom = 1 + (x @ matP @ (x.T))
        g = (matP @ (x.T))/denom
        
        # Update weights
        weights = weights + (y - (x @ weights))*g
        
        # Update matrix P
        matP = (identity(m) - outer(g, x))*matP
    
    return weights

# Error function
def error(y_computed, y_expected):
    """
        Computes error between computed output and expected output
        
        :param y_computed: computed output
        :param y_expected: expected output
        :param x_data: numpy.ndarray
        :param y_data:numpy.ndarray
        :return: mean square distance between the two vectors
    """
    
    leng = len(y_computed)
    return sum((y_computed-y_expected)**2)/leng

# Hidden coefficients
alpha = -0.749
a = 0.926
b = -1.113

# Number of training samples
n_train = 20

# Generate training data
x_train = uniform(-5,5,n_train)
noise = randn(n_train)*0.2
y_train = a*((x_train-alpha)**2)+b+noise

# Continuous time axis for regression ploting
t = linspace(-5,5,1001)

# Affine regression
affine_weights = RLS(2, x_train, y_train)
y_affine = affine_weights[0] + (affine_weights[1]*t)
y_affine_train = affine_weights[0] + (affine_weights[1]*x_train)
error_affine = error(y_affine_train, y_train)
print("Affine regression training error: "+str(error_affine))

# Quadratic regression
quadratic_weights = RLS(3, x_train, y_train)
y_quadratic = quadratic_weights[0] + (quadratic_weights[1]*t) + (quadratic_weights[2]*t*t)
y_quadratic_train = quadratic_weights[0] + (quadratic_weights[1]*x_train) + (quadratic_weights[2]*x_train*x_train)
error_quadratic = error(y_quadratic_train, y_train)
print("Quadratic regression training error: "+str(error_quadratic))

# Lagrange interpolation
lagrange_coeffs = lagrange(x_train,y_train)
y_lagrange = lagrange_coeffs(t)
y_lagrange_train = lagrange_coeffs(x_train)
errorLagrange = error(y_lagrange_train, y_train)
print("Lagrange interpolation training error: "+str(errorLagrange))

# Number of test samples
n_test = 20

# Training data
x_test = uniform(-5,5,n_test)
noise = randn(n_test)*0.2
y_test = a*((x_test-alpha)**2)+b+noise

# Test regression models
# Affine regression
y_affine_test = affine_weights[0] + (affine_weights[1]*x_test)
error_affine_test = error(y_affine_test, y_test)
print("Affine regression testing error: "+str(error_affine_test))
# Quadratic regression
y_quadratic_test = quadratic_weights[0] + (quadratic_weights[1]*x_test) + (quadratic_weights[2]*x_test*x_test)
error_quadratic_test = error(y_quadratic_test, y_test)
print("Quadratic regression testing error: "+str(error_quadratic_test))
# Lagrange interpolation
y_lagrange_test = lagrange_coeffs(x_test)
error_lagrange_test = error(y_lagrange_test, y_train)
print("Lagrange interpolation training error: "+str(error_lagrange_test))

plt.close("all")
fig1, ax1 = plt.subplots()
ax1.plot(x_train, y_train, 'kx', label='Training data')
ax1.plot(x_test, y_test, 'mx', label='Test data')
ax1.plot(t, y_affine, 'b-', label='Affine regression', linewidth=1)
ax1.plot(t, y_quadratic, 'g-', label='Quadratic regression', linewidth=1)
ax1.plot(t, y_lagrange, 'r-', label='Lagrange interpolation', linewidth=1)
ax1.grid()
ax1.legend()
plt.ylim(floor(min(y_train))-1, ceil(max(y_train))+1)
fig1.set_size_inches(fig_size)
fig1.show()
fig1.savefig('poly_reg.png', dpi=200)
