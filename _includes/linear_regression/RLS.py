# The Recursive Least Squares algorithm for linear regression

#import numpy as np
from numpy import zeros, identity, outer, sqrt
from numpy.random import randn, uniform
import matplotlib.pyplot as plt

# Parameters for proper figure export
fig_width_pt = 2*252.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)+1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt      # height in inches
fig_height = fig_width/golden_mean  # width in inches
fig_size =  [fig_width,fig_height]

# RLS function
def RLS(m, x_data, y_data, delta):
    """
        Applies Recursive Least Squares algorithm
        
        :param m: weight vector dimension
        :param x_data: hidden function input
        :param y_data: hidden function output
        :param delta: factor to initialize matrix P
        :type m: int
        :param x_data: numpy.ndarray
        :param y_data:numpy.ndarray
        :param delta: float
        :return weights: weights estimated by the RLS algorithm
    """
    
    n = x_data.shape[1]
    
    histo_weights = zeros((n+1,m))
    
    #weights = randn(m)
    weights = zeros(m)
    histo_weights[0,:] = weights
    matP = identity(m)*delta
    
    for k in range(n):
        # Prepare data
        x = x_data[:,k]
        y = y_data[k]
        
        # Update gain vector g
        denom = 1 + (x @ matP @ (x.T))
        g = (matP @ (x.T))/denom
        
        # Update weights
        weights = weights + (y - (x @ weights))*g
        histo_weights[k+1,:] = weights
        
        # Update matrix P
        matP = (identity(m) - outer(g, x)) @ matP
    
    return histo_weights

# Data length
n = 100

# Data input dimension
m = 4

# Generate regression weights
act_weights = randn(m)
print('Actual weights:')
print(act_weights)

# Generate data input
x = uniform(-10, 10, (m, n))

# Generate data output
sigma_noise = 1
y = zeros(n)
for k in range(n):
    noise = randn()*sigma_noise
    y[k] = (act_weights @ x[:,k]) + noise

histo_weights = RLS(m, x, y, 0.1)
print('Estimated weights:')
print(histo_weights[-1])

# Plot evolution of weights over time
plt.close('all')
fig1, ax1 = plt.subplots()
ax1.plot(histo_weights)
ax1.grid()
fig1.set_size_inches(fig_size)
fig1.show()
fig1.savefig('linear_RLS.png', dpi=200)
