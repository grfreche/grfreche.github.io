# Illustrative example of Single Neuron Classifier for quadratic separation
# We work in R^2 with the ellipse of equation x^2+2y^2-2x*y-x+y-5=0

from numpy import sqrt, array, linspace, zeros, exp, arange, meshgrid
from numpy.random import uniform, randn
import matplotlib.pyplot as plt

# Parameters for proper figure export
fig_width_pt = 2*252.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)+1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt      # height in inches
fig_height = fig_width/golden_mean  # width in inches
fig_size =  [fig_width,fig_height]

def assignLabel(w, x):
    """
        Given weights w and input x, assigns corresponding label y
        
        :param w: weights
        :param x: input
        :type w: numpy.ndarray
        :type x: numpy.ndarray
        :return res:
    """
    if (((w[0]*x[0]*x[0])+(w[1]*x[1]*x[1])+(w[2]*x[0]*x[1])+(w[3]*x[0])+(w[4]*x[1])+w[5])>0):
        res = 1
    else:
        res = 0
    return res

def sigm(x):
    """
        Sigmoid function
        
        :param x: input
        :type x: float
        :return: sigmoid(x)
    """
    return 1/(1+exp(-x))

def feed_forward(w, x):
    """
        Single Neuron Classifier feed-forward function
        
        :param w: weights
        :param x: input
        :type w: numpy.ndarray
        :type x: numpy.ndarray
        :return res: neuron output
    """
    return sigm((w[0]*x[0]*x[0])+(w[1]*x[1]*x[1])+(w[2]*x[0]*x[1])+(w[3]*x[0])+(w[4]*x[1])+w[5])

# Weights correponding to our fixed ellipsoid
act_weights = array([1,2,-2,-1,1,-5])

# Preparing ellipoid contour for plot
delta = 0.025
xrange = arange(-5,5, delta)
yrange = arange(-5,5,delta)
X, Y = meshgrid(xrange,yrange)
Z = (act_weights[0]*X**2) + (act_weights[1]*Y**2) + (act_weights[2]*X*Y) + (act_weights[3]*X) + (act_weights[4]*Y) + act_weights[5]

# Generate training data
n = 800
x = uniform(-5, 5, (2,n))
y = zeros(n);
for k in range(n):
    y[k] = assignLabel(act_weights, x[:,k])

# Separate points for plot
x0 = x[:, (y==0)]
x1 = x[:, (y==1)]

# Estimate weights
mu = 1
estim_weights = randn(6)
for k in range(n):
    neur_resp = feed_forward(estim_weights, x[:,k])
    xtilde = array([x[0,k]**2, x[1,k]**2, x[0,k]*x[1,k], x[0,k], x[1,k], 1])
    estim_weights = estim_weights - 2 * mu *(neur_resp-y[k])*neur_resp*(1-neur_resp)*xtilde

# Estimated ellipse equation 
Z_estim = (estim_weights[0]*X**2) + (estim_weights[1]*Y**2) + (estim_weights[2]*X*Y) + (estim_weights[3]*X) + (estim_weights[4]*Y) + estim_weights[5]

# Plot the illustration
plt.close('all')
fig1, ax1 = plt.subplots()
ax1.contour(X,Y, Z, [0], colors='k', linewidths=1)
ax1.contour(X,Y, Z_estim, [0], colors='r', linewidths=1)
ax1.plot(x0[0,],x0[1,], 'bx')
ax1.plot(x1[0,],x1[1,], 'mx')
ax1.grid()
fig1.set_size_inches(fig_size)
fig1.show()
fig1.savefig('SNC_quadratic_illust.png', dpi=200)
