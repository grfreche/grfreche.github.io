# Illustrative example of Single Neuron Classifier for affine separation
# We work in R^2 with the hyperplane of equation -6+2x+3y = 0
# i.e. y = -(2/3)*x + 2

from numpy import sqrt, array, linspace, zeros, exp
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
        :return res: res = 1 if x^T.w > 0, 0 otherwise
    """
    if ((w[0]+w[1]*x[0]+w[2]*x[1])>0):
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
    return sigm(w[0]+w[1]*x[0]+w[2]*x[1])

# Weights correponding to our fixed hyperplane
act_weights = array([-6,2,3])

# Explicit equation of the hyperplane for plot
t = linspace(-20,20,1001)
yt = -(2/3)*t + 2

# Generate training data
n = 300
x = uniform(-20,20, (2,n))
y = zeros(n);
for k in range(n):
    y[k] = assignLabel(act_weights, x[:,k])

# Separate points for plot
x0 = x[:, (y==0)]
x1 = x[:, (y==1)]

# Estimate weights
mu = 1
estim_weights = randn(3)
for k in range(n):
    neur_resp = feed_forward(estim_weights, x[:,k])
    xtilde = array([1, x[0,k], x[1,k]])
    estim_weights = estim_weights - 2 * mu *(neur_resp-y[k])*neur_resp*(1-neur_resp)*xtilde

# Estimated separator
yt_estim = - (estim_weights[1]/estim_weights[2])*t - (estim_weights[0]/estim_weights[2])

# Plot the illustration
plt.close('all')
fig1, ax1 = plt.subplots()
ax1.plot(t,yt,'k-')
ax1.plot(t,yt_estim,'r-')
ax1.plot(x0[0,:],x0[1,:], 'bx', linewidth=1)
ax1.plot(x1[0,:],x1[1,:], 'mx', linewidth=1)
ax1.grid()
plt.xlim(-20, 20)
plt.ylim(-20, 20)
fig1.set_size_inches(fig_size)
fig1.show()
fig1.savefig('SNC_affine_illust.png', dpi=200)
