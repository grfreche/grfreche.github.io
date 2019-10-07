# Apply weighted Recursive Least Squares to determine autoregressive model of a speech signal

from numpy import sqrt, linspace, zeros, identity, outer, log10
from numpy.random import randn
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Parameters for proper figure export
fig_width_pt = 2*252.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)+1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt      # height in inches
fig_height = fig_width/golden_mean  # width in inches
fig_size =  [fig_width,fig_height]


# Weighted RLS function
def RLS_autoregressive(data, m, lamb, delta):
    """
        Applies Recursive Least Squares algorithm with forgetting factor
        on autoregressive signals
        
        :param data: the autogressive signal on which the RLS algorithm is applied
        :param m: order of the autoregressive model
        :param lamb: the forgetting factor lambda
        :param delta: factor to initialize matrix P
        :type data: numpy.ndarray
        :type m: int
        :type lamb: float
        :type delta: float
        :return histo_weights: a table keeping track of all the weights estimated
        during the algorithm
        :return error_weights: error between signal sample and estimated sample
    """
    
    # Data length
    n = len(data)
    
    # Initialization of parameters returned by the function
    histo_weights = zeros((n-m,m))
    error_weights = zeros(n-m)
    
    # Recursive Least Squares algorithm
    # Generate estimated weights
    estim_weights = zeros(m)
    
    histo_weights[0,:] = estim_weights
    # Initialize matrix P
    P = identity(m)*delta
    
    for k in range(n-m-1):
        # Convert data
        x = data[k:k+m]
        y = data[k+m]
        
        # Update gain vector g
        denom = lamb + (x @ P @ (x.T))
        g = (P @ (x.T))/denom
        
        # Update weights
        estim_weights = estim_weights + (y - (x @ estim_weights))*g
        histo_weights[k+1,:] = estim_weights
        
        # Update matrix P
        P = (identity(m) - outer(g, x))@P/lamb
        
        # Compute error
        error_weights[k] = (y - (x @ estim_weights))**2
    
    return histo_weights, error_weights


# Read the wav file containing the speech signal
fs, data = wavfile.read('hello.wav')
# Keep the signal between 0 s and 0.6 s
data = data[:26460,0]
# Time variable
t = linspace(0,len(data)/fs,len(data))

# Plot and save the speech signal
plt.close("all")
fig1, ax1 = plt.subplots()
ax1.plot(t, data,'k-', linewidth=1)
ax1.grid()
ax1.set_xlabel('time (s)')
ax1.set_ylabel('signal amplitude')
fig1.set_size_inches(fig_size)
fig1.show()
fig1.savefig('speech_signal.png', dpi=200)

# Order of the autoregressive model
m = 10

# Factor for initialization of matrix P
delta = 100

# RLS with forgetting factor 1
lamb = 1
histo_weights1, error_weights1 = RLS_autoregressive(data, m, lamb, delta)

# RLS with forgetting factor 0.92
lamb = 0.95
histo_weights2, error_weights2 = RLS_autoregressive(data, m, lamb, delta)

# RLS with forgetting factor 0.89
lamb = 0.90
histo_weights3, error_weights3 = RLS_autoregressive(data, m, lamb, delta)

# Time axis for RLS weights
t_autoreg = t[:-m]

# Plot weight estimation evolution over time for forgetting factors 1 and 0.89
fig2, ax2 = plt.subplots(2)
ax2[0].plot(t_autoreg, histo_weights1,'-', linewidth=1.5)
ax2[0].grid()
ax2[0].set_ylabel('$\lambda = 1$')
ax2[1].plot(t_autoreg, histo_weights3,'-', linewidth=1.5)
ax2[1].grid()
ax2[1].set_ylabel('$\lambda = 0.90$')
ax2[1].set_xlabel('time (s)')
fig2.set_size_inches(fig_size)
fig2.show()
fig2.savefig('estimated_weights_evolution.png', dpi=200)


# Average error on time windows for cleaner plot
win_leng = 50
error_weights = zeros((3,len(error_weights1)))
error_weights[0,:] = error_weights1
error_weights[1,:] = error_weights2
error_weights[2,:] = error_weights3

error_weights_average = error_weights[:,::win_leng]
for k in range(1,win_leng):
    error_weights_average = error_weights_average + error_weights[:,k::win_leng]
error_weights_average = error_weights_average/win_leng

# Plot error evolution over time
fig3, ax3 = plt.subplots()
ax3.plot(t_autoreg[::win_leng], log10(error_weights_average[0,:]),'r-', linewidth=1.5, label='$\lambda = 1$')
ax3.plot(t_autoreg[::win_leng], log10(error_weights_average[1,:]),'b-', linewidth=1.5, label='$\lambda = 0.95$')
ax3.plot(t_autoreg[::win_leng], log10(error_weights_average[2,:]),'g-', linewidth=1.5,
label='$\lambda = 0.90$')
ax3.grid()
ax3.set_xlabel('time (s)')
ax3.set_ylabel('$\log_{10}($error$)$')
ax3.legend()
fig3.set_size_inches(fig_size)
fig3.show()
fig3.savefig('autoregressive_error.png', dpi=200)
