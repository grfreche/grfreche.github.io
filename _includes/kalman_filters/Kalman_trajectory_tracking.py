# Application of Kalman filter for trajectory tracking

# import numpy as np
from numpy import sqrt, zeros, array, identity, arange
from numpy.random import randn
import matplotlib.pyplot as plt

import datetime

# Parameters for proper figure export
fig_width_pt = 2*252.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)+1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt      # height in inches
fig_height = fig_width/golden_mean  # width in inches
fig_size =  [fig_width,fig_height]

start = datetime.datetime.now()

# Discretized time step and number of steps
DT = 1
niter = 300

# Generate random acceleration
sigma_a = 0.2
accel = randn(niter)*sigma_a

# Generate true position and speed with initial random speed
true_pos = zeros(niter+1)
true_speed = zeros(niter+1)
sigma_speed = 2
true_speed[0] = randn()*sigma_speed

for k in range(niter):
    true_pos[k+1] = true_pos[k] + DT*true_speed[k] + (DT*DT/2)*accel[k]
    true_speed[k+1] = true_speed[k] + DT*accel[k]

# Generate noisy observations
sigma_noise = 30
obs_noise = randn(niter)*sigma_noise
obs_pos = zeros(niter+1)
for k in range(niter):
    obs_pos[k+1] = true_pos[k+1] + obs_noise[k]

# Fixed model matrices
matF = array([[1, DT], [0, 1]])
matRv = array([[(DT**4)/4, (DT**3)/2], [(DT**3)/2, DT*DT]])*(sigma_a**2)
matH = array([[1, 0]])
matRu = sigma_noise*sigma_noise

# Keep track of estimated states over time
estim_state = zeros((2,niter+1))

# Kalman filter
posterior_innov_cov = zeros((2,2))

# Store Kalman gains
gains = zeros((2,niter))

for k in range(niter):
    # Prior innovation covariance matrix update
    prior_innov_cov = (matF @ posterior_innov_cov @ (matF.T)) + matRv
    
    # Kalman gain update
    kalman_gain = (prior_innov_cov @ (matH.T))/((matH @ prior_innov_cov @ (matH.T))+matRu)
    gains[0,k] = kalman_gain[0]
    gains[1,k] = kalman_gain[1]
    
    # Posterior innovation covariance matrix update
    posterior_innov_cov = (identity(2)-(kalman_gain @ matH)) @ prior_innov_cov
    
    # Previous posterior state estimate
    posterior_estim_state = array([[estim_state[0,k]],[estim_state[1,k]]])
    
    # New prior state estimate
    prior_estim_state = matF @ posterior_estim_state
    
    # New posterior state estimate
    posterior_estim_state = prior_estim_state + (kalman_gain*(obs_pos[k+1] - (matH @ prior_estim_state)))
    
    # Store this new posterior estimate
    estim_state[:,k+1] = posterior_estim_state[:,0]

stop = datetime.datetime.now()
duration = stop - start

print("Program duration: "+str(duration.microseconds/1000000)+" sec")

# Plot true, observed and estimated positions
plt.close('all')
time_ax = arange(niter+1)*DT

fig1, ax1 = plt.subplots()
ax1.plot(time_ax, true_pos, '-', label='True position', linewidth=1)
ax1.plot(time_ax, estim_state[0,:], '-', label='Estimated position', linewidth=1)
ax1.plot(time_ax, obs_pos, '--', label='Observed position', linewidth=1)
ax1.grid()
ax1.legend()
fig1.set_size_inches(fig_size)
fig1.show()
fig1.savefig('kalman_position.png', dpi=200)

# Plot true and estimated speeds
fig2, ax2 = plt.subplots()
ax2.plot(time_ax, true_speed, '-', label='True speed', linewidth=1)
ax2.plot(time_ax, estim_state[1,:], '-', label='Estimated speed', linewidth=1)
ax2.grid()
ax2.legend()
fig2.set_size_inches(fig_size)
fig2.show()
fig2.savefig('kalman_speed.png', dpi=200)

# Compute error true-observed and error true-estimated
err_true_obs = (true_pos-obs_pos)**2
err_true_estim = (true_pos-estim_state[0,:])**2

fig3, ax3 = plt.subplots()
ax3.plot(time_ax, err_true_obs, '-', label='Error true-observed', linewidth=1)
ax3.plot(time_ax, err_true_estim, '-', label='Error true-estimated', linewidth=1)
ax3.grid()
ax3.legend()
fig3.set_size_inches(fig_size)
fig3.show()
fig3.savefig('kalman_error.png', dpi=200)

# Plot evolution of Kalman gain
fig4, ax4 = plt.subplots()
ax4.plot(gains.T)
ax4.grid()
fig4.set_size_inches(fig_size)
fig4.show()
