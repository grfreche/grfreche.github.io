import numpy as np
import matplotlib.pyplot as plt

# Parameters for proper figure export
fig_width_pt = 2*252.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)+1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt      # height in inches
fig_height = fig_width/golden_mean  # width in inches
fig_size =  [fig_width,fig_height]

#we work in R^2, the hyperplane has equation -6+2x+3y = 0
#i.e. y = -(2/3)*x + 2

def assignLabel(w, x):
    """
        Given weights w and input x, assigns corresponding label y
        
        :param w: weights
        :param x: input
        :type w: numpy.ndarray
        :type x: numpy.ndarray
        :return res:
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
    return 1/(1+np.exp(-x))

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

def SNC_affine_error(act_weights, n_train):
# Generate training data
    x = np.random.uniform(-20,20, (2,n_train))
    y = np.zeros(n_train);
    for k in range(n_train):
        y[k] = assignLabel(act_weights, x[:,k])
    
    #Separate points for plot
    x0 = x[:, (y==0)]
    x1 = x[:, (y==1)]
    
    #Estimate weights
    mu = 1
    estim_weights = np.random.randn(3)
    histo_weights = np.zeros((3,n_train+1))
    histo_weights[:,0] = estim_weights
    for k in range(n_train):
        neur_resp = feed_forward(estim_weights, x[:,k])
        xtilde = np.array([1, x[0,k], x[1,k]])
        estim_weights = estim_weights - 2 * mu *(neur_resp-y[k])*neur_resp*(1-neur_resp)*xtilde
        histo_weights[:,k+1] = estim_weights
    
    #Estimated training labels
    y_estim = np.zeros(n_train);
    for k in range(n_train):
        y_estim[k] = assignLabel(estim_weights, x[:,k])
    
    training_error = np.sum(y!=y_estim)/n_train
    
    # Generate test data
    n_test = 10000
    x_test = np.random.uniform(-20,20, (2,n_test))
    y_test = np.zeros(n_test)
    y_test_estim = np.zeros(n_test)
    for k in range(n_test):
        y_test[k] = assignLabel(act_weights, x_test[:,k])
        y_test_estim[k] = assignLabel(estim_weights, x_test[:,k])
    
    test_error = np.sum(y_test!=y_test_estim)/n_test
    
    return training_error, test_error


act_weights = np.array([-6,2,3])
n_trains = np.arange(100, 2001, 100)
n_iter = 200
mean_training_errors = np.zeros(len(n_trains))
mean_test_errors = np.zeros(len(n_trains))

for i in range(len(n_trains)):
    n_train = n_trains[i]
    mean_training_error = 0
    mean_test_error = 0
    
    for k in range(n_iter):
        training_error, test_error = SNC_affine_error(act_weights, n_train)
        mean_training_error = mean_training_error + training_error
        mean_test_error = mean_test_error + test_error
        
    mean_training_error = 100*mean_training_error / n_iter
    mean_test_error = 100*mean_test_error / n_iter
    
    mean_training_errors[i] = mean_training_error
    mean_test_errors[i] = mean_test_error
    
    
    # Print errors
    print ('Number of training examples: '+str(n_train))
    print('Mean training error: '+str(mean_training_error))
    print('Mean test error: '+str(mean_test_error))

plt.close('all')
fig1, ax1 = plt.subplots()
ax1.plot(n_trains, mean_training_errors, 'b-x', label='Training error')
ax1.plot(n_trains, mean_test_errors, 'k-x', label='Test error')
ax1.grid()
ax1.set_xlabel('Number of training examples')
ax1.set_ylabel('Percentage of mislabeled examples')
ax1.legend()
fig1.set_size_inches(fig_size)
fig1.show()
fig1.savefig('SNC_affine_error.png', dpi=200)
