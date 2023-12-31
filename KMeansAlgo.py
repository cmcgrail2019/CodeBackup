
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: solution.ipynb

import numpy as np
from helper_functions import *

def get_initial_means(array, k):
    """
    Picks k random points from the 2D array
    (without replacement) to use as initial
    cluster means

    params:
    array = numpy.ndarray[numpy.ndarray[float]] - m x n | datapoints x features

    k = int

    returns:
    initial_means = numpy.ndarray[numpy.ndarray[float]]
    """

    mean_list = np.random.choice(array.shape[0], k, replace=False)

    return array[mean_list]


########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def k_means_step(X, k, means):
    """
    A single update/step of the K-means algorithm
    Based on a input X and current mean estimate,
    predict clusters for each of the pixels and
    calculate new means.
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n | pixels x features (already flattened)
    k = int
    means = numpy.ndarray[numpy.ndarray[float]] - k x n

    returns:
    (new_means, clusters)
    new_means = numpy.ndarray[numpy.ndarray[float]] - k x n
    clusters = numpy.ndarray[int] - m sized vector
    """


    m=np.size(X,0)
    z=np.sum(np.square(X),axis=1)
    t=np.sum(np.square(means),axis=1)

    d = np.sqrt(np.reshape(z,(m,1)) + np.reshape(t,(1,k)) - np.dot(X,means.T)*2)

    clusters=np.argmin(d,axis=1)
    n=np.size(X,1)
    new_means=np.zeros((k,n))
    for i in range(k):
        xcen=X[clusters==i]
        new_means[i,:]=np.mean(xcen,axis=0)


    return (new_means,clusters)



########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def k_means_segment(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - r x c x ch
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - r x c x ch
    """
    r, c, ch = image_values.shape
    flat_img = np.reshape(image_values, [-1, ch])

    #print('image val', image_values.shape)
    #print('flat', flat_img.shape)
    if np.size(initial_means)==1:
        initial_means = get_initial_means(flat_img,k)
    old_means = initial_means
    (new_means,clusters)=k_means_step(flat_img, k,old_means)

    #While not converged
    while np.sum(old_means - new_means) != 0:
        old_means=new_means
        (new_means,clusters)=k_means_step(flat_img, k,old_means)
    #Updated image vals based upon convergence
    updated_image_vals = np.reshape(new_means[clusters], (r, c, ch))
    return updated_image_vals

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

"""
Make sure to put #export (first line in this cell) only
if you call/use this function elsewhere in the code
"""
def compute_sigma(X, MU):
    """
    Calculate covariance matrix, based in given X and MU values

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n

    returns:
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    """
    #Initialize vars
    m=np.size(X,0)
    k,n=MU.shape
    SIGMA=np.zeros((k,n,n))
    mean=np.mean(X,0)

    #Compute row sigma
    for i in range(k):
        SIGMA[i,:,:]=np.dot((X-MU[i,:]).T,(X-MU[i,:]))/m
    return SIGMA

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def initialize_parameters(X, k):
    """
    Return initial values for training of the GMM
    Set component mean to a random
    pixel's value (without replacement),
    based on the mean calculate covariance matrices,
    and set each component mixing coefficient (PIs)
    to a uniform values
    (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int

    returns:
    (MU, SIGMA, PI)
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    """

    m=np.size(X,0)
    n=np.size(X,1)
    initial=np.random.choice(m,k,replace=False)
    PI=np.ones(k)/k
    MU=X[initial]
    m=np.size(X,0)
    SIGMA=np.zeros((k,n,n))


    for i in range(k):
        SIGMA[i,:,:]=np.dot((X-MU[i,:]).T,(X-MU[i,:]))/m

    return (MU,SIGMA,PI)





########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def prob(x, mu, sigma):
    """Calculate the probability of x (a single
    data point or an array of data points) under the
    component with the given mean and covariance.
    The function is intended to compute multivariate
    normal distribution, which is given by N(x;MU,SIGMA).

    params:
    x = numpy.ndarray[float] (for single datapoint)
        or numpy.ndarray[numpy.ndarray[float]] (for array of datapoints)
    mu = numpy.ndarray[float]
    sigma = numpy.ndarray[numpy.ndarray[float]]

    returns:
    probability = float (for single datapoint)
                or numpy.ndarray[float] (for array of datapoints)
    """

    #If only one point
    if np.size(x) == 3:
        D = np.size(x)
        diff = x - mu
        det = np.linalg.det(sigma)
        sigma_inv = np.linalg.inv(sigma)
        p = np.exp(-0.5*np.dot(np.dot(diff,sigma_inv),diff))/(np.sqrt(np.absolute(det))*(np.power(2*np.pi,D/2)))
        return p

    #If multiple points
    else:
        p_vals = np.ones(len(x),)
        i = 0
        for x_vals in x:
            D = np.size(x_vals)
            diff = x_vals - mu
            det = np.linalg.det(sigma)
            sigma_inv = np.linalg.inv(sigma)
            p = np.exp(-0.5*np.dot(np.dot(diff,sigma_inv),diff))/(np.sqrt(np.absolute(det))*(np.power(2*np.pi,D/2)))
            p_vals[i] = p
            i = i+1
        return p_vals


########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def E_step(X,MU,SIGMA,PI,k):
    """
    E-step - Expectation
    Calculate responsibility for each
    of the data points, for the given
    MU, SIGMA and PI.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    k = int

    returns:
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    m,n = X.shape
    r = np.zeros((k,m))
    for i in range(k):
        diff=X-MU[i,:]
        sig=SIGMA[i,:,:]
        det=np.linalg.det(sig)
        inv=np.linalg.inv(sig)
        r[i,:]=PI[i]*np.exp(np.sum(-0.5*np.dot(diff,inv)*diff,axis=1))/(np.sqrt(np.absolute(det))*(np.power(2*np.pi,n/2)))
    return r/np.sum(r,axis=0)

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def M_step(X, r, k):
    """
    M-step - Maximization
    Calculate new MU, SIGMA and PI matrices
    based on the given responsibilities.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    r = numpy.ndarray[numpy.ndarray[float]] - k x m
    k = int

    returns:
    (new_MU, new_SIGMA, new_PI)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k
    """
    m,n = X.shape
    new_MU = np.zeros((k,n))
    new_SIGMA = np.zeros((k,n,n))
    new_PI = np.zeros(k)

    for i in range(k):
        Nk=np.sum(r[i,:])
        mu_num=np.sum(np.reshape(r[i,:],(m,1))*X,axis=0)
        new_MU[i,:]=mu_num/Nk
        new_PI[i]=Nk/m
        diff=X-new_MU[i,:]
        new_SIGMA[i,:,:] = np.dot(r[i,:] * diff.T, diff)/Nk

    return (new_MU, new_SIGMA, new_PI)

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def likelihood(X, PI, MU, SIGMA, k):
    """Calculate a log likelihood of the
    trained model based on the following
    formula for posterior probability:

    log(Pr(X | mixing, mean, stdev)) = sum((i=1 to m), log(sum((j=1 to k),
                                      mixing_j * N(x_i | mean_j,stdev_j))))

    Make sure you are using natural log, instead of log base 2 or base 10.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    k = int

    returns:
    log_likelihood = float
    """
    m,n=X.shape
    prob=np.zeros((k,m))
    for i in range(k):
        diff=X-MU[i,:]
        sig=SIGMA[i,:,:]
        det=np.linalg.det(sig)
        inv=np.linalg.inv(sig)
        prob[i,:]=PI[i]*np.exp(np.sum(-0.5*np.dot(diff,inv)*diff,axis=1))/(np.sqrt(np.absolute(det))*(np.power(2*np.pi,n/2)))
    return np.sum(np.log(np.sum(prob,axis=0)))

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION!

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def train_model(X, k, convergence_function, initial_values = None):
    """
    Train the mixture model using the
    expectation-maximization algorithm.
    E.g., iterate E and M steps from
    above until convergence.
    If the initial_values are None, initialize them.
    Else it's a tuple of the format (MU, SIGMA, PI).
    Convergence is reached when convergence_function
    returns terminate as True,
    see default convergence_function example
    in `helper_functions.py`

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    convergence_function = func
    initial_values = None or (MU, SIGMA, PI)

    returns:
    (new_MU, new_SIGMA, new_PI, responsibility)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    #Initialize Parameters
    m,n=X.shape
    if initial_values==None:
        initial_values=initialize_parameters(X, k)
    (MU,SIGMA,PI)=initial_values
    prev_likelihood=likelihood(X, PI, MU, SIGMA, k)

    #Conduct first EM pass
    r=E_step(X,MU,SIGMA,PI,k)
    (new_MU, new_SIGMA, new_PI)=M_step(X, r, k)
    new_likelihood=likelihood(X, new_PI, new_MU, new_SIGMA, k)

    count=0
    #Eval if converged
    count,terminate=convergence_function(prev_likelihood,new_likelihood,count)

    #if not converged, re-estimate vals
    while not terminate:
        (MU,SIGMA,PI)=(new_MU, new_SIGMA, new_PI)
        prev_likelihood=new_likelihood
        r=E_step(X,MU,SIGMA,PI,k)
        (new_MU, new_SIGMA, new_PI)=M_step(X, r, k)
        new_likelihood=likelihood(X, new_PI, new_MU, new_SIGMA, k)
        count,terminate=convergence_function(prev_likelihood,new_likelihood,count)

    return (new_MU, new_SIGMA, new_PI, r)

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def cluster(r):
    """
    Based on a given responsibilities matrix
    return an array of cluster indices.
    Assign each datapoint to a cluster based,
    on component with a max-likelihood
    (maximum responsibility value).

    params:
    r = numpy.ndarray[numpy.ndarray[float]] - k x m - responsibility matrix

    return:
    clusters = numpy.ndarray[int] - m x 1
    """
    clusters=np.argmax(r,axis=0)
    return clusters

########## DON'T WRITE

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def segment(X, MU, k, r):
    """
    Segment the X matrix into k components.
    Returns a matrix where each data point is
    replaced with its max-likelihood component mean.
    E.g., return the original matrix where each pixel's
    intensity replaced with its max-likelihood
    component mean. (the shape is still mxn, not
    original image size)

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    k = int
    r = numpy.ndarray[numpy.ndarray[float]] - k x m - responsibility matrix

    returns:
    new_X = numpy.ndarray[numpy.ndarray[float]] - m x n
    """

    clusters=cluster(r)
    new_X=MU[clusters]
    return new_X

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def best_segment(X,k,iters):
    """Determine the best segmentation
    of the image by repeatedly
    training the model and
    calculating its likelihood.
    Return the segment with the
    highest likelihood.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    iters = int

    returns:
    (likelihood, segment)
    likelihood = float
    segment = numpy.ndarray[numpy.ndarray[float]]
    """
    #Initialize Vars
    best = float('-inf')
    MU = np.zeros((k, X.shape[1]))
    R = np.zeros((k, X.shape[0]))

    #check convergence and update model parameters
    for _ in range(iters):
        mu, sigma, pi, r = train_model(X, k, default_convergence)
        lk = likelihood(X, pi, mu, sigma, k)
        if lk > best:
            best = lk
            MU, R = mu, r
    return best, segment(X, MU, k, R)

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def improved_initialization(X,k):
    """
    Initialize the training
    process by setting each
    component mean using some algorithm that
    you think might give better means to start with,
    based on the mean calculate covariance matrices,
    and set each component mixing coefficient (PIs)
    to a uniform values
    (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int

    returns:
    (MU, SIGMA, PI)
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    """
    #Initialize parameters
    initial_means=get_initial_means(X,k)
    old_means=initial_means
    (new_means,clusters)=k_means_step(X, k,old_means)


    #Iterate for convergence
    while np.sum(old_means - new_means) != 0:
        old_means=new_means
        (new_means,clusters)=k_means_step(X, k,old_means)

    PI=np.ones(k)/k
    m=np.size(X,0)
    n=np.size(X,1)
    initial=np.random.choice(m,k,replace=False)
    MU=new_means
    SIGMA=np.zeros((k,n,n))

    #Calc for sigma
    for i in range(k):
        SIGMA[i,:,:]=np.dot((X-MU[i,:]).T,(X-MU[i,:]))/m

    return (MU,SIGMA,PI)

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def new_convergence_function(previous_variables, new_variables, conv_ctr,
                             conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:
    previous_variables = [numpy.ndarray[float]]
                         containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]]
                    containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    (conv_crt, converged)
    conv_ctr = int
    converged = boolean
    """
    a1 = abs(previous_variables[0])*0.9 < abs(new_variables[0])
    a2 = abs(new_variables[0]) < abs(previous_variables[0])*1.1
    b1 = abs(previous_variables[1])*0.9 < abs(new_variables[1])
    b2 = abs(new_variables[1]) < abs(previous_variables[1])*1.1
    c1 = abs(previous_variables[2])*0.9 < abs(new_variables[2])
    c2 = abs(new_variables[2]) < abs(previous_variables[2])*1.1

    count = a1.all() and a2.all() and b1.all() and b2.all() and c1.all() and c2.all()


    if count:
        conv_ctr = conv_ctr + 1
    else:
        conv_ctr = 0

    return conv_ctr, conv_ctr > conv_ctr_cap


def train_model_improved(X, k, convergence_function, initial_values = None):
    """
    Train the mixture model using the
    expectation-maximization algorithm.
    E.g., iterate E and M steps from
    above until convergence.
    If the initial_values are None, initialize them.
    Else it's a tuple of the format (MU, SIGMA, PI).
    Convergence is reached when convergence_function
    returns terminate as True. Use new_convergence_fuction
    implemented above.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    convergence_function = func
    initial_values = None or (MU, SIGMA, PI)

    returns:
    (new_MU, new_SIGMA, new_PI, responsibility)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    #Intialize vars (initial set)
    m,n=X.shape
    if initial_values==None:
        initial_values=initialize_parameters(X, k)
    (MU,SIGMA,PI)=initial_values
    prev_variables=[MU,SIGMA,PI]


    #Conduct first E Step
    r=E_step(X,MU,SIGMA,PI,k)

    #Conduct first M Step
    (new_MU, new_SIGMA, new_PI)=M_step(X, r, k)
    new_variables=[new_MU, new_SIGMA, new_PI]


    #Iterate for convergence
    count=0
    count,terminate=convergence_function(prev_variables,new_variables,count)
    while not terminate:
        (MU,SIGMA,PI)=(new_MU, new_SIGMA, new_PI)
        prev_variables=new_variables
        r = E_step(X,MU,SIGMA,PI,k)
        (new_MU, new_SIGMA, new_PI) = M_step(X, r, k)
        new_variables = [new_MU, new_SIGMA, new_PI]
        count,terminate=convergence_function(prev_variables,new_variables,count)

    return (new_MU, new_SIGMA, new_PI, r)


########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
# Unittest below will check both of the functions at the same time.
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def bayes_info_criterion(X, PI, MU, SIGMA, k):
    """
    See description above
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    k = int

    return:
    bayes_info_criterion = int
    """
    m,n=X.shape

    return (np.log(m)*(k*(1 + n + n * (n+1)/2)-1)) - 2*likelihood(X,PI,MU,SIGMA,k)

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######
################ END OF LOCAL TEST CODE SECTION ######################

def BIC_likelihood_model_test(image_matrix, comp_means):
    """Returns the number of components
    corresponding to the minimum BIC
    and maximum likelihood with respect
    to image_matrix and comp_means.

    params:
    image_matrix = numpy.ndarray[numpy.ndarray[float]] - m x n
    comp_means = list(numpy.ndarray[numpy.ndarray[float]]) - list(k x n) (means for each value of k)

    returns:
    (n_comp_min_bic, n_comp_max_likelihood)
    n_comp_min_bic = int
    n_comp_max_likelihood = int
    """
    #Initalize Vars
    n_comp_min_bic=0
    min_bic=float('inf')
    n_comp_max_likelihood=0
    max_likelihood=float('-inf')
    m,n=image_matrix.shape


    #Iterate to determine best likelihood and BIC
    for i in range(len(comp_means)):
        MU = comp_means[i]
        k = np.size(MU,axis=0)
        PI = np.ones(k)/k
        SIGMA = np.zeros((k,n,n))
        for i in range(k):
            SIGMA[i,:,:] = np.dot((image_matrix-MU[i,:]).T,(image_matrix-MU[i,:]))/m
        (new_MU, new_SIGMA, new_PI, r)=train_model_improved(image_matrix, k, new_convergence_function, initial_values = (MU,SIGMA,PI))
        bic=bayes_info_criterion(image_matrix,new_PI, new_MU, new_SIGMA, k)

        #Eval criteria
        if bic<min_bic:
            min_bic=bic
            n_comp_min_bic=k

        the_likelihood=likelihood(image_matrix, new_PI, new_MU, new_SIGMA, k)
        if the_likelihood>max_likelihood:
            max_likelihood=the_likelihood
            n_comp_max_likelihood = k

    return (n_comp_min_bic, n_comp_max_likelihood)

def return_your_name():
    return 'Chase McGrail'