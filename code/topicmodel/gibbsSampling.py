import numpy as np

def gibbs_sample(X_init, conditional_distribution, num_samples, burn=0):
    # burn = number of iterations used for burn-in
    # conditional_distribution(dimension, current value) should return a value X_i ~ p(z_i|Z_{-i}=X_{-i})
    dim = X_init.shape[0]
    X = X_init
    sample_index = 0
    samples_out = np.zeros([dim,num_samples])
    for i in range(burn + num_samples):
        for d in range(dim):
            X[d]=conditional_distribution(d, X)
        if i >= burn:
            samples_out[:,sample_index]= X[:]
            sample_index += 1
    return samples_out
