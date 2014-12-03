import numpy as np

def gibbs_sample(X_init, conditional_distribution, num_samples, burn=0):
    dim = X_init.shape[0]
    sample_index = 0
    samples_out = np.zeros([dim,num_samples])
    for i in range(burn + num_samples):

        if i >= burn:
            sample_index += 1

    return samples_out
