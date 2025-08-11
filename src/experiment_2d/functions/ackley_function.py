import numpy as np
import autograd.numpy as ag_np


def ackley_function(theta):
    """
    Ackley Function
    
    Parameters:
        theta (array-like): Input vector of length d
    
    Returns:
        float: Value of the Ackley function
    """
    d = len(theta)
    term1 = -20 * ag_np.exp(-0.2 * ag_np.sqrt(ag_np.mean(theta**2)))
    term2 = -ag_np.exp(ag_np.mean(ag_np.cos(2 * ag_np.pi * theta)))
    return term1 + term2 + 20 + ag_np.e
