import numpy as np
import autograd.numpy as ag_np


def levy_function(theta):
    """
    Levy Function
    
    Parameters:
        theta (array-like): Input vector of length d
    
    Returns:
        float: Value of the Levy function
    """
    d = len(theta)
    w = 1 + (theta - 1) / 4  
    
    term1 = ag_np.sin(ag_np.pi * w[0])**2
    
    middle_terms = ag_np.sum((w[:-1] - 1)**2 * (1 + 10 * ag_np.sin(ag_np.pi * w[:-1] + 1)**2))
    
    last_term = (w[-1] - 1)**2 * (1 + ag_np.sin(2 * ag_np.pi * w[-1])**2)
    
    return term1 + middle_terms + last_term
