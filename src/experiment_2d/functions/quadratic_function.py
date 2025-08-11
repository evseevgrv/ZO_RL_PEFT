import numpy as np 
import autograd.numpy as ag_np


def quadratic_function(theta):
    """
    Quadratic Function
    
    Parameters:
        theta (array-like): Input vector of length d
    
    Returns:
        float: Value of the Quadratic function
    """
    return 0.5 * ag_np.sum(theta**2)
