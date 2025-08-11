import numpy as np
import autograd.numpy as ag_np


def rosenbrock_function(theta):
    """
    Rosenbrock Function
    
    Parameters:
        theta (array-like): Input vector of length d
    
    Returns:
        float: Value of the Rosenbrock function
    """
    d = len(theta)
    total = 0
    for i in range(d - 1):
        x_i = theta[i]
        x_ip1 = theta[i + 1]
        term1 = 100 * (x_ip1 - x_i**2)**2
        term2 = (1 - x_i)**2
        total += term1 + term2
    return total
