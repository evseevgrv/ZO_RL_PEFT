import numpy as np


def ackley_function(theta):
    """
    Ackley Function
    
    Parameters:
        theta (array-like): Input vector of length d
    
    Returns:
        float: Value of the Ackley function
    """
    d = len(theta)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.mean(theta**2)))
    term2 = -np.exp(np.mean(np.cos(2 * np.pi * theta)))
    return term1 + term2 + 20 + np.e

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
    
    term1 = np.sin(np.pi * w[0])**2
    
    middle_terms = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    
    last_term = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    
    return term1 + middle_terms + last_term

def quadratic_function(theta):
    """
    Quadratic Function
    
    Parameters:
        theta (array-like): Input vector of length d
    
    Returns:
        float: Value of the Quadratic function
    """
    return 0.5 * np.sum(theta**2)

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
