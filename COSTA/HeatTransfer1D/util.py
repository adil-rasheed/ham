"""
utils.py

Written by Sindre Stenen Blakseth, 2020.

Collection of helper functions.
"""

########################################################################################################################
# Package import.

import numpy as np
from scipy.integrate import fixed_quad

########################################################################################################################
# Function evaluation.

def straight_line_eval(x, x_1, y_1, x_2, y_2):
    """
    Evaluating the straight line passing through (x_1, y_1) and (x_2, y_2) at x.
    """
    slope = (y_2 - y_1) / (x_2 - x_1)
    return y_1 + slope*(x - x_1)

########################################################################################################################
# Normalization/standardization.

def normalize(x, x_min, x_max):
    """
    Normalizing x in [x_min, x_max] -> x_normalized in [-1, 1].
    """
    return 2*(x - x_min)/(x_max - x_min) - 1

def unnormalize(x_normalized, x_min, x_max):
    """
    Unnormalizing x_normalized in [-1, 1] -> x in [x_min, x_max].
    """
    return 0.5*(x_normalized + 1)*(x_max - x_min) + x_min

def z_normalize(x, mean, std):
    """
    :param x: Numpy array containing data points.
    :param mean: Pre-calculated mean to be used for normalization.
    :param std: Pre-calculated standard deviation to be used for normalization.
    :return: Z-scores of the data points in x.
    """
    if mean is None:
        mean = np.mean(x)
    if std is None:
        std  = np.std(x)
    return (x - mean) / std

def z_unnormalize(x_normalized, mean, std):
    """
    :param x_normalized: Numpy array containing z-normalized data points.
    :param mean: Mean of original, unnormalized data.
    :param std: Standard deviation of original, unnormalized data.
    :return: Unnormalized data.
    """
    return (x_normalized * std) + mean


########################################################################################################################
# Norm calculations.

# Linearize numerical solution.
def linearize_between_nodes(x, nodes, T_num):
    assert nodes.shape == T_num.shape
    if type(x) is np.ndarray:
        evals = np.zeros(x.shape[0])
        for i, x_value in enumerate(x):
            for j in range(nodes.shape[0]-1):
                if nodes[j] <= x_value <= nodes[j + 1]:
                    evals[i] = straight_line_eval(x_value, nodes[j], T_num[j], nodes[j+1], T_num[j+1])
                    break
        return evals
    else:
        for i in range(nodes.shape[0]-1):
            if nodes[i] <= x <= nodes[i + 1]:
                return straight_line_eval(x, nodes[i], T_num[i], nodes[i+1], T_num[i+1])

def get_L2_norm(cell_faces, function):
    # Integrate square of function over each cell.
    sq_int_parts = np.zeros(cell_faces.shape[0] + 1)
    for i in range(0, cell_faces.shape[0]-1):
        sq_int_parts[i] = fixed_quad(func = lambda x_lam: (function(x_lam))**2,
                                     a = cell_faces[i], b = cell_faces[i+1], n = 5)[0]

    # Sum integrals of individual cells, and take sqrt of sum.
    norm = np.sqrt(np.sum(sq_int_parts))

    return norm

def get_disc_L2_norm(vector):
    return np.sqrt(np.sum(vector**2))

########################################################################################################################

def main():
    pass

########################################################################################################################

if __name__ == "__main__":
    main()

########################################################################################################################