# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def build_poly(x, degree):
    """
    polynomial basis functions for input data x, for j=0 up to j=degree.
    input x is not standardized
    """
    if degree <= 0:
        raise ValueError("degree should be >= 1")
    x, _, _ = standardize(x)
    px = np.vstack([x**j for j in range(0,degree+1)])
    return px.T