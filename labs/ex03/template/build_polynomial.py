# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np
from helpers import standardize

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    if degree <= 0:
        raise ValueError("degree should be >= 1")
    x, _, _ = standardize(x)
    px = np.vstack([x**j for j in range(0,degree+1)])
    return px.T