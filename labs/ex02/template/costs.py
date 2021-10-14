# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_error(y, tx, w):
    return y - tx @ w

def compute_loss(y, tx, w):
    """Calculate the loss with MSE.
    """
    e = compute_error(y, tx, w)

    return np.sum((e ** 2), axis=0) / (2 * len(y)) 