# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_error(y, tx, w):
    # print(tx.shape, w.shape)
    return y - tx @ w

def compute_loss(y, tx, w):
    """Calculate the loss with MSE.
    """
    e = compute_error(y, tx, w)

    return np.sum((e ** 2), axis=0) / (2 * len(y)) 

def least_squares(y, tx):
    """calculate the least squares solution.
    Return optimal weight, mse loss"""
    w = np.linalg.lstsq(tx, y, rcond=None)[0] # if X is not full rank, w minimize MAE
    mse = compute_loss(y, tx, w)
    return w, mse