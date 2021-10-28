# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lambda_I = lambda_* (2 * tx.shape[0]) * np.identity(tx.shape[1])
    A = tx.T @ tx + lambda_I
    w = np.linalg.lstsq(A, tx.T.dot(y), rcond=None)[0]
    # w = np.linalg.solve(A, tx.T.dot(y))
    return w