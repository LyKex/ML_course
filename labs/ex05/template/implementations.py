# -*- coding: utf-8 -*-
"""ML functions implementation"""
import numpy as np
from utility import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression with GD"""
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * grad
        print(
            "Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss
            )
        )

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression with SGD"""
    w = initial_w
    n_iter = 0
    batch_size = 1
    for batch_y, batch_x in batch_iter(y, tx, batch_size, num_batches=max_iters):
        grad = compute_gradient(batch_y, batch_x, w)
        loss = compute_mse(batch_y, batch_x, w)
        w = w - gamma * grad

        n_iter += 1
        if n_iter % 100 == 0:
            print("SGD ({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters, l=loss))

    return w, loss


def least_squares(y, tx):
    """least square using normal equation"""
    A = tx.T @ tx
    w = np.linalg.lstsq(A, tx.T.dot(y), rcond=None)[0]
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """ridge regression using normal equation"""
    lambda_I = lambda_ * (2 * tx.shape[0]) * np.identity(tx.shape[1])
    A = tx.T @ tx + lambda_I
    w = np.linalg.lstsq(A, tx.T.dot(y), rcond=None)[0]
    return w, compute_mse(y, tx, w)

def logstic_regression(y, tx, initial_w, max_iters, gamma):
    """logistic regression with gradient descent"""
    # init parameters
    w = initial_w
    
    # start the logistic regression
    for iter in range(max_iters):
        # get reg loss and update w.
        ls, w = learning_by_gradient(y, tx, w, gamma, np.inf)
        if iter % 100 == 0:
            print(
                "Current iteration={i}, loss: {ls_reg}".format(
                    i=iter, ls_reg=ls
                )
            )

    return w, ls


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """regularized logistic regression with gradient descent"""
    # init parameters
    w = initial_w
    
    # start the logistic regression
    for iter in range(max_iters):
        # get reg loss and update w.
        ls, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_, np.inf)
        if iter % 100 == 0:
            print(
                "Current iteration={i}, loss: {ls_reg}".format(
                    i=iter, ls_reg=ls
                )
            )

    return w, ls
