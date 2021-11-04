# -*- coding: utf-8 -*-
"""ML utility functions"""
import numpy as np
import matplotlib.pyplot as plt

# ----------------------#
### data processing ###
# ----------------------#
def standardize(x):
    """Standardize the train data set."""
    mean_x = x.mean(axis=0)
    std_x = x.std(axis=0)
    x = x - mean_x
    x = x / std_x
    return x, mean_x, std_x


def standardize_test(x, mean_x, std_x):
    """Standardize the test data set."""
    x = x - mean_x
    x = x / std_x
    return x


def build_poly(tx, degree):
    """
    Generate polynomial basis functions for input data tx, for j=1 up to j=degree,
    without combination of features.
    eg. Input = [x1, x2] degree =2, Output = [1, x1, x1**2, x2, x2**2, x1x2]
    """
    if degree <= 0:
        raise ValueError("degree should be larger than 0")
    px = np.hstack([tx ** j for j in range(1, degree + 1)])
    return px


def feature_expansion(x):
    """Polynomial feature expansion of degree 2 with combination"""
    indices = np.triu_indices(x.shape[1])  # triangular matrices indices
    x_poly = np.multiply(
        x[:, indices[0]], x[:, indices[1]]
    )  # multiply the elements with triangular indices
    return x_poly


def add_shift(tx):
    """add shift i.e. ones to the first column of x"""
    return np.hstack([np.ones((tx.shape[0], 1)), tx])


# ----------------------#
###     Training    ###
# ----------------------#
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = (batch_num * batch_size) % len(y)
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""

    # get k'th subgroup in test, others in train:
    x_te = x[k_indices[k]]
    y_te = y[k_indices[k]]
    # print('test x shape:{}, test y shape:{}'.format(x_te.shape, y_te.shape))
    k_fold = k_indices.shape[0]
    tr_fold = np.r_[np.arange(0, k), np.arange(k + 1, k_fold)]
    x_tr = x[k_indices[tr_fold].flatten()]
    y_tr = y[k_indices[tr_fold].flatten()]
    # print('train x shape:{}, train y shape:{}'.format(x_tr.shape, y_tr.shape))

    # form data with polynomial degree
    px_tr = build_poly(x_tr, degree)
    px_te = build_poly(x_te, degree)

    # ridge regression
    w = ridge_regression(y_tr, px_tr, lambda_)

    # calculate the loss for train and test data
    mse_tr = compute_mse(y_tr, px_tr, w)
    mse_te = compute_mse(y_te, px_te, w)
    loss_tr = np.sqrt(2 * mse_tr)
    loss_te = np.sqrt(2 * mse_te)

    return loss_tr, loss_te


# ----------------------#
###   model building  ###
# ----------------------#

## helper functions for linear model ##


def compute_error(y, tx, w):
    return y - tx @ w


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = compute_error(y, tx, w)
    return e.dot(e) / (2 * len(e))


def compute_mae(y, tx, w):
    """compute the loss by mae (more stable for outliers)"""
    e = compute_error(y, tx, w)
    return np.abs(e).sum() / (2 * len(e))


def compute_gradient(y, tx, w):
    """gradient linear regression GD with MSE loss"""
    e = compute_error(y, tx, w)
    n = tx.shape[0]
    return (-1 / n) * (tx.T @ e)


## linear regression with gradient descent ##
def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm for linear regression"""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print(
            "Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return losses, ws


## linear regression with stochastic gradient descent ##
def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm for linear regression"""
    ws = [initial_w]
    losses = []
    w = initial_w
    n_iter = 0
    for batch_y, batch_x in batch_iter(y, tx, batch_size, num_batches=max_iters):
        grad = compute_gradient(batch_y, batch_x, w)
        loss = compute_mse(batch_y, batch_x, w)
        w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
        n_iter += 1
        print("SGD ({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters, l=loss))
    return losses, ws


def linear_sgd(y, x, max_iters=100, maxstep=0.5, gamma=0.1, batch_size=1):
    """
    Linear regression with SGD
    """
    D = x.shape[1]
    w_initial = np.ones(D)

    ws = [w_initial]
    losses = []
    w = w_initial
    n_iter = 0
    min_loss = np.inf
    w_opt = w_initial
    for batch_y, batch_x in batch_iter(y, x, batch_size, num_batches=max_iters):
        grad = compute_gradient(batch_y, batch_x, w)
        grad = np.clip(grad, -maxstep, maxstep)  # limit step size
        loss = compute_mse(batch_y, batch_x, w)
        w = w - gamma * grad
        ws.append(w)
        losses.append(loss)
        if loss < min_loss:
            min_loss = loss
            w_opt = w
        n_iter += 1
        if n_iter % 100 == 0:
            print("SGD ({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters, l=loss))

    return w_opt


## linear regression with normal equations ##
def least_square(y, tx):
    """least square using normal equation"""
    A = tx.T @ tx
    w = np.linalg.lstsq(A, tx.T.dot(y), rcond=None)[0]
    return w


## ridge regression with normal equations ##
def ridge_regression(y, tx, lambda_):
    """ridge regression using normal equation"""
    lambda_I = lambda_ * (2 * tx.shape[0]) * np.identity(tx.shape[1])
    A = tx.T @ tx + lambda_I
    w = np.linalg.lstsq(A, tx.T.dot(y), rcond=None)[0]
    return w


## helper functions for logistic model ##
def sigmoid(t):
    """apply the sigmoid function on t."""
    # lower bound at -100 to precent overflow
    t[np.where(t < -100)] = -100
    sig = 1 / (1 + np.exp(-t))
    # add small number to make sure no exact -1 or 1
    return np.clip(sig, -1 + 1e-15, 1 - 1e-15)


def compute_loss_sigmoid(y, tx, w):
    """compute the loss: negative log likelihood."""
    # y, w = y.flatten(), w.flatten()
    yp = sigmoid(tx @ w)
    l = np.log(yp).dot(y) + np.log(1 - yp).dot(1 - y)
    return -l


def compute_gradient_sigmoid(y, tx, w):
    """compute the gradient of loss."""
    # y = y.flatten()
    # print(tx.T.shape, (sigmoid(tx @ w) - y).shape)
    return tx.T @ (sigmoid(tx @ w) - y)


def compute_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    pred = sigmoid(tx @ w)
    pred = np.diag(pred.flatten())
    # element wise product, not matrix product
    S = np.multiply(pred, 1 - pred)
    hess = tx.T @ S @ tx
    return hess


def logistic_regression_newton(y, tx, w):
    """return the loss, gradient, and Hessian."""
    loss = compute_loss_sigmoid(y, tx, w)
    grad = compute_gradient_sigmoid(y, tx, w)
    hess = compute_hessian(y, tx, w)
    return loss, grad, hess


## logistic regression with newton method ##


def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, grad, hess = logistic_regression_newton(y, tx, w)
    z = np.linalg.solve(hess, grad)
    w = w - gamma * z
    return loss, w


def logistic_regression_newton(y, tx, max_iter=100, threshold=1e-8, gamma=1.0):
    """
    Logistic regression with Newton's method. Return optimal loss and weight.

    max_iter: max iteration
    threshold: convergence threshold
    gamma: step size

    Note: will not work in this project for the number of data points exceeds
    the capability for normal laptop to store the Hessian matrix.
    """
    # init parameters
    losses = []
    w = np.zeros(tx.shape[1])
    weights = []
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_newton_method(y, tx, w, gamma)
        # log info
        if iter % 1 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        weights.append(w)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # chose optimal
    id_opt = np.argmin(losses)

    print("final loss={l}".format(l=losses[-1]))
    print("min loss={l}".format(l=losses[id_opt]))
    return losses[-1], weights[id_opt]


## logistic regression with gradient descent ##


def learning_by_gradient(y, tx, w, gamma, maxstep):
    """
    Do one step of gradient descent, using the logistic regression.
    Return the loss and updated w.
    """
    loss = compute_loss_sigmoid(y, tx, w)
    grad = compute_gradient_sigmoid(y, tx, w)
    grad = np.clip(grad, -maxstep, maxstep)  # limit step size
    # update w
    w = w - gamma * grad
    return loss, w


def logstic_regression(
    y,
    tx,
    max_iter=300,
    gamma=0.1,
    pred_threshold=0.5,
    conv_threshold=1e-6,
    isplot=False,
    maxstep=0.5,
):
    """
    logistic regression with gradient descent

    max_iter: maximal number of iteration for GD
    gamma: step size, w = w - gamma * grad
    pred_threshold: tx @ w < pred_threshold --> non-higgs; others --> higgs
    conv_threshold: convergence threshold
    maxstep: max stepsize on gradient
    isplot: whethre to plot optimization process
    """
    # init parameters
    losses = []  # cross entropy loss
    accrs = []
    min_loss = np.inf
    w = np.zeros(tx.shape[1])
    # start the logistic regression
    for iter in range(max_iter):
        # decrease step size dynamically
        if iter % 10 == 0:
            gamma = 0.8 * gamma
        # get reg loss and update w.
        ls, w = learning_by_gradient(y, tx, w, gamma, maxstep)
        losses.append(ls)
        if isplot:
            accrs.append([iter, get_accuracy(pred_logistic(tx, w, pred_threshold), y)])
        if iter % 100 == 0:
            print(
                "Current iteration={i}, loss: {ls_reg}".format(
                    i=iter, ls_reg=ls
                )
            )

        # store best weright
        if ls < min_loss:
            min_loss = ls
            w_opt = w
        # converge criterion
        if iter > 10 and np.abs(losses[-1] - losses[-2]) < conv_threshold:
            print("gradient descent converged at iteration {}".format(iter))
            break
    accr = get_accuracy(pred_logistic(tx, w_opt, pred_threshold), y)
    print("final loss={}".format(losses[-1]))
    print("min loss={}".format(min_loss))
    print("accuracy {}".format(accr))

    if isplot:
        accrs = np.array(accrs)
        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(losses, label="cross entropy loss")
        ax2.plot(accrs[:, 0], accrs[:, 1], label="accuracy", color="r")
        ax1.set_title("optimization for logistic regression with GD")
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("loss")
        ax2.set_ylabel("accuracy", color="r")
        ax1.legend(loc="center right")
        plt.savefig("logistic_gd_optimization")
    return w_opt


## Regularized logistic regression using gradient descent ##


def penalized_logistic_regression(y, tx, w, lambda_):
    """
    return the loss, gradient. Note this loss is not for
    evaluation
    """
    # w = w.flatten()
    loss = compute_loss_sigmoid(y, tx, w) + lambda_ * (w.T @ w)
    grad = compute_gradient_sigmoid(y, tx, w) + 2 * lambda_ * w
    # hess = calculate_hessian(y, tx, w)
    return loss, grad


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_, maxstep):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    grad = np.clip(grad, -maxstep, maxstep)  # limit step size
    # update w
    w = w - gamma * grad
    return loss, w


def logistic_regression_penalized_gradient_descent(
    y,
    tx,
    max_iter=300,
    gamma=0.1,
    lambda_=0.1,
    pred_threshold=0.5,
    conv_threshold=1e-6,
    isplot=False,
    maxstep=0.5,
):
    """
    regularized logistic regression with gradient descent

    max_iter: maximal number of iteration for GD
    gamma: step size, w = w - gamma * grad
    lambda_: coefficient for penalizing term in regularized logistic regression
    pred_threshold: tx @ w < pred_threshold --> non-higgs; others --> higgs
    conv_threshold: convergence threshold
    maxstep: max stepsize on gradient
    isplot: whethre to plot optimization process
    """
    # init parameters
    losses_reg = []  # loss with regularization term
    losses = []  # cross entropy loss
    accrs = []
    min_loss = np.inf
    w = np.zeros(tx.shape[1])
    # start the logistic regression
    for iter in range(max_iter):
        # decrease step size dynamically
        if iter % 10 == 0:
            gamma = 0.8 * gamma
        # get reg loss and update w.
        ls_reg, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_, maxstep)
        losses_reg.append(ls_reg)
        if isplot:
            # log info for plotting
            losses.append(compute_loss_sigmoid(y, tx, w))
            accrs.append([iter, get_accuracy(pred_logistic(tx, w, pred_threshold), y)])
        if iter % 100 == 0:
            print(
                "Current iteration={i}, regularized loss: {ls_reg}".format(
                    i=iter, ls_reg=ls_reg
                )
            )

        # store best weright
        if ls_reg < min_loss:
            min_loss = ls_reg
            w_opt = w
        # converge criterion
        if iter > 1 and np.abs(losses_reg[-1] - losses_reg[-2]) < conv_threshold:
            print("gradient descent converged at iteration {}".format(iter))
            break
    accr = get_accuracy(pred_logistic(tx, w_opt, pred_threshold), y)
    print("final loss={}".format(losses_reg[-1]))
    print("min loss={}".format(min_loss))
    print("accuracy {}".format(accr))

    if isplot:
        accrs = np.array(accrs)
        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(losses, label="cross entropy loss")
        ax1.plot(losses_reg, label="regularized loss")
        ax2.plot(accrs[:, 0], accrs[:, 1], label="accuracy", color="r")
        ax1.set_title("regularized logistic regression with GD")
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("loss")
        ax2.set_ylabel("accuracy", color="r")
        ax1.legend(loc="center right")
        plt.savefig("logistic_gd_optimization")
    return w_opt


# ----------------------------#
### Evaluation and tuning ###
# ---------------------------#
def pred_logistic(x, w, threshold=0.5, for_submission=False):
    """
    Return prediction from logistic regression model
    labels are (0, 1) by default
    """
    pred = sigmoid(x @ w)
    threshold_y = np.ones(pred.shape[0])
    if for_submission:
        threshold_y[np.where(pred < threshold)] = -1
    else:
        threshold_y[np.where(pred < threshold)] = 0
    return threshold_y


def pred_linear(x, w, threshold=0):
    """
    Return prediction from linear model
    labels are (-1, 1)
    """
    pred = x @ w
    threshold_y = np.ones(pred.shape[0])
    threshold_y[np.where(pred < threshold)] = -1
    return threshold_y


def get_accuracy(pred, true):
    """return accuracy of prediction"""
    assert len(pred) == len(true)
    return (pred == true).sum() / len(true)


def cv_eval(y, x, k_fold, model, seed=1, islinear=False, threshold=0.5, **kwargs):
    """
    return the training loss, test loss and accuracy by cross validation

    model: the method for training model, should return only weight
    """
    k_indices = build_k_indices(y, k_fold, seed)
    tr_loss = []
    te_loss = []
    accrs = []
    for k in range(k_fold):
        # print("Begin training on fold {}".format(k))
        # get k'th subgroup in test set, others in train set
        x_te = x[k_indices[k]]
        y_te = y[k_indices[k]]

        tr_fold = np.r_[np.arange(0, k), np.arange(k + 1, k_fold)]
        x_tr = x[k_indices[tr_fold].flatten()]
        y_tr = y[k_indices[tr_fold].flatten()]

        # train the model
        w = model(y_tr, x_tr, **kwargs)

        # calculate the loss and accuracy
        if islinear:
            pred = pred_linear(x_te, w, threshold=threshold)
        else:
            pred = pred_logistic(x_te, w, threshold=threshold)
        accrs.append(get_accuracy(pred, y_te))
        tr_loss.append(compute_loss_sigmoid(y_tr, x_tr, w))
        te_loss.append(compute_loss_sigmoid(y_te, x_te, w))

    return np.mean(tr_loss), np.mean(te_loss), np.mean(accrs)


def cv_eval_notrain(y, x, w, k_fold, threshold, seed=1, islinear=False):
    """
    return the training loss, test loss and accuracy by cross validation
    given trained model w
    """
    k_indices = build_k_indices(y, k_fold, seed)
    tr_loss = []
    te_loss = []
    accrs = []
    for k in range(k_fold):
        # print("Begin training on fold {}".format(k))
        # get k'th subgroup in test set, others in train set
        x_te = x[k_indices[k]]
        y_te = y[k_indices[k]]

        tr_fold = np.r_[np.arange(0, k), np.arange(k + 1, k_fold)]
        x_tr = x[k_indices[tr_fold].flatten()]
        y_tr = y[k_indices[tr_fold].flatten()]

        # calculate the loss and accuracy
        if islinear:
            pred = pred_linear(x_te, w, threshold=threshold)
            tr_loss.append(compute_mse(y_tr, x_tr, w))
            te_loss.append(compute_mse(y_te, x_te, w))
        else:
            pred = pred_logistic(x_te, w, threshold=threshold)
            tr_loss.append(compute_loss_sigmoid(y_tr, x_tr, w))
            te_loss.append(compute_loss_sigmoid(y_te, x_te, w))
        accrs.append(get_accuracy(pred, y_te))

    return np.mean(tr_loss), np.mean(te_loss), np.mean(accrs)
