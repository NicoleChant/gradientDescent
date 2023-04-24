import numpy as np
from typing import Optional, Callable

def h(X, z, w):
    return z * X + w

def loss(X, y, z, w):
    return np.mean((y - h(X, z, w))**2)

def r2_score(y, y_pred):
    # TODO
    pass

def gradient(X, y, z, w):
    y_hat = h(X, z, w)
    d_z = -2 * np.dot(X, y - y_hat)
    d_w = -2 * np.dot(np.ones(X.shape), y - y_hat)
    return d_z, d_w

def gradient_descent(X, y, z0: float, w0: float,
                     epochs: int,
                     batch_size: Optional[int] = None,
                     learning_rate: float = 0.01):
    z, w = z0, w0
    for _ in range(epochs):
        cur_loss = loss(X, y, z, w)
        yield z, w, cur_loss

        d_z, d_w = gradient(X, y, z, w)
        z -= learning_rate * d_z
        w -= learning_rate * d_w


def gradient_descent_minibatch(X, y, z0: float, w0: float,
                               epochs: int,
                               batch_size: int,
                               learning_rate: float):
    index = np.array(X.index)
    z, w = z0, w0

    for _ in range(int(epochs)):
        np.random.shuffle(index)
        for i in range(index.shape[0] // batch_size):

            cur_loss = loss(X, y, z, w)
            yield z, w, cur_loss

            index_slice = index[i * batch_size:(i + 1) * batch_size]
            mini_batch_X = X[X.index.isin(index_slice)]
            mini_batch_y = y[y.index.isin(index_slice)]

            d_z, d_w = gradient(mini_batch_X, mini_batch_y, z, w)
            z -= learning_rate * d_z
            w -= learning_rate * d_w

def stochastic_gradient_descent(X, y, z0: float, w0: float, epochs: int,
                                batch_size: Optional[int] = None,
                                learning_rate: float = 0.01):
    # TODO
    pass


solvers: dict[str, Callable] = {
        "normal": gradient_descent,
        "minibatch": gradient_descent_minibatch,
        "stochastic": stochastic_gradient_descent,
    }
