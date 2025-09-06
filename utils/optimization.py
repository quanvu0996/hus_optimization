import numpy as np


def take_gradient_step(weights, grad, lr):
    """Basic gradient descent step"""
    return weights - lr * grad


def optimizer_gd(df, w, lr, objective_fn, **obj_kwargs):
    """Gradient Descent optimizer"""
    value, grad, _ = objective_fn(df, w, **obj_kwargs)
    w_new = take_gradient_step(w, grad, lr)
    return w_new, value, grad


def optimizer_sgd(df, w, lr, objective_fn, batch_size=1, rng=None, **obj_kwargs):
    """Stochastic Gradient Descent optimizer"""
    if rng is None:
        rng = np.random.default_rng()
    # Sample random rows (days)
    n = len(df)
    idx = rng.choice(n, size=min(batch_size, n), replace=False)
    df_batch = df.iloc[idx]
    value, grad, _ = objective_fn(df_batch, w, **obj_kwargs)
    w_new = take_gradient_step(w, grad, lr)
    return w_new, value, grad


def optimizer_minibatch_gd(df, w, lr, objective_fn, batch_size=16, rng=None, **obj_kwargs):
    """Mini-batch Gradient Descent optimizer"""
    return optimizer_sgd(df, w, lr, objective_fn, batch_size=batch_size, rng=rng, **obj_kwargs)


def optimizer_newton(df, w, lr, objective_fn, **obj_kwargs):
    """Newton's method optimizer"""
    value, grad, hess = objective_fn(df, w, **obj_kwargs)
    if hess is None:
        # Fallback to GD if Hessian not available
        w_new = take_gradient_step(w, grad, lr)
        return w_new, value, grad
    # Regularize Hessian to ensure invertibility
    reg = 1e-6
    hess_reg = hess + reg * np.eye(len(w))
    try:
        step_dir = np.linalg.solve(hess_reg, grad)
    except np.linalg.LinAlgError:
        step_dir = grad
    w_new = w - lr * step_dir
    return w_new, value, grad


def optimizer_nesterov(df, w, lr, objective_fn, momentum_state, momentum=0.9, **obj_kwargs):
    """Nesterov Accelerated Gradient optimizer"""
    v_prev = momentum_state.get("v", np.zeros_like(w))
    lookahead_w = w - momentum * v_prev
    value, grad, _ = objective_fn(df, lookahead_w, **obj_kwargs)
    v = momentum * v_prev + lr * grad
    w_new = w - v
    momentum_state["v"] = v
    return w_new, value, grad


def optimizer_adam(df, w, lr, objective_fn, adam_state, beta1=0.9, beta2=0.999, eps=1e-8, t=1, **obj_kwargs):
    """Adam optimizer"""
    m = adam_state.get("m", np.zeros_like(w))
    v = adam_state.get("v", np.zeros_like(w))
    value, grad, _ = objective_fn(df, w, **obj_kwargs)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    w_new = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    adam_state["m"] = m
    adam_state["v"] = v
    adam_state["t"] = t
    return w_new, value, grad


def optimizer_adagrad(df, w, lr, objective_fn, adagrad_state, eps=1e-8, **obj_kwargs):
    """Adagrad optimizer"""
    # Adagrad: adaptive learning rate based on sum of squared gradients
    G = adagrad_state.get("G", np.zeros_like(w))
    value, grad, _ = objective_fn(df, w, **obj_kwargs)
    G += grad ** 2
    w_new = w - lr * grad / (np.sqrt(G) + eps)
    adagrad_state["G"] = G
    return w_new, value, grad


# Threaded versions for app2.py
def opt_step_gd(df, w, lr, objective_fn):
    """Threaded version of GD for app2.py"""
    value, grad, _ = objective_fn(df, w)
    w_new = take_gradient_step(w, grad, lr)
    return w_new, value


def opt_step_sgd(df, w, lr, objective_fn, batch_size=1, rng=None):
    """Threaded version of SGD for app2.py"""
    if rng is None:
        rng = np.random.default_rng()
    n = len(df)
    idx = rng.choice(n, size=min(batch_size, n), replace=False)
    df_batch = df.iloc[idx]
    value, grad, _ = objective_fn(df_batch, w)
    w_new = take_gradient_step(w, grad, lr)
    return w_new, value


def opt_step_minibatch(df, w, lr, objective_fn, batch_size=16, rng=None):
    """Threaded version of mini-batch GD for app2.py"""
    return opt_step_sgd(df, w, lr, objective_fn, batch_size=batch_size, rng=rng)


def opt_step_newton(df, w, lr, objective_fn):
    """Threaded version of Newton for app2.py"""
    value, grad, hess = objective_fn(df, w)
    if hess is None:
        return take_gradient_step(w, grad, lr), value
    reg = 1e-6
    hess_reg = hess + reg * np.eye(len(w))
    try:
        step_dir = np.linalg.solve(hess_reg, grad)
    except np.linalg.LinAlgError:
        step_dir = grad
    w_new = w - lr * step_dir
    return w_new, value


def opt_step_nesterov(df, w, lr, objective_fn, state):
    """Threaded version of Nesterov for app2.py"""
    v_prev = state.get("v", np.zeros_like(w))
    momentum = state.get("momentum", 0.9)
    lookahead_w = w - momentum * v_prev
    value, grad, _ = objective_fn(df, lookahead_w)
    v = momentum * v_prev + lr * grad
    w_new = w - v
    state["v"] = v
    return w_new, value


def opt_step_adam(df, w, lr, objective_fn, state):
    """Threaded version of Adam for app2.py"""
    beta1 = state.get("beta1", 0.9)
    beta2 = state.get("beta2", 0.999)
    eps = state.get("eps", 1e-8)
    t = int(state.get("t", 0)) + 1
    m = state.get("m", np.zeros_like(w))
    v = state.get("v", np.zeros_like(w))

    value, grad, _ = objective_fn(df, w)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    w_new = w - lr * m_hat / (np.sqrt(v_hat) + eps)

    state.update({"m": m, "v": v, "t": t})
    return w_new, value


def opt_step_adagrad(df, w, lr, objective_fn, state):
    """Threaded version of Adagrad for app2.py"""
    eps = state.get("eps", 1e-8)
    G = state.get("G", np.zeros_like(w))
    value, grad, _ = objective_fn(df, w)
    G += grad ** 2
    w_new = w - lr * grad / (np.sqrt(G) + eps)
    state["G"] = G
    return w_new, value
