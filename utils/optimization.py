import torch
import numpy as np
from scipy.optimize import minimize

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


# ------------------------------------------------------------
# Optional: Torch-based single-step optimizers (use provided grads)
# ------------------------------------------------------------
def _torch_single_step(df, w, lr, objective_fn, torch_state, opt_ctor, opt_kwargs=None):
    if opt_kwargs is None:
        opt_kwargs = {}
    # Compute value and gradient using provided objective_fn (numpy)
    value, grad, _ = objective_fn(df, w)
    # Initialize param and optimizer if needed
    param = torch_state.get("param")
    optimizer = torch_state.get("optimizer")
    w_tensor = torch.tensor(w, dtype=torch.float32)
    if param is None or param.numel() != w_tensor.numel():
        param = torch.nn.Parameter(w_tensor.clone().detach())
        optimizer = opt_ctor([param], lr=lr, **opt_kwargs)
    else:
        # Sync param with current w
        with torch.no_grad():
            param.copy_(w_tensor)
    # Set gradient from numpy gradient
    if param.grad is not None:
        param.grad.detach_()
        param.grad.zero_()
    param.grad = torch.tensor(grad, dtype=torch.float32)
    optimizer.step()
    w_new = param.detach().cpu().numpy().astype(float)
    torch_state["param"] = param
    torch_state["optimizer"] = optimizer
    return w_new, float(value), grad

def optimizer_torch_adam(df, w, lr, objective_fn, torch_state, eps=1e-8, betas=(0.9, 0.999)):
    return _torch_single_step(df, w, lr, objective_fn, torch_state, torch.optim.Adam, {"eps": eps, "betas": betas})

def optimizer_torch_adagrad(df, w, lr, objective_fn, torch_state, lr_decay=0.0, eps=1e-10):
    return _torch_single_step(df, w, lr, objective_fn, torch_state, torch.optim.Adagrad, {"lr_decay": lr_decay, "eps": eps})

def optimizer_torch_sgd(df, w, lr, objective_fn, torch_state):
    return _torch_single_step(df, w, lr, objective_fn, torch_state, torch.optim.SGD, {})

def optimizer_torch_nesterov(df, w, lr, objective_fn, torch_state, momentum=0.9):
    return _torch_single_step(df, w, lr, objective_fn, torch_state, torch.optim.SGD, {"momentum": momentum, "nesterov": True})



# ------------------------------------------------------------
# Optional: SciPy-based single-step optimizers (minimize with maxiter=1)
# ------------------------------------------------------------
def _scipy_step(df, w, objective_fn, method="BFGS", use_hess=False):
    def fun(x):
        val, _, _ = objective_fn(df, x)
        return float(val)
    def jac(x):
        _, g, _ = objective_fn(df, x)
        return g
    def hess(x):
        _, _, H = objective_fn(df, x)
        return H
    options = {"maxiter": 1, "disp": False}
    if use_hess:
        res = minimize(fun, w, jac=jac, hess=hess, method=method, options=options)
    else:
        res = minimize(fun, w, jac=jac, method=method, options=options)
    x_new = res.x.astype(float)
    val_new = float(res.fun)
    _, g_new, _ = objective_fn(df, x_new)
    return x_new, val_new, g_new

def optimizer_scipy_bfgs(df, w, lr_unused, objective_fn):
    return _scipy_step(df, w, objective_fn, method="BFGS", use_hess=False)

def optimizer_scipy_cg(df, w, lr_unused, objective_fn):
    return _scipy_step(df, w, objective_fn, method="CG", use_hess=False)

def optimizer_scipy_newton_cg(df, w, lr_unused, objective_fn):
    # Newton-CG uses Hessian-vector products; we provide hess if available
    return _scipy_step(df, w, objective_fn, method="Newton-CG", use_hess=True)

def optimizer_scipy_trust_ncg(df, w, lr_unused, objective_fn):
    return _scipy_step(df, w, objective_fn, method="trust-ncg", use_hess=True)
