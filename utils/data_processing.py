import numpy as np
import pandas as pd


def compute_statistics(df_returns: pd.DataFrame, weights: np.ndarray):
    """Return mean and covariance for selected assets and the current weights"""
    # Means per asset
    mu = df_returns.mean(axis=0).values  # shape (k,)
    # Covariance matrix
    Sigma = np.cov(df_returns.values, rowvar=False)  # shape (k,k)
    # Portfolio stats
    port_mean = float(mu @ weights)
    port_var = float(weights @ Sigma @ weights)
    port_std = float(np.sqrt(max(port_var, 1e-16)))
    return mu, Sigma, port_mean, port_std, port_var

class Markowitz:
    def __init__(self, df_returns: pd.DataFrame, lam: float = 1.0):
        self.df_returns = df_returns
        self.lam = lam

        # Means per asset
        self.mu = df_returns.mean(axis=0).values  # shape (k,)
        # Covariance matrix
        self.Sigma = np.cov(df_returns.values, rowvar=False)  # shape (k,k)
    
    def get_objective(self, weights: np.ndarray):
        port_mean = float(self.mu @ weights)
        value = -port_mean + self.lam * float(weights @ self.Sigma @ weights)
        return value
    
    def get_gradient(self, weights: np.ndarray):
        grad = -self.mu + 2.0 * self.lam * (self.Sigma @ weights)
        return grad

    def get_hessian(self, weights: np.ndarray):
        hess = 2.0 * self.lam * self.Sigma
        return hess

def objective_markowitz(df_returns: pd.DataFrame, weights: np.ndarray, lam: float = 1.0):
    """Minimize: -mu^T w + lam * w^T Sigma w  (i.e., trade-off return vs variance)"""
    mu, Sigma, port_mean, _, _ = compute_statistics(df_returns, weights)
    value = -port_mean + lam * float(weights @ Sigma @ weights)
    # Gradient: -mu + 2*lam*Sigma w
    grad = -mu + 2.0 * lam * (Sigma @ weights)
    # Hessian: 2*lam*Sigma
    hess = 2.0 * lam * Sigma
    return value, grad, hess


def objective_sharpe(df_returns: pd.DataFrame, weights: np.ndarray, r_f: float = 0.0, eps: float = 1e-6):
    """Maximize Sharpe = (mu_p - r_f) / sigma_p -> Minimize negative Sharpe: -(mu_p - r_f) / sigma_p"""
    mu, Sigma, port_mean, port_std, _ = compute_statistics(df_returns, weights)
    # Handle degenerate std
    denom = max(port_std, eps)
    excess_return = port_mean - r_f
    value = -excess_return / denom

    # Gradient of -(mu-r_f)/σ: -(mu' * σ - (mu-r_f) * σ') / σ^2
    # mu' = mu vector; σ' = (Sigma w)/σ
    Sigma_w = Sigma @ weights
    grad = -(mu * denom - excess_return * (Sigma_w / denom)) / (denom ** 2)
    # Hessian is complicated; use an approximation (identity scaled) for Newton fallback
    hess = None
    return value, grad, hess
