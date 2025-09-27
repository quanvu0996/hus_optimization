"""
Portfolio management prototype
- Market class reads CSV of prices (n x m)
- Agent class optimizes Markowitz on last k days and executes trades if fee payable
- Simulation loop computes NAV over time
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

class Market:
    """
    Market reads a CSV file with shape (n_days, m_assets).
    CSV: each column is one asset (header optional), each row is daily close price.
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, header=0)
        # Ensure numeric and drop NaNs (simple treatment)
        df = df.select_dtypes(include=[np.number]).fillna(method='ffill').fillna(method='bfill')
        self.prices = df.values.astype(float)  # shape (n, m)
        self.n, self.m = self.prices.shape

    def get_data_t(self, t):
        """
        Return prices observed at beginning of day t: X[0:t], shape (t, m).
        Accept t in [0, n]. If t==0 returns empty array shape (0, m).
        """
        if t < 0 or t > self.n:
            raise ValueError("t out of range")
        return self.prices[:t].copy()  # up to day t-1 inclusive

    def price_at(self, t):
        """
        Return price vector at day t (0-indexed)
        """
        return self.prices[t].copy()


class MarkowitzAgent:
    """
    Agent with cash + m assets holdings.
    - cash: C_t
    - shares: q_{t,i}
    - r_fee: transaction fee rate (proportional)
    - lambda_: risk aversion coefficient
    - k: window for returns & covariance
    """
    def __init__(self, m_assets, initial_capital=1_000_000.0, r_fee=0.001, lambda_=10.0, k=20):
        self.m = m_assets
        self.C = float(initial_capital)     # cash in currency units
        self.q = np.zeros(self.m, dtype=float)  # holdings (fractional shares allowed)
        self.r_fee = float(r_fee)
        self.lambda_ = float(lambda_)
        self.k = int(k)

    def nav(self, prices):
        """Compute NAV given current cash and prices (prices: shape (m,))"""
        return self.C + float(np.dot(self.q, prices))

    def _estimate_mean_cov(self, price_hist):
        """
        price_hist: array shape (t, m) where t >= k
        Return mean returns vector (m,) and covariance matrix (m,m) computed from last k days
        using simple returns r_j = (p_j - p_{j-1})/p_{j-1}.
        """
        t = price_hist.shape[0]
        if t < 2:
            # Not enough data -> zero mean and small diag covariance
            mu = np.zeros(self.m)
            cov = np.eye(self.m) * 1e-6
            return mu, cov

        # compute simple daily returns r_{j} for j=1..t-1
        rets = (price_hist[1:] - price_hist[:-1]) / (price_hist[:-1] + 1e-12)  # shape (t-1, m)
        # take last k returns
        last = rets[-self.k:] if rets.shape[0] >= self.k else rets
        mu = np.mean(last, axis=0)
        # sample covariance (rows = observations)
        if last.shape[0] >= 2:
            cov = np.cov(last, rowvar=False, bias=False)  # shape (m, m)
        else:
            cov = np.eye(self.m) * 1e-6
        # regularize covariance to avoid singularity
        cov += np.eye(self.m) * 1e-8
        return mu, cov

    def run_eval_t(self, price_hist):
        """
        Given X[0:t] (price history up to day t-1), compute Markowitz-optimal weights w (m,)
        Constraints: w_i >= 0, sum(w) <= 1  (allow cash remainder)
        Objective: maximize w^T mu - lambda * w^T Sigma w
        We solve via scipy.optimize.minimize by minimizing negative objective.
        Returns w (m,) in [0,1], sum<=1
        """
        mu, cov = self._estimate_mean_cov(price_hist)
        m = self.m
        lam = self.lambda_

        # objective to minimize
        def obj(w):
            # w as numpy array
            w = np.asarray(w)
            return - (w @ mu - lam * (w @ cov @ w))

        # initial guess: proportional to positive mu (or uniform small)
        w0 = np.maximum(mu, 0.0)
        if w0.sum() <= 0:
            w0 = np.ones(m) / m * 0.1
        else:
            w0 = w0 / w0.sum() * 0.5  # start with 50% allocated to risky assets

        # constraints: w_i >=0, sum(w) <= 1
        bounds = [(0.0, 1.0) for _ in range(m)]
        cons = ({'type': 'ineq', 'fun': lambda w: 1.0 - np.sum(w)})  # 1 - sum(w) >= 0 -> sum(w) <=1

        res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol':1e-9, 'disp': False, 'maxiter': 200})
        if not res.success:
            # fallback: project to feasible region
            w = np.clip(w0, 0, 1)
            s = w.sum()
            if s > 1:
                w = w / s
        else:
            w = res.x
        # numerical safety
        w = np.maximum(w, 0.0)
        if w.sum() > 1.0:
            w = w / w.sum() * 1.0
        return w

    def change_portfolio(self, w_target, prices):
        """
        Execute change from current holdings (q, C) to target weight allocation w_target (m,).
        - prices: current price vector at day t (used to compute desired holdings)
        - fees: r_fee * traded value
        Behavior:
          * compute NAV
          * compute dollar target A_i = w_i * NAV
          * desired shares q_target = A_i / p_i
          * traded value = sum |q_target - q| * p_i
          * fee = r_fee * traded_value
          * if C >= fee: execute: C <- C - fee + residual_cash (1 - sum w) * NAV
            q <- q_target
          * else: do nothing (skip trade)
        Returns: dict with info (executed:bool, fee, turnover, nav_before, nav_after)
        """
        prices = np.asarray(prices, dtype=float)
        nav_before = self.nav(prices)
        # desired allocations in currency
        A = w_target * nav_before
        # desired shares
        q_target = A / (prices + 1e-12)
        turnover = float(np.sum(np.abs(q_target - self.q) * prices))
        fee = float(self.r_fee * turnover)

        executed = False
        if self.C >= fee - 1e-12:
            # execute
            # subtract fee from cash
            self.C -= fee
            # set holdings to target
            self.q = q_target.copy()
            # set residual cash as remaining part of NAV not invested in risky assets
            invested = float(np.sum(self.q * prices))
            # residual from allocation (numerical): nav_before - invested is cash (should match (1-sum w)*NAV)
            residual_cash = nav_before - invested
            # avoid negative due to numeric
            self.C = max(self.C + residual_cash, 0.0)
            executed = True
        else:
            # cannot execute trade due to insufficient cash to pay fee
            executed = False
            fee = 0.0
            turnover = 0.0

        nav_after = self.nav(prices)
        return {
            "executed": executed,
            "fee": fee,
            "turnover": turnover,
            "nav_before": nav_before,
            "nav_after": nav_after
        }


def simulate(market: Market, agent: MarkowitzAgent, start_capital=None, start_day=None, verbose=False):
    """
    Run simulation:
      - initialize agent (optionally set initial capital)
      - loop t from k to n-2 (so we can realize price at t+1)
    Returns history dict with NAV series, actions, fees, etc.
    """
    n, m = market.n, market.m
    if start_capital is not None:
        agent.C = float(start_capital)
    # initialize holdings zero (cash only)
    agent.q = np.zeros(m)

    # choose starting day
    t0 = agent.k
    if start_day is not None:
        t0 = max(t0, start_day)

    nav_history = []
    fee_history = []
    turnover_history = []
    w_history = []
    days = []

    for t in range(t0, n-1):  # need p_{t+1} to compute next-day NAV
        X_t = market.get_data_t(t)  # X[0:t]
        # compute weights
        w = agent.run_eval_t(X_t)  # shape (m,)
        prices_t = market.price_at(t)  # prices at day t (we execute trade at end of day t)
        info = agent.change_portfolio(w, prices_t)
        # realize next-day NAV at price p_{t+1}
        prices_tp1 = market.price_at(t+1)
        nav_next = agent.nav(prices_tp1)
        nav_history.append(nav_next)
        fee_history.append(info["fee"])
        turnover_history.append(info["turnover"])
        w_history.append(w.copy())
        days.append(t+1)  # NAV at day t+1
        if verbose:
            print(f"t={t}: executed={info['executed']} fee={info['fee']:.2f} nav={nav_next:.2f}")
    return {
        "days": np.array(days),
        "nav": np.array(nav_history),
        "fee": np.array(fee_history),
        "turnover": np.array(turnover_history),
        "weights": np.vstack(w_history) if w_history else np.empty((0, m))
    }


market = Market("data/df_price_adjusted.csv")
agent = MarkowitzAgent(m_assets=market.m, initial_capital=1_000.0, r_fee=0.001, lambda_=20.0, k=20)
res = simulate(market, agent, start_capital=1_000.0, verbose=True)