import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Optimization Demo", layout="wide")

# ---------------------------
# Utilities: objectives & grads
# ---------------------------

def compute_statistics(df_returns: pd.DataFrame, weights: np.ndarray):
    # Return mean and covariance for selected assets and the current weights
    # Means per asset
    mu = df_returns.mean(axis=0).values  # shape (k,)
    # Covariance matrix
    Sigma = np.cov(df_returns.values, rowvar=False)  # shape (k,k)
    # Portfolio stats
    port_mean = float(mu @ weights)
    port_var = float(weights @ Sigma @ weights)
    port_std = float(np.sqrt(max(port_var, 1e-16)))
    return mu, Sigma, port_mean, port_std, port_var


def objective_markowitz(df_returns: pd.DataFrame, weights: np.ndarray, lam: float = 1.0):
    # Minimize: -mu^T w + lam * w^T Sigma w  (i.e., trade-off return vs variance)
    mu, Sigma, port_mean, _, _ = compute_statistics(df_returns, weights)
    value = -port_mean + lam * float(weights @ Sigma @ weights)
    # Gradient: -mu + 2*lam*Sigma w
    grad = -mu + 2.0 * lam * (Sigma @ weights)
    # Hessian: 2*lam*Sigma
    hess = 2.0 * lam * Sigma
    return value, grad, hess


def objective_sharpe(df_returns: pd.DataFrame, weights: np.ndarray, r_f: float = 0.0, eps: float = 1e-6):
    # Maximize Sharpe = (mu_p - r_f) / sigma_p -> Minimize negative Sharpe: -(mu_p - r_f) / sigma_p
    # We return value and gradient for minimization
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


# ---------------------------
# Optimizers
# ---------------------------

def take_gradient_step(weights, grad, lr):
    return weights - lr * grad


def optimizer_gd(df, w, lr, objective_fn, **obj_kwargs):
    value, grad, _ = objective_fn(df, w, **obj_kwargs)
    w_new = take_gradient_step(w, grad, lr)
    return w_new, value, grad


def optimizer_sgd(df, w, lr, objective_fn, batch_size=1, rng=None, **obj_kwargs):
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
    return optimizer_sgd(df, w, lr, objective_fn, batch_size=batch_size, rng=rng, **obj_kwargs)


def optimizer_newton(df, w, lr, objective_fn, **obj_kwargs):
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
    v_prev = momentum_state.get("v", np.zeros_like(w))
    lookahead_w = w - momentum * v_prev
    value, grad, _ = objective_fn(df, lookahead_w, **obj_kwargs)
    v = momentum * v_prev + lr * grad
    w_new = w - v
    momentum_state["v"] = v
    return w_new, value, grad


def optimizer_adam(df, w, lr, objective_fn, adam_state, beta1=0.9, beta2=0.999, eps=1e-8, t=1, **obj_kwargs):
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


# ---------------------------
# UI
# ---------------------------

st.title("Demo: Tối ưu hóa danh mục (Modern Portfolio Theory)")
st.write("Hệ thống demo tối ưu hóa danh mục với các thuật toán GD, mini-batch GD, SGD, Newton, Nesterov, Adam. Chọn hàm mục tiêu: Markowitz hoặc Sharpe. Không ràng buộc trọng số (cho phép bán khống/đòn bẩy).")

# Example synthetic returns if none provided
st.sidebar.header("Input dữ liệu")
seed = st.sidebar.number_input("Seed sinh dữ liệu (nếu không upload)", value=42)
rng = np.random.default_rng(int(seed))

uploaded = st.sidebar.file_uploader("Upload CSV df_return (n x m)", type=["csv"]) 

if uploaded is not None:
    df_all = pd.read_csv(uploaded)
else:
    # synthetic data: 500 days x 6 assets
    n_days, m_assets = 500, 6
    # create random correlated returns
    A = rng.normal(size=(m_assets, m_assets))
    cov = A @ A.T / m_assets
    mean = rng.normal(loc=0.0005, scale=0.002, size=m_assets)
    X = rng.multivariate_normal(mean, cov, size=n_days)
    df_all = pd.DataFrame(X, columns=[f"Asset_{i+1}" for i in range(m_assets)])

assets = list(df_all.columns)
short_list = st.sidebar.multiselect("Chọn short-list cổ phiếu", assets, default=assets[:2])

if len(short_list) < 1:
    st.warning("Chọn ít nhất 1 cổ phiếu trong short-list.")
    st.stop()

k = len(short_list)
df = df_all[short_list]

# Objective selection
objective_name = st.sidebar.selectbox("Hàm mục tiêu", ["Markowitz (mean-variance)", "Sharpe ratio"])
lam = st.sidebar.number_input("Lambda (chỉ dùng cho Markowitz)", value=1.0, step=0.1)
r_f = st.sidebar.number_input("Risk-free rate (for Sharpe ratio)", value=0.0, step=0.001)

# Optimizer selection
opt_name = st.sidebar.selectbox(
    "Thuật toán tối ưu",
    [
        "GD",
        "mini-batch GD",
        "SGD",
        "Newton",
        "Nesterov accelerated",
        "Adam",
    ],
)

# Hyperparameters
lr = st.sidebar.number_input("Learning rate", value=0.1, step=0.01)
batch_size = st.sidebar.number_input("Kích thước batch (cho mini-batch/SGD)", value=16, step=1, min_value=1)

# State initialization
if "weights" not in st.session_state or st.sidebar.button("Khởi tạo lại trọng số"):
    st.session_state.weights = rng.normal(size=k)

if "momentum_state" not in st.session_state:
    st.session_state.momentum_state = {}
if "adam_state" not in st.session_state:
    st.session_state.adam_state = {"t": 0}

# Choose objective function
if objective_name.startswith("Markowitz"):
    objective_fn = lambda DF, W: objective_markowitz(DF, W, lam=lam)
else:
    objective_fn = lambda DF, W: objective_sharpe(DF, W, r_f=r_f)

# Display current weights and objective value
w = st.session_state.weights.astype(float)
val, grad, _ = objective_fn(df, w)

# Initialize histories when needed (on first load or when k changes)
if "hist_k" not in st.session_state or st.session_state.get("hist_k") != k or "weight_history" not in st.session_state:
    st.session_state.hist_k = k
    st.session_state.weight_history = [w.copy()]
    st.session_state.objective_history = [float(val)]

st.subheader("Trạng thái hiện tại")
st.write({"weights": w, "objective": float(val)})

col1, col2 = st.columns([1, 1])
with col1:
    run = st.button("RUN_STEP")
with col2:
    reset = st.button("Reset về ngẫu nhiên")

if reset:
    st.session_state.weights = rng.normal(size=k)
    # reset histories
    w0 = st.session_state.weights.astype(float)
    v0, _, _ = objective_fn(df, w0)
    st.session_state.hist_k = k
    st.session_state.weight_history = [w0.copy()]
    st.session_state.objective_history = [float(v0)]
    # st.experimental_rerun()

if run:
    if opt_name == "GD":
        w_new, value, grad = optimizer_gd(df, w, lr, objective_fn)
    elif opt_name == "mini-batch GD":
        w_new, value, grad = optimizer_minibatch_gd(df, w, lr, objective_fn, batch_size=int(batch_size), rng=rng)
    elif opt_name == "SGD":
        w_new, value, grad = optimizer_sgd(df, w, lr, objective_fn, batch_size=1, rng=rng)
    elif opt_name == "Newton":
        w_new, value, grad = optimizer_newton(df, w, lr, objective_fn)
    elif opt_name == "Nesterov accelerated":
        w_new, value, grad = optimizer_nesterov(df, w, lr, objective_fn, st.session_state.momentum_state)
    elif opt_name == "Adam":
        st.session_state.adam_state["t"] = int(st.session_state.adam_state.get("t", 0)) + 1
        w_new, value, grad = optimizer_adam(df, w, lr, objective_fn, st.session_state.adam_state, t=st.session_state.adam_state["t"]) 
    else:
        st.stop()

    st.session_state.weights = w_new
    # append histories
    st.session_state.weight_history.append(w_new.copy())
    st.session_state.objective_history.append(float(value))

    st.success(f"Step done. Objective: {value:.6f}")
    st.write({"weights": w_new, "objective": float(value)})

# Plot contour for 2-asset case and objective history side by side
if k == 2:
    # Create two columns for the charts
    chart_col1, chart_col2 = st.columns([1, 1])
    
    with chart_col1:
        st.subheader("Lịch sử giá trị hàm mục tiêu")
        if "objective_history" in st.session_state and len(st.session_state.objective_history) > 0:
            st.line_chart(pd.DataFrame({"objective": st.session_state.objective_history}))
        else:
            st.write("Chưa có lịch sử. Hãy nhấn RUN_STEP để bắt đầu.")
    
    with chart_col2:
        st.subheader("Contour của hàm mục tiêu (2 tài sản)")
        w1 = np.linspace(-2.0, 2.0, 10)
        w2 = np.linspace(-2.0, 2.0, 10)
        W1, W2 = np.meshgrid(w1, w2)

        Z = np.zeros_like(W1)
        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                ww = np.array([W1[i, j], W2[i, j]], dtype=float)
                z, _, _ = objective_fn(df, ww)
                Z[i, j] = z

        fig, ax = plt.subplots(figsize=(6, 5))
        cs = ax.contourf(W1, W2, Z, levels=30, cmap="viridis")
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label("Objective value")
        ax.contour(W1, W2, Z, colors="k", linewidths=0.5, levels=15)
        ax.set_xlabel(f"Trọng số {short_list[0]}")
        ax.set_ylabel(f"Trọng số {short_list[1]}")
        ax.set_title("Mặt đồng mức hàm mục tiêu")

        # draw path from history (previous in white), and current in red
        hist = np.array(st.session_state.weight_history, dtype=float)
        if hist.shape[1] == 2 and len(hist) >= 1:
            if len(hist) > 1:
                ax.plot(hist[:, 0], hist[:, 1], color="white", linewidth=1.8, alpha=0.95, label="History path")
                ax.scatter(hist[:-1, 0], hist[:-1, 1], color="white", s=18)
            curr = hist[-1]
            ax.scatter([curr[0]], [curr[1]], color="red", s=60, label="Current w")
        ax.legend()
        st.pyplot(fig)

else:
    # For k≠2, show only objective history chart
    if "objective_history" in st.session_state and len(st.session_state.objective_history) > 0:
        st.subheader("Lịch sử giá trị hàm mục tiêu")
        st.line_chart(pd.DataFrame({"objective": st.session_state.objective_history}))

st.caption("Lưu ý: Bài toán không ràng buộc, trọng số có thể âm (bán khống) hoặc >1 (đòn bẩy).")
