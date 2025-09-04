import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Optimizer Comparison", layout="wide")

# ---------------------------
# Utilities: objectives & grads
# ---------------------------

def compute_statistics(df_returns: pd.DataFrame, weights: np.ndarray):
    mu = df_returns.mean(axis=0).values
    Sigma = np.cov(df_returns.values, rowvar=False)
    port_mean = float(mu @ weights)
    port_var = float(weights @ Sigma @ weights)
    port_std = float(np.sqrt(max(port_var, 1e-16)))
    return mu, Sigma, port_mean, port_std, port_var


def objective_markowitz(df_returns: pd.DataFrame, weights: np.ndarray, lam: float = 1.0):
    mu, Sigma, port_mean, _, _ = compute_statistics(df_returns, weights)
    value = -port_mean + lam * float(weights @ Sigma @ weights)
    grad = -mu + 2.0 * lam * (Sigma @ weights)
    hess = 2.0 * lam * Sigma
    return value, grad, hess


def objective_sharpe(df_returns: pd.DataFrame, weights: np.ndarray, r_f: float = 0.0, eps: float = 1e-6):
    mu, Sigma, port_mean, port_std, _ = compute_statistics(df_returns, weights)
    denom = max(port_std, eps)
    excess_return = port_mean - r_f
    value = -excess_return / denom
    Sigma_w = Sigma @ weights
    grad = -(mu * denom - excess_return * (Sigma_w / denom)) / (denom ** 2)
    hess = None
    return value, grad, hess


# ---------------------------
# Optimizers
# ---------------------------

def take_gradient_step(weights, grad, lr):
    return weights - lr * grad


def opt_step_gd(df, w, lr, objective_fn):
    value, grad, _ = objective_fn(df, w)
    w_new = take_gradient_step(w, grad, lr)
    return w_new, value


def opt_step_sgd(df, w, lr, objective_fn, batch_size=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n = len(df)
    idx = rng.choice(n, size=min(batch_size, n), replace=False)
    df_batch = df.iloc[idx]
    value, grad, _ = objective_fn(df_batch, w)
    w_new = take_gradient_step(w, grad, lr)
    return w_new, value


def opt_step_minibatch(df, w, lr, objective_fn, batch_size=16, rng=None):
    return opt_step_sgd(df, w, lr, objective_fn, batch_size=batch_size, rng=rng)


def opt_step_newton(df, w, lr, objective_fn):
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
    v_prev = state.get("v", np.zeros_like(w))
    momentum = state.get("momentum", 0.9)
    lookahead_w = w - momentum * v_prev
    value, grad, _ = objective_fn(df, lookahead_w)
    v = momentum * v_prev + lr * grad
    w_new = w - v
    state["v"] = v
    return w_new, value


def opt_step_adam(df, w, lr, objective_fn, state):
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


# ---------------------------
# UI
# ---------------------------

st.title("So sánh các thuật toán tối ưu danh mục")
st.write("Chạy nhiều thuật toán song song theo thời gian thực và so sánh: vị trí danh mục trên contour map, giá trị hàm mục tiêu theo iteration và theo thời gian.")

# Data section
st.sidebar.header("Input dữ liệu")
seed = st.sidebar.number_input("Seed sinh dữ liệu (nếu không upload)", value=42)
rng = np.random.default_rng(int(seed))

uploaded = st.sidebar.file_uploader("Upload CSV df_return (n x m)", type=["csv"]) 

if uploaded is not None:
    df_all = pd.read_csv(uploaded)
else:
    n_days, m_assets = 500, 6
    A = rng.normal(size=(m_assets, m_assets))
    cov = A @ A.T / m_assets
    mean = rng.normal(loc=0.0005, scale=0.002, size=m_assets)
    X = rng.multivariate_normal(mean, cov, size=n_days)
    df_all = pd.DataFrame(X, columns=[f"Asset_{i+1}" for i in range(m_assets)])

assets = list(df_all.columns)
short_list = st.sidebar.multiselect("Chọn short-list cổ phiếu (nên chọn 2 để vẽ contour)", assets, default=assets[:2])
if len(short_list) < 1:
    st.warning("Chọn ít nhất 1 cổ phiếu trong short-list.")
    st.stop()

k = len(short_list)
df = df_all[short_list]

# Objectives & hyperparams
objective_name = st.sidebar.selectbox("Hàm mục tiêu", ["Markowitz (mean-variance)", "Sharpe ratio"])
lam = st.sidebar.number_input("Lambda (cho Markowitz)", value=1.0, step=0.1)
r_f = st.sidebar.number_input("Risk-free rate (cho Sharpe)", value=0.0, step=0.001)

lr = st.sidebar.number_input("Learning rate", value=0.1, step=0.01)
batch_size = st.sidebar.number_input("Batch size (mini-batch/SGD)", value=16, step=1, min_value=1)
max_iters = st.sidebar.number_input("Số vòng lặp (iterations)", value=50, step=10, min_value=1)
update_delay = st.sidebar.number_input("Độ trễ cập nhật (giây)", value=0.0, step=0.05, min_value=0.0)

# Optimizer selection
all_opts = [
    "GD",
    "mini-batch GD",
    "SGD",
    "Newton",
    "Nesterov accelerated",
    "Adam",
]
selected_opts = st.sidebar.multiselect("Chọn thuật toán để so sánh", all_opts, default=["GD", "Adam", "Nesterov accelerated"])

# Prepare objective function
if objective_name.startswith("Markowitz"):
    def objective_fn(DF, W):
        return objective_markowitz(DF, W, lam=lam)
else:
    def objective_fn(DF, W):
        return objective_sharpe(DF, W, r_f=r_f)

# Initialize session state for comparison
if "cmp_state" not in st.session_state or st.sidebar.button("Khởi tạo lại state"):
    st.session_state.cmp_state = {}

# Create per-optimizer states
opt_configs = {}
for name in selected_opts:
    state = st.session_state.cmp_state.get(name, {})
    state.setdefault("w", rng.normal(size=k))
    state.setdefault("weights_hist", [state["w"].copy()])
    state.setdefault("obj_hist", [])
    state.setdefault("time_hist", [])
    # method-specific states
    if name == "Nesterov accelerated":
        state.setdefault("v", np.zeros(k))
        state.setdefault("momentum", 0.9)
    if name == "Adam":
        state.setdefault("m", np.zeros(k))
        state.setdefault("v", np.zeros(k))
        state.setdefault("t", 0)
        state.setdefault("beta1", 0.9)
        state.setdefault("beta2", 0.999)
        state.setdefault("eps", 1e-8)
    opt_configs[name] = state

col_run, _ = st.columns([1, 3])
with col_run:
    run = st.button("RUN")

# Placeholders for live charts
top_row = st.container()
col1, col2 = st.columns([1, 1])

# Prepare static layout and placeholders (single render surfaces)
with top_row:
    st.subheader("Biểu đồ 1: Contour và quỹ đạo các thuật toán (k=2)")
    contour_placeholder = top_row.empty()

with col1:
    st.subheader("Biểu đồ 2: Objective theo iteration")
    iter_placeholder = col1.empty()

with col2:
    st.subheader("Biểu đồ 3: Objective theo thời gian (giây)")
    time_placeholder = col2.empty()

if run:
    # Live update loop interleaving optimizers
    start_time = time.perf_counter()

    for it in range(int(max_iters)):
        iter_start = time.perf_counter()
        for name in selected_opts:
            state = opt_configs[name]
            w = state["w"].astype(float)
            # dispatch step
            if name == "GD":
                w_new, obj_val = opt_step_gd(df, w, lr, objective_fn)
            elif name == "mini-batch GD":
                w_new, obj_val = opt_step_minibatch(df, w, lr, objective_fn, batch_size=int(batch_size), rng=rng)
            elif name == "SGD":
                w_new, obj_val = opt_step_sgd(df, w, lr, objective_fn, batch_size=1, rng=rng)
            elif name == "Newton":
                w_new, obj_val = opt_step_newton(df, w, lr, objective_fn)
            elif name == "Nesterov accelerated":
                w_new, obj_val = opt_step_nesterov(df, w, lr, objective_fn, state)
            elif name == "Adam":
                w_new, obj_val = opt_step_adam(df, w, lr, objective_fn, state)
            else:
                continue

            # update state
            state["w"] = w_new
            state["weights_hist"].append(w_new.copy())
            state["obj_hist"].append(float(obj_val))
            state["time_hist"].append(time.perf_counter() - start_time)

        # live drawing into placeholders
        if k == 2:
            w1 = np.linspace(-2.0, 2.0, 50)
            w2 = np.linspace(-2.0, 2.0, 50)
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
            ax.set_title("Mặt đồng mức + quỹ đạo")
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_opts) or 1))
            for color, name in zip(colors, selected_opts):
                hist = np.array(opt_configs[name]["weights_hist"], dtype=float)
                if hist.shape[1] == 2 and len(hist) >= 1:
                    ax.plot(hist[:, 0], hist[:, 1], color=color, linewidth=1.8, label=name)
                    ax.scatter(hist[-1, 0], hist[-1, 1], color=color, s=50)
            ax.legend()
            contour_placeholder.pyplot(fig)
        else:
            contour_placeholder.info("Contour chỉ khả dụng khi chọn đúng 2 cổ phiếu.")

        # iteration chart
        df_iter = pd.DataFrame({name: opt_configs[name]["obj_hist"] for name in selected_opts})
        df_iter.index.name = "iteration"
        iter_placeholder.line_chart(df_iter)

        # time chart
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_opts) or 1))
        for color, name in zip(colors, selected_opts):
            t_hist = np.array(opt_configs[name]["time_hist"]) if len(opt_configs[name]["time_hist"]) > 0 else np.array([0.0])
            y_hist = np.array(opt_configs[name]["obj_hist"]) if len(opt_configs[name]["obj_hist"]) > 0 else np.array([np.nan])
            ax2.plot(t_hist, y_hist, color=color, label=name)
        ax2.set_xlabel("Thời gian (s)")
        ax2.set_ylabel("Objective")
        ax2.legend()
        time_placeholder.pyplot(fig2)

        if update_delay > 0:
            time.sleep(float(update_delay))

    # persist states back to session
    for name in selected_opts:
        st.session_state.cmp_state[name] = opt_configs[name]

else:
    # If not running, show last results if any
    st.subheader("Biểu đồ 1: Contour và quỹ đạo các thuật toán (k=2)")
    if k == 2 and len(selected_opts) > 0 and any("weights_hist" in st.session_state.cmp_state.get(name, {}) for name in selected_opts):
        w1 = np.linspace(-2.0, 2.0, 50)
        w2 = np.linspace(-2.0, 2.0, 50)
        W1, W2 = np.meshgrid(w1, w2)
        Z = np.zeros_like(W1)
        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                ww = np.array([W1[i, j], W2[i, j]], dtype=float)
                z, _, _ = objective_fn(df, ww)
                Z[i, j] = z
        fig, ax = plt.subplots(figsize=(6, 5))
        cs = ax.contourf(W1, W2, Z, levels=30, cmap="viridis")
        fig.colorbar(cs, ax=ax).set_label("Objective value")
        ax.contour(W1, W2, Z, colors="k", linewidths=0.5, levels=15)
        ax.set_xlabel(f"Trọng số {short_list[0]}")
        ax.set_ylabel(f"Trọng số {short_list[1]}")
        ax.set_title("Mặt đồng mức + quỹ đạo")
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_opts) or 1))
        for color, name in zip(colors, selected_opts):
            hist = np.array(st.session_state.cmp_state.get(name, {}).get("weights_hist", []), dtype=float)
            if hist.ndim == 2 and hist.shape[1] == 2 and len(hist) > 0:
                ax.plot(hist[:, 0], hist[:, 1], color=color, linewidth=1.8, label=name)
                ax.scatter(hist[-1, 0], hist[-1, 1], color=color, s=50)
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Nhấn RUN để bắt đầu so sánh (Contour chỉ khả dụng cho 2 cổ phiếu).")

    st.subheader("Biểu đồ 2: Objective theo iteration")
    if len(selected_opts) > 0:
        df_iter = pd.DataFrame({name: st.session_state.cmp_state.get(name, {}).get("obj_hist", []) for name in selected_opts})
        df_iter.index.name = "iteration"
        st.line_chart(df_iter)

    st.subheader("Biểu đồ 3: Objective theo thời gian (giây)")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_opts) or 1))
    for color, name in zip(colors, selected_opts):
        t_hist = np.array(st.session_state.cmp_state.get(name, {}).get("time_hist", []))
        y_hist = np.array(st.session_state.cmp_state.get(name, {}).get("obj_hist", []))
        if len(t_hist) > 0 and len(y_hist) > 0:
            ax2.plot(t_hist, y_hist, color=color, label=name)
    ax2.set_xlabel("Thời gian (s)")
    ax2.set_ylabel("Objective")
    ax2.legend()
    st.pyplot(fig2)
