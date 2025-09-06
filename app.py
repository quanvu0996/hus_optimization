import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import utility functions
from utils.data_processing import objective_markowitz, objective_sharpe
from utils.optimization import (
    optimizer_gd, optimizer_sgd, optimizer_minibatch_gd, optimizer_newton,
    optimizer_nesterov, optimizer_adam, optimizer_adagrad
)

st.set_page_config(page_title="Portfolio Optimization Demo", layout="wide")


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
    # synthetic data: user can specify number of days and assets
    n_days = st.sidebar.number_input("Số ngày (n_days)", value=500, min_value=100, step=50)
    m_assets = st.sidebar.number_input("Số tài sản (m_assets)", value=10, min_value=2, step=1)
    # create random correlated returns
    A = rng.normal(size=(m_assets, m_assets))
    cov = A @ A.T / m_assets
    mean = rng.normal(loc=0.0005, scale=0.002, size=m_assets)
    X = rng.multivariate_normal(mean, cov, size=n_days)
    df_all = pd.DataFrame(X, columns=[f"Asset_{i+1}" for i in range(m_assets)])

assets = list(df_all.columns)
short_list = st.sidebar.multiselect("Chọn short-list cổ phiếu", assets, default=assets[:2])

# If no assets selected, use all assets
if len(short_list) == 0:
    short_list = assets

k = len(short_list)
df = df_all[short_list]
mu = df.mean(axis=0).values  # shape (k,)
Sigma = np.cov(df.values, rowvar=False)  # shape (k,k)

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
        "Adagrad",
    ],
)


# Hyperparameters
lr = st.sidebar.number_input("Learning rate", value=0.1, step=0.01)
batch_size = st.sidebar.number_input("Kích thước batch (cho mini-batch/SGD)", value=16, step=1, min_value=1)

inp_ws = st.sidebar.text_input("Initial weights (comma-separated)", value="")
    

# State initialization - ensure weights match current k
reset = st.sidebar.button("Khởi tạo lại trọng số")
if "weights" not in st.session_state or len(st.session_state.weights) != k or reset:
    if inp_ws !="": 
        st.session_state.weights = np.array(list(map(float, inp_ws.split(","))))
    else:
        st.session_state.weights = rng.normal(size=k)

if "momentum_state" not in st.session_state:
    st.session_state.momentum_state = {}
if "adam_state" not in st.session_state:
    st.session_state.adam_state = {"t": 0}
if "adagrad_state" not in st.session_state:
    st.session_state.adagrad_state = {}

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

# st.subheader("Trạng thái hiện tại")
# st.write({"weights": w, "objective": float(val)})

col1, col2 = st.columns([1, 1])
with col1:
    run = st.button("RUN_STEP")
# with col2:
#     reset = st.button("Reset về ngẫu nhiên")

if reset:
    # st.session_state.weights = rng.normal(size=k)
    # reset histories
    w0 = st.session_state.weights.astype(float)
    v0, _, _ = objective_fn(df, w0)
    st.session_state.hist_k = k
    st.session_state.weight_history = [w0.copy()]
    st.session_state.objective_history = [float(v0)]
    # st.experimental_rerun()

w_new, value = None, None
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
    elif opt_name == "Adagrad":
        w_new, value, grad = optimizer_adagrad(df, w, lr, objective_fn, st.session_state.adagrad_state)
    else:
        st.stop()

    st.session_state.weights = w_new
    # append histories
    st.session_state.weight_history.append(w_new.copy())
    st.session_state.objective_history.append(float(value))

    # st.success(f"Step done. Objective: {value:.6f}")

# Plot contour for 2-asset case and objective history side by side
# if k == 2:
# Create two columns for the charts
chart_col1, chart_col2 = st.columns([1, 1])

with chart_col1:
    st.subheader("Lịch sử giá trị hàm mục tiêu")
    if "objective_history" in st.session_state and len(st.session_state.objective_history) > 0:
        st.line_chart(pd.DataFrame({"objective": st.session_state.objective_history}))
    else:
        st.write("Chưa có lịch sử. Hãy nhấn RUN_STEP để bắt đầu.")
    chart_col1.subheader("Trạng thái hiện tại")
    chart_col1.write({"weights": w_new, "objective": value, "mu": mu, "Sigma":Sigma})

with chart_col2:
    if k==2:
        st.subheader("Contour của hàm mục tiêu (2 tài sản)")
        w1 = np.linspace(-3.0, 3.0, 10)
        w2 = np.linspace(-3.0, 3.0, 10)
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
        chart_col2.info("Contour chỉ khả dụng khi chọn đúng 2 cổ phiếu.")

# else:
#     # For k≠2, show only objective history chart
#     if "objective_history" in st.session_state and len(st.session_state.objective_history) > 0:
#         st.subheader("Lịch sử giá trị hàm mục tiêu")
#         st.line_chart(pd.DataFrame({"objective": st.session_state.objective_history}))

bottom_row = st.container()

st.caption("Lưu ý: Bài toán không ràng buộc, trọng số có thể âm (bán khống) hoặc >1 (đòn bẩy).")
