import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading

# Import utility functions
from utils.data_processing import objective_markowitz, objective_sharpe
from utils.optimization import (
    opt_step_gd, opt_step_sgd, opt_step_minibatch, opt_step_newton,
    opt_step_nesterov, opt_step_adam, opt_step_adagrad
)

st.set_page_config(page_title="Portfolio Optimizer Comparison", layout="wide")


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
    n_days = st.sidebar.number_input("Số ngày (n_days)", value=500, min_value=100, step=50)
    m_assets = st.sidebar.number_input("Số tài sản (m_assets)", value=10, min_value=2, step=1)
    A = rng.normal(size=(m_assets, m_assets))
    cov = A @ A.T / m_assets
    mean = rng.normal(loc=0.0005, scale=0.002, size=m_assets)
    X = rng.multivariate_normal(mean, cov, size=n_days)
    df_all = pd.DataFrame(X, columns=[f"Asset_{i+1}" for i in range(m_assets)])

assets = list(df_all.columns)
short_list = st.sidebar.multiselect("Chọn short-list cổ phiếu (nên chọn 2 để vẽ contour)", assets, default=assets[:2])

# If no assets selected, use all assets
if len(short_list) == 0:
    short_list = assets

k = len(short_list)
df = df_all[short_list]

# Objectives & hyperparams
objective_name = st.sidebar.selectbox("Hàm mục tiêu", ["Markowitz (mean-variance)", "Sharpe ratio"])
lam = st.sidebar.number_input("Lambda (cho Markowitz)", value=1.0, step=0.1)
r_f = st.sidebar.number_input("Risk-free rate (cho Sharpe)", value=0.0, step=0.001)

lr = st.sidebar.number_input("Learning rate", value=0.1, step=0.01)
batch_size = st.sidebar.number_input("Batch size (mini-batch/SGD)", value=16, step=1, min_value=1)
max_iters = st.sidebar.number_input("Số vòng lặp (iterations)", value=50, step=10, min_value=1)
update_delay = st.sidebar.number_input("Độ trễ cập nhật (giây)", value=0.01, step=0.05, min_value=0.0)

# Optimizer selection
all_opts = [
    "GD",
    "mini-batch GD",
    "SGD",
    "Newton",
    "Nesterov accelerated",
    "Adam",
    "Adagrad",
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

# Create per-optimizer states - ensure weights match current k
opt_configs = {}
for name in selected_opts:
    state = st.session_state.cmp_state.get(name, {})
    # Check if weights need to be resized
    if "w" not in state or len(state["w"]) != k:
        state["w"] = rng.normal(size=k)
        state["weights_hist"] = [state["w"].copy()]
        state["obj_hist"] = []
        state["time_hist"] = []
    else:
        state.setdefault("weights_hist", [state["w"].copy()])
        state.setdefault("obj_hist", [])
        state.setdefault("time_hist", [])
    
    # method-specific states
    if name == "Nesterov accelerated":
        if "v" not in state or len(state["v"]) != k:
            state["v"] = np.zeros(k)
        state.setdefault("momentum", 0.9)
    if name == "Adam":
        if "m" not in state or len(state["m"]) != k:
            state["m"] = np.zeros(k)
        if "v" not in state or len(state["v"]) != k:
            state["v"] = np.zeros(k)
        state.setdefault("t", 0)
        state.setdefault("beta1", 0.9)
        state.setdefault("beta2", 0.999)
        state.setdefault("eps", 1e-8)
    if name == "Adagrad":
        if "G" not in state or len(state["G"]) != k:
            state["G"] = np.zeros(k)
        state.setdefault("eps", 1e-8)
    opt_configs[name] = state

col_run, _ = st.columns([1, 3])
with col_run:
    run = st.button("RUN")

# Placeholders for live charts
top_row = st.container()
col1, col2, col3 = st.columns([1, 1, 1])

# Prepare static layout and placeholders (single render surfaces)
with col3:
    st.subheader("Biểu đồ 1: Contour và quỹ đạo các thuật toán (k=2)")
    contour_placeholder = col3.empty()

with col1:
    st.subheader("Biểu đồ 2: Objective theo iteration")
    iter_placeholder = col1.empty()

with col2:
    st.subheader("Biểu đồ 3: Objective theo thời gian (giây)")
    time_placeholder = col2.empty()

if run:
    # Check if any optimizers are selected
    if len(selected_opts) == 0:
        st.error("Vui lòng chọn ít nhất một thuật toán để so sánh.")
    else:
        st.info(f"Đang chạy {len(selected_opts)} thuật toán: {', '.join(selected_opts)}")
        
        # Threaded execution: one worker thread per optimizer
        run_start = time.perf_counter()

        locks = {name: threading.Lock() for name in selected_opts}
        stop_flags = {name: False for name in selected_opts}
        error_flags = {name: None for name in selected_opts}

        def worker(name: str):
            try:
                state = opt_configs[name]
                for it in range(int(max_iters)):
                    w = state["w"].astype(float)
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
                    elif name == "Adagrad":
                        w_new, obj_val = opt_step_adagrad(df, w, lr, objective_fn, state)
                    else:
                        break

                    with locks[name]:
                        state["w"] = w_new
                        state["weights_hist"].append(w_new.copy())
                        state["obj_hist"].append(float(obj_val))
                        state["time_hist"].append(time.perf_counter() - run_start)

                    if update_delay > 0:
                        time.sleep(float(update_delay))

                stop_flags[name] = True
            except Exception as e:
                error_flags[name] = str(e)
                stop_flags[name] = True

        threads = []
        for name in selected_opts:
            t = threading.Thread(target=worker, args=(name,), daemon=True)
            t.start()
            threads.append(t)

    # Precompute contour grid once per run (only if k=2)
    if k == 2:
        w1 = np.linspace(-3.0, 3.0, 10)
        w2 = np.linspace(-3.0, 3.0, 10)
        W1, W2 = np.meshgrid(w1, w2)
        Z = np.zeros_like(W1)
        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                ww = np.array([W1[i, j], W2[i, j]], dtype=float)
                z, _, _ = objective_fn(df, ww)
                Z[i, j] = z

    # Main refresh loop: update charts every 0.5s until all threads complete
    max_wait_time = 60  # Maximum 60 seconds
    start_wait = time.perf_counter()
    last_update = 0
    
    while not all(stop_flags.values()) and (time.perf_counter() - start_wait) < max_wait_time:
        current_time = time.perf_counter()
        
        # Update charts every 0.5 seconds
        if current_time - last_update >= 0.5:
            try:
                # contour (only if k=2)
                if k == 2:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    cs = ax.contourf(W1, W2, Z, levels=30, cmap="viridis")
                    fig.colorbar(cs, ax=ax).set_label("Objective value")
                    ax.contour(W1, W2, Z, colors="k", linewidths=0.5, levels=15)
                    ax.set_xlabel(f"Trọng số {short_list[0]}")
                    ax.set_ylabel(f"Trọng số {short_list[1]}")
                    ax.set_title("Mặt đồng mức + quỹ đạo")
                    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_opts) or 1))
                    for color, name in zip(colors, selected_opts):
                        with locks[name]:
                            hist = np.array(opt_configs[name]["weights_hist"], dtype=float)
                        if hist.ndim == 2 and hist.shape[1] == 2 and len(hist) > 0:
                            ax.plot(hist[:, 0], hist[:, 1], color=color, linewidth=1.8, label=name)
                            ax.scatter(hist[-1, 0], hist[-1, 1], color=color, s=50)
                    ax.legend()
                    contour_placeholder.pyplot(fig)
                else:
                    contour_placeholder.info("Contour chỉ khả dụng khi chọn đúng 2 cổ phiếu.")

                # iteration chart (always show)
                with pd.option_context('mode.copy_on_write', True):
                    # Ensure all arrays have the same length by padding with NaN
                    max_len = max(len(opt_configs[name]["obj_hist"]) for name in selected_opts) if selected_opts else 0
                    if max_len > 0:
                        data_dict = {}
                        for name in selected_opts:
                            hist = opt_configs[name]["obj_hist"]
                            if len(hist) < max_len:
                                # Pad with NaN to match max length
                                padded_hist = hist + [np.nan] * (max_len - len(hist))
                            else:
                                padded_hist = hist
                            data_dict[name] = padded_hist
                        df_iter = pd.DataFrame(data_dict)
                        df_iter.index.name = "iteration"
                        iter_placeholder.line_chart(df_iter)

                # time chart (always show)
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                colors = plt.cm.tab10(np.linspace(0, 1, len(selected_opts) or 1))
                has_data = False
                for color, name in zip(colors, selected_opts):
                    with locks[name]:
                        t_hist = np.array(opt_configs[name]["time_hist"]) if len(opt_configs[name]["time_hist"]) > 0 else np.array([0.0])
                        y_hist = np.array(opt_configs[name]["obj_hist"]) if len(opt_configs[name]["obj_hist"]) > 0 else np.array([np.nan])
                    if len(t_hist) > 0 and len(y_hist) > 0 and not np.all(np.isnan(y_hist)):
                        ax2.plot(t_hist, y_hist, color=color, label=name)
                        has_data = True
                ax2.set_xlabel("Thời gian (s)")
                ax2.set_ylabel("Objective")
                if has_data:
                    ax2.legend()
                time_placeholder.pyplot(fig2)
                
                last_update = current_time
                
            except Exception as e:
                st.error(f"Lỗi khi vẽ biểu đồ: {str(e)}")
                break
        
        # Small sleep to prevent busy waiting
        time.sleep(0.1)

    # Ensure threads finished
    for t in threads:
        t.join(timeout=2.0)
    
    # Check for errors
    for name, error in error_flags.items():
        if error:
            st.error(f"Lỗi trong thuật toán {name}: {error}")

    # Persist final results to session state
    for name in selected_opts:
        st.session_state.cmp_state[name] = opt_configs[name]
        
    st.success("Hoàn thành!")