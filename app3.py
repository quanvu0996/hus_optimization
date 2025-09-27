import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as spo

# Import utility functions
from utils.data_processing import objective_markowitz, Markowitz
from utils.optimization import (
    opt_step_gd, opt_step_sgd, opt_step_minibatch, opt_step_newton,
    opt_step_nesterov, opt_step_adam, opt_step_adagrad
)

st.set_page_config(page_title="Portfolio Optimization with Multiple Lambdas", layout="wide")

# ---------------------------
# UI
# ---------------------------

st.title("Tối ưu danh mục với nhiều Lambda")
st.write("Tải lên dữ liệu lợi nhuận hoặc giả lập, chọn short-list cổ phiếu, thuật toán, nhiều lambda, và tỷ lệ train-test. Chạy tối ưu trên train và đánh giá trên train/test.")

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

# Short list input
# short_list_input = st.sidebar.text_input("Danh sách short-list cổ phiếu (comma-separated)", value=",".join(assets[:2]))
if uploaded:
    short_list = st.sidebar.multiselect("Chọn short-list cổ phiếu ", assets, default=["HDB", "KDH", "VIC", "MWG", "VCB", "FPT"])
else:
    short_list = st.sidebar.multiselect("Chọn short-list cổ phiếu ", assets, default=assets[:2])
# short_list = [x.strip() for x in short_list_input.split(",") if x.strip() in assets]

# If no valid assets, use first two
if len(short_list) == 0:
    short_list = assets[:2]

k = len(short_list)
df = df_all[short_list]

# Optimizer selection
all_opts = [
    "GD",
    "mini-batch GD",
    "SGD",
    "Newton",
    "Nesterov accelerated",
    "Adam",
    "Adagrad",
    "SLSQP",
]
selected_opt = st.sidebar.selectbox("Chọn thuật toán tối ưu", all_opts)

# Lambdas input
lambdas_input = st.sidebar.text_input("Các giá trị lambda (comma-separated)", value="0.5,1.0,1.5")
lambda_list = [float(x.strip()) for x in lambdas_input.split(",") if x.strip()]

# Train-test split ratio
test_size = st.sidebar.number_input("Tỷ lệ test set (0-1)", value=0.3, min_value=0.0, max_value=1.0, step=0.05)

# Hyperparams
lr = st.sidebar.number_input("Learning rate", value=0.1, step=0.01)
batch_size = st.sidebar.number_input("Batch size (mini-batch/SGD)", value=16, step=1, min_value=1)
max_iters = st.sidebar.number_input("Số vòng lặp (iterations)", value=50, step=10, min_value=1)

# Initial weights
if "shared_initial_weights" not in st.session_state or len(st.session_state.shared_initial_weights) != k or st.sidebar.button("Reset weights"):
    st.session_state.shared_initial_weights = rng.normal(size=k)

inp_ws = st.sidebar.text_input("Initial weights (comma-separated)", value="")
if inp_ws != "": 
    st.session_state.shared_initial_weights = np.array([float(x.strip()) for x in inp_ws.split(",") if x.strip()])

# RUN button
run = st.button("RUN")

# Placeholders for charts
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Biểu đồ 1: Objective theo iteration")
    iter_placeholder = col1.empty()

with col2:
    st.subheader("Biểu đồ 2: Return vs Risk trên Test")
    test_placeholder = col2.empty()

with col3:
    st.subheader("Biểu đồ 3: Return vs Risk trên Train")
    train_placeholder = col3.empty()

# Containers for tables
st.subheader("Kết quả chi tiết")
col_table1, col_table2 = st.columns(2)

with col_table1:
    st.subheader("Bảng danh mục")
    portfolio_placeholder = col_table1.empty()

with col_table2:
    st.subheader("Bảng tóm tắt kết quả")
    summary_placeholder = col_table2.empty()

# Additional sections for means and covs
st.subheader("Vector lợi nhuận kỳ vọng")
col_mean_train, col_mean_test = st.columns(2)
mean_train_placeholder = col_mean_train.empty()
mean_test_placeholder = col_mean_test.empty()

st.subheader("Ma trận hiệp phương sai")
col_cov_train, col_cov_test = st.columns(2)
cov_train_placeholder = col_cov_train.empty()
cov_test_placeholder = col_cov_test.empty()

if run:
    if len(lambda_list) == 0:
        st.error("Vui lòng nhập ít nhất một giá trị lambda.")
    elif len(short_list) < 2:
        st.error("Short-list phải có ít nhất 2 cổ phiếu.")
    else:
        st.info(f"Đang chạy thuật toán {selected_opt} với {len(lambda_list)} lambda: {', '.join(map(str, lambda_list))}")
        
        # Split data sequentially (time series)
        n = len(df)
        train_end = int(n * (1 - test_size))
        df = df.sort_index(ascending=True)
        df_train = df.iloc[:train_end]
        df_test = df.iloc[train_end:] if test_size > 0 else pd.DataFrame(columns=df.columns)
        
        # Compute mean and cov for train and test
        mean_train = df_train.mean().values
        cov_train = df_train.cov().values
        mean_test = df_test.mean().values if not df_test.empty else np.zeros(k)
        cov_test = df_test.cov().values if not df_test.empty else np.zeros((k, k))
        
        # Display means
        with col_mean_train:
            st.subheader("Train")
            mean_train_df = pd.DataFrame(mean_train, index=short_list, columns=["Mean Return"])
            mean_train_placeholder.dataframe(mean_train_df)
        
        with col_mean_test:
            st.subheader("Test")
            mean_test_df = pd.DataFrame(mean_test, index=short_list, columns=["Mean Return"])
            mean_test_placeholder.dataframe(mean_test_df)
        
        # Display covs
        with col_cov_train:
            st.subheader("Train")
            cov_train_df = pd.DataFrame(cov_train, index=short_list, columns=short_list)
            cov_train_placeholder.dataframe(cov_train_df)
        
        with col_cov_test:
            st.subheader("Test")
            cov_test_df = pd.DataFrame(cov_test, index=short_list, columns=short_list)
            cov_test_placeholder.dataframe(cov_test_df)
        
        # Results storage
        obj_hists = {}  # lambda -> list of obj values
        final_ws = {}   # lambda -> final w
        train_returns = []
        train_risks = []
        test_returns = []
        test_risks = []
        labels = []
        
        for lam in lambda_list:
            # Prepare Markowitz
            markowitz = Markowitz(df_train, lam=lam)
            mu = markowitz.mu
            Sigma = markowitz.Sigma
            
            if selected_opt == "SLSQP":
                def obj_fn(w):
                    return lam * np.dot(w.T, np.dot(Sigma, w)) - np.dot(mu, w)
                
                cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
                bnds = ((0, None),) * k
                initial_w = np.ones(k) / k
                
                res = spo.minimize(obj_fn, initial_w, method='SLSQP', bounds=bnds, constraints=cons)
                
                if res.success:
                    final_w = res.x
                    obj_val = markowitz.get_objective(final_w)
                else:
                    st.error(f"Tối ưu hóa thất bại cho lambda={lam}: {res.message}")
                    continue
                
                obj_hist = [obj_val] * int(max_iters)  # Horizontal line for plot
                
            else:
                # Prepare objective for other methods
                def objective_fn(DF, W):
                    return objective_markowitz(DF, W, lam=lam)
                
                # Optimizer state
                state = {}
                state["w"] = st.session_state.shared_initial_weights.copy().astype(float)
                state["obj_hist"] = []
                
                # Method-specific states
                if selected_opt == "Nesterov accelerated":
                    state["v"] = np.zeros(k)
                    state["momentum"] = 0.9
                elif selected_opt == "Adam":
                    state["m"] = np.zeros(k)
                    state["v"] = np.zeros(k)
                    state["t"] = 0
                    state["beta1"] = 0.9
                    state["beta2"] = 0.999
                    state["eps"] = 1e-8
                elif selected_opt == "Adagrad":
                    state["G"] = np.zeros(k)
                    state["eps"] = 1e-8
                
                # Run optimization
                for it in range(int(max_iters)):
                    w = state["w"]
                    if selected_opt == "GD":
                        w_new, obj_val = opt_step_gd(df_train, w, lr, objective_fn)
                    elif selected_opt == "mini-batch GD":
                        w_new, obj_val = opt_step_minibatch(df_train, w, lr, objective_fn, batch_size=int(batch_size), rng=rng)
                    elif selected_opt == "SGD":
                        w_new, obj_val = opt_step_sgd(df_train, w, lr, objective_fn, batch_size=1, rng=rng)
                    elif selected_opt == "Newton":
                        w_new, obj_val = opt_step_newton(df_train, w, lr, objective_fn)
                    elif selected_opt == "Nesterov accelerated":
                        w_new, obj_val = opt_step_nesterov(df_train, w, lr, objective_fn, state)
                    elif selected_opt == "Adam":
                        w_new, obj_val = opt_step_adam(df_train, w, lr, objective_fn, state)
                    elif selected_opt == "Adagrad":
                        w_new, obj_val = opt_step_adagrad(df_train, w, lr, objective_fn, state)
                    else:
                        break
                    
                    # Use markowitz.get_objective for consistent obj
                    obj_val = markowitz.get_objective(w_new)
                    state["w"] = w_new
                    state["obj_hist"].append(float(obj_val))
                
                final_w = state["w"]
                obj_hist = state["obj_hist"]
                obj_val = obj_hist[-1] if obj_hist else 0.0
            
            # Store results
            obj_hists[lam] = obj_hist
            final_ws[lam] = final_w
            
            # Evaluate on train
            ret_train = np.dot(final_w, mean_train)
            risk_train = np.sqrt(np.dot(final_w.T, np.dot(cov_train, final_w)))
            train_returns.append(ret_train)
            train_risks.append(risk_train)
            
            # Evaluate on test
            ret_test = np.dot(final_w, mean_test)
            risk_test = np.sqrt(np.dot(final_w.T, np.dot(cov_test, final_w))) if not df_test.empty else 0.0
            test_returns.append(ret_test)
            test_risks.append(risk_test)
            
            labels.append(f"λ={lam}")
        
        # Plot 1: Objective vs Iteration
        fig_iter, ax_iter = plt.subplots(figsize=(6, 4))
        for lam, hist in obj_hists.items():
            iterations = np.arange(len(hist))
            ax_iter.plot(iterations, hist, label=f"λ={lam}", linewidth=2)
        ax_iter.set_xlabel("Iteration")
        ax_iter.set_ylabel("Objective")
        ax_iter.legend()
        iter_placeholder.pyplot(fig_iter)
        plt.close(fig_iter)
        
        # Plot 2: Return vs Risk on Test
        fig_test, ax_test = plt.subplots(figsize=(6, 4))
        ax_test.scatter(test_risks, test_returns)
        for i, txt in enumerate(labels):
            ax_test.annotate(txt, (test_risks[i], test_returns[i]))
        ax_test.set_xlabel("Risk (Std Dev)")
        ax_test.set_ylabel("Expected Return")
        ax_test.set_title("Test Set")
        test_placeholder.pyplot(fig_test)
        plt.close(fig_test)
        
        # Plot 3: Return vs Risk on Train
        fig_train, ax_train = plt.subplots(figsize=(6, 4))
        ax_train.scatter(train_risks, train_returns)
        for i, txt in enumerate(labels):
            ax_train.annotate(txt, (train_risks[i], train_returns[i]))
        ax_train.set_xlabel("Risk (Std Dev)")
        ax_train.set_ylabel("Expected Return")
        ax_train.set_title("Train Set")
        train_placeholder.pyplot(fig_train)
        plt.close(fig_train)
        
        # Portfolio table
        portfolio_df = pd.DataFrame(
            data=[final_ws[lam] for lam in lambda_list],
            index=lambda_list,
            columns=short_list
        )
        portfolio_df.index.name = 'Lambda'
        portfolio_placeholder.dataframe(portfolio_df)
        
        # Summary table
        summary_df = pd.DataFrame({
            'Lambda': lambda_list,
            'mu_train': train_returns,
            'sigma_train': train_risks,
            'mu_test': test_returns,
            'sigma_test': test_risks
        })
        summary_placeholder.dataframe(summary_df)
        
        st.success("Hoàn thành!")