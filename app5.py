"""
Compare giữa các thuật toán
"""

import time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
from utils.data_processing import objective_markowitz, objective_sharpe
from utils.optimization import (
    optimizer_gd, optimizer_sgd, optimizer_minibatch_gd, optimizer_newton,
    optimizer_nesterov, optimizer_adam, optimizer_adagrad,
    optimizer_torch_adam, optimizer_torch_adagrad, optimizer_torch_sgd, optimizer_torch_nesterov,
    optimizer_scipy_bfgs, optimizer_scipy_cg, optimizer_scipy_newton_cg, optimizer_scipy_trust_ncg
)

st.set_page_config(page_title="Portfolio Optimizer Benchmark", layout="wide")

# ---------------------------
# UI
# ---------------------------

st.title("Benchmark: So sánh hiệu suất các thuật toán tối ưu")
st.write("So sánh toàn diện các thuật toán về thời gian chạy và chất lượng nghiệm (loss/convergence). Chạy song song và hiển thị kết quả dưới dạng bảng và biểu đồ.")

# Data section
st.sidebar.header("Input dữ liệu")
seed = st.sidebar.number_input("Seed sinh dữ liệu", value=42)
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
short_list = st.sidebar.multiselect("Chọn short-list cổ phiếu", assets, default=assets[:2])

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
max_iters = st.sidebar.number_input("Số vòng lặp tối đa", value=100, step=10, min_value=1)
tolerance = st.sidebar.number_input("Ngưỡng hội tụ (tolerance)", value=1e-6, step=1e-7, format="%.2e")

# Optimizer selection
all_opts = [
    "GD",
    "mini-batch GD", 
    "SGD",
    "Newton",
    "Nesterov accelerated",
    "Adam",
    "Adagrad",
    "Torch Adam",
    "Torch Adagrad",
    "Torch SGD", 
    "Torch Nesterov",
    "SciPy BFGS",
    "SciPy CG",
    "SciPy Newton-CG",
    "SciPy Trust-NCG",
]

selected_opts = st.sidebar.multiselect(
    "Chọn thuật toán để benchmark", 
    all_opts, 
    default=["GD", "Adam", "Newton", "Torch Adam", "SciPy BFGS"]
)

# Prepare objective function
if objective_name.startswith("Markowitz"):
    def objective_fn(DF, W):
        return objective_markowitz(DF, W, lam=lam)
else:
    def objective_fn(DF, W):
        return objective_sharpe(DF, W, r_f=r_f)

# Benchmark execution
run_benchmark = st.sidebar.button("🚀 CHẠY BENCHMARK")

if run_benchmark:
    if len(selected_opts) == 0:
        st.error("Vui lòng chọn ít nhất một thuật toán để benchmark.")
    else:
        st.info(f"Đang benchmark {len(selected_opts)} thuật toán: {', '.join(selected_opts)}")
        
        # Initialize shared weights for fair comparison
        initial_weights = rng.normal(size=k)
        
        # Results storage
        results = {}
        
        def run_optimizer(name):
            """Run a single optimizer and return results"""
            try:
                # Initialize state
                state = {}
                w = initial_weights.copy()
                
                # Initialize method-specific states
                if name == "Nesterov accelerated":
                    state["v"] = np.zeros(k)
                    state["momentum"] = 0.9
                elif name == "Adam":
                    state["m"] = np.zeros(k)
                    state["v"] = np.zeros(k)
                    state["t"] = 0
                    state["beta1"] = 0.9
                    state["beta2"] = 0.999
                    state["eps"] = 1e-8
                elif name == "Adagrad":
                    state["G"] = np.zeros(k)
                    state["eps"] = 1e-8
                elif name in ["Torch Adam", "Torch Adagrad", "Torch SGD", "Torch Nesterov"]:
                    state["torch_state"] = {}
                
                # Track progress
                obj_history = []
                time_history = []
                start_time = time.perf_counter()
                
                prev_obj = float('inf')
                converged = False
                iterations = 0
                
                for it in range(max_iters):
                    # Run optimizer step
                    if name == "GD":
                        w_new, obj_val, grad = optimizer_gd(df, w, lr, objective_fn)
                    elif name == "mini-batch GD":
                        w_new, obj_val, grad = optimizer_minibatch_gd(df, w, lr, objective_fn, batch_size=int(batch_size), rng=rng)
                    elif name == "SGD":
                        w_new, obj_val, grad = optimizer_sgd(df, w, lr, objective_fn, batch_size=1, rng=rng)
                    elif name == "Newton":
                        w_new, obj_val, grad = optimizer_newton(df, w, lr, objective_fn)
                    elif name == "Nesterov accelerated":
                        w_new, obj_val, grad = optimizer_nesterov(df, w, lr, objective_fn, state)
                    elif name == "Adam":
                        state["t"] = int(state.get("t", 0)) + 1
                        w_new, obj_val, grad = optimizer_adam(df, w, lr, objective_fn, state, t=state["t"])
                    elif name == "Adagrad":
                        w_new, obj_val, grad = optimizer_adagrad(df, w, lr, objective_fn, state)
                    elif name == "Torch Adam":
                        w_new, obj_val, grad = optimizer_torch_adam(df, w, lr, objective_fn, state["torch_state"])
                    elif name == "Torch Adagrad":
                        w_new, obj_val, grad = optimizer_torch_adagrad(df, w, lr, objective_fn, state["torch_state"])
                    elif name == "Torch SGD":
                        w_new, obj_val, grad = optimizer_torch_sgd(df, w, lr, objective_fn, state["torch_state"])
                    elif name == "Torch Nesterov":
                        w_new, obj_val, grad = optimizer_torch_nesterov(df, w, lr, objective_fn, state["torch_state"])
                    elif name == "SciPy BFGS":
                        w_new, obj_val, grad = optimizer_scipy_bfgs(df, w, lr, objective_fn)
                    elif name == "SciPy CG":
                        w_new, obj_val, grad = optimizer_scipy_cg(df, w, lr, objective_fn)
                    elif name == "SciPy Newton-CG":
                        w_new, obj_val, grad = optimizer_scipy_newton_cg(df, w, lr, objective_fn)
                    elif name == "SciPy Trust-NCG":
                        w_new, obj_val, grad = optimizer_scipy_trust_ncg(df, w, lr, objective_fn)
                    else:
                        break
                    
                    # Update state
                    w = w_new
                    obj_history.append(float(obj_val))
                    time_history.append(time.perf_counter() - start_time)
                    iterations = it + 1
                    
                    # Check convergence
                    if abs(obj_val - prev_obj) < tolerance:
                        converged = True
                        break
                    prev_obj = obj_val
                
                total_time = time.perf_counter() - start_time
                final_obj = obj_history[-1] if obj_history else float('inf')
                
                return {
                    'name': name,
                    'iterations': iterations,
                    'converged': converged,
                    'total_time': total_time,
                    'final_objective': final_obj,
                    'objective_history': obj_history,
                    'time_history': time_history,
                    'final_weights': w,
                    'error': None
                }
                
            except Exception as e:
                return {
                    'name': name,
                    'iterations': 0,
                    'converged': False,
                    'total_time': 0,
                    'final_objective': float('inf'),
                    'objective_history': [],
                    'time_history': [],
                    'final_weights': None,
                    'error': str(e)
                }
        
        # Run benchmarks in parallel
        with ThreadPoolExecutor(max_workers=min(len(selected_opts), 8)) as executor:
            future_to_opt = {executor.submit(run_optimizer, name): name for name in selected_opts}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            completed = 0
            for future in as_completed(future_to_opt):
                result = future.result()
                results[result['name']] = result
                completed += 1
                progress_bar.progress(completed / len(selected_opts))
                status_text.text(f"Hoàn thành {completed}/{len(selected_opts)}: {result['name']}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.success(f"✅ Benchmark hoàn thành! Đã test {len(selected_opts)} thuật toán.")
        
        # Results table
        st.subheader("📊 Kết quả tổng quan")
        
        table_data = []
        for name in selected_opts:
            result = results[name]
            if result['error']:
                table_data.append({
                    'Thuật toán': name,
                    'Lỗi': result['error'],
                    'Iterations': '-',
                    'Thời gian (s)': '-',
                    'Objective cuối': '-',
                    'Hội tụ': '-'
                })
            else:
                table_data.append({
                    'Thuật toán': name,
                    'Lỗi': '',
                    'Iterations': result['iterations'],
                    'Thời gian (s)': f"{result['total_time']:.4f}",
                    'Objective cuối': f"{result['final_objective']:.6f}",
                    'Hội tụ': '✅' if result['converged'] else '❌'
                })
        
        results_df = pd.DataFrame(table_data)
        st.dataframe(results_df, use_container_width=True)
        
        # Performance charts
        st.subheader("📈 Biểu đồ hiệu suất")
        
        # Filter successful results for plotting
        successful_results = {k: v for k, v in results.items() if not v['error']}
        
        if successful_results:
            # Chart 1: Final objective vs Time
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Objective cuối vs Thời gian")
                names = list(successful_results.keys())
                times = [successful_results[name]['total_time'] for name in names]
                objs = [successful_results[name]['final_objective'] for name in names]
                
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                scatter = ax1.scatter(times, objs, s=100, alpha=0.7)
                ax1.set_xlabel('Thời gian (giây)')
                ax1.set_ylabel('Objective cuối')
                ax1.set_title('Hiệu suất: Thời gian vs Chất lượng')
                ax1.grid(True, alpha=0.3)
                
                # Add labels
                for i, name in enumerate(names):
                    ax1.annotate(name, (times[i], objs[i]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
                
                st.pyplot(fig1)
                plt.close(fig1)
            
            with col2:
                st.subheader("Convergence speed")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                for name, result in successful_results.items():
                    if result['objective_history']:
                        ax2.plot(result['objective_history'], label=name, linewidth=2)
                ax2.set_xlabel('Iterations')
                ax2.set_ylabel('Objective value')
                ax2.set_title('Tốc độ hội tụ')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
                plt.close(fig2)
            
            # Chart 3: Time vs Iterations
            st.subheader("Thời gian vs Số iterations")
            names = list(successful_results.keys())
            iterations = [successful_results[name]['iterations'] for name in names]
            times = [successful_results[name]['total_time'] for name in names]
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            bars = ax3.bar(range(len(names)), times, alpha=0.7)
            ax3.set_xlabel('Thuật toán')
            ax3.set_ylabel('Thời gian (giây)')
            ax3.set_title('Thời gian chạy của các thuật toán')
            ax3.set_xticks(range(len(names)))
            ax3.set_xticklabels(names, rotation=45, ha='right')
            
            # Add iteration count on bars
            for i, (bar, iter_count) in enumerate(zip(bars, iterations)):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{iter_count} iter', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
            
            # Best performers
            st.subheader("🏆 Top performers")
            
            # Best by final objective
            best_obj = min(successful_results.items(), key=lambda x: x[1]['final_objective'])
            # Best by time
            best_time = min(successful_results.items(), key=lambda x: x[1]['total_time'])
            # Best by iterations to converge
            converged_results = {k: v for k, v in successful_results.items() if v['converged']}
            best_convergence = min(converged_results.items(), key=lambda x: x[1]['iterations']) if converged_results else None
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Chất lượng tốt nhất",
                    f"{best_obj[0]}",
                    f"Objective: {best_obj[1]['final_objective']:.6f}"
                )
            
            with col2:
                st.metric(
                    "Nhanh nhất",
                    f"{best_time[0]}",
                    f"{best_time[1]['total_time']:.4f}s"
                )
            
            with col3:
                if best_convergence:
                    st.metric(
                        "Hội tụ nhanh nhất",
                        f"{best_convergence[0]}",
                        f"{best_convergence[1]['iterations']} iterations"
                    )
                else:
                    st.metric("Hội tụ nhanh nhất", "Không có", "Chưa hội tụ")
        
        else:
            st.error("Không có thuật toán nào chạy thành công!")
        
        # Store results in session state for potential export
        st.session_state.benchmark_results = results

# Export results
if st.sidebar.button("📁 Export kết quả"):
    if 'benchmark_results' in st.session_state:
        results = st.session_state.benchmark_results
        
        # Create exportable DataFrame
        export_data = []
        for name, result in results.items():
            export_data.append({
                'Algorithm': name,
                'Iterations': result['iterations'],
                'Converged': result['converged'],
                'Total_Time_s': result['total_time'],
                'Final_Objective': result['final_objective'],
                'Error': result['error'] or ''
            })
        
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="Tải xuống CSV",
            data=csv,
            file_name=f"benchmark_results_{int(time.time())}.csv",
            mime="text/csv"
        )
    else:
        st.warning("Chưa có kết quả để export. Hãy chạy benchmark trước.")

st.caption("💡 Tip: SciPy optimizers thường chậm hơn nhưng chính xác hơn. Torch optimizers có thể có hiệu suất khác nhau tùy thuộc vào cấu hình.")
