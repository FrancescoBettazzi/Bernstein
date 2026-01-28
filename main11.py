import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
from scipy.integrate import trapezoid

# Assume these modules are present in your environment
from bernstein import create_ecdf, calculate_bernstein_cdf, calculate_bernstein_pdf
from KumaraswamyDist import KumaraswamyDist

# =============================================================================
# 1. CONFIGURATION AND PARAMETERS
# =============================================================================

scelta_dist = 'k'  # 'n', 'u', 'e', 'k'
M = 100  # Number of samples
NUM_SIMULATIONS = 10
num_points = 500  # Graph resolution

# --- HEURISTIC N CALCULATION ---
N_pdf = math.ceil(M / math.log(M, 2))
N_cdf = math.ceil(M / math.log(M, 2)) ** 2

# Initialize Distribution
distribuzione = None
nome_dist = ""

# GAUSSIAN
if scelta_dist == 'n':
    mu = 0
    sigma = 1
    distribuzione = stats.norm(loc=mu, scale=sigma)
    nome_dist = f"Normal(mu={mu}, sigma={sigma})"
# UNIFORM
elif scelta_dist == 'u':
    uni_a = 5
    uni_b = 15
    distribuzione = stats.uniform(loc=uni_a, scale=(uni_b - uni_a))
    nome_dist = f"Uniform[{uni_a}, {uni_b}]"
elif scelta_dist == 'e':
    lambda_param = 0.5
    distribuzione = stats.expon(loc=0, scale=1 / lambda_param)
    nome_dist = f"Exponential(lambda={lambda_param})"
# KUMARASWAMY
elif scelta_dist == 'k':
    '''
    campana
    k_a = 2
    k_b = 5
    
    decrescente
    k_a = 1
    k_b = 3
    
    forma a U
    k_a = 0.5
    k_b = 0.5
    '''

    k_a = 1
    k_b = 3

    distribuzione = KumaraswamyDist(a=k_a, b=k_b)
    nome_dist = f"Kumaraswamy(a={k_a}, b={k_b})"

# =============================================================================
# 2. DATA STRUCTURES
# =============================================================================

sim_data = {
    'x_grids': [],
    'cdf_N_cdf': [], 'pdf_N_pdf': [],
    'cdf_M': [], 'pdf_M': [],
    'pdf_conn_to_cdf': []
}

# Structures for Boxplots
boxplot_data = {
    'cdf_M_diff_ecdf': [],  # |CDF_est - ECDF| on grid
    'cdf_N_diff_ecdf': [],
    'pdf_M_nll_samples': [],  # -Log(PDF_est(samples)) -> Individual NLL contributions
    'pdf_N_nll_samples': []
}

# Scalar Metrics (Medians)
scalar_metrics = {
    'wd_true_M': [], 'wd_emp_M': [],
    'wd_true_N': [], 'wd_emp_N': [],
    'kl_M': [], 'nll_M': [],  # CHANGED: ll_M -> nll_M
    'kl_N': [], 'nll_N': []   # CHANGED: ll_N -> nll_N
}

# Metrics for Bias-Variance Curves (Fig 3)
max_n_test = int(N_cdf * 1.5)
range_N = np.unique(np.arange(5, max_n_test, 5).astype(int))
errors_wd_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))
errors_kl_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))
errors_nll_matrix = np.zeros((NUM_SIMULATIONS, len(range_N))) # CHANGED name

global_min_x, global_max_x = float('inf'), float('-inf')

# =============================================================================
# 3. SIMULATION LOOP
# =============================================================================

for i in range(NUM_SIMULATIONS):
    print(f"Cycle {i + 1}/{NUM_SIMULATIONS}")

    # A. Generate Samples
    campioni = distribuzione.rvs(size=M)
    campioni_ordinati = np.sort(campioni)
    ecdf = create_ecdf(campioni)

    # B. Local Support
    a, b = campioni_ordinati[0], campioni_ordinati[-1]
    global_min_x = min(global_min_x, a)
    global_max_x = max(global_max_x, b)

    # C. Local Grid
    curr_asse_x = np.linspace(a, b, num_points)
    sim_data['x_grids'].append(curr_asse_x)

    # True Values
    cdf_true_loc = distribuzione.cdf(curr_asse_x)
    pdf_true_loc = distribuzione.pdf(curr_asse_x)
    ecdf_vals_grid = ecdf(curr_asse_x)

    # -----------------------------------------------------------
    # PART 1: Estimations
    # -----------------------------------------------------------

    # === Case N = M ===
    cdf_stima_M = calculate_bernstein_cdf(ecdf, M, a, b, curr_asse_x)
    pdf_stima_M = calculate_bernstein_pdf(ecdf, M, a, b, curr_asse_x)
    pdf_stima_M_samples = calculate_bernstein_pdf(ecdf, M, a, b, campioni)

    sim_data['cdf_M'].append(cdf_stima_M)
    sim_data['pdf_M'].append(pdf_stima_M)

    # --- BOXPLOT DATA ---
    diff_cdf_M = np.abs(ecdf_vals_grid - cdf_stima_M)
    boxplot_data['cdf_M_diff_ecdf'].append(diff_cdf_M)

    # CHANGED: Store Negative Log Probs (Positive Cost)
    # Using epsilon to avoid log(0)
    nll_samples_M = -np.log(pdf_stima_M_samples + 1e-12)
    boxplot_data['pdf_M_nll_samples'].append(nll_samples_M)

    # --- SCALAR METRICS ---
    wd_true_M = trapezoid(np.abs(cdf_true_loc - cdf_stima_M), curr_asse_x)
    # print("wd_true_M:", wd_true_M)
    wd_emp_M = trapezoid(diff_cdf_M, curr_asse_x)
    scalar_metrics['wd_true_M'].append(wd_true_M)
    scalar_metrics['wd_emp_M'].append(wd_emp_M)

    kl_M = entropy(pk=pdf_true_loc, qk=pdf_stima_M + 1e-12)
    # CHANGED: Average NLL calculation
    nll_M = np.mean(nll_samples_M)
    scalar_metrics['kl_M'].append(kl_M)
    scalar_metrics['nll_M'].append(nll_M)

    # === Case N Heuristic ===

    # CDF (N_cdf)
    cdf_stima_N = calculate_bernstein_cdf(ecdf, int(N_cdf), a, b, curr_asse_x)
    pdf_conn_to_cdf = calculate_bernstein_pdf(ecdf, int(N_cdf), a, b, curr_asse_x)
    sim_data['cdf_N_cdf'].append(cdf_stima_N)
    sim_data['pdf_conn_to_cdf'].append(pdf_conn_to_cdf)

    diff_cdf_N = np.abs(ecdf_vals_grid - cdf_stima_N)
    boxplot_data['cdf_N_diff_ecdf'].append(diff_cdf_N)

    wd_true_N = trapezoid(np.abs(cdf_true_loc - cdf_stima_N), curr_asse_x)
    wd_emp_N = trapezoid(diff_cdf_N, curr_asse_x)
    scalar_metrics['wd_true_N'].append(wd_true_N)
    scalar_metrics['wd_emp_N'].append(wd_emp_N)

    # PDF (N_pdf)
    pdf_stima_N = calculate_bernstein_pdf(ecdf, int(N_pdf), a, b, curr_asse_x)
    pdf_stima_N_samples = calculate_bernstein_pdf(ecdf, int(N_pdf), a, b, campioni)
    sim_data['pdf_N_pdf'].append(pdf_stima_N)

    # CHANGED: Store Negative Log Probs
    nll_samples_N = -np.log(pdf_stima_N_samples + 1e-12)
    boxplot_data['pdf_N_nll_samples'].append(nll_samples_N)

    kl_N = entropy(pk=pdf_true_loc, qk=pdf_stima_N + 1e-12)
    # CHANGED: Average NLL
    nll_N = np.mean(nll_samples_N)
    scalar_metrics['kl_N'].append(kl_N)
    scalar_metrics['nll_N'].append(nll_N)

    # -----------------------------------------------------------
    # PART 2: Bias-Variance Analysis
    # -----------------------------------------------------------
    for idx_n, n_val in enumerate(range_N):
        cdf_temp = calculate_bernstein_cdf(ecdf, int(n_val), a, b, curr_asse_x)
        pdf_temp = calculate_bernstein_pdf(ecdf, int(n_val), a, b, curr_asse_x)
        pdf_temp_samples = calculate_bernstein_pdf(ecdf, int(n_val), a, b, campioni)
        errors_wd_matrix[i, idx_n] = trapezoid(np.abs(ecdf_vals_grid - cdf_temp), curr_asse_x)
        errors_kl_matrix[i, idx_n] = entropy(pk=pdf_true_loc, qk=pdf_temp + 1e-12)
        # CHANGED: Calculate Mean NLL for matrix
        errors_nll_matrix[i, idx_n] = -np.mean(np.log(pdf_temp_samples + 1e-12))

# =============================================================================
# 4. POST-PROCESSING AND MEDIANS
# =============================================================================
def get_stats(metric_list):
    return np.median(metric_list), np.std(metric_list)

# Calcolo statistiche finali
mu_wd_true_M, std_wd_true_M = get_stats(scalar_metrics['wd_true_M'])
mu_wd_emp_M, std_wd_emp_M = get_stats(scalar_metrics['wd_emp_M'])
mu_wd_true_N, std_wd_true_N = get_stats(scalar_metrics['wd_true_N'])
mu_wd_emp_N, std_wd_emp_N = get_stats(scalar_metrics['wd_emp_N'])

mu_kl_M, std_kl_M = get_stats(scalar_metrics['kl_M'])
mu_kl_N, std_kl_N = get_stats(scalar_metrics['kl_N'])

mu_nll_M, std_nll_M = get_stats(scalar_metrics['nll_M'])
mu_nll_N, std_nll_N = get_stats(scalar_metrics['nll_N'])

'''med_wd_true_M = np.median(scalar_metrics['wd_true_M'])
med_wd_emp_M = np.median(scalar_metrics['wd_emp_M'])
med_wd_true_N = np.median(scalar_metrics['wd_true_N'])
med_wd_emp_N = np.median(scalar_metrics['wd_emp_N'])

med_kl_M = np.median(scalar_metrics['kl_M'])
med_nll_M = np.median(scalar_metrics['nll_M']) # CHANGED
med_kl_N = np.median(scalar_metrics['kl_N'])
med_nll_N = np.median(scalar_metrics['nll_N']) # CHANGED'''

# Average Curves
avg_curve_wd = np.mean(errors_wd_matrix, axis=0)
avg_curve_kl = np.mean(errors_kl_matrix, axis=0)
avg_curve_nll = np.mean(errors_nll_matrix, axis=0)  # CHANGED

best_n_wd = range_N[np.argmin(avg_curve_wd)]
best_n_kl = range_N[np.argmin(avg_curve_kl)]
# CHANGED: argmin instead of argmax because we want to MINIMIZE NLL
best_n_nll = range_N[np.argmin(avg_curve_nll)]

if scelta_dist == 'k':
    asse_x_generale = np.linspace(max(0.0001, global_min_x), min(0.9999, global_max_x), num_points)
else:
    asse_x_generale = np.linspace(global_min_x, global_max_x, num_points)

cdf_vera = distribuzione.cdf(asse_x_generale)
pdf_vera = distribuzione.pdf(asse_x_generale)


def plot_spaghetti(ax, x_list, y_list, y_true, label_true, title, color_true='k'):
    for x, y in zip(x_list, y_list):
        ax.plot(x, y, 'k-', linewidth=0.5, alpha=0.3)
    ax.plot(asse_x_generale, y_true, color=color_true, linewidth=2, linestyle='-', label=label_true)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

def add_gt_line(ax, med_ref, std_dev = None, label_prefix="GT"):
    ax.axhline(med_ref, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(1.02, med_ref, f"{label_prefix}\nMedian: {med_ref:.4f}\nStd Dev: {std_dev:.4f}",
            transform=ax.get_yaxis_transform(),
            color='red', fontsize=8, va='center')

# =============================================================================
# 5. VISUALIZATION - FIGURE 1: FOCUS CDF
# =============================================================================

fig1, ax1 = plt.subplots(3, 2, sharey='row', figsize=(12, 18))
fig1.suptitle(f"CDF Analysis: {nome_dist} (M={M})", fontsize=14)

# CDF Plots
plot_spaghetti(ax1[0, 0], sim_data['x_grids'], sim_data['cdf_M'], cdf_vera, 'True CDF', f"CDF (N=M={M})")
plot_spaghetti(ax1[0, 1], sim_data['x_grids'], sim_data['cdf_N_cdf'], cdf_vera, 'True CDF', f"CDF (N={int(N_cdf)})")

# PDF (Derivative of CDF) Plots
plot_spaghetti(ax1[1, 0], sim_data['x_grids'], sim_data['pdf_M'], pdf_vera, 'True PDF', f"Derivative PDF (N=M={M})")
plot_spaghetti(ax1[1, 1], sim_data['x_grids'], sim_data['pdf_conn_to_cdf'], pdf_vera, 'True PDF', f"Derivative PDF (N={int(N_cdf)})")

# Boxplots Diff vs ECDF (Wasserstein Empirica visuale)
# ax1[2, 0].boxplot(boxplot_data['cdf_M_diff_ecdf'], medianprops=dict(color='k', linewidth=1.5))
ax1[2, 0].boxplot(scalar_metrics['wd_emp_M'], medianprops=dict(color='k', linewidth=1.5))
add_gt_line(ax1[2, 0], mu_wd_true_M, std_wd_true_M,  label_prefix="Est CDF vs True")
add_gt_line(ax1[2, 0], mu_wd_emp_M, std_wd_emp_M,  label_prefix="Est CDF vs ECDF")
# ax1[2, 0].set_title(f"Diff vs ECDF (N=M)\nMed WD(True): {med_wd_true_M:.4f} | Med WD(Emp): {med_wd_emp_M:.4f}", fontsize=9)
ax1[2, 0].set_title(f"Est CDF vs ECDF (N=M={M})", fontsize=10, fontweight='bold')  # \nAvg WD(True): {mu_wd_true_M:.4f} (std {std_wd_true_M:.4f})")
ax1[2, 0].set_ylabel("Wasserstein distance")
ax1[2, 0].grid(True, alpha=0.3)

# ax1[2, 1].boxplot(boxplot_data['cdf_N_diff_ecdf'], medianprops=dict(color='k', linewidth=1.5))
ax1[2, 1].boxplot(scalar_metrics['wd_emp_N'], medianprops=dict(color='k', linewidth=1.5))
add_gt_line(ax1[2, 1], mu_wd_true_N, std_wd_true_N,  label_prefix="Est CDF vs True")
add_gt_line(ax1[2, 1], mu_wd_emp_N, std_wd_emp_N,  label_prefix="Est CDF vs ECDF")
# ax1[2, 1].set_title(f"Diff vs ECDF (N={int(N_cdf)})\nMed WD(True): {med_wd_true_N:.4f} | Med WD(Emp): {med_wd_emp_N:.4f}", fontsize=9)
ax1[2, 1].set_title(f"Est CDF vs ECDF (N={int(N_cdf)})", fontsize=10, fontweight='bold')  # \nAvg WD(True): {mu_wd_N:.4f} (std {std_wd_N:.4f})")
ax1[2, 1].set_ylabel("Wasserstein distance")
ax1[2, 1].grid(True, alpha=0.3)

for ax in ax1.flat:
    ax.tick_params(labelleft=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# =============================================================================
# 6. VISUALIZATION - FIGURE 2: FOCUS PDF (SMART ZOOM)
# =============================================================================

fig2, ax2 = plt.subplots(3, 2, sharey='row', figsize=(12, 18))
fig2.suptitle(f"PDF Analysis: {nome_dist} (M={M})", fontsize=14)

# --- Row 1: Spaghetti Plots (Unchanged) ---
plot_spaghetti(ax2[0, 0], sim_data['x_grids'], sim_data['pdf_M'], pdf_vera,
               'True PDF', f"PDF Estimator (N=M={M})")
plot_spaghetti(ax2[0, 1], sim_data['x_grids'], sim_data['pdf_N_pdf'], pdf_vera,
               'True PDF', f"PDF Estimator (N={int(N_pdf)})")

# --- Row 2: BOXPLOTS PDF (NLL Samples) ---

# NLL Boxplots (Distance from Samples)
# Filter extremes for better visualization if needed
# all_nll = np.concatenate(boxplot_data['pdf_M_nll_samples'] + boxplot_data['pdf_N_nll_samples'])
# y_max_nll = np.percentile(all_nll, 95) * 1.2

# 2,1: N=M
'''ax2[1, 0].boxplot(boxplot_data['pdf_M_nll_samples'], patch_artist=True,
                  # flierprops=dict(marker='o', markersize=3, alpha=0.5), # Make them subtle
                  boxprops=dict(facecolor='salmon'),
                  medianprops=dict(color='red'))'''

ax2[1, 0].boxplot(scalar_metrics['nll_M'], medianprops=dict(color='k', linewidth=1.5))
add_gt_line(ax2[1, 0], mu_nll_M, std_nll_M,  label_prefix="Est PDF vs Samples")
# title_pdf_M = (f"Neg Log-Likelihood (N=M)\n"
#                f"Med KL: {med_kl_M:.4f} | Med NLL: {med_nll_M:.4f}")
# ax2[1, 0].set_title(title_pdf_M, fontsize=9)
ax2[1, 0].set_title(f"Est PDF vs Samples (N=M={M})", fontsize=10, fontweight='bold')  # \nMean KL: {mu_kl_M:.4f} | Mean NLL: {mu_nll_M:.4f}")
ax2[1, 0].set_ylabel("Negative log-likelihood")
# ax2[1, 0].set_xlabel("Simulation ID")
# ax2[1, 0].set_ylim(top=y_max_nll)
ax2[1, 0].grid(True, alpha=0.3)

# 2,2: N=N_pdf
'''ax2[1, 1].boxplot(boxplot_data['pdf_N_nll_samples'], patch_artist=True,
                  boxprops=dict(facecolor='orange'),
                  medianprops=dict(color='darkorange'))'''
ax2[1, 1].boxplot(scalar_metrics['nll_N'], medianprops=dict(color='k', linewidth=1.5))
add_gt_line(ax2[1, 1], mu_nll_N, std_nll_N,  label_prefix="Est PDF vs Samples")
# title_pdf_N = (f"Neg Log-Likelihood (N={int(N_pdf)})\n"
#                f"Med KL: {med_kl_N:.4f} | Med NLL: {med_nll_N:.4f}")
# ax2[1, 1].set_title(title_pdf_N, fontsize=9)
ax2[1, 1].set_title(f"Est PDF vs Samples (N={int(N_pdf)})", fontsize=10, fontweight='bold')  # \nMean KL: {mu_kl_N:.4f} | Mean NLL: {mu_nll_N:.4f}")
ax2[1, 1].set_ylabel("Negative log-likelihood")
# ax2[1, 1].set_xlabel("Simulation ID")
# ax2[1, 1].set_ylim(top=y_max_nll)
ax2[1, 1].grid(True, alpha=0.3)

# TODO: NUOVO
ax2[2, 0].boxplot(scalar_metrics['kl_M'], medianprops=dict(color='k', linewidth=1.5))
add_gt_line(ax2[2, 0], mu_kl_M, std_kl_M,  label_prefix="Est PDF vs True")
ax2[2, 0].set_title(f"Est PDF vs True (N=M={M})", fontsize=10, fontweight='bold')  # \nMean KL: {mu_kl_M:.4f} | Mean NLL: {mu_nll_M:.4f}")
ax2[2, 0].set_ylabel("Kullback–Leibler divergence")
ax2[2, 0].grid(True, alpha=0.3)

ax2[2, 1].boxplot(scalar_metrics['kl_N'], medianprops=dict(color='k', linewidth=1.5))
add_gt_line(ax2[2, 1], mu_kl_N, std_kl_N,  label_prefix="Est PDF vs True")
ax2[2, 1].set_title(f"Est PDF vs True (N={N_pdf})", fontsize=10, fontweight='bold')  # \nMean KL: {mu_kl_M:.4f} | Mean NLL: {mu_nll_M:.4f}")
ax2[2, 1].set_ylabel("Kullback–Leibler divergence")
ax2[2, 1].grid(True, alpha=0.3)

for ax in ax2.flat:
    ax.tick_params(labelleft=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# =============================================================================
# 7. VISUALIZATION - FIGURE 3: BIAS-VARIANCE
# =============================================================================

fig3, ax3 = plt.subplots(1, 3, figsize=(18, 5)) # Changed to 1,3 to include NLL plot
fig3.suptitle(f"Metric Sensitivity vs Degree N (Avg over {NUM_SIMULATIONS} runs) - {nome_dist}", fontsize=14)

# Wasserstein
ax3[0].plot(range_N, avg_curve_wd, 'b-o', label='Mean WD')
ax3[0].axvline(best_n_wd, color='r', linestyle='--', label=f'Best N={best_n_wd}')
ax3[0].axvline(N_cdf, color='orange', linestyle=':', linewidth=2, label=f'Heur N={int(N_cdf)}')
ax3[0].set_title("CDF Distance (Wasserstein)")
ax3[0].set_xlabel("Degree N")
ax3[0].grid(True, alpha=0.3)
ax3[0].legend()

# KL Divergence
ax3[1].plot(range_N, avg_curve_kl, 'g-o', label='Mean KL')
ax3[1].axvline(best_n_kl, color='r', linestyle='--', label=f'Best N={best_n_kl}')
ax3[1].axvline(N_pdf, color='orange', linestyle=':', linewidth=2, label=f'Heur N={int(N_pdf)}')
ax3[1].set_title("PDF Distance (KL Divergence)")
ax3[1].set_xlabel("Degree N")
ax3[1].grid(True, alpha=0.3)
ax3[1].legend()

# CHANGED: Average NLL Plot (Minimization)
ax3[2].plot(range_N, avg_curve_nll, 'm-o', label='Mean NLL')
ax3[2].axvline(best_n_nll, color='r', linestyle='--', label=f'Best N={best_n_nll}')
ax3[2].axvline(N_pdf, color='orange', linestyle=':', linewidth=2, label=f'Heur N={int(N_pdf)}')
ax3[2].set_title("Sample Fit (Neg Log Likelihood)")
ax3[2].set_xlabel("Degree N")
ax3[2].grid(True, alpha=0.3)
ax3[2].legend()

plt.tight_layout()
plt.show()
