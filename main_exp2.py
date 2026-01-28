import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
from scipy.integrate import trapezoid

# Modulo specifico per Bernstein Exponentials (domino [0, inf))
from bernstein_exp import create_ecdf, calculate_bernstein_exp_cdf, calculate_bernstein_exp_pdf

# =============================================================================
# 1. CONFIGURATION AND PARAMETERS
# =============================================================================

# Scegli la distribuzione in base alle immagini fornite:
# 'erlang', 'weibull', 'lognormal'
scelta_dist = 'weibull'

M = 100  # Numero di campioni
NUM_SIMULATIONS = 10  # Numero di simulazioni Monte Carlo
num_points = 500  # Risoluzione grafici

# --- HEURISTIC N CALCULATION ---
# Calcolo gradi N euristici in base a M
N_pdf = math.ceil(M / math.log(M, 2))
N_cdf = math.ceil(M / math.log(M, 2)) ** 2

# Inizializzazione Variabili Distribuzione
distribuzione = None
nome_dist = ""

# --- CONFIGURAZIONE DISTRIBUZIONI (da immagini) ---

if scelta_dist == 'erlang':
    # Erlang(n, lambda). Immagine: media 1 => scala = 1/n
    n_erlang = 5  # Esempio: n=5 (puoi cambiare questo valore)
    # In scipy: Gamma(a=n, scale=1/lambda_rate).
    # Se media=1, scale=1/n.
    distribuzione = stats.gamma(a=n_erlang, scale=1 / n_erlang)
    nome_dist = f"Erlang(n={n_erlang}, mean=1)"

elif scelta_dist == 'weibull':
    # Immagine Weibull: (1, 1.5) o (1, 0.5). Assumiamo (Scale=1, Shape=k)
    '''
    w_shape = 1.5
    
    w_shape = 0.5
    '''

    w_shape = 0.5  # k
    w_scale = 1.0  # lambda
    # Scipy weibull_min: c=shape, scale=scale
    distribuzione = stats.weibull_min(c=w_shape, scale=w_scale)
    nome_dist = f"Weibull(scale={w_scale}, shape={w_shape})"

elif scelta_dist == 'lognormal':
    # Immagine Lognormal: (1, 1.8). Assumiamo (Scale=1, Shape=sigma)
    # Parametri (Scale, s). Scale = exp(mu). Se Scale=1 => mu=0.
    '''
    ln_shape = 1.8
    ln_shape = 0.8
    ln_shape = 0.2
    '''
    ln_shape = 1.8  # sigma (s)
    ln_scale = 1.0  # exp(mu)
    distribuzione = stats.lognorm(s=ln_shape, scale=ln_scale)
    nome_dist = f"Lognormal(s={ln_shape}, scale={ln_scale})"

visual_xlim = distribuzione.ppf(0.999) * 1.5
# visual_xlim = 10
# print(f"Limite visuale asse X impostato a: {visual_xlim:.2f}")
# =============================================================================
# 2. DATA STRUCTURES
# =============================================================================

sim_data = {
    'x_grids': [],
    'cdf_N_cdf': [], 'pdf_N_pdf': [],
    'cdf_M': [], 'pdf_M': [],
    'pdf_conn_to_cdf': []
}

# Scalar Metrics
scalar_metrics = {
    'wd_true_M': [], 'wd_emp_M': [],
    'wd_true_N': [], 'wd_emp_N': [],
    'kl_M': [], 'nll_M': [],
    'kl_N': [], 'nll_N': []
}

# Metrics for Bias-Variance Curves
max_n_test = int(N_cdf * 1.5)
range_N = np.unique(np.arange(5, max_n_test, 5).astype(int))
errors_wd_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))
errors_kl_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))
errors_nll_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))

# Definiamo i limiti globali per i plot (dominio positivo)
# Usiamo la ppf per trovare dove finisce la massa significativa
true_max_x = distribuzione.ppf(0.999)
global_max_x = true_max_x

# =============================================================================
# 3. SIMULATION LOOP
# =============================================================================

for i in range(NUM_SIMULATIONS):
    print(f"Cycle {i + 1}/{NUM_SIMULATIONS}")

    # A. Generazione Campioni
    campioni = distribuzione.rvs(size=M)
    # Assicuriamo che siano positivi (dovrebbero esserlo per queste distribuzioni)
    campioni = np.abs(campioni)
    campioni_ordinati = np.sort(campioni)

    # ECDF function creation (Step function)
    ecdf = create_ecdf(campioni)

    # B. Definizione Griglia Locale
    # Per Bernstein Exp il dominio è [0, inf).
    # Usiamo un range ragionevole basato sui dati correnti e sulla verità
    max_curr = max(campioni[-1], true_max_x)
    curr_asse_x = np.linspace(1e-9, max_curr, num_points)
    sim_data['x_grids'].append(curr_asse_x)

    # Aggiorniamo il max globale per i plot finali
    if max_curr > global_max_x:
        global_max_x = max_curr

    # C. Parametro di Trasformazione (Euristica Adattiva)
    # λ_trans = 1 / mean(samples) è una buona scelta standard per mappare i dati in [0,1]
    scale_param = 1.0 / np.mean(campioni)

    # True Values su griglia locale
    cdf_true_loc = distribuzione.cdf(curr_asse_x)
    pdf_true_loc = distribuzione.pdf(curr_asse_x)
    # Opzionale: limita i valori infiniti per la stabilità dei calcoli successivi
    # pdf_true_loc = np.clip(pdf_true_loc, 0, 1e10)
    ecdf_vals_grid = ecdf(curr_asse_x)

    # -----------------------------------------------------------
    # PART 1: Estimations (Using Bernstein EXP)
    # -----------------------------------------------------------

    # === Case N = M ===
    cdf_stima_M = calculate_bernstein_exp_cdf(ecdf, M, curr_asse_x, scale=scale_param)
    pdf_stima_M = calculate_bernstein_exp_pdf(ecdf, M, curr_asse_x, scale=scale_param)

    # PDF sui campioni per NLL
    pdf_stima_M_samples = calculate_bernstein_exp_pdf(ecdf, M, campioni, scale=scale_param)

    sim_data['cdf_M'].append(cdf_stima_M)
    sim_data['pdf_M'].append(pdf_stima_M)

    # --- METRICS CALCULATION (N=M) ---
    diff_cdf_M = np.abs(ecdf_vals_grid - cdf_stima_M)
    nll_samples_M = -np.log(pdf_stima_M_samples + 1e-12)

    wd_true_M = trapezoid(np.abs(cdf_true_loc - cdf_stima_M), curr_asse_x)
    wd_emp_M = trapezoid(diff_cdf_M, curr_asse_x)
    kl_M = entropy(pk=pdf_true_loc, qk=pdf_stima_M + 1e-12)
    nll_M = np.mean(nll_samples_M)

    scalar_metrics['wd_true_M'].append(wd_true_M)
    scalar_metrics['wd_emp_M'].append(wd_emp_M)
    scalar_metrics['kl_M'].append(kl_M)
    scalar_metrics['nll_M'].append(nll_M)

    # === Case N Heuristic ===

    # CDF (N_cdf)
    cdf_stima_N = calculate_bernstein_exp_cdf(ecdf, int(N_cdf), curr_asse_x, scale=scale_param)
    pdf_conn_to_cdf = calculate_bernstein_exp_pdf(ecdf, int(N_cdf), curr_asse_x, scale=scale_param)

    sim_data['cdf_N_cdf'].append(cdf_stima_N)
    sim_data['pdf_conn_to_cdf'].append(pdf_conn_to_cdf)

    diff_cdf_N = np.abs(ecdf_vals_grid - cdf_stima_N)
    wd_true_N = trapezoid(np.abs(cdf_true_loc - cdf_stima_N), curr_asse_x)
    wd_emp_N = trapezoid(diff_cdf_N, curr_asse_x)

    scalar_metrics['wd_true_N'].append(wd_true_N)
    scalar_metrics['wd_emp_N'].append(wd_emp_N)

    # PDF (N_pdf)
    pdf_stima_N = calculate_bernstein_exp_pdf(ecdf, int(N_pdf), curr_asse_x, scale=scale_param)
    pdf_stima_N_samples = calculate_bernstein_exp_pdf(ecdf, int(N_pdf), campioni, scale=scale_param)

    sim_data['pdf_N_pdf'].append(pdf_stima_N)

    nll_samples_N = -np.log(pdf_stima_N_samples + 1e-12)
    kl_N = entropy(pk=pdf_true_loc, qk=pdf_stima_N + 1e-12)
    nll_N = np.mean(nll_samples_N)

    scalar_metrics['kl_N'].append(kl_N)
    scalar_metrics['nll_N'].append(nll_N)

    # -----------------------------------------------------------
    # PART 2: Bias-Variance Analysis (Loop over N)
    # -----------------------------------------------------------
    for idx_n, n_val in enumerate(range_N):
        cdf_temp = calculate_bernstein_exp_cdf(ecdf, int(n_val), curr_asse_x, scale=scale_param)
        pdf_temp = calculate_bernstein_exp_pdf(ecdf, int(n_val), curr_asse_x, scale=scale_param)
        pdf_temp_samples = calculate_bernstein_exp_pdf(ecdf, int(n_val), campioni, scale=scale_param)

        errors_wd_matrix[i, idx_n] = trapezoid(np.abs(cdf_true_loc - cdf_temp), curr_asse_x)
        errors_kl_matrix[i, idx_n] = entropy(pk=pdf_true_loc, qk=pdf_temp + 1e-12)
        errors_nll_matrix[i, idx_n] = -np.mean(np.log(pdf_temp_samples + 1e-12))


# =============================================================================
# 4. POST-PROCESSING AND MEDIANS
# =============================================================================

def get_stats(metric_list):
    return np.median(metric_list), np.std(metric_list)


# Calcolo statistiche
mu_wd_true_M, std_wd_true_M = get_stats(scalar_metrics['wd_true_M'])
mu_wd_emp_M, std_wd_emp_M = get_stats(scalar_metrics['wd_emp_M'])
mu_wd_true_N, std_wd_true_N = get_stats(scalar_metrics['wd_true_N'])
mu_wd_emp_N, std_wd_emp_N = get_stats(scalar_metrics['wd_emp_N'])

mu_kl_M, std_kl_M = get_stats(scalar_metrics['kl_M'])
mu_kl_N, std_kl_N = get_stats(scalar_metrics['kl_N'])

mu_nll_M, std_nll_M = get_stats(scalar_metrics['nll_M'])
mu_nll_N, std_nll_N = get_stats(scalar_metrics['nll_N'])

# Curve medie
avg_curve_wd = np.mean(errors_wd_matrix, axis=0)
avg_curve_kl = np.mean(errors_kl_matrix, axis=0)
avg_curve_nll = np.mean(errors_nll_matrix, axis=0)

best_n_wd = range_N[np.argmin(avg_curve_wd)]
best_n_kl = range_N[np.argmin(avg_curve_kl)]
best_n_nll = range_N[np.argmin(avg_curve_nll)]

# Asse X Generale per il plot Ground Truth
asse_x_generale = np.linspace(1e-9, global_max_x, num_points)
cdf_vera = distribuzione.cdf(asse_x_generale)
pdf_vera = distribuzione.pdf(asse_x_generale)


# Helper Functions per il Plotting
def plot_spaghetti(ax, x_list, y_list, y_true, label_true, title, color_true='k', x_max=None):
    ax.plot(asse_x_generale, y_true, color=color_true, linewidth=2, linestyle='-', label=label_true)
    for x, y in zip(x_list, y_list):
        ax.plot(x, y, 'k-', linewidth=0.5, alpha=0.3)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

    # === AGGIUNTA ===
    if x_max is not None:
        ax.set_xlim(0, x_max)
    # ================


def add_gt_line(ax, med_ref, std_dev=None, label_prefix="GT"):
    ax.axhline(med_ref, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(1.02, med_ref, f"{label_prefix}\nMedian: {med_ref:.4f}\nStd Dev: {std_dev:.4f}",
            transform=ax.get_yaxis_transform(),
            color='red', fontsize=8, va='center')


# =============================================================================
# 5. VISUALIZATION - FIGURE 1: FOCUS CDF
# =============================================================================

fig1, ax1 = plt.subplots(3, 2, sharey='row', figsize=(12, 18))
fig1.suptitle(f"CDF Analysis: {nome_dist} (M={M})", fontsize=14)

# Row 0: Spaghetti
# print("sim_data['cdf_M']", sim_data['cdf_M'])
plot_spaghetti(ax1[0, 0], sim_data['x_grids'], sim_data['cdf_M'], cdf_vera, 'True CDF', f"CDF (N=M={M})", x_max=visual_xlim)
plot_spaghetti(ax1[0, 1], sim_data['x_grids'], sim_data['cdf_N_cdf'], cdf_vera, 'True CDF', f"CDF (N={int(N_cdf)})", x_max=visual_xlim)

# Row 1: Derivative PDF
plot_spaghetti(ax1[1, 0], sim_data['x_grids'], sim_data['pdf_M'], pdf_vera, 'True PDF', f"Derivative PDF (N=M={M})", x_max=visual_xlim)
plot_spaghetti(ax1[1, 1], sim_data['x_grids'], sim_data['pdf_conn_to_cdf'], pdf_vera, 'True PDF',
               f"Derivative PDF (N={int(N_cdf)})", x_max=visual_xlim)

# Row 2: Boxplots WD
ax1[2, 0].boxplot(scalar_metrics['wd_emp_M'], medianprops=dict(color='k', linewidth=1.5))
add_gt_line(ax1[2, 0], mu_wd_true_M, std_wd_true_M, label_prefix="Est CDF vs True")
add_gt_line(ax1[2, 0], mu_wd_emp_M, std_wd_emp_M, label_prefix="Est CDF vs ECDF")
ax1[2, 0].set_title(f"Est CDF vs ECDF (N=M={M})", fontsize=10, fontweight='bold')
ax1[2, 0].set_ylabel("Wasserstein distance")
ax1[2, 0].grid(True, alpha=0.3)

ax1[2, 1].boxplot(scalar_metrics['wd_emp_N'], medianprops=dict(color='k', linewidth=1.5))
add_gt_line(ax1[2, 1], mu_wd_true_N, std_wd_true_N, label_prefix="Est CDF vs True")
add_gt_line(ax1[2, 1], mu_wd_emp_N, std_wd_emp_N, label_prefix="Est CDF vs ECDF")
ax1[2, 1].set_title(f"Est CDF vs ECDF (N={int(N_cdf)})", fontsize=10, fontweight='bold')
ax1[2, 1].set_ylabel("Wasserstein distance")
ax1[2, 1].grid(True, alpha=0.3)

# Fix Scale shared
ylim_0 = ax1[2, 0].get_ylim()
ylim_1 = ax1[2, 1].get_ylim()
global_min = min(ylim_0[0], ylim_1[0])
global_max = max(ylim_0[1], ylim_1[1])
ax1[2, 0].set_ylim(global_min, global_max)
ax1[2, 1].set_ylim(global_min, global_max)

for ax in ax1.flat:
    ax.tick_params(labelleft=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# =============================================================================
# 6. VISUALIZATION - FIGURE 2: FOCUS PDF
# =============================================================================

fig2, ax2 = plt.subplots(3, 2, sharey='row', figsize=(12, 18))
fig2.suptitle(f"PDF Analysis: {nome_dist} (M={M})", fontsize=14)

# Row 0: Spaghetti
plot_spaghetti(ax2[0, 0], sim_data['x_grids'], sim_data['pdf_M'], pdf_vera,
               'True PDF', f"PDF Estimator (N=M={M})", x_max=visual_xlim)
plot_spaghetti(ax2[0, 1], sim_data['x_grids'], sim_data['pdf_N_pdf'], pdf_vera,
               'True PDF', f"PDF Estimator (N={int(N_pdf)})", x_max=visual_xlim)

# Row 1: NLL
ax2[1, 0].boxplot(scalar_metrics['nll_M'], medianprops=dict(color='k', linewidth=1.5))
add_gt_line(ax2[1, 0], mu_nll_M, std_nll_M, label_prefix="Est PDF vs Samples")
ax2[1, 0].set_title(f"Est PDF vs Samples (N=M={M})", fontsize=10, fontweight='bold')
ax2[1, 0].set_ylabel("Negative log-likelihood")
ax2[1, 0].grid(True, alpha=0.3)

ax2[1, 1].boxplot(scalar_metrics['nll_N'], medianprops=dict(color='k', linewidth=1.5))
add_gt_line(ax2[1, 1], mu_nll_N, std_nll_N, label_prefix="Est PDF vs Samples")
ax2[1, 1].set_title(f"Est PDF vs Samples (N={int(N_pdf)})", fontsize=10, fontweight='bold')
ax2[1, 1].set_ylabel("Negative log-likelihood")
ax2[1, 1].grid(True, alpha=0.3)

# Fix Scale shared NLL
ylim_0 = ax2[1, 0].get_ylim()
ylim_1 = ax2[1, 1].get_ylim()
global_min = min(ylim_0[0], ylim_1[0])
global_max = max(ylim_0[1], ylim_1[1])
ax2[1, 0].set_ylim(global_min, global_max)
ax2[1, 1].set_ylim(global_min, global_max)

# Row 2: KL Divergence
ax2[2, 0].boxplot(scalar_metrics['kl_M'], medianprops=dict(color='k', linewidth=1.5))
add_gt_line(ax2[2, 0], mu_kl_M, std_kl_M, label_prefix="Est PDF vs True")
ax2[2, 0].set_title(f"Est PDF vs True (N=M={M})", fontsize=10, fontweight='bold')
ax2[2, 0].set_ylabel("Kullback–Leibler divergence")
ax2[2, 0].grid(True, alpha=0.3)

ax2[2, 1].boxplot(scalar_metrics['kl_N'], medianprops=dict(color='k', linewidth=1.5))
add_gt_line(ax2[2, 1], mu_kl_N, std_kl_N, label_prefix="Est PDF vs True")
ax2[2, 1].set_title(f"Est PDF vs True (N={N_pdf})", fontsize=10, fontweight='bold')
ax2[2, 1].set_ylabel("Kullback–Leibler divergence")
ax2[2, 1].grid(True, alpha=0.3)

# Fix Scale shared KL
ylim_0 = ax2[2, 0].get_ylim()
ylim_1 = ax2[2, 1].get_ylim()
global_min = min(ylim_0[0], ylim_1[0])
global_max = max(ylim_0[1], ylim_1[1])
ax2[2, 0].set_ylim(global_min, global_max)
ax2[2, 1].set_ylim(global_min, global_max)

for ax in ax2.flat:
    ax.tick_params(labelleft=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# =============================================================================
# 7. VISUALIZATION - FIGURE 3: BIAS-VARIANCE
# =============================================================================

fig3, ax3 = plt.subplots(1, 3, figsize=(18, 5))
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

# NLL
ax3[2].plot(range_N, avg_curve_nll, 'm-o', label='Mean NLL')
ax3[2].axvline(best_n_nll, color='r', linestyle='--', label=f'Best N={best_n_nll}')
ax3[2].axvline(N_pdf, color='orange', linestyle=':', linewidth=2, label=f'Heur N={int(N_pdf)}')
ax3[2].set_title("Sample Fit (Neg Log Likelihood)")
ax3[2].set_xlabel("Degree N")
ax3[2].grid(True, alpha=0.3)
ax3[2].legend()

plt.tight_layout()
plt.show()
