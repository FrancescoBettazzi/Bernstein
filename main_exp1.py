import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
from scipy.integrate import trapezoid

# =============================================================================
# IMPORT MODULO BERNSTEIN EXP (Dominio 0 -> Infinito)
# =============================================================================
# Assicurati che il file bernstein_exp.py contenga:
# create_ecdf, calculate_bernstein_exp_cdf, calculate_bernstein_exp_pdf
from bernstein_exp import create_ecdf, calculate_bernstein_exp_cdf, calculate_bernstein_exp_pdf

# =============================================================================
# 1. CONFIGURAZIONE E PARAMETRI
# =============================================================================

# Scegli la distribuzione qui:
# 'exp'        : Exponential
# 'erlang'     : Erlang(n=3, mean=1)
# 'weibull_1.5': Weibull(scale=1, shape=1.5)
# 'weibull_0.5': Weibull(scale=1, shape=0.5)
# 'ln_1.8'     : Lognormal(mu=1, sigma=1.8) -> Heavy Tail
# 'ln_0.8'     : Lognormal(mu=1, sigma=0.8)
# 'ln_0.2'     : Lognormal(mu=1, sigma=0.2)
scelta_dist = 'ln_1.8'

M = 100  # Numero campioni
NUM_SIMULATIONS = 10  # Numero di run per i boxplot
num_points = 500  # Risoluzione grafici

# --- CALCOLO N EURISTICI ---
# (Rimangono validi come punto di partenza)
N_pdf = math.ceil(M / math.log(M, 2))
N_cdf = math.ceil(M / math.log(M, 2)) ** 2

# Inizializzazione Distribuzione
distribuzione = None
nome_dist = ""

if scelta_dist == 'exp':
    lam = 1.0
    distribuzione = stats.expon(scale=1 / lam)
    nome_dist = f"Exponential(lambda={lam})"

elif scelta_dist == 'erlang':
    # Erlang(n, lambda) con media 1 => media = n/lambda = 1 => lambda = n
    # In scipy gamma: a=shape(n), scale=1/lambda
    n_erlang = 3
    lam_erlang = 3.0
    distribuzione = stats.gamma(a=n_erlang, scale=1 / lam_erlang)
    nome_dist = f"Erlang(n={n_erlang}, mean=1)"

elif scelta_dist == 'weibull_1.5':
    # Weibull(scale=1, shape=1.5)
    distribuzione = stats.weibull_min(c=1.5, scale=1.0)
    nome_dist = "Weibull(scale=1, shape=1.5)"

elif scelta_dist == 'weibull_0.5':
    # Weibull(scale=1, shape=0.5)
    distribuzione = stats.weibull_min(c=0.5, scale=1.0)
    nome_dist = "Weibull(scale=1, shape=0.5)"

elif scelta_dist == 'ln_1.8':
    # Lognormal(mu=1, sigma=1.8)
    # Scipy: s=sigma, scale=exp(mu)
    mu, sigma = 1.0, 1.8
    distribuzione = stats.lognorm(s=sigma, scale=np.exp(mu))
    nome_dist = f"Lognormal(mu={mu}, sigma={sigma})"

elif scelta_dist == 'ln_0.8':
    mu, sigma = 1.0, 0.8
    distribuzione = stats.lognorm(s=sigma, scale=np.exp(mu))
    nome_dist = f"Lognormal(mu={mu}, sigma={sigma})"

elif scelta_dist == 'ln_0.2':
    mu, sigma = 1.0, 0.2
    distribuzione = stats.lognorm(s=sigma, scale=np.exp(mu))
    nome_dist = f"Lognormal(mu={mu}, sigma={sigma})"

# =============================================================================
# 2. STRUTTURE DATI
# =============================================================================

sim_data = {
    'x_grids': [],
    'cdf_N_cdf': [], 'pdf_N_pdf': [],
    'cdf_M': [], 'pdf_M': [],
    'pdf_conn_to_cdf': []
}

boxplot_data = {
    'cdf_M_diff_ecdf': [],
    'cdf_N_diff_ecdf': [],
    'pdf_M_logprobs': [],
    'pdf_N_logprobs': []
}

scalar_metrics = {
    'wd_true_M': [], 'wd_emp_M': [],
    'wd_true_N': [], 'wd_emp_N': [],
    'kl_M': [], 'll_M': [],
    'kl_N': [], 'll_N': []
}

# Metriche per curve Bias-Variance
max_n_test = int(N_cdf * 1.5)
range_N = np.arange(5, max_n_test, 5)
errors_wd_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))
errors_kl_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))
errors_ll_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))

# Per gestire il plot range dinamico
max_x_observed = 0

# =============================================================================
# 3. LOOP DI SIMULAZIONE
# =============================================================================

for i in range(NUM_SIMULATIONS):
    # A. Generazione Campioni
    campioni = distribuzione.rvs(size=M)
    campioni = np.sort(campioni)  # Utile averli ordinati

    # Creazione ECDF (Step function)
    ecdf_func = create_ecdf(campioni)

    # B. Parametri per Bernstein Exp
    # Euristica: scale = 1 / media empirica
    # Questo mappa la media dei dati al centro della trasformazione z ~ 1 - exp(-1)
    mean_val = np.mean(campioni)
    bernstein_scale = 1.0 / mean_val if mean_val > 1e-9 else 1.0

    # C. Definizione Griglia Locale
    # Poiché siamo su [0, inf), prendiamo fino a un po' oltre il massimo campione
    # o un quantile teorico alto per vedere la coda.
    x_max_loc = max(campioni[-1] * 1.5, distribuzione.ppf(0.99))
    if x_max_loc > max_x_observed: max_x_observed = x_max_loc

    curr_asse_x = np.linspace(0, x_max_loc, num_points)
    sim_data['x_grids'].append(curr_asse_x)

    # Valori Veri
    cdf_true_loc = distribuzione.cdf(curr_asse_x)
    pdf_true_loc = distribuzione.pdf(curr_asse_x)

    # Valori ECDF su griglia (per metriche empiriche)
    ecdf_vals_grid = ecdf_func(curr_asse_x)

    # -----------------------------------------------------------
    # PARTE 1: Stime (USANDO calculate_bernstein_exp_...)
    # -----------------------------------------------------------

    # === Caso N = M ===
    cdf_stima_M = calculate_bernstein_exp_cdf(ecdf_func, M, curr_asse_x, scale=bernstein_scale)
    pdf_stima_M = calculate_bernstein_exp_pdf(ecdf_func, M, curr_asse_x, scale=bernstein_scale)
    pdf_stima_M_samples = calculate_bernstein_exp_pdf(ecdf_func, M, campioni, scale=bernstein_scale)

    sim_data['cdf_M'].append(cdf_stima_M)
    sim_data['pdf_M'].append(pdf_stima_M)

    # Boxplot & Metrics M
    diff_cdf_M = np.abs(ecdf_vals_grid - cdf_stima_M)
    boxplot_data['cdf_M_diff_ecdf'].append(diff_cdf_M)

    log_probs_M = np.log(pdf_stima_M_samples + 1e-12)
    boxplot_data['pdf_M_logprobs'].append(log_probs_M)

    scalar_metrics['wd_true_M'].append(trapezoid(np.abs(cdf_true_loc - cdf_stima_M), curr_asse_x))
    scalar_metrics['wd_emp_M'].append(trapezoid(diff_cdf_M, curr_asse_x))
    scalar_metrics['kl_M'].append(entropy(pk=pdf_true_loc, qk=pdf_stima_M + 1e-12))
    scalar_metrics['ll_M'].append(np.sum(log_probs_M))

    # === Caso N Euristici ===

    # CDF (N_cdf)
    cdf_stima_N = calculate_bernstein_exp_cdf(ecdf_func, int(N_cdf), curr_asse_x, scale=bernstein_scale)
    pdf_conn_to_cdf = calculate_bernstein_exp_pdf(ecdf_func, int(N_cdf), curr_asse_x, scale=bernstein_scale)

    sim_data['cdf_N_cdf'].append(cdf_stima_N)
    sim_data['pdf_conn_to_cdf'].append(pdf_conn_to_cdf)

    diff_cdf_N = np.abs(ecdf_vals_grid - cdf_stima_N)
    boxplot_data['cdf_N_diff_ecdf'].append(diff_cdf_N)

    scalar_metrics['wd_true_N'].append(trapezoid(np.abs(cdf_true_loc - cdf_stima_N), curr_asse_x))
    scalar_metrics['wd_emp_N'].append(trapezoid(diff_cdf_N, curr_asse_x))

    # PDF (N_pdf)
    pdf_stima_N = calculate_bernstein_exp_pdf(ecdf_func, int(N_pdf), curr_asse_x, scale=bernstein_scale)
    pdf_stima_N_samples = calculate_bernstein_exp_pdf(ecdf_func, int(N_pdf), campioni, scale=bernstein_scale)

    sim_data['pdf_N_pdf'].append(pdf_stima_N)

    log_probs_N = np.log(pdf_stima_N_samples + 1e-12)
    boxplot_data['pdf_N_logprobs'].append(log_probs_N)

    scalar_metrics['kl_N'].append(entropy(pk=pdf_true_loc, qk=pdf_stima_N + 1e-12))
    scalar_metrics['ll_N'].append(np.sum(log_probs_N))

    # -----------------------------------------------------------
    # PARTE 2: Analisi Bias-Variance
    # -----------------------------------------------------------
    for idx_n, n_val in enumerate(range_N):
        cdf_temp = calculate_bernstein_exp_cdf(ecdf_func, int(n_val), curr_asse_x, scale=bernstein_scale)
        pdf_temp = calculate_bernstein_exp_pdf(ecdf_func, int(n_val), curr_asse_x, scale=bernstein_scale)
        pdf_temp_samples = calculate_bernstein_exp_pdf(ecdf_func, int(n_val), campioni, scale=bernstein_scale)

        errors_wd_matrix[i, idx_n] = trapezoid(np.abs(cdf_true_loc - cdf_temp), curr_asse_x)
        errors_kl_matrix[i, idx_n] = entropy(pk=pdf_true_loc, qk=pdf_temp + 1e-12)
        errors_ll_matrix[i, idx_n] = np.sum(np.log(pdf_temp_samples + 1e-12))

    print(f"Sim {i + 1}/{NUM_SIMULATIONS} done. Scale used: {bernstein_scale:.3f}")

# =============================================================================
# 4. POST-PROCESSING
# =============================================================================

# Calcolo Mediane
med_wd_true_M = np.median(scalar_metrics['wd_true_M'])
med_wd_emp_M = np.median(scalar_metrics['wd_emp_M'])
med_wd_true_N = np.median(scalar_metrics['wd_true_N'])
med_wd_emp_N = np.median(scalar_metrics['wd_emp_N'])

med_kl_M = np.median(scalar_metrics['kl_M'])
med_ll_M = np.median(scalar_metrics['ll_M'])
med_kl_N = np.median(scalar_metrics['kl_N'])
med_ll_N = np.median(scalar_metrics['ll_N'])

# Curve Medie Bias-Variance
avg_curve_wd = np.mean(errors_wd_matrix, axis=0)
avg_curve_kl = np.mean(errors_kl_matrix, axis=0)
best_n_wd = range_N[np.argmin(avg_curve_wd)]
best_n_kl = range_N[np.argmin(avg_curve_kl)]

# Asse X Generale per i plot "Veri" (fino al max osservato)
asse_x_generale = np.linspace(0, max_x_observed, num_points)
cdf_vera = distribuzione.cdf(asse_x_generale)
pdf_vera = distribuzione.pdf(asse_x_generale)


def plot_spaghetti(ax, x_list, y_list, y_true, label_true, title, color_true='r'):
    for x, y in zip(x_list, y_list):
        ax.plot(x, y, 'k-', linewidth=0.5, alpha=0.3)
    ax.plot(asse_x_generale, y_true, color=color_true, linewidth=2, linestyle='-', label=label_true)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')


# =============================================================================
# 5. VISUALIZZAZIONE - FIGURE 1: FOCUS CDF (Exp Domain)
# =============================================================================

fig1, ax1 = plt.subplots(3, 2, figsize=(14, 15))
fig1.suptitle(f"Exponential-Bernstein CDF Analysis: {nome_dist} (M={M})", fontsize=16)

# Riga 1: Spaghetti CDF
plot_spaghetti(ax1[0, 0], sim_data['x_grids'], sim_data['cdf_M'], cdf_vera, 'True CDF', f"CDF Exp (N=M={M})")
plot_spaghetti(ax1[0, 1], sim_data['x_grids'], sim_data['cdf_N_cdf'], cdf_vera, 'True CDF', f"CDF Exp (N={int(N_cdf)})")

# Riga 2: Spaghetti PDF derivata da CDF
plot_spaghetti(ax1[1, 0], sim_data['x_grids'], sim_data['pdf_M'], pdf_vera, 'True PDF', f"Deriv PDF (N=M={M})",
               color_true='g')
plot_spaghetti(ax1[1, 1], sim_data['x_grids'], sim_data['pdf_conn_to_cdf'], pdf_vera, 'True PDF',
               f"Deriv PDF (N={int(N_cdf)})", color_true='g')

# Riga 3: Boxplots Errori
ax1[2, 0].boxplot(boxplot_data['cdf_M_diff_ecdf'], patch_artist=True, boxprops=dict(facecolor='lightblue'),
                  medianprops=dict(color='blue'))
ax1[2, 0].set_title(f"Diff vs ECDF (N=M)\nWD(True): {med_wd_true_M:.4f} | WD(Emp): {med_wd_emp_M:.4f}", fontsize=9)
ax1[2, 0].set_ylabel("|ECDF - Est CDF|")

ax1[2, 1].boxplot(boxplot_data['cdf_N_diff_ecdf'], patch_artist=True, boxprops=dict(facecolor='lightgreen'),
                  medianprops=dict(color='green'))
ax1[2, 1].set_title(f"Diff vs ECDF (N={int(N_cdf)})\nWD(True): {med_wd_true_N:.4f} | WD(Emp): {med_wd_emp_N:.4f}",
                    fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# =============================================================================
# 6. VISUALIZZAZIONE - FIGURE 2: FOCUS PDF (Exp Domain)
# =============================================================================

fig2, ax2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle(f"Exponential-Bernstein PDF Analysis: {nome_dist}", fontsize=16)

plot_spaghetti(ax2[0, 0], sim_data['x_grids'], sim_data['pdf_M'], pdf_vera, 'True PDF', f"PDF Est (N=M={M})",
               color_true='b')
plot_spaghetti(ax2[0, 1], sim_data['x_grids'], sim_data['pdf_N_pdf'], pdf_vera, 'True PDF', f"PDF Est (N={int(N_pdf)})",
               color_true='b')

ax2[1, 0].boxplot(boxplot_data['pdf_M_logprobs'], patch_artist=True, boxprops=dict(facecolor='salmon'),
                  medianprops=dict(color='red'))
ax2[1, 0].set_title(f"Log-Probs (N=M)\nKL(True): {med_kl_M:.4f} | LL: {med_ll_M:.2f}", fontsize=9)
ax2[1, 0].set_ylabel("Log( f(x) )")

ax2[1, 1].boxplot(boxplot_data['pdf_N_logprobs'], patch_artist=True, boxprops=dict(facecolor='orange'),
                  medianprops=dict(color='darkorange'))
ax2[1, 1].set_title(f"Log-Probs (N={int(N_pdf)})\nKL(True): {med_kl_N:.4f} | LL: {med_ll_N:.2f}", fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# =============================================================================
# 7. FIGURE 3: BIAS-VARIANCE
# =============================================================================

fig3, ax3 = plt.subplots(1, 2, figsize=(12, 6))
fig3.suptitle(f"Metric Sensitivity vs Degree N (Exp Transform)", fontsize=16)

ax3[0].plot(range_N, avg_curve_wd, 'b-o', label='Avg WD')
ax3[0].axvline(best_n_wd, color='r', linestyle='--', label=f'Best N={best_n_wd}')
ax3[0].axvline(N_cdf, color='orange', linestyle=':', label=f'Heur N={int(N_cdf)}')
ax3[0].set_title("CDF Error (Wasserstein)")
ax3[0].set_xlabel("N")
ax3[0].legend()
ax3[0].grid(True, alpha=0.3)

ax3[1].plot(range_N, avg_curve_kl, 'g-o', label='Avg KL')
ax3[1].axvline(best_n_kl, color='r', linestyle='--', label=f'Best N={best_n_kl}')
ax3[1].axvline(N_pdf, color='orange', linestyle=':', label=f'Heur N={int(N_pdf)}')
ax3[1].set_title("PDF Error (KL Divergence)")
ax3[1].set_xlabel("N")
ax3[1].legend()
ax3[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
