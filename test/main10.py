import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
from scipy.integrate import trapezoid

# Assumo che questi moduli siano presenti nel tuo ambiente
# Se create_ecdf restituisce un oggetto step function (es. statsmodels ECDF), ecdf(x) funziona.
from bernstein import create_ecdf, calculate_bernstein_cdf, calculate_bernstein_pdf
from KumaraswamyDist import KumaraswamyDist

# =============================================================================
# 1. CONFIGURAZIONE E PARAMETRI
# =============================================================================

scelta_dist = 'k'  # 'n', 'u', 'e', 'k'
M = 100  # Numero campioni
NUM_SIMULATIONS = 10
num_points = 500  # Risoluzione grafici

# --- CALCOLO N EURISTICI ---
N_pdf = math.ceil(M / math.log(M, 2))
N_cdf = math.ceil(M / math.log(M, 2)) ** 2

# Inizializzazione Distribuzione
distribuzione = None
nome_dist = ""

# GAUSSIANA
if scelta_dist == 'n':
    mu = 0
    sigma = 1
    distribuzione = stats.norm(loc=mu, scale=sigma)
    nome_dist = f"Normal(mu={mu}, sigma={sigma})"
# UNIFORME
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
    k_a = 2
    k_b = 5
    distribuzione = KumaraswamyDist(a=k_a, b=k_b)
    nome_dist = f"Kumaraswamy(a={k_a}, b={k_b})"

# =============================================================================
# 2. STRUTTURE DATI
# =============================================================================

sim_data = {
    'x_grids': [],
    'cdf_N_cdf': [], 'pdf_N_pdf': [],
    'cdf_M': [], 'pdf_M': [],
    'pdf_conn_to_cdf': []
}

# Strutture per salvare i vettori da plottare nei Boxplot
boxplot_data = {
    'cdf_M_diff_ecdf': [],  # |CDF_stima - ECDF| sulla griglia
    'cdf_N_diff_ecdf': [],
    'pdf_M_logprobs': [],  # Log(PDF_stima(campioni))
    'pdf_N_logprobs': []
}

# Strutture per salvare le metriche SCALARI per i titoli (Mediane)
scalar_metrics = {
    'wd_true_M': [], 'wd_emp_M': [],  # Wasserstein vs True, vs ECDF
    'wd_true_N': [], 'wd_emp_N': [],
    'kl_M': [], 'll_M': [],  # KL vs True, LogLikelihood sum
    'kl_N': [], 'll_N': []
}

# Metriche per curve Bias-Variance (Fig 3)
max_n_test = int(N_cdf * 1.5)
range_N = np.arange(5, max_n_test, 5)
errors_wd_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))
errors_kl_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))
errors_ll_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))

global_min_x, global_max_x = float('inf'), float('-inf')

# =============================================================================
# 3. LOOP DI SIMULAZIONE
# =============================================================================

for i in range(NUM_SIMULATIONS):
    print(f"Simulation cycle {i + 1}/{NUM_SIMULATIONS}")

    # A. Generazione Campioni
    campioni = distribuzione.rvs(size=M)
    campioni_ordinati = np.sort(campioni)
    ecdf = create_ecdf(campioni)  # Assumiamo restituisca un callable

    # B. Supporto Locale
    a, b = campioni_ordinati[0], campioni_ordinati[-1]
    global_min_x = min(global_min_x, a)
    global_max_x = max(global_max_x, b)

    # C. Griglia Locale
    curr_asse_x = np.linspace(a, b, num_points)
    sim_data['x_grids'].append(curr_asse_x)

    # Valori Veri
    cdf_true_loc = distribuzione.cdf(curr_asse_x)
    pdf_true_loc = distribuzione.pdf(curr_asse_x)

    # Valori ECDF sulla griglia (per confronto Boxplot CDF)
    ecdf_vals_grid = ecdf(curr_asse_x)

    # -----------------------------------------------------------
    # PARTE 1: Stime
    # -----------------------------------------------------------

    # === Caso N = M ===
    cdf_stima_M = calculate_bernstein_cdf(ecdf, M, a, b, curr_asse_x)
    pdf_stima_M = calculate_bernstein_pdf(ecdf, M, a, b, curr_asse_x)
    # Per LL servono valori sui campioni
    pdf_stima_M_samples = calculate_bernstein_pdf(ecdf, M, a, b, campioni)

    sim_data['cdf_M'].append(cdf_stima_M)
    sim_data['pdf_M'].append(pdf_stima_M)

    # --- BOXPLOT DATA (MODIFICA 1) ---
    # CDF: Differenza con ECDF
    diff_cdf_M = np.abs(ecdf_vals_grid - cdf_stima_M)
    boxplot_data['cdf_M_diff_ecdf'].append(diff_cdf_M)

    # PDF: Log-Probabilities dei campioni
    # Aggiungiamo epsilon per evitare log(0)
    log_probs_M = np.log(pdf_stima_M_samples + 1e-12)
    boxplot_data['pdf_M_logprobs'].append(log_probs_M)

    # --- SCALAR METRICS (MODIFICA 2) ---
    # CDF Metrics
    wd_true_M = trapezoid(np.abs(cdf_true_loc - cdf_stima_M), curr_asse_x)
    wd_emp_M = trapezoid(diff_cdf_M, curr_asse_x)
    scalar_metrics['wd_true_M'].append(wd_true_M)
    scalar_metrics['wd_emp_M'].append(wd_emp_M)

    # PDF Metrics
    kl_M = entropy(pk=pdf_true_loc, qk=pdf_stima_M + 1e-12)
    ll_M = np.sum(log_probs_M)
    scalar_metrics['kl_M'].append(kl_M)
    scalar_metrics['ll_M'].append(ll_M)

    # === Caso N Euristici ===

    # CDF (N_cdf)
    cdf_stima_N = calculate_bernstein_cdf(ecdf, int(N_cdf), a, b, curr_asse_x)
    pdf_conn_to_cdf = calculate_bernstein_pdf(ecdf, int(N_cdf), a, b, curr_asse_x)

    sim_data['cdf_N_cdf'].append(cdf_stima_N)
    sim_data['pdf_conn_to_cdf'].append(pdf_conn_to_cdf)

    # Boxplot CDF N
    diff_cdf_N = np.abs(ecdf_vals_grid - cdf_stima_N)
    boxplot_data['cdf_N_diff_ecdf'].append(diff_cdf_N)

    # Scalars CDF N
    wd_true_N = trapezoid(np.abs(cdf_true_loc - cdf_stima_N), curr_asse_x)
    wd_emp_N = trapezoid(diff_cdf_N, curr_asse_x)
    scalar_metrics['wd_true_N'].append(wd_true_N)
    scalar_metrics['wd_emp_N'].append(wd_emp_N)

    # PDF (N_pdf)
    pdf_stima_N = calculate_bernstein_pdf(ecdf, int(N_pdf), a, b, curr_asse_x)
    pdf_stima_N_samples = calculate_bernstein_pdf(ecdf, int(N_pdf), a, b, campioni)

    sim_data['pdf_N_pdf'].append(pdf_stima_N)

    # Boxplot PDF N
    log_probs_N = np.log(pdf_stima_N_samples + 1e-12)
    boxplot_data['pdf_N_logprobs'].append(log_probs_N)

    # Scalars PDF N
    kl_N = entropy(pk=pdf_true_loc, qk=pdf_stima_N + 1e-12)
    ll_N = np.sum(log_probs_N)
    scalar_metrics['kl_N'].append(kl_N)
    scalar_metrics['ll_N'].append(ll_N)

    # -----------------------------------------------------------
    # PARTE 2: Analisi Bias-Variance (Invariata)
    # -----------------------------------------------------------
    for idx_n, n_val in enumerate(range_N):
        cdf_temp = calculate_bernstein_cdf(ecdf, int(n_val), a, b, curr_asse_x)
        pdf_temp = calculate_bernstein_pdf(ecdf, int(n_val), a, b, curr_asse_x)
        pdf_temp_samples = calculate_bernstein_pdf(ecdf, int(n_val), a, b, campioni)

        errors_wd_matrix[i, idx_n] = trapezoid(np.abs(cdf_true_loc - cdf_temp), curr_asse_x)
        errors_kl_matrix[i, idx_n] = entropy(pk=pdf_true_loc, qk=pdf_temp + 1e-12)
        errors_ll_matrix[i, idx_n] = np.sum(np.log(pdf_temp_samples + 1e-12))

# =============================================================================
# 4. POST-PROCESSING E CALCOLO MEDIANE
# =============================================================================

# Mediane per i titoli dei grafici
med_wd_true_M = np.median(scalar_metrics['wd_true_M'])
med_wd_emp_M = np.median(scalar_metrics['wd_emp_M'])
med_wd_true_N = np.median(scalar_metrics['wd_true_N'])
med_wd_emp_N = np.median(scalar_metrics['wd_emp_N'])

med_kl_M = np.median(scalar_metrics['kl_M'])
med_ll_M = np.median(scalar_metrics['ll_M'])
med_kl_N = np.median(scalar_metrics['kl_N'])
med_ll_N = np.median(scalar_metrics['ll_N'])

# Curve medie Bias-Variance
avg_curve_wd = np.mean(errors_wd_matrix, axis=0)
avg_curve_kl = np.mean(errors_kl_matrix, axis=0)
avg_curve_ll = np.mean(errors_ll_matrix, axis=0)
best_n_wd = range_N[np.argmin(avg_curve_wd)]
best_n_kl = range_N[np.argmin(avg_curve_kl)]
best_n_ll = range_N[np.argmax(avg_curve_ll)]

if scelta_dist == 'k':
    asse_x_generale = np.linspace(max(0.0001, global_min_x), min(0.9999, global_max_x), num_points)
else:
    asse_x_generale = np.linspace(global_min_x, global_max_x, num_points)

cdf_vera = distribuzione.cdf(asse_x_generale)
pdf_vera = distribuzione.pdf(asse_x_generale)


def plot_spaghetti(ax, x_list, y_list, y_true, label_true, title, color_true='r'):
    for x, y in zip(x_list, y_list):
        ax.plot(x, y, 'k-', linewidth=0.5, alpha=0.3)
    ax.plot(asse_x_generale, y_true, color=color_true, linewidth=2, linestyle='-', label=label_true)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')


# --- FUNZIONE DI SUPPORTO PER LA LINEA ROSSA ---
def add_gt_line(ax, med_ref, label_prefix="GT"):
    """
    Calcola la mediana globale di tutti gli errori della Ground Truth
    e disegna una linea orizzontale.
    """
    # Concatena tutti gli array di tutti i cicli di simulazione
    # all_ref_errors = np.concatenate(ref_data_list)
    # med_ref = np.median(all_ref_errors)

    ax.axhline(med_ref, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    # Aggiunge testo annotazione (coordinate trasformate per stare a destra)
    ax.text(1.02, med_ref, f"{label_prefix}\nMed: {med_ref:.3f}",
            transform=ax.get_yaxis_transform(),
            color='red', fontsize=8, va='center')

# =============================================================================
# 5. VISUALIZZAZIONE - FIGURE 1: FOCUS CDF
# =============================================================================

fig1, ax1 = plt.subplots(3, 2, figsize=(14, 15))
fig1.suptitle(f"CDF Analysis: {nome_dist} (M={M})", fontsize=16)

# Riga 1 e 2: Spaghetti Plots
plot_spaghetti(ax1[0, 0], sim_data['x_grids'], sim_data['cdf_M'], cdf_vera, 'True CDF', f"CDF (N=M={M})")
plot_spaghetti(ax1[0, 1], sim_data['x_grids'], sim_data['cdf_N_cdf'], cdf_vera, 'True CDF', f"CDF (N={int(N_cdf)})")
plot_spaghetti(ax1[1, 0], sim_data['x_grids'], sim_data['pdf_M'], pdf_vera, 'True PDF', f"Derivative PDF (N=M={M})",
               color_true='g')
plot_spaghetti(ax1[1, 1], sim_data['x_grids'], sim_data['pdf_conn_to_cdf'], pdf_vera, 'True PDF',
               f"Derivative PDF (N={int(N_cdf)})", color_true='g')

# --- RIGA 3: BOXPLOTS PER SIMULAZIONE (CDF vs ECDF) ---

# 3,1: N=M
ax1[2, 0].boxplot(boxplot_data['cdf_M_diff_ecdf'], patch_artist=True,
                  boxprops=dict(facecolor='lightblue'),
                  medianprops=dict(color='blue'))
add_gt_line(ax1[2, 0], med_wd_true_M, label_prefix="True vs ECDF")
# Titolo con le Mediane (True WD e Emp WD)
title_cdf_M = (f"Diff vs ECDF (N=M)\n"
               f"Med WD(True): {med_wd_true_M:.4f} | Med WD(Emp): {med_wd_emp_M:.4f}")
ax1[2, 0].set_title(title_cdf_M, fontsize=9)
ax1[2, 0].set_ylabel("|ECDF - Est CDF|")
ax1[2, 0].set_xlabel("Simulation ID")
ax1[2, 0].grid(True, alpha=0.3)

# 3,2: N=N_cdf
ax1[2, 1].boxplot(boxplot_data['cdf_N_diff_ecdf'], patch_artist=True,
                  boxprops=dict(facecolor='lightgreen'),
                  medianprops=dict(color='green'))
add_gt_line(ax1[2, 1], med_wd_true_N, label_prefix="True vs ECDF")
title_cdf_N = (f"Diff vs ECDF (N={int(N_cdf)})\n"
               f"Med WD(True): {med_wd_true_N:.4f} | Med WD(Emp): {med_wd_emp_N:.4f}")
ax1[2, 1].set_title(title_cdf_N, fontsize=9)
ax1[2, 1].set_ylabel("|ECDF - Est CDF|")
ax1[2, 1].set_xlabel("Simulation ID")
ax1[2, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# =============================================================================
# 6. VISUALIZZAZIONE - FIGURE 2: FOCUS PDF
# =============================================================================

fig2, ax2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle(f"PDF Analysis: {nome_dist} (M={M})", fontsize=16)

# Riga 1: Spaghetti PDF
plot_spaghetti(ax2[0, 0], sim_data['x_grids'], sim_data['pdf_M'], pdf_vera, 'True PDF', f"PDF Estimator (N=M={M})",
               color_true='b')
plot_spaghetti(ax2[0, 1], sim_data['x_grids'], sim_data['pdf_N_pdf'], pdf_vera, 'True PDF',
               f"PDF Estimator (N={int(N_pdf)})", color_true='b')

# --- RIGA 2: BOXPLOTS PER SIMULAZIONE (PDF: Log-Likelihood Samples) ---

# 2,1: N=M
# ax2[1, 0].boxplot(boxplot_data['pdf_M_logprobs'], patch_artist=True,
#                   boxprops=dict(facecolor='salmon'),
#                   medianprops=dict(color='red'))
# add_gt_line(ax2[1, 0], med_kl_M, label_prefix="True vs Hist")
# Titolo con Mediane (True KL e Emp LL)
title_pdf_M = (f"Log-Probs of Samples (N=M)\n"
               f"Med KL(True): {med_kl_M:.4f} | Med LL(Sum): {med_ll_M:.2f}")
ax2[1, 0].set_title(title_pdf_M, fontsize=9)
ax2[1, 0].set_ylabel("Log( f_est(x_i) )")
ax2[1, 0].set_xlabel("Simulation ID")
ax2[1, 0].grid(True, alpha=0.3)

# 2,2: N=N_pdf
# ax2[1, 1].boxplot(boxplot_data['pdf_N_logprobs'], patch_artist=True,
#                   boxprops=dict(facecolor='orange'),
#                   medianprops=dict(color='darkorange'))
# add_gt_line(ax2[1, 1], med_kl_N, label_prefix="True vs Hist")
title_pdf_N = (f"Log-Probs of Samples (N={int(N_pdf)})\n"
               f"Med KL(True): {med_kl_N:.4f} | Med LL(Sum): {med_ll_N:.2f}")
ax2[1, 1].set_title(title_pdf_N, fontsize=9)
ax2[1, 1].set_ylabel("Log( f_est(x_i) )")
ax2[1, 1].set_xlabel("Simulation ID")
ax2[1, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# =============================================================================
# 7. VISUALIZZAZIONE - FIGURE 3: BIAS-VARIANCE (Invariata)
# =============================================================================

fig3, ax3 = plt.subplots(1, 2, figsize=(12, 6))
fig3.suptitle(f"Metric Sensitivity vs Degree N (Avg over {NUM_SIMULATIONS} runs)", fontsize=16)

ax3[0].plot(range_N, avg_curve_wd, 'b-o', markersize=4, label='Avg WD')
ax3[0].axvline(best_n_wd, color='r', linestyle='--', label=f'Best N={best_n_wd}')
ax3[0].axvline(N_cdf, color='orange', linestyle=':', linewidth=2, label=f'Heur N={int(N_cdf)}')
ax3[0].set_title("CDF Error (Wasserstein)")
ax3[0].set_xlabel("Degree N")
ax3[0].grid(True, alpha=0.3)
ax3[0].legend()

ax3[1].plot(range_N, avg_curve_kl, 'g-o', markersize=4, label='Avg KL')
ax3[1].axvline(best_n_kl, color='r', linestyle='--', label=f'Best N={best_n_kl}')
ax3[1].axvline(N_pdf, color='orange', linestyle=':', linewidth=2, label=f'Heur N={int(N_pdf)}')
ax3[1].set_title("PDF Error (KL Divergence)")
ax3[1].set_xlabel("Degree N")
ax3[1].grid(True, alpha=0.3)
ax3[1].legend()

'''
ax3[2].plot(range_N, avg_curve_ll, 'm-o', markersize=4, label='Avg Log-Likelihood')
ax3[2].axvline(best_n_ll, color='r', linestyle='--', label=f'Best N={best_n_ll}')
ax3[2].axvline(N_pdf, color='orange', linestyle=':', linewidth=2, label=f'Heur N={int(N_pdf)}')
ax3[2].set_title("Sample Fit (Log-Likelihood)")
ax3[2].set_xlabel("Degree N")
ax3[2].grid(True, alpha=0.3)
ax3[2].legend()'''

plt.tight_layout()
plt.show()
