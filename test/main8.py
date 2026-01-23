import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
from scipy.integrate import trapezoid

# Assumo che questi moduli siano presenti nel tuo ambiente
from bernstein import create_ecdf, calculate_bernstein_cdf, calculate_bernstein_pdf
from KumaraswamyDist import KumaraswamyDist

# =============================================================================
# 1. CONFIGURAZIONE E PARAMETRI
# =============================================================================

scelta_dist = 'k'  # 'n', 'u', 'e', 'k'
M = 100  # Numero campioni
NUM_SIMULATIONS = 10  # Aumentato leggermente per avere boxplot più significativi
num_points = 500  # Risoluzione grafici

# --- CALCOLO N EURISTICI ---
# N.B. N_pdf e N_cdf sono spesso diversi
N_pdf = math.ceil(M / math.log(M, 2))
N_cdf = math.ceil(M / math.log(M, 2)) ** 2

# Inizializzazione Distribuzione
distribuzione = None
nome_dist = ""

if scelta_dist == 'n':
    mu = 0;
    sigma = 1
    distribuzione = stats.norm(loc=mu, scale=sigma)
    nome_dist = f"Normal(mu={mu}, sigma={sigma})"
elif scelta_dist == 'u':
    uni_a = 5;
    uni_b = 15
    distribuzione = stats.uniform(loc=uni_a, scale=(uni_b - uni_a))
    nome_dist = f"Uniform[{uni_a}, {uni_b}]"
elif scelta_dist == 'e':
    lambda_param = 0.5
    distribuzione = stats.expon(loc=0, scale=1 / lambda_param)
    nome_dist = f"Exponential(lambda={lambda_param})"
elif scelta_dist == 'k':
    k_a = 2;
    k_b = 5
    distribuzione = KumaraswamyDist(a=k_a, b=k_b)
    nome_dist = f"Kumaraswamy(a={k_a}, b={k_b})"

# =============================================================================
# 2. STRUTTURE DATI
# =============================================================================

sim_data = {
    'x_grids': [],
    'cdf_N_cdf': [], 'pdf_N_pdf': [],  # Casi ottimali rispettivi
    'cdf_M': [], 'pdf_M': [],  # Caso N=M
    'pdf_conn_to_cdf': []  # NUOVO: PDF calcolata usando N_cdf (per coerenza visiva Fig 1)
}

metrics = {
    'wd_ecdf_N': [], 'wd_true_N': [], 'kl_N': [], 'll_N': [],
    'wd_ecdf_M': [], 'wd_true_M': [], 'kl_M': [], 'll_M': []
}

global_min_x, global_max_x = float('inf'), float('-inf')

# --- STRUTTURE PER ANALISI ROBUSTA BIAS-VARIANCE ---
max_n_test = int(N_cdf * 1.5)
range_N = np.arange(5, max_n_test, 5)

errors_wd_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))
errors_kl_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))
errors_ll_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))

# =============================================================================
# 3. LOOP DI SIMULAZIONE
# =============================================================================

for i in range(NUM_SIMULATIONS):
    print(f"Simulation cycle {i + 1}/{NUM_SIMULATIONS}")

    # A. Generazione Campioni
    campioni = distribuzione.rvs(size=M)
    campioni_ordinati = np.sort(campioni)
    ecdf = create_ecdf(campioni)

    # B. Supporto Locale
    a, b = campioni_ordinati[0], campioni_ordinati[-1]
    if a < global_min_x: global_min_x = a
    if b > global_max_x: global_max_x = b

    # C. Griglia Locale
    curr_asse_x = np.linspace(a, b, num_points)
    sim_data['x_grids'].append(curr_asse_x)

    # Valori Veri/Empirici
    ecdf_values = ecdf(curr_asse_x)
    pdf_true_loc = distribuzione.pdf(curr_asse_x)
    cdf_true_loc = distribuzione.cdf(curr_asse_x)

    # -----------------------------------------------------------
    # PARTE 1: Stime Puntuali (N Euristici vs N=M)
    # -----------------------------------------------------------

    # --- Caso N_cdf (Euristico per CDF) ---
    cdf_stima_N = calculate_bernstein_cdf(ecdf, int(N_cdf), a, b, curr_asse_x)
    # NUOVO: Calcolo la PDF usando N_cdf solo per mostrarla nel grafico CDF focus
    pdf_stima_conn_to_cdf = calculate_bernstein_pdf(ecdf, int(N_cdf), a, b, curr_asse_x)

    sim_data['cdf_N_cdf'].append(cdf_stima_N)
    sim_data['pdf_conn_to_cdf'].append(pdf_stima_conn_to_cdf)

    # Metriche CDF (usando N_cdf)
    metrics['wd_ecdf_N'].append(trapezoid(np.abs(ecdf_values - cdf_stima_N), curr_asse_x))
    metrics['wd_true_N'].append(trapezoid(np.abs(cdf_true_loc - cdf_stima_N), curr_asse_x))

    # --- Caso N_pdf (Euristico per PDF) ---
    # Nota: Usiamo questo N diverso per calcolare le metriche PDF ottimali
    pdf_stima_N = calculate_bernstein_pdf(ecdf, int(N_pdf), a, b, curr_asse_x)
    pdf_vals_campioni_N = calculate_bernstein_pdf(ecdf, int(N_pdf), a, b, campioni)

    sim_data['pdf_N_pdf'].append(pdf_stima_N)

    metrics['kl_N'].append(entropy(pk=pdf_true_loc, qk=pdf_stima_N + 1e-12))
    metrics['ll_N'].append(np.sum(np.log(pdf_vals_campioni_N + 1e-12)))

    # --- Caso N = M (Overfitting) ---
    cdf_stima_M = calculate_bernstein_cdf(ecdf, M, a, b, curr_asse_x)
    pdf_stima_M = calculate_bernstein_pdf(ecdf, M, a, b, curr_asse_x)
    pdf_vals_campioni_M = calculate_bernstein_pdf(ecdf, M, a, b, campioni)

    sim_data['cdf_M'].append(cdf_stima_M)
    sim_data['pdf_M'].append(pdf_stima_M)

    metrics['wd_ecdf_M'].append(trapezoid(np.abs(ecdf_values - cdf_stima_M), curr_asse_x))
    metrics['wd_true_M'].append(trapezoid(np.abs(cdf_true_loc - cdf_stima_M), curr_asse_x))
    metrics['kl_M'].append(entropy(pk=pdf_true_loc, qk=pdf_stima_M + 1e-12))
    metrics['ll_M'].append(np.sum(np.log(pdf_vals_campioni_M + 1e-12)))

    # -----------------------------------------------------------
    # PARTE 2: Analisi Bias-Variance
    # -----------------------------------------------------------
    for idx_n, n_val in enumerate(range_N):
        cdf_temp = calculate_bernstein_cdf(ecdf, int(n_val), a, b, curr_asse_x)
        pdf_temp = calculate_bernstein_pdf(ecdf, int(n_val), a, b, curr_asse_x)
        pdf_temp_samples = calculate_bernstein_pdf(ecdf, int(n_val), a, b, campioni)

        errors_wd_matrix[i, idx_n] = trapezoid(np.abs(cdf_true_loc - cdf_temp), curr_asse_x)
        errors_kl_matrix[i, idx_n] = entropy(pk=pdf_true_loc, qk=pdf_temp + 1e-12)
        errors_ll_matrix[i, idx_n] = np.sum(np.log(pdf_temp_samples + 1e-12))

# =============================================================================
# 4. POST-PROCESSING
# =============================================================================

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


# Funzione Helper per Spaghetti Plot
def plot_spaghetti(ax, x_list, y_list, y_true, label_true, title, color_true='r'):
    for x, y in zip(x_list, y_list):
        ax.plot(x, y, 'k-', linewidth=0.5, alpha=0.3)
    ax.plot(asse_x_generale, y_true, color=color_true, linewidth=2, linestyle='-', label=label_true)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')


# =============================================================================
# 5. VISUALIZZAZIONE - FIGURE 1: FOCUS CDF (3x2)
# =============================================================================

fig1, ax1 = plt.subplots(3, 2, figsize=(14, 15))
fig1.suptitle(f"CDF Analysis: {nome_dist} (M={M})", fontsize=16)

# --- RIGA 1: CDF Spaghetti ---
# 1,1: CDF con N=M
plot_spaghetti(ax1[0, 0], sim_data['x_grids'], sim_data['cdf_M'], cdf_vera, 'True CDF',
               f"CDF Estimator (N=M={M})")

# 1,2: CDF con N=N_cdf
plot_spaghetti(ax1[0, 1], sim_data['x_grids'], sim_data['cdf_N_cdf'], cdf_vera, 'True CDF',
               f"CDF Estimator (Heuristic N={int(N_cdf)})")

# --- RIGA 2: PDF Corrispondenti ---
# 2,1: PDF con N=M
plot_spaghetti(ax1[1, 0], sim_data['x_grids'], sim_data['pdf_M'], pdf_vera, 'True PDF',
               f"Derivative PDF (N=M={M})", color_true='g')

# 2,2: PDF con N=N_cdf (Nota: qui usiamo N_cdf anche per la PDF per coerenza colonna)
plot_spaghetti(ax1[1, 1], sim_data['x_grids'], sim_data['pdf_conn_to_cdf'], pdf_vera, 'True PDF',
               f"Derivative PDF (N={int(N_cdf)})", color_true='g')

# --- RIGA 3: Boxplots Errori CDF (Wasserstein) ---

# 3,1: Errori per N=M
data_err_M = metrics['wd_ecdf_M']
med_wd_ecdf_M = np.median(metrics['wd_ecdf_M'])
med_wd_true_M = np.median(metrics['wd_true_M'])

ax1[2, 0].boxplot(data_err_M, patch_artist=True, boxprops=dict(facecolor='lightblue'))
ax1[2, 0].set_title(f"WD Error Distribution (N=M)\nMedian vs ECDF: {med_wd_ecdf_M:.4f} | vs True: {med_wd_true_M:.4f}",
                    fontsize=10)
ax1[2, 0].set_ylabel("Wasserstein Distance")
ax1[2, 0].set_xticks([])
ax1[2, 0].grid(True, alpha=0.3)

# 3,2: Errori per N=N_cdf
data_err_N = metrics['wd_ecdf_N']
med_wd_ecdf_N = np.median(metrics['wd_ecdf_N'])
med_wd_true_N = np.median(metrics['wd_true_N'])

ax1[2, 1].boxplot(data_err_N, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
ax1[2, 1].set_title(
    f"WD Error Distribution (N={int(N_cdf)})\nMedian vs ECDF: {med_wd_ecdf_N:.4f} | vs True: {med_wd_true_N:.4f}",
    fontsize=10)
ax1[2, 1].set_ylabel("Wasserstein Distance")
ax1[2, 1].set_xticks([])
ax1[2, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# =============================================================================
# 6. VISUALIZZAZIONE - FIGURE 2: FOCUS PDF (2x2)
# =============================================================================

fig2, ax2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle(f"PDF Analysis: {nome_dist} (M={M})", fontsize=16)

# --- RIGA 1: PDF Spaghetti ---
# 1,1: PDF con N=M
plot_spaghetti(ax2[0, 0], sim_data['x_grids'], sim_data['pdf_M'], pdf_vera, 'True PDF',
               f"PDF Estimator (N=M={M})", color_true='b')

# 1,2: PDF con N=N_pdf (Qui usiamo l'N ottimale per le PDF)
plot_spaghetti(ax2[0, 1], sim_data['x_grids'], sim_data['pdf_N_pdf'], pdf_vera, 'True PDF',
               f"PDF Estimator (Heuristic N={int(N_pdf)})", color_true='b')

# --- RIGA 2: Boxplots Errori PDF (Log-Likelihood) ---

# 2,1: Errori per N=M
data_ll_M = metrics['ll_M']
med_ll_M = np.median(metrics['ll_M'])
med_kl_M = np.median(metrics['kl_M'])

ax2[1, 0].boxplot(data_ll_M, patch_artist=True, boxprops=dict(facecolor='salmon'))
ax2[1, 0].set_title(f"Log-Likelihood Dist. (N=M)\nMedian LL: {med_ll_M:.2f} | Median KL: {med_kl_M:.4f}", fontsize=10)
ax2[1, 0].set_ylabel("Log-Likelihood (Higher is better)")
ax2[1, 0].set_xticks([])
ax2[1, 0].grid(True, alpha=0.3)

# 2,2: Errori per N=N_pdf
data_ll_N = metrics['ll_N']
med_ll_N = np.median(metrics['ll_N'])
med_kl_N = np.median(metrics['kl_N'])

ax2[1, 1].boxplot(data_ll_N, patch_artist=True, boxprops=dict(facecolor='orange'))
ax2[1, 1].set_title(f"Log-Likelihood Dist. (N={int(N_pdf)})\nMedian LL: {med_ll_N:.2f} | Median KL: {med_kl_N:.4f}",
                    fontsize=10)
ax2[1, 1].set_ylabel("Log-Likelihood (Higher is better)")
ax2[1, 1].set_xticks([])
ax2[1, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# =============================================================================
# 7. VISUALIZZAZIONE - FIGURE 3: BIAS-VARIANCE (Invariata)
# =============================================================================

fig3, ax3 = plt.subplots(1, 3, figsize=(18, 6))
fig3.suptitle(f"Metric Sensitivity vs Degree N (Avg over {NUM_SIMULATIONS} runs)", fontsize=16)

# Plot 1: Wasserstein (CDF vs TRUE)
ax3[0].plot(range_N, avg_curve_wd, 'b-o', markersize=4, label='Avg WD (vs True CDF)')
ax3[0].axvline(best_n_wd, color='r', linestyle='--', label=f'Best N={best_n_wd}')
ax3[0].axvline(N_cdf, color='orange', linestyle=':', linewidth=2, label=f'Heuristic N={int(N_cdf)}')
ax3[0].set_title("CDF Error (Wasserstein)")
ax3[0].set_xlabel("Degree N")
ax3[0].set_ylabel("Distance (Lower is better)")
ax3[0].legend()
ax3[0].grid(True, alpha=0.3)

# Plot 2: KL Divergence (PDF vs TRUE)
ax3[1].plot(range_N, avg_curve_kl, 'g-o', markersize=4, label='Avg KL Divergence')
ax3[1].axvline(best_n_kl, color='r', linestyle='--', label=f'Best N={best_n_kl}')
ax3[1].axvline(N_pdf, color='orange', linestyle=':', linewidth=2, label=f'Heuristic N={int(N_pdf)}')
ax3[1].set_title("PDF Error (KL Divergence)")
ax3[1].set_xlabel("Degree N")
ax3[1].set_ylabel("Divergence (Lower is better)")
ax3[1].legend()
ax3[1].grid(True, alpha=0.3)

# Plot 3: Log-Likelihood (Samples vs Est PDF)
ax3[2].plot(range_N, avg_curve_ll, 'm-o', markersize=4, label='Avg Log-Likelihood')
ax3[2].axvline(best_n_ll, color='r', linestyle='--', label=f'Best N={best_n_ll}')
ax3[2].axvline(N_pdf, color='orange', linestyle=':', linewidth=2, label=f'Heuristic N={int(N_pdf)}')
ax3[2].set_title("Sample Fit (Log-Likelihood)")
ax3[2].set_xlabel("Degree N")
ax3[2].set_ylabel("LL (Higher is better)")
ax3[2].legend()
ax3[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
