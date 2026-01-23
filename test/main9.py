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
NUM_SIMULATIONS = 10
num_points = 500  # Risoluzione grafici

# --- CALCOLO N EURISTICI ---
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
    'cdf_N_cdf': [], 'pdf_N_pdf': [],
    'cdf_M': [], 'pdf_M': [],
    'pdf_conn_to_cdf': []
}

# NUOVO: Strutture per salvare gli errori puntuali (vettori) per ogni simulazione
# Ogni elemento della lista sarà un array di lunghezza num_points (es. 500)
pointwise_errors = {
    'cdf_M': [],  # |CDF_stima - CDF_vera| punto per punto
    'cdf_N': [],
    'pdf_M': [],  # |PDF_stima - PDF_vera| punto per punto
    'pdf_N': []
}

# Metriche scalari per Bias-Variance (restano invariate per Fig 3)
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
    ecdf = create_ecdf(campioni)

    # B. Supporto Locale
    a, b = campioni_ordinati[0], campioni_ordinati[-1]
    if a < global_min_x: global_min_x = a
    if b > global_max_x: global_max_x = b

    # C. Griglia Locale
    curr_asse_x = np.linspace(a, b, num_points)
    sim_data['x_grids'].append(curr_asse_x)

    # Valori Veri
    cdf_true_loc = distribuzione.cdf(curr_asse_x)
    pdf_true_loc = distribuzione.pdf(curr_asse_x)

    # -----------------------------------------------------------
    # PARTE 1: Stime e Errori Puntuali
    # -----------------------------------------------------------

    # --- Caso N = M (Overfitting) ---
    cdf_stima_M = calculate_bernstein_cdf(ecdf, M, a, b, curr_asse_x)
    pdf_stima_M = calculate_bernstein_pdf(ecdf, M, a, b, curr_asse_x)

    sim_data['cdf_M'].append(cdf_stima_M)
    sim_data['pdf_M'].append(pdf_stima_M)

    # Salva errore puntuale (valore assoluto della differenza)
    pointwise_errors['cdf_M'].append(np.abs(cdf_true_loc - cdf_stima_M))
    pointwise_errors['pdf_M'].append(np.abs(pdf_true_loc - pdf_stima_M))

    # --- Caso N Euristici ---
    # CDF (N_cdf)
    cdf_stima_N = calculate_bernstein_cdf(ecdf, int(N_cdf), a, b, curr_asse_x)
    pdf_stima_conn_to_cdf = calculate_bernstein_pdf(ecdf, int(N_cdf), a, b, curr_asse_x)  # Solo visuale

    sim_data['cdf_N_cdf'].append(cdf_stima_N)
    sim_data['pdf_conn_to_cdf'].append(pdf_stima_conn_to_cdf)

    pointwise_errors['cdf_N'].append(np.abs(cdf_true_loc - cdf_stima_N))

    # PDF (N_pdf)
    pdf_stima_N = calculate_bernstein_pdf(ecdf, int(N_pdf), a, b, curr_asse_x)
    sim_data['pdf_N_pdf'].append(pdf_stima_N)

    pointwise_errors['pdf_N'].append(np.abs(pdf_true_loc - pdf_stima_N))

    # -----------------------------------------------------------
    # PARTE 2: Analisi Bias-Variance (come prima)
    # -----------------------------------------------------------
    pdf_vals_campioni = calculate_bernstein_pdf(ecdf, int(N_pdf), a, b, campioni)  # Per LogLikelihood

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


def plot_spaghetti(ax, x_list, y_list, y_true, label_true, title, color_true='r'):
    for x, y in zip(x_list, y_list):
        ax.plot(x, y, 'k-', linewidth=0.5, alpha=0.3)
    ax.plot(asse_x_generale, y_true, color=color_true, linewidth=2, linestyle='-', label=label_true)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')


# =============================================================================
# 5. VISUALIZZAZIONE - FIGURE 1: FOCUS CDF
# =============================================================================

fig1, ax1 = plt.subplots(3, 2, figsize=(14, 15))
fig1.suptitle(f"CDF Analysis: {nome_dist} (M={M})", fontsize=16)

# Riga 1 e 2: Spaghetti Plots (uguale a prima)
plot_spaghetti(ax1[0, 0], sim_data['x_grids'], sim_data['cdf_M'], cdf_vera, 'True CDF', f"CDF (N=M={M})")
plot_spaghetti(ax1[0, 1], sim_data['x_grids'], sim_data['cdf_N_cdf'], cdf_vera, 'True CDF', f"CDF (N={int(N_cdf)})")
plot_spaghetti(ax1[1, 0], sim_data['x_grids'], sim_data['pdf_M'], pdf_vera, 'True PDF', f"Derivative PDF (N=M={M})",
               color_true='g')
plot_spaghetti(ax1[1, 1], sim_data['x_grids'], sim_data['pdf_conn_to_cdf'], pdf_vera, 'True PDF',
               f"Derivative PDF (N={int(N_cdf)})", color_true='g')

# --- RIGA 3: BOXPLOTS PER SIMULAZIONE (CDF) ---
# Mostriamo la distribuzione dell'errore assoluto (|F(x) - F_n(x)|) per ogni simulazione

# 3,1: Errore Puntuale N=M
ax1[2, 0].boxplot(pointwise_errors['cdf_M'], patch_artist=True,
                  boxprops=dict(facecolor='lightblue'),
                  medianprops=dict(color='blue'))
ax1[2, 0].set_title(f"Pointwise Absolute Error Dist. per Simulation (N=M)", fontsize=10)
ax1[2, 0].set_ylabel("|True CDF - Est CDF|")
ax1[2, 0].set_xlabel("Simulation ID")
ax1[2, 0].grid(True, alpha=0.3)

# 3,2: Errore Puntuale N=N_cdf
ax1[2, 1].boxplot(pointwise_errors['cdf_N'], patch_artist=True,
                  boxprops=dict(facecolor='lightgreen'),
                  medianprops=dict(color='green'))
ax1[2, 1].set_title(f"Pointwise Absolute Error Dist. per Simulation (N={int(N_cdf)})", fontsize=10)
ax1[2, 1].set_ylabel("|True CDF - Est CDF|")
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

# --- RIGA 2: BOXPLOTS PER SIMULAZIONE (PDF) ---
# Mostriamo la distribuzione dell'errore assoluto (|f(x) - f_n(x)|) per ogni simulazione
# Nota: La Log-Likelihood è scalare per simulazione, quindi usiamo l'errore puntuale L1 qui per avere boxplot

# 2,1: Errore Puntuale N=M
ax2[1, 0].boxplot(pointwise_errors['pdf_M'], patch_artist=True,
                  boxprops=dict(facecolor='salmon'),
                  medianprops=dict(color='red'))
ax2[1, 0].set_title(f"Pointwise Absolute Error Dist. per Simulation (N=M)", fontsize=10)
ax2[1, 0].set_ylabel("|True PDF - Est PDF|")
ax2[1, 0].set_xlabel("Simulation ID")
ax2[1, 0].grid(True, alpha=0.3)

# 2,2: Errore Puntuale N=N_pdf
ax2[1, 1].boxplot(pointwise_errors['pdf_N'], patch_artist=True,
                  boxprops=dict(facecolor='orange'),
                  medianprops=dict(color='darkorange'))
ax2[1, 1].set_title(f"Pointwise Absolute Error Dist. per Simulation (N={int(N_pdf)})", fontsize=10)
ax2[1, 1].set_ylabel("|True PDF - Est PDF|")
ax2[1, 1].set_xlabel("Simulation ID")
ax2[1, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# =============================================================================
# 7. VISUALIZZAZIONE - FIGURE 3: BIAS-VARIANCE (Invariata)
# =============================================================================

fig3, ax3 = plt.subplots(1, 3, figsize=(18, 6))
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

ax3[2].plot(range_N, avg_curve_ll, 'm-o', markersize=4, label='Avg Log-Likelihood')
ax3[2].axvline(best_n_ll, color='r', linestyle='--', label=f'Best N={best_n_ll}')
ax3[2].axvline(N_pdf, color='orange', linestyle=':', linewidth=2, label=f'Heur N={int(N_pdf)}')
ax3[2].set_title("Sample Fit (Log-Likelihood)")
ax3[2].set_xlabel("Degree N")
ax3[2].grid(True, alpha=0.3)
ax3[2].legend()

plt.tight_layout()
plt.show()
