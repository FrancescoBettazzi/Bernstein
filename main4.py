import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
from scipy.integrate import trapezoid
from bernstein2 import create_ecdf, calculate_bernstein_cdf, calculate_bernstein_pdf
from KumaraswamyDist import KumaraswamyDist

# =============================================================================
# 1. CONFIGURAZIONE E PARAMETRI
# =============================================================================

scelta_dist = 'k'  # 'n', 'u', 'e', 'k'
M = 100  # Numero campioni
NUM_SIMULATIONS = 10  # Numero cicli
num_points = 500  # Risoluzione grafici

# --- CALCOLO N EURISTICI ---
N_pdf = math.ceil(M / math.log(M, 2))  # N euristico per PDF
N_cdf = math.ceil(M / math.log(M, 2)) ** 2  # N euristico per CDF

# Inizializzazione Distribuzione
distribuzione = None
nome_dist = ""

if scelta_dist == 'n':
    mu = 0
    sigma = 1
    distribuzione = stats.norm(loc=mu, scale=sigma)
    nome_dist = f"Normal(mu={mu}, sigma={sigma})"
elif scelta_dist == 'u':
    # In scipy uniform loc=start, scale=width (quindi b - a)
    uni_a = 5
    uni_b = 15
    distribuzione = stats.uniform(loc=uni_a, scale=(uni_b - uni_a))
    nome_dist = f"Uniform[{uni_a}, {uni_b}]"
elif scelta_dist == 'e':
    lambda_param = 0.5
    distribuzione = stats.expon(loc=0, scale=1 / lambda_param)
    nome_dist = f"Exponential(lambda={lambda_param})"
elif scelta_dist == 'k':
    k_a = 2
    k_b = 5
    distribuzione = KumaraswamyDist(a=k_a, b=k_b)
    nome_dist = f"Kumaraswamy(a={k_a}, b={k_b})"

# =============================================================================
# 2. STRUTTURE DATI (Stile originale)
# =============================================================================

# Liste per salvare i risultati delle simulazioni (per i grafici "spaghetti")
sim_data = {
    'x_grids': [],
    'cdf_N': [], 'pdf_N': [],  # N euristico
    'cdf_M': [], 'pdf_M': []  # N = M
}

# Liste per le metriche di accuratezza (per i titoli dei grafici)
metrics = {
    'wd_N': [], 'wd_M': [],  # Wasserstein Distance (CDF)
    'kl_N': [], 'kl_M': []  # KL Divergence (PDF)
}

# Limiti globali per i plot finali
global_min_x, global_max_x = float('inf'), float('-inf')

# --- STRUTTURE PER ANALISI ROBUSTA BIAS-VARIANCE ---
# Range di N da testare (da 5 a un po' oltre il necessario per la CDF)
max_n_test = int(N_cdf * 1.5)
range_N = np.arange(5, max_n_test, 5)

# Matrici per salvare gli errori: Righe=Simulazioni, Colonne=Valori di N
errors_wd_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))
errors_kl_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))

# =============================================================================
# 3. LOOP DI SIMULAZIONE
# =============================================================================

for i in range(NUM_SIMULATIONS):
    print(f"Starting cycle n. {i + 1}/{NUM_SIMULATIONS}")

    # A. Generazione Campioni
    campioni = distribuzione.rvs(size=M)
    campioni_ordinati = np.sort(campioni)
    ecdf = create_ecdf(campioni)

    # B. Supporto Locale
    a, b = campioni_ordinati[0], campioni_ordinati[-1]

    if a < global_min_x: global_min_x = a
    if b > global_max_x: global_max_x = b

    # C. Griglia Locale e Ground Truth
    curr_asse_x = np.linspace(a, b, num_points)
    sim_data['x_grids'].append(curr_asse_x)

    ecdf_values = ecdf(curr_asse_x)  # Per Wasserstein contro ECDF (o Teorica)
    pdf_true_loc = distribuzione.pdf(curr_asse_x)  # Per KL contro Teorica
    cdf_true_loc = distribuzione.cdf(curr_asse_x)  # Per Bias-Variance contro Teorica

    # -------------------------------------------------------------------------
    # PARTE 1: Calcolo casi specifici (N Euristico vs N=M) per i Grafici 1
    # -------------------------------------------------------------------------

    # Caso 1: N Ottimale (Euristico)
    cdf_stima = calculate_bernstein_cdf(ecdf, int(N_cdf), a, b, curr_asse_x)
    pdf_stima = calculate_bernstein_pdf(ecdf, int(N_pdf), a, b, curr_asse_x)

    sim_data['cdf_N'].append(cdf_stima)
    sim_data['pdf_N'].append(pdf_stima)

    # Calcolo Metriche N Ottimale
    wd_n = trapezoid(np.abs(ecdf_values - cdf_stima), curr_asse_x)  # Vs ECDF
    metrics['wd_N'].append(wd_n)

    kl_n = entropy(pk=pdf_true_loc, qk=pdf_stima + 1e-12)  # Vs True PDF
    metrics['kl_N'].append(kl_n)

    # Caso 2: N = M
    cdf_stima_BM = calculate_bernstein_cdf(ecdf, M, a, b, curr_asse_x)
    pdf_stima_BM = calculate_bernstein_pdf(ecdf, M, a, b, curr_asse_x)

    sim_data['cdf_M'].append(cdf_stima_BM)
    sim_data['pdf_M'].append(pdf_stima_BM)

    # Calcolo Metriche N = M
    wd_m = trapezoid(np.abs(ecdf_values - cdf_stima_BM), curr_asse_x)
    metrics['wd_M'].append(wd_m)

    kl_m = entropy(pk=pdf_true_loc, qk=pdf_stima_BM + 1e-12)
    metrics['kl_M'].append(kl_m)

    # -------------------------------------------------------------------------
    # PARTE 2: Analisi di Sensibilità ROBUSTA (Bias-Variance su tutti gli N)
    # -------------------------------------------------------------------------
    for idx_n, n_val in enumerate(range_N):
        # Calcolo stimatori temporanei
        cdf_temp = calculate_bernstein_cdf(ecdf, int(n_val), a, b, curr_asse_x)
        pdf_temp = calculate_bernstein_pdf(ecdf, int(n_val), a, b, curr_asse_x)

        # IMPORTANTE: Per l'analisi Bias-Variance confrontiamo con la VERITÀ (Teorica)
        # 1. Wasserstein (CDF Teorica vs Bernstein)
        wd_err_bv = trapezoid(np.abs(cdf_true_loc - cdf_temp), curr_asse_x)
        errors_wd_matrix[i, idx_n] = wd_err_bv

        # 2. KL Divergence (PDF Teorica vs Bernstein)
        kl_err_bv = entropy(pk=pdf_true_loc, qk=pdf_temp + 1e-12)
        errors_kl_matrix[i, idx_n] = kl_err_bv

# =============================================================================
# 4. POST-PROCESSING E MEDIE
# =============================================================================

# Medie per i titoli dei grafici Spaghetti (Metriche puntuali)
avg_wd_N = np.mean(metrics['wd_N'])
avg_wd_M = np.mean(metrics['wd_M'])
avg_kl_N = np.mean(metrics['kl_N'])
avg_kl_M = np.mean(metrics['kl_M'])

# Medie per i grafici Bias-Variance (Medie sulle simulazioni)
avg_curve_wd = np.mean(errors_wd_matrix, axis=0)
avg_curve_kl = np.mean(errors_kl_matrix, axis=0)

# Trova i minimi nelle curve medie
best_n_wd = range_N[np.argmin(avg_curve_wd)]
best_n_kl = range_N[np.argmin(avg_curve_kl)]

# Asse X globale per il plot della verità teorica
if scelta_dist == 'k':
    asse_x_generale = np.linspace(max(0.0001, global_min_x), min(0.9999, global_max_x), num_points)
else:
    asse_x_generale = np.linspace(global_min_x, global_max_x, num_points)

cdf_vera = distribuzione.cdf(asse_x_generale)
pdf_vera = distribuzione.pdf(asse_x_generale)

# =============================================================================
# 5. VISUALIZZAZIONE
# =============================================================================

# FIGURE 1: Simulation Spaghetti Plots con Metriche nei Titoli
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"Bernstein Estimation - {nome_dist} (M={M})", fontsize=16)


def plot_simulations(ax, title, x_list, y_list, y_true, label_true):
    for x, y in zip(x_list, y_list):
        ax.plot(x, y, 'k-', linewidth=0.8, alpha=0.4)
    ax.plot(asse_x_generale, y_true, 'k-', linewidth=2.5, label=label_true)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


# 1. CDF (N=M)
plot_simulations(axes[0, 0], f"CDF (N=M={M}) - Avg WD: {avg_wd_M:.5f}",
                 sim_data['x_grids'], sim_data['cdf_M'], cdf_vera, 'Theoretical CDF')

# 2. CDF (Optimal N)
plot_simulations(axes[0, 1], f"CDF (N={int(N_cdf)}) - Avg WD: {avg_wd_N:.5f}",
                 sim_data['x_grids'], sim_data['cdf_N'], cdf_vera, 'Theoretical CDF')

# 3. PDF (N=M)
plot_simulations(axes[1, 0], f"PDF (N=M={M}) - Avg KL: {avg_kl_M:.5f}",
                 sim_data['x_grids'], sim_data['pdf_M'], pdf_vera, 'Theoretical PDF')

# 4. PDF (Optimal N)
plot_simulations(axes[1, 1], f"PDF (N={int(N_pdf)}) - Avg KL: {avg_kl_N:.5f}",
                 sim_data['x_grids'], sim_data['pdf_N'], pdf_vera, 'Theoretical PDF')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# FIGURE 2: Bias-Variance Tradeoff (Analisi robusta)
fig2, ax2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle(f"Error Analysis vs Degree N (Avg over {NUM_SIMULATIONS} runs)", fontsize=16)

# Plot 1: Wasserstein (CDF)
ax2[0].plot(range_N, avg_curve_wd, 'b-o', markersize=4, label='Avg Wasserstein Dist.')
ax2[0].axvline(best_n_wd, color='r', linestyle='--', label=f'Optimal N (sim)={best_n_wd}')
ax2[0].axvline(N_cdf, color='orange', linestyle=':', linewidth=2, label=f'Heuristic N={int(N_cdf)}')
ax2[0].set_title("CDF Accuracy (Bias-Variance)")
ax2[0].set_xlabel("Degree N")
ax2[0].set_ylabel("Error (Wasserstein)")
ax2[0].legend()
ax2[0].grid(True, alpha=0.3)

# Plot 2: KL Divergence (PDF)
ax2[1].plot(range_N, avg_curve_kl, 'g-o', markersize=4, label='Avg KL Divergence')
ax2[1].axvline(best_n_kl, color='r', linestyle='--', label=f'Optimal N (sim)={best_n_kl}')
ax2[1].axvline(N_pdf, color='orange', linestyle=':', linewidth=2, label=f'Heuristic N={int(N_pdf)}')
ax2[1].set_title("PDF Accuracy (Bias-Variance)")
ax2[1].set_xlabel("Degree N")
ax2[1].set_ylabel("Error (KL Div)")
ax2[1].legend()
ax2[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
