import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
from scipy.integrate import trapezoid

# Assumo che questi moduli siano presenti nel tuo ambiente come da script originale
from bernstein import create_ecdf, calculate_bernstein_cdf, calculate_bernstein_pdf
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
# 2. STRUTTURE DATI
# =============================================================================

# Liste per salvare i risultati delle simulazioni (grafici "spaghetti")
sim_data = {
    'x_grids': [],
    'cdf_N': [], 'pdf_N': [],  # N euristico
    'cdf_M': [], 'pdf_M': []  # N = M
}

# Liste per le metriche di accuratezza
# MODIFICA: Aggiunte chiavi per WD vs True e LogLikelihood (LL)
metrics = {
    'wd_ecdf_N': [], 'wd_true_N': [], 'kl_N': [], 'll_N': [],
    'wd_ecdf_M': [], 'wd_true_M': [], 'kl_M': [], 'll_M': []
}

# Limiti globali
global_min_x, global_max_x = float('inf'), float('-inf')

# --- STRUTTURE PER ANALISI ROBUSTA BIAS-VARIANCE ---
max_n_test = int(N_cdf * 1.5)
range_N = np.arange(5, max_n_test, 5)

# Matrici per salvare gli errori/metriche: Righe=Simulazioni, Colonne=Valori di N
errors_wd_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))  # WD vs True CDF
errors_kl_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))  # KL vs True PDF
errors_ll_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))  # MODIFICA: Log-Likelihood

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

    # Valori Veri/Empirici sulla griglia
    ecdf_values = ecdf(curr_asse_x)
    pdf_true_loc = distribuzione.pdf(curr_asse_x)
    cdf_true_loc = distribuzione.cdf(curr_asse_x)

    # -------------------------------------------------------------------------
    # PARTE 1: Calcolo casi specifici (N Euristico vs N=M)
    # -------------------------------------------------------------------------

    # --- Caso 1: N Ottimale (Euristico) ---
    cdf_stima_N = calculate_bernstein_cdf(ecdf, int(N_cdf), a, b, curr_asse_x)
    pdf_stima_N = calculate_bernstein_pdf(ecdf, int(N_pdf), a, b, curr_asse_x)

    # MODIFICA: Calcolo PDF sui punti campione per LogLikelihood
    pdf_vals_campioni_N = calculate_bernstein_pdf(ecdf, int(N_pdf), a, b, campioni)

    sim_data['cdf_N'].append(cdf_stima_N)
    sim_data['pdf_N'].append(pdf_stima_N)

    # Calcolo Metriche N Ottimale
    # 1. WD vs ECDF (esistente)
    metrics['wd_ecdf_N'].append(trapezoid(np.abs(ecdf_values - cdf_stima_N), curr_asse_x))

    # 2. WD vs TRUE CDF (MODIFICA RICHIESTA)
    metrics['wd_true_N'].append(trapezoid(np.abs(cdf_true_loc - cdf_stima_N), curr_asse_x))

    # 3. KL vs True PDF (esistente)
    metrics['kl_N'].append(entropy(pk=pdf_true_loc, qk=pdf_stima_N + 1e-12))

    # 4. Log-Likelihood (MODIFICA RICHIESTA)
    # Somma dei log delle probabilità sui campioni
    metrics['ll_N'].append(np.sum(np.log(pdf_vals_campioni_N + 1e-12)))

    # --- Caso 2: N = M ---
    cdf_stima_M = calculate_bernstein_cdf(ecdf, M, a, b, curr_asse_x)
    pdf_stima_M = calculate_bernstein_pdf(ecdf, M, a, b, curr_asse_x)

    # MODIFICA: Calcolo PDF sui punti campione per LogLikelihood
    pdf_vals_campioni_M = calculate_bernstein_pdf(ecdf, M, a, b, campioni)

    sim_data['cdf_M'].append(cdf_stima_M)
    sim_data['pdf_M'].append(pdf_stima_M)

    # Calcolo Metriche N = M
    metrics['wd_ecdf_M'].append(trapezoid(np.abs(ecdf_values - cdf_stima_M), curr_asse_x))
    metrics['wd_true_M'].append(trapezoid(np.abs(cdf_true_loc - cdf_stima_M), curr_asse_x))
    metrics['kl_M'].append(entropy(pk=pdf_true_loc, qk=pdf_stima_M + 1e-12))
    metrics['ll_M'].append(np.sum(np.log(pdf_vals_campioni_M + 1e-12)))

    # -------------------------------------------------------------------------
    # PARTE 2: Analisi di Sensibilità ROBUSTA (Bias-Variance su tutti gli N)
    # -------------------------------------------------------------------------
    for idx_n, n_val in enumerate(range_N):
        # Stima su griglia (per WD e KL)
        cdf_temp = calculate_bernstein_cdf(ecdf, int(n_val), a, b, curr_asse_x)
        pdf_temp = calculate_bernstein_pdf(ecdf, int(n_val), a, b, curr_asse_x)

        # Stima sui campioni (per LogLikelihood)
        pdf_temp_samples = calculate_bernstein_pdf(ecdf, int(n_val), a, b, campioni)

        # 1. Wasserstein (CDF Teorica vs Bernstein)
        wd_err_bv = trapezoid(np.abs(cdf_true_loc - cdf_temp), curr_asse_x)
        errors_wd_matrix[i, idx_n] = wd_err_bv

        # 2. KL Divergence (PDF Teorica vs Bernstein)
        kl_err_bv = entropy(pk=pdf_true_loc, qk=pdf_temp + 1e-12)
        errors_kl_matrix[i, idx_n] = kl_err_bv

        # 3. Log-Likelihood (MODIFICA RICHIESTA)
        ll_val_bv = np.sum(np.log(pdf_temp_samples + 1e-12))
        errors_ll_matrix[i, idx_n] = ll_val_bv

# =============================================================================
# 4. POST-PROCESSING E MEDIE
# =============================================================================

# Medie metriche puntuali
avg_wd_ecdf_N = np.mean(metrics['wd_ecdf_N'])
avg_wd_true_N = np.mean(metrics['wd_true_N'])
avg_kl_N = np.mean(metrics['kl_N'])
avg_ll_N = np.mean(metrics['ll_N'])

avg_wd_ecdf_M = np.mean(metrics['wd_ecdf_M'])
avg_wd_true_M = np.mean(metrics['wd_true_M'])
avg_kl_M = np.mean(metrics['kl_M'])
avg_ll_M = np.mean(metrics['ll_M'])

# Medie curve Bias-Variance
avg_curve_wd = np.mean(errors_wd_matrix, axis=0)
avg_curve_kl = np.mean(errors_kl_matrix, axis=0)
avg_curve_ll = np.mean(errors_ll_matrix, axis=0)

# Trova i migliori N
best_n_wd = range_N[np.argmin(avg_curve_wd)]
best_n_kl = range_N[np.argmin(avg_curve_kl)]
best_n_ll = range_N[np.argmax(avg_curve_ll)]  # Per LL cerchiamo il MASSIMO, non il minimo

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

# FIGURE 1: Simulation Spaghetti Plots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"Bernstein Estimation - {nome_dist} (M={M})", fontsize=16)


def plot_simulations(ax, title, x_list, y_list, y_true, label_true):
    for x, y in zip(x_list, y_list):
        ax.plot(x, y, 'k-', linewidth=0.8, alpha=0.4)
    ax.plot(asse_x_generale, y_true, 'r-', linewidth=2.5, label=label_true)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()


# 1. CDF (N=M)
t1 = f"CDF (N=M) | WD(True): {avg_wd_true_M:.4f} | WD(ECDF): {avg_wd_ecdf_M:.4f}"
plot_simulations(axes[0, 0], t1, sim_data['x_grids'], sim_data['cdf_M'], cdf_vera, 'True CDF')

# 2. CDF (Optimal N)
t2 = f"CDF (N={int(N_cdf)}) | WD(True): {avg_wd_true_N:.4f} | WD(ECDF): {avg_wd_ecdf_N:.4f}"
plot_simulations(axes[0, 1], t2, sim_data['x_grids'], sim_data['cdf_N'], cdf_vera, 'True CDF')

# 3. PDF (N=M)
t3 = f"PDF (N=M) | KL: {avg_kl_M:.4f} | LL: {avg_ll_M:.2f}"
plot_simulations(axes[1, 0], t3, sim_data['x_grids'], sim_data['pdf_M'], pdf_vera, 'True PDF')

# 4. PDF (Optimal N)
t4 = f"PDF (N={int(N_pdf)}) | KL: {avg_kl_N:.4f} | LL: {avg_ll_N:.2f}"
plot_simulations(axes[1, 1], t4, sim_data['x_grids'], sim_data['pdf_N'], pdf_vera, 'True PDF')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# FIGURE 2: Bias-Variance Tradeoff (Analisi robusta)
# Modificato a 3 subplots per includere LogLikelihood
fig2, ax2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle(f"Metric Sensitivity vs Degree N (Avg over {NUM_SIMULATIONS} runs)", fontsize=16)

# Plot 1: Wasserstein (CDF vs TRUE)
ax2[0].plot(range_N, avg_curve_wd, 'b-o', markersize=4, label='Avg WD (vs True CDF)')
ax2[0].axvline(best_n_wd, color='r', linestyle='--', label=f'Best N={best_n_wd}')
ax2[0].axvline(N_cdf, color='orange', linestyle=':', linewidth=2, label=f'Heuristic N={int(N_cdf)}')
ax2[0].set_title("CDF Error (Wasserstein)")
ax2[0].set_xlabel("Degree N")
ax2[0].set_ylabel("Distance (Lower is better)")
ax2[0].legend()
ax2[0].grid(True, alpha=0.3)

# Plot 2: KL Divergence (PDF vs TRUE)
ax2[1].plot(range_N, avg_curve_kl, 'g-o', markersize=4, label='Avg KL Divergence')
ax2[1].axvline(best_n_kl, color='r', linestyle='--', label=f'Best N={best_n_kl}')
ax2[1].axvline(N_pdf, color='orange', linestyle=':', linewidth=2, label=f'Heuristic N={int(N_pdf)}')
ax2[1].set_title("PDF Error (KL Divergence)")
ax2[1].set_xlabel("Degree N")
ax2[1].set_ylabel("Divergence (Lower is better)")
ax2[1].legend()
ax2[1].grid(True, alpha=0.3)

# Plot 3: Log-Likelihood (Samples vs Est PDF)
ax2[2].plot(range_N, avg_curve_ll, 'm-o', markersize=4, label='Avg Log-Likelihood')
ax2[2].axvline(best_n_ll, color='r', linestyle='--', label=f'Best N={best_n_ll}')
ax2[2].axvline(N_pdf, color='orange', linestyle=':', linewidth=2, label=f'Heuristic N={int(N_pdf)}')
ax2[2].set_title("Sample Fit (Log-Likelihood)")
ax2[2].set_xlabel("Degree N")
ax2[2].set_ylabel("LL (Higher is better)")
ax2[2].legend()
ax2[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
