import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
from scipy.integrate import trapezoid
# Assicurati che questi import funzionino nel tuo ambiente
from bernstein import create_ecdf, calculate_bernstein_cdf, calculate_bernstein_pdf
from KumaraswamyDist import KumaraswamyDist

# =============================================================================
# 1. CONFIGURAZIONE E PARAMETRI
# =============================================================================

scelta_dist = 'k'  # 'n', 'u', 'e', 'k'
M = 100  # Numero campioni
NUM_SIMULATIONS = 10  # Numero cicli (boxplot avranno 10 entry sulle X)
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

# Dati per grafici "Spaghetti" (Curve)
sim_curves = {
    'x_grids': [],
    'cdf_N': [], 'pdf_N': [],  # N euristico
    'cdf_M': [], 'pdf_M': []  # N = M
}

# Dati Scalari per le mediane nei titoli
scalar_metrics = {
    'wd_ecdf_N': [], 'wd_true_N': [], 'kl_N': [], 'nll_N': [],
    'wd_ecdf_M': [], 'wd_true_M': [], 'kl_M': [], 'nll_M': []
}

# Dati Vettoriali per i Boxplot (liste di liste/array)
# Ogni elemento della lista esterna rappresenta una simulazione
boxplot_data = {
    'cdf_err_ecdf_N': [],  # Differenze assolute |Stima - ECDF| sulla griglia
    'cdf_err_ecdf_M': [],
    'pdf_nll_samples_N': [],  # Negative Log Likelihood dei singoli campioni
    'pdf_nll_samples_M': []
}

# Limiti globali
global_min_x, global_max_x = float('inf'), float('-inf')

# =============================================================================
# 3. LOOP DI SIMULAZIONE
# =============================================================================

for i in range(NUM_SIMULATIONS):
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
    sim_curves['x_grids'].append(curr_asse_x)

    # Valori di riferimento sulla griglia
    ecdf_values = ecdf(curr_asse_x)
    pdf_true_loc = distribuzione.pdf(curr_asse_x)
    cdf_true_loc = distribuzione.cdf(curr_asse_x)


    # --- FUNZIONE HELPER PER LOG-LIKELIHOOD ---
    def calc_nll_samples(pdf_func_vals):
        # Evitiamo log(0) o negativi aggiungendo epsilon
        return -np.log(np.maximum(pdf_func_vals, 1e-12))


    # -------------------------------------------------------------------------
    # CASO 1: N OTTIMALE (Euristico)
    # -------------------------------------------------------------------------
    # Stime su Griglia
    cdf_stima_N = calculate_bernstein_cdf(ecdf, int(N_cdf), a, b, curr_asse_x)
    pdf_stima_N = calculate_bernstein_pdf(ecdf, int(N_pdf), a, b, curr_asse_x)
    # Stima PDF sui Campioni (per LogLikelihood)
    pdf_stima_N_samples = calculate_bernstein_pdf(ecdf, int(N_pdf), a, b, campioni_ordinati)

    sim_curves['cdf_N'].append(cdf_stima_N)
    sim_curves['pdf_N'].append(pdf_stima_N)

    # -- Metriche CDF --
    # 1. Vs ECDF
    abs_err_ecdf_N = np.abs(ecdf_values - cdf_stima_N)  # Vettore per Boxplot
    wd_ecdf_N = trapezoid(abs_err_ecdf_N, curr_asse_x)  # Scalare
    # 2. Vs TRUE CDF
    wd_true_N = trapezoid(np.abs(cdf_true_loc - cdf_stima_N), curr_asse_x)

    # -- Metriche PDF --
    # 1. KL Divergence (Vs True PDF)
    kl_N = entropy(pk=pdf_true_loc, qk=pdf_stima_N + 1e-12)
    # 2. Log Likelihood (Vs Samples)
    nll_samples_N = calc_nll_samples(pdf_stima_N_samples)  # Vettore per Boxplot
    mean_nll_N = np.mean(nll_samples_N)  # Scalare (Mean NLL)

    # Salvataggio Metriche N
    scalar_metrics['wd_ecdf_N'].append(wd_ecdf_N)
    scalar_metrics['wd_true_N'].append(wd_true_N)
    scalar_metrics['kl_N'].append(kl_N)
    scalar_metrics['nll_N'].append(mean_nll_N)

    boxplot_data['cdf_err_ecdf_N'].append(abs_err_ecdf_N)
    boxplot_data['pdf_nll_samples_N'].append(nll_samples_N)

    # -------------------------------------------------------------------------
    # CASO 2: N = M
    # -------------------------------------------------------------------------
    # Stime su Griglia
    cdf_stima_M = calculate_bernstein_cdf(ecdf, M, a, b, curr_asse_x)
    pdf_stima_M = calculate_bernstein_pdf(ecdf, M, a, b, curr_asse_x)
    # Stima PDF sui Campioni
    pdf_stima_M_samples = calculate_bernstein_pdf(ecdf, M, a, b, campioni_ordinati)

    sim_curves['cdf_M'].append(cdf_stima_M)
    sim_curves['pdf_M'].append(pdf_stima_M)

    # -- Metriche CDF --
    abs_err_ecdf_M = np.abs(ecdf_values - cdf_stima_M)
    wd_ecdf_M = trapezoid(abs_err_ecdf_M, curr_asse_x)
    wd_true_M = trapezoid(np.abs(cdf_true_loc - cdf_stima_M), curr_asse_x)

    # -- Metriche PDF --
    kl_M = entropy(pk=pdf_true_loc, qk=pdf_stima_M + 1e-12)
    nll_samples_M = calc_nll_samples(pdf_stima_M_samples)
    mean_nll_M = np.mean(nll_samples_M)

    # Salvataggio Metriche M
    scalar_metrics['wd_ecdf_M'].append(wd_ecdf_M)
    scalar_metrics['wd_true_M'].append(wd_true_M)
    scalar_metrics['kl_M'].append(kl_M)
    scalar_metrics['nll_M'].append(mean_nll_M)

    boxplot_data['cdf_err_ecdf_M'].append(abs_err_ecdf_M)
    boxplot_data['pdf_nll_samples_M'].append(nll_samples_M)

# =============================================================================
# 4. CALCOLO MEDIANE GLOBALI (Per i titoli)
# =============================================================================

med_wd_ecdf_N = np.median(scalar_metrics['wd_ecdf_N'])
med_wd_true_N = np.median(scalar_metrics['wd_true_N'])
med_kl_N = np.median(scalar_metrics['kl_N'])
med_nll_N = np.median(scalar_metrics['nll_N'])

med_wd_ecdf_M = np.median(scalar_metrics['wd_ecdf_M'])
med_wd_true_M = np.median(scalar_metrics['wd_true_M'])
med_kl_M = np.median(scalar_metrics['kl_M'])
med_nll_M = np.median(scalar_metrics['nll_M'])

# Generazione Verità Teorica Globale per plotting
if scelta_dist == 'k':
    asse_x_generale = np.linspace(max(0.0001, global_min_x), min(0.9999, global_max_x), num_points)
else:
    asse_x_generale = np.linspace(global_min_x, global_max_x, num_points)

cdf_vera = distribuzione.cdf(asse_x_generale)
pdf_vera = distribuzione.pdf(asse_x_generale)


# =============================================================================
# 5. VISUALIZZAZIONE
# =============================================================================

def plot_spaghetti(ax, x_list, y_list, y_true, label_true, title, color_est='k'):
    for x, y in zip(x_list, y_list):
        ax.plot(x, y, color=color_est, linewidth=0.8, alpha=0.35)
    ax.plot(asse_x_generale, y_true, 'r-', linewidth=2, label=label_true)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_simulation_boxplots(ax, data_list, title, y_label):
    # data_list è una lista di array. Ogni array corrisponde a una simulazione.
    # Creiamo un boxplot per ogni simulazione sull'asse X (1...10)
    ax.boxplot(data_list, showfliers=True,
               flierprops=dict(marker='o', markerfacecolor='r', markersize=2, alpha=0.5),
               medianprops=dict(color='blue'))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Simulation Index")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3, axis='y')


# --- FIGURE 1: CDF Analysis (3 rows x 2 cols) ---
fig1, ax1 = plt.subplots(3, 2, figsize=(14, 15))
fig1.suptitle(f"CDF Analysis: {nome_dist} (M={M})", fontsize=16)

# Row 1: CDF Curves
title_cdf_M = f"CDF (N=M={M})\nMedian WD[ECDF]: {med_wd_ecdf_M:.2e} | WD[True]: {med_wd_true_M:.2e}"
plot_spaghetti(ax1[0, 0], sim_curves['x_grids'], sim_curves['cdf_M'], cdf_vera, 'True CDF', title_cdf_M)

title_cdf_N = f"CDF (N={int(N_cdf)})\nMedian WD[ECDF]: {med_wd_ecdf_N:.2e} | WD[True]: {med_wd_true_N:.2e}"
plot_spaghetti(ax1[0, 1], sim_curves['x_grids'], sim_curves['cdf_N'], cdf_vera, 'True CDF', title_cdf_N)

# Row 2: PDF Curves associated to above
title_pdf_M = f"PDF (N=M={M}) - Associated"
plot_spaghetti(ax1[1, 0], sim_curves['x_grids'], sim_curves['pdf_M'], pdf_vera, 'True PDF', title_pdf_M)

title_pdf_N = f"PDF (N={int(N_pdf)}) - Associated"
plot_spaghetti(ax1[1, 1], sim_curves['x_grids'], sim_curves['pdf_N'], pdf_vera, 'True PDF', title_pdf_N)

# Row 3: Boxplots of Errors vs ECDF
# Nota: Ogni box rappresenta la distribuzione dell'errore assoluto (|F - Fn|) per quella specifica simulazione
plot_simulation_boxplots(ax1[2, 0], boxplot_data['cdf_err_ecdf_M'],
                         f"Error Distribution vs ECDF (N=M)\n(Medians: ECDF_WD={med_wd_ecdf_M:.2e}, True_WD={med_wd_true_M:.2e})",
                         "Abs Error |ECDF - Est|")

plot_simulation_boxplots(ax1[2, 1], boxplot_data['cdf_err_ecdf_N'],
                         f"Error Distribution vs ECDF (N={int(N_cdf)})\n(Medians: ECDF_WD={med_wd_ecdf_N:.2e}, True_WD={med_wd_true_N:.2e})",
                         "Abs Error |ECDF - Est|")

plt.tight_layout(rect=[0, 0.03, 1, 0.98])

# --- FIGURE 2: PDF Analysis (2 rows x 2 cols) ---
fig2, ax2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle(f"PDF Analysis: {nome_dist} (M={M})", fontsize=16)

# Row 1: PDF Curves (Ripetute per chiarezza focus)
title_pdf_M_f2 = f"PDF (N=M={M})\nMedian KL: {med_kl_M:.3f} | Median Mean-NLL: {med_nll_M:.3f}"
plot_spaghetti(ax2[0, 0], sim_curves['x_grids'], sim_curves['pdf_M'], pdf_vera, 'True PDF', title_pdf_M_f2)

title_pdf_N_f2 = f"PDF (N={int(N_pdf)})\nMedian KL: {med_kl_N:.3f} | Median Mean-NLL: {med_nll_N:.3f}"
plot_spaghetti(ax2[0, 1], sim_curves['x_grids'], sim_curves['pdf_N'], pdf_vera, 'True PDF', title_pdf_N_f2)

# Row 2: Boxplots of Negative Log Likelihood (NLL) on Samples
# Nota: Ogni box rappresenta la distribuzione della NLL sui 100 campioni di quella simulazione
plot_simulation_boxplots(ax2[1, 0], boxplot_data['pdf_nll_samples_M'],
                         f"NLL Distribution on Samples (N=M)\n(Medians: KL={med_kl_M:.3f}, NLL={med_nll_M:.3f})",
                         "Negative Log Likelihood")

plot_simulation_boxplots(ax2[1, 1], boxplot_data['pdf_nll_samples_N'],
                         f"NLL Distribution on Samples (N={int(N_pdf)})\n(Medians: KL={med_kl_N:.3f}, NLL={med_nll_N:.3f})",
                         "Negative Log Likelihood")

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()
