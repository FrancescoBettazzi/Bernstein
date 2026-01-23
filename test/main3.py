import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
from scipy.integrate import trapezoid
from bernstein2 import create_ecdf, calculate_bernstein_cdf, calculate_bernstein_pdf
from KumaraswamyDist import KumaraswamyDist

# SCELTA DISTRIBUZIONE
# 'n' = Normal, 'u' = Uniform, 'e' = Exponential, 'k' = Kumaraswamy
scelta_dist = 'k'

M = 100  # n. campioni
N_pdf = math.ceil(M / math.log(M, 2))  # grado BP per PDF
N_cdf = math.ceil(M / math.log(M, 2)) ** 2  # grado BP per CDF

distribuzione = None
nome_dist = ""

if scelta_dist == 'n':
    media = 0
    std_dev = 1
    distribuzione = stats.norm(loc=media, scale=std_dev)
    nome_dist = "Normal(0, 1)"
elif scelta_dist == 'u':
    distribuzione = stats.uniform(loc=5, scale=10)
    nome_dist = "Uniform[5, 15]"
elif scelta_dist == 'e':
    distribuzione = stats.expon(loc=0, scale=1/0.5)
    nome_dist = "Exponential(lambda=0.5)"
elif scelta_dist == 'k':
    # Parametri di forma a e b
    a_param = 2
    b_param = 5
    distribuzione = KumaraswamyDist(a=a_param, b=b_param)
    nome_dist = f"Kumaraswamy(a={a_param}, b={b_param})"

# array_campioni = []
# array_cdf_stima = []
# array_pdf_stima = []
# array_cdf_stima_BM = []
# array_pdf_stima_BM = []

# Liste per salvare i risultati delle simulazioni
sim_data = {
    'x_grids': [],
    'cdf_N': [], 'pdf_N': [],
    'cdf_M': [], 'pdf_M': []
}

# Liste per le metriche di accuratezza
metrics = {
    'wd_N': [], 'wd_M': [],   # Wasserstein Distance (CDF)
    'kl_N': [], 'kl_M': []    # KL Divergence (PDF)
}

array_asse_x = []

a_min = 0
b_max = 1
num_points = 500

ecdf = None
curr_asse_x = None
a = None
b = None

# Loop di simulazione
for i in range(10):
    print("starting cycle n.", (i+1))
    campioni = distribuzione.rvs(size=M)
    campioni_ordinati = np.sort(campioni)
    # array_campioni.append(campioni_ordinati)

    ecdf = create_ecdf(campioni)

    # Supporto locale [a, b]
    a, b = campioni_ordinati[0], campioni_ordinati[-1]
    # a = campioni_ordinati.min()
    # b = campioni_ordinati.max()

    if a < a_min: a_min = a
    if b > b_max: b_max = b

    # Generazione asse x locale
    curr_asse_x = np.linspace(a, b, num_points)
    # array_asse_x.append(curr_asse_x)
    sim_data['x_grids'].append(curr_asse_x)

    # --- CALCOLO RIFERIMENTI LOCALI PER LE METRICHE ---
    # 1. Valori ECDF su griglia locale (per Wasserstein)
    # L'oggetto ecdf è callable e restituisce i valori a gradino
    ecdf_values = ecdf(curr_asse_x)

    # 2. PDF Vera su griglia locale (per KL Divergence)
    pdf_true_loc = distribuzione.pdf(curr_asse_x)
    # --------------------------------------------------

    # Calcolo vettorizzato (senza loop interni)
    # Passiamo l'intero array curr_asse_x
    cdf_stima = calculate_bernstein_cdf(ecdf, int(N_cdf), a, b, curr_asse_x)
    pdf_stima = calculate_bernstein_pdf(ecdf, int(N_pdf), a, b, curr_asse_x)

    # array_cdf_stima.append(cdf_stima)
    # array_pdf_stima.append(pdf_stima)
    sim_data['cdf_N'].append(cdf_stima)
    sim_data['pdf_N'].append(pdf_stima)

    # Calcolo Metriche N Ottimale
    # Wasserstein (Integrale della differenza assoluta tra le CDF)
    wd_n = trapezoid(np.abs(ecdf_values - cdf_stima), curr_asse_x)
    metrics['wd_N'].append(wd_n)
    # KL Divergence (Vera PDF || Stima Bernstein)
    # Aggiungiamo 1e-12 alla stima per evitare log(0)
    kl_n = entropy(pk=pdf_true_loc, qk=pdf_stima + 1e-12)
    metrics['kl_N'].append(kl_n)

    # Calcolo vettorizzato caso M
    cdf_stima_BM = calculate_bernstein_cdf(ecdf, M, a, b, curr_asse_x)
    pdf_stima_BM = calculate_bernstein_pdf(ecdf, M, a, b, curr_asse_x)

    # array_cdf_stima_BM.append(cdf_stima_BM)
    # array_pdf_stima_BM.append(pdf_stima_BM)

    sim_data['cdf_M'].append(cdf_stima_BM)
    sim_data['pdf_M'].append(pdf_stima_BM)

    # Calcolo Metriche N = M
    wd_m = trapezoid(np.abs(ecdf_values - cdf_stima_BM), curr_asse_x)
    metrics['wd_M'].append(wd_m)

    kl_m = entropy(pk=pdf_true_loc, qk=pdf_stima_BM + 1e-12)
    metrics['kl_M'].append(kl_m)

# Calcolo medie metriche
avg_wd_N = np.mean(metrics['wd_N'])
avg_wd_M = np.mean(metrics['wd_M'])
avg_kl_N = np.mean(metrics['kl_N'])
avg_kl_M = np.mean(metrics['kl_M'])

# Preparazione dati teorici globali
if scelta_dist == 'k':
    # Piccolo margine epsilon per evitare 0 e 1 esatti se causano problemi logaritmici
    asse_x_generale = np.linspace(max(0.0001, a_min), min(0.9999, b_max), num_points)
else:
    asse_x_generale = np.linspace(a_min, b_max, num_points)

cdf_vera = distribuzione.cdf(asse_x_generale)
pdf_vera = distribuzione.pdf(asse_x_generale)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"Bernstein Estimation - {nome_dist} (M={M})", fontsize=16)

# Helper function per plottare
def plot_simulations(ax, title, x_list, y_list, y_true, label_true):
    # Plot delle 10 simulazioni
    for x, y in zip(x_list, y_list):
        ax.plot(x, y, 'k-', linewidth=0.8, alpha=0.4)
    # Plot della teorica
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
plt.show()

# =============================================================================
# NUOVA SEZIONE: Analisi di Sensibilità del parametro N (Bias-Variance Tradeoff)
# =============================================================================

print("\nStarting analysis of optimal degree N...")

if asse_x_generale is None or ecdf is None:
    raise ValueError("Error: Simulation data missing.")

# Definiamo un range di N da testare
range_N = np.arange(5, int(N_cdf*1.5), 5)

# Liste per salvare gli errori
errori_wd = []  # Wasserstein Distance (sulla CDF)
errori_kl = []  # KL Divergence (sulla PDF)

# Usiamo l'ultimo set di dati generato nel ciclo precedente
# (campioni, a, b, asse_x_generale sono già definiti dall'ultima iterazione)

# Ricalcoliamo i valori VERI sulla griglia locale per confronto preciso
cdf_vera_loc = distribuzione.cdf(asse_x_generale)
pdf_vera_loc = distribuzione.pdf(asse_x_generale)

for n_test in range_N:
    # 1. Calcolo Stime con grado n_test
    # Nota: passiamo n_test come grado del polinomio
    cdf_est = calculate_bernstein_cdf(ecdf, int(n_test), a, b, curr_asse_x)
    pdf_est = calculate_bernstein_pdf(ecdf, int(n_test), a, b, curr_asse_x)

    # 2. Calcolo Errori rispetto alla distribuzione VERA (Oracolo)

    # Wasserstein (Errore medio assoluto sulla CDF)
    # Calcoliamo l'area tra la stima e la CDF vera
    wd_val = trapezoid(np.abs(cdf_vera_loc - cdf_est), curr_asse_x)
    errori_wd.append(wd_val)

    # KL Divergence (Entropia relativa sulla PDF)
    # Aggiungiamo epsilon per stabilità numerica
    kl_val = entropy(pk=pdf_vera_loc, qk=pdf_est + 1e-12)
    errori_kl.append(kl_val)

# Trova l'N che minimizza l'errore (il "punto dolce")
best_n_wd = range_N[np.argmin(errori_wd)]
best_n_kl = range_N[np.argmin(errori_kl)]

print("-" * 60)
print(f"OPTIMAL DEGREE ANALYSIS (Sample Size M={M})")
print("-" * 60)
print("CDF ESTIMATION (Wasserstein Distance):")
print(f"  -> Found Optimal N (via loop):  {best_n_wd}")
print(f"  -> Heuristic N (calculated):    {int(N_cdf)}  [Formula: (M / log M)^2]")
print("-" * 60)
print("PDF ESTIMATION (KL Divergence):")
print(f"  -> Found Optimal N (via loop):  {best_n_kl}")
print(f"  -> Heuristic N (calculated):    {int(N_pdf)}   [Formula: M / log M]")
print("-" * 60)


# =============================================================================
# PLOT ANALISI N
# =============================================================================

fig2, ax2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle(f"Error Analysis vs Degree N (Bias-Variance Tradeoff)", fontsize=16)

# Plot 1: Wasserstein Error (CDF)
ax2[0].plot(range_N, errori_wd, 'b-o', label='Wasserstein Dist.')
ax2[0].axvline(best_n_wd, color='r', linestyle='--', label=f'Optimal N={best_n_wd}')
# Aggiungiamo anche la linea dell'euristico per confronto visivo
ax2[0].axvline(N_cdf, color='orange', linestyle=':', alpha=0.8, label=f'Heuristic N={int(N_cdf)}')

ax2[0].set_title("CDF Accuracy vs Polynomial Degree")
ax2[0].set_xlabel("Polynomial Degree (N)")
ax2[0].set_ylabel("Error (Wasserstein Distance)")
ax2[0].grid(True, alpha=0.3)
ax2[0].legend()

# Plot 2: KL Error (PDF)
ax2[1].plot(range_N, errori_kl, 'g-o', label='KL Divergence')
ax2[1].axvline(best_n_kl, color='r', linestyle='--', label=f'Optimal N={best_n_kl}')
# Aggiungiamo anche la linea dell'euristico per confronto visivo
ax2[1].axvline(N_pdf, color='orange', linestyle=':', alpha=0.8, label=f'Heuristic N={int(N_pdf)}')

ax2[1].set_title("PDF Accuracy vs Polynomial Degree")
ax2[1].set_xlabel("Polynomial Degree (N)")
ax2[1].set_ylabel("Error (KL Divergence)")
ax2[1].grid(True, alpha=0.3)
ax2[1].legend()

# Annotation Box
text_str = (
    "Bias Region (Underfitting): Low N\n"
    "Variance Region (Overfitting): High N"
)
ax2[0].text(0.95, 0.05, text_str, transform=ax2[0].transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()
