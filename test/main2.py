import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
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
    wd_n = np.trapz(np.abs(ecdf_values - cdf_stima), curr_asse_x)
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
    wd_m = np.trapz(np.abs(ecdf_values - cdf_stima_BM), curr_asse_x)
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
fig.suptitle(f"Stima di Bernstein - {nome_dist} (M={M})", fontsize=16)

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
plot_simulations(axes[0, 0], f"CDF (N=M={M}) - Wd: {avg_wd_M:.5f}",
                 sim_data['x_grids'], sim_data['cdf_M'], cdf_vera, 'CDF Teorica')

# 2. CDF (N Ottimale)
plot_simulations(axes[0, 1], f"CDF (N={int(N_cdf)}) - Wd: {avg_wd_N:.5f}",
                 sim_data['x_grids'], sim_data['cdf_N'], cdf_vera, 'CDF Teorica')

# 3. PDF (N=M)
plot_simulations(axes[1, 0], f"PDF (N=M={M}) - KL: {avg_kl_M:.5f}",
                 sim_data['x_grids'], sim_data['pdf_M'], pdf_vera, 'PDF Teorica')

# 4. PDF (N Ottimale)
plot_simulations(axes[1, 1], f"PDF (N={int(N_pdf)}) - KL: {avg_kl_N:.5f}",
                 sim_data['x_grids'], sim_data['pdf_N'], pdf_vera, 'PDF Teorica')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
