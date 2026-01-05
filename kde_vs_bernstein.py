import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
from scipy.stats import gaussian_kde
from bernstein2 import create_ecdf, calculate_bernstein_pdf
from KumaraswamyDist import KumaraswamyDist

# =============================================================================
# 1. CONFIGURAZIONE
# =============================================================================

# Parametri Simulazione
M = 100  # Numero campioni (pochi campioni = KDE soffre di più)
NUM_SIMULATIONS = 50  # Numero di run per la media degli errori
NUM_POINTS = 500  # Risoluzione griglia

# Scelta parametri Kumaraswamy
# Usiamo parametri che spingono la massa verso i bordi per evidenziare il difetto della KDE
# Caso A: a=0.8, b=0.8 (Forma a "U", massa su entrambi i bordi 0 e 1)
# Caso B: a=2, b=5 (Massa vicino a 0, il tuo caso standard)
a_param = 2
b_param = 5

distribuzione = KumaraswamyDist(a=a_param, b=b_param)
nome_dist = f"Kumaraswamy(a={a_param}, b={b_param})"

# Euristica per Bernstein (N Ottimale per PDF)
N_pdf = math.ceil(M / math.log(M, 2))  # N euristico per PDF

# Liste per salvare gli errori
kl_bernstein = []
kl_kde = []

# Liste per salvare l'ultimo run (per il plot qualitativo)
last_run_data = {}

# =============================================================================
# 2. LOOP DI CONFRONTO (BERNSTEIN vs KDE)
# =============================================================================

print(f"Confronto Bernstein (N={N_pdf}) vs Standard KDE (Gaussian)")
print(f"Distribuzione: {nome_dist} | Campioni M={M}")
print("-" * 60)

for i in range(NUM_SIMULATIONS):
    if (i + 1) % 10 == 0: print(f"Simulazione {i + 1}/{NUM_SIMULATIONS}...")

    # 1. Generazione Dati
    campioni = distribuzione.rvs(size=M)
    campioni_ordinati = np.sort(campioni)

    # Supporto locale (min/max campionari)
    a, b = campioni_ordinati[0], campioni_ordinati[-1]

    # 2. Griglia di Valutazione (Strettamente [0, 1] per il calcolo errore)
    x_eval = np.linspace(0.0001, 0.9999, NUM_POINTS)

    # 3. Ground Truth
    pdf_true = distribuzione.pdf(x_eval)

    # ---------------------------------------------------------
    # METODO A: BERNSTEIN
    # ---------------------------------------------------------
    ecdf = create_ecdf(campioni)
    pdf_bern = calculate_bernstein_pdf(ecdf, N_pdf, a, b, x_eval)

    # Calcolo KL Divergence Bernstein
    # Aggiungiamo epsilon per evitare log(0)
    kl_b = entropy(pk=pdf_true, qk=pdf_bern + 1e-12)
    kl_bernstein.append(kl_b)

    # ---------------------------------------------------------
    # METODO B: KERNEL DENSITY ESTIMATION (KDE)
    # ---------------------------------------------------------
    # Scipy usa "Scott's Rule" o "Silverman's Rule" per la bandwidth automatica
    kde_func = gaussian_kde(campioni)
    pdf_kde = kde_func(x_eval)

    # Calcolo KL Divergence KDE
    kl_k = entropy(pk=pdf_true, qk=pdf_kde + 1e-12)
    kl_kde.append(kl_k)

    # Salviamo dati ultimo giro per il plot
    if i == NUM_SIMULATIONS - 1:
        last_run_data = {
            'campioni': campioni,
            'pdf_bern': pdf_bern,
            'pdf_kde': pdf_kde,
            'pdf_true': pdf_true,
            'x_eval': x_eval,
            'kde_func': kde_func  # Salviamo la funzione per plottare fuori dai bordi
        }

# =============================================================================
# 3. RISULTATI E PLOT
# =============================================================================

mean_kl_bern = np.mean(kl_bernstein)
mean_kl_kde = np.mean(kl_kde)

print("\n" + "=" * 40)
print("RISULTATI MEDI (KL Divergence - Lower is Better)")
print("=" * 40)
print(f"Bernstein (N={N_pdf}):  {mean_kl_bern:.5f}")
print(f"Standard KDE:        {mean_kl_kde:.5f}")
diff_pct = ((mean_kl_kde - mean_kl_bern) / mean_kl_kde) * 100
print(f"Miglioramento:       {diff_pct:.1f}%")
print("=" * 40)

# --- VISUALIZZAZIONE ---

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f"Bernstein vs KDE Comparison - {nome_dist} (M={M})", fontsize=16)

# PLOT 1: Qualitativo (Boundary Bias)
# Creiamo una griglia estesa per mostrare dove la KDE sbaglia (valori < 0 e > 1)
x_extended = np.linspace(-0.2, 1.2, 600)
pdf_true_ext = np.zeros_like(x_extended)
# Popoliamo la PDF vera solo dentro [0,1]
mask_inside = (x_extended >= 0) & (x_extended <= 1)
pdf_true_ext[mask_inside] = distribuzione.pdf(x_extended[mask_inside])

# Calcoliamo la KDE sulla griglia estesa
pdf_kde_ext = last_run_data['kde_func'](x_extended)

# Bernstein è definita solo tra [a, b] campionari, la estendiamo a 0 altrove per il plot
# (Nota: Bernstein nativamente non esce dal supporto [0,1] riscalato)
# Per semplicità riusiamo i dati salvati su [0,1]
ax[0].plot(last_run_data['x_eval'], last_run_data['pdf_true'], 'k-', lw=2, label='True PDF')
ax[0].plot(last_run_data['x_eval'], last_run_data['pdf_bern'], 'b-', lw=2, label=f'Bernstein (N={N_pdf})')
ax[0].plot(x_extended, pdf_kde_ext, 'r--', lw=2, label='Standard KDE')

# Evidenziare il "Boundary Leakage" della KDE
ax[0].fill_between(x_extended, pdf_kde_ext, 0,
                   where=(x_extended < 0) | (x_extended > 1),
                   color='red', alpha=0.2, label='KDE Leakage (Error)')

ax[0].set_xlim(-0.15, 1.15)
ax[0].set_title("Boundary Bias Visualized")
ax[0].set_xlabel("x")
ax[0].set_ylabel("Density")
ax[0].legend()
ax[0].grid(True, alpha=0.3)
ax[0].text(0.05, 0.95, "Bernstein respects [0,1] limits.\nKDE leaks probability mass.",
           transform=ax[0].transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# PLOT 2: Bar Chart degli Errori
methods = ['Bernstein', 'Standard KDE']
errors = [mean_kl_bern, mean_kl_kde]
colors = ['blue', 'red']

bars = ax[1].bar(methods, errors, color=colors, alpha=0.7, width=0.5)
ax[1].set_title("Average Estimation Error (KL Divergence)")
ax[1].set_ylabel("KL Divergence (Lower is Better)")
ax[1].grid(axis='y', alpha=0.3)

# Aggiungi etichette valore sulle barre
for bar in bars:
    height = bar.get_height()
    ax[1].text(bar.get_x() + bar.get_width() / 2., height,
               f'{height:.4f}',
               ha='center', va='bottom')

plt.tight_layout()
plt.show()
