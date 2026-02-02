import os
from datetime import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
from scipy.integrate import trapezoid

# Assume these modules are present in your environment
from bernstein import create_ecdf, calculate_bernstein_cdf, calculate_bernstein_pdf
from KumaraswamyDist import KumaraswamyDist

# =============================================================================
# 1. CONFIGURATION AND PARAMETERS
# =============================================================================

scelta_dist = 'k'  # 'n', 'u', 'k'
M = 100  # Number of samples
NUM_SIMULATIONS = 10
num_points = 500  # Graph resolution

# --- HEURISTIC N CALCULATION ---
N_pdf = math.ceil(M / math.log(M, 2))
N_cdf = math.ceil(M / math.log(M, 2)) ** 2

# Initialize Distribution
distribuzione = None
nome_dist = ""
dist_string = ""

# GAUSSIAN
if scelta_dist == 'n':
    mu = 0
    sigma = 1
    distribuzione = stats.norm(loc=mu, scale=sigma)
    nome_dist = f"Normal(mu={mu}, sigma={sigma})"
    dist_string = f"gaussian_mu{mu}_sigma{sigma}"
# UNIFORM
elif scelta_dist == 'u':
    uni_a = 5
    uni_b = 15
    distribuzione = stats.uniform(loc=uni_a, scale=(uni_b - uni_a))
    nome_dist = f"Uniform[{uni_a}, {uni_b}]"
    dist_string = f"uniform_a{uni_a}_b{uni_b}"
# KUMARASWAMY
elif scelta_dist == 'k':
    '''
    campana
    k_a = 2
    k_b = 5
    
    decrescente
    k_a = 1
    k_b = 3
    
    forma a U
    k_a = 0.5
    k_b = 0.5
    '''

    k_a = 1
    k_b = 3

    distribuzione = KumaraswamyDist(a=k_a, b=k_b)
    nome_dist = f"Kumaraswamy(a={k_a}, b={k_b})"
    dist_string = f"kumaraswamy_a{k_a}_b{k_b}"

dist_string += f"_M{M}_runs{NUM_SIMULATIONS}"

# =============================================================================
# 2. DATA STRUCTURES
# =============================================================================

sim_data = {
    'x_grids': [],
    'cdf_N_cdf': [], 'pdf_N_pdf': [],
    'cdf_M': [], 'pdf_M': [],
    'pdf_conn_to_cdf': []
}

# Structures for Boxplots
boxplot_data = {
    'cdf_M_diff_ecdf': [],  # |CDF_est - ECDF| on grid
    'cdf_N_diff_ecdf': [],
    'pdf_M_nll_samples': [],  # -Log(PDF_est(samples)) -> Individual NLL contributions
    'pdf_N_nll_samples': []
}

# Scalar Metrics (Medians)
scalar_metrics = {
    'wd_true_M': [], 'wd_emp_M': [],
    'wd_true_N': [], 'wd_emp_N': [],
    'kl_M': [], 'nll_M': [],  # CHANGED: ll_M -> nll_M
    'kl_N': [], 'nll_N': []   # CHANGED: ll_N -> nll_N
}

# Metrics for Bias-Variance Curves (Fig 3)
max_n_test = int(N_cdf * 1.5)
range_N = np.unique(np.arange(5, max_n_test, 5).astype(int))
errors_wd_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))
errors_kl_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))
errors_nll_matrix = np.zeros((NUM_SIMULATIONS, len(range_N))) # CHANGED name

global_min_x, global_max_x = float('inf'), float('-inf')

# =============================================================================
# 3. SIMULATION LOOP
# =============================================================================

for i in range(NUM_SIMULATIONS):
    print(f"Cycle {i + 1}/{NUM_SIMULATIONS}")

    # A. Generate Samples
    campioni = distribuzione.rvs(size=M)
    campioni_ordinati = np.sort(campioni)
    ecdf = create_ecdf(campioni)

    # B. Local Support
    a, b = campioni_ordinati[0], campioni_ordinati[-1]
    global_min_x = min(global_min_x, a)
    global_max_x = max(global_max_x, b)

    # C. Local Grid
    curr_asse_x = np.linspace(a, b, num_points)
    sim_data['x_grids'].append(curr_asse_x)

    # True Values
    cdf_true_loc = distribuzione.cdf(curr_asse_x)
    pdf_true_loc = distribuzione.pdf(curr_asse_x)
    ecdf_vals_grid = ecdf(curr_asse_x)

    # -----------------------------------------------------------
    # PART 1: Estimations
    # -----------------------------------------------------------

    # === Case N = M ===
    cdf_stima_M = calculate_bernstein_cdf(ecdf, M, a, b, curr_asse_x)
    pdf_stima_M = calculate_bernstein_pdf(ecdf, M, a, b, curr_asse_x)
    pdf_stima_M_samples = calculate_bernstein_pdf(ecdf, M, a, b, campioni)

    sim_data['cdf_M'].append(cdf_stima_M)
    sim_data['pdf_M'].append(pdf_stima_M)

    # --- BOXPLOT DATA ---
    diff_cdf_M = np.abs(ecdf_vals_grid - cdf_stima_M)
    boxplot_data['cdf_M_diff_ecdf'].append(diff_cdf_M)

    # CHANGED: Store Negative Log Probs (Positive Cost)
    # Using epsilon to avoid log(0)
    nll_samples_M = -np.log(pdf_stima_M_samples + 1e-12)
    boxplot_data['pdf_M_nll_samples'].append(nll_samples_M)

    # --- SCALAR METRICS ---
    wd_true_M = trapezoid(np.abs(cdf_true_loc - cdf_stima_M), curr_asse_x)
    # print("wd_true_M:", wd_true_M)
    wd_emp_M = trapezoid(diff_cdf_M, curr_asse_x)
    scalar_metrics['wd_true_M'].append(wd_true_M)
    scalar_metrics['wd_emp_M'].append(wd_emp_M)

    kl_M = entropy(pk=pdf_true_loc, qk=pdf_stima_M + 1e-12)
    # CHANGED: Average NLL calculation
    nll_M = np.mean(nll_samples_M)
    scalar_metrics['kl_M'].append(kl_M)
    scalar_metrics['nll_M'].append(nll_M)

    # === Case N Heuristic ===

    # CDF (N_cdf)
    cdf_stima_N = calculate_bernstein_cdf(ecdf, int(N_cdf), a, b, curr_asse_x)
    pdf_conn_to_cdf = calculate_bernstein_pdf(ecdf, int(N_cdf), a, b, curr_asse_x)
    sim_data['cdf_N_cdf'].append(cdf_stima_N)
    sim_data['pdf_conn_to_cdf'].append(pdf_conn_to_cdf)

    diff_cdf_N = np.abs(ecdf_vals_grid - cdf_stima_N)
    boxplot_data['cdf_N_diff_ecdf'].append(diff_cdf_N)

    wd_true_N = trapezoid(np.abs(cdf_true_loc - cdf_stima_N), curr_asse_x)
    wd_emp_N = trapezoid(diff_cdf_N, curr_asse_x)
    scalar_metrics['wd_true_N'].append(wd_true_N)
    scalar_metrics['wd_emp_N'].append(wd_emp_N)

    # PDF (N_pdf)
    pdf_stima_N = calculate_bernstein_pdf(ecdf, int(N_pdf), a, b, curr_asse_x)
    pdf_stima_N_samples = calculate_bernstein_pdf(ecdf, int(N_pdf), a, b, campioni)
    sim_data['pdf_N_pdf'].append(pdf_stima_N)

    # CHANGED: Store Negative Log Probs
    nll_samples_N = -np.log(pdf_stima_N_samples + 1e-12)
    boxplot_data['pdf_N_nll_samples'].append(nll_samples_N)

    kl_N = entropy(pk=pdf_true_loc, qk=pdf_stima_N + 1e-12)
    # CHANGED: Average NLL
    nll_N = np.mean(nll_samples_N)
    scalar_metrics['kl_N'].append(kl_N)
    scalar_metrics['nll_N'].append(nll_N)

    # -----------------------------------------------------------
    # PART 2: Bias-Variance Analysis
    # -----------------------------------------------------------
    for idx_n, n_val in enumerate(range_N):
        cdf_temp = calculate_bernstein_cdf(ecdf, int(n_val), a, b, curr_asse_x)
        pdf_temp = calculate_bernstein_pdf(ecdf, int(n_val), a, b, curr_asse_x)
        pdf_temp_samples = calculate_bernstein_pdf(ecdf, int(n_val), a, b, campioni)
        errors_wd_matrix[i, idx_n] = trapezoid(np.abs(ecdf_vals_grid - cdf_temp), curr_asse_x)
        errors_kl_matrix[i, idx_n] = entropy(pk=pdf_true_loc, qk=pdf_temp + 1e-12)
        # CHANGED: Calculate Mean NLL for matrix
        errors_nll_matrix[i, idx_n] = -np.mean(np.log(pdf_temp_samples + 1e-12))

# =============================================================================
# 4. POST-PROCESSING AND STATS
# =============================================================================
def get_full_stats(metric_list):
    """Restituisce Media, Mediana, Std Dev"""
    return np.mean(metric_list), np.median(metric_list), np.std(metric_list)

# Calcolo statistiche complete (Mean, Median, Std)
mean_wd_true_M, med_wd_true_M, std_wd_true_M = get_full_stats(scalar_metrics['wd_true_M'])
mean_wd_emp_M,  med_wd_emp_M,  std_wd_emp_M  = get_full_stats(scalar_metrics['wd_emp_M'])

mean_wd_true_N, med_wd_true_N, std_wd_true_N = get_full_stats(scalar_metrics['wd_true_N'])
mean_wd_emp_N,  med_wd_emp_N,  std_wd_emp_N  = get_full_stats(scalar_metrics['wd_emp_N'])

mean_kl_M, med_kl_M, std_kl_M = get_full_stats(scalar_metrics['kl_M'])
mean_kl_N, med_kl_N, std_kl_N = get_full_stats(scalar_metrics['kl_N'])

mean_nll_M, med_nll_M, std_nll_M = get_full_stats(scalar_metrics['nll_M'])
mean_nll_N, med_nll_N, std_nll_N = get_full_stats(scalar_metrics['nll_N'])

# Average Curves (invariato)
avg_curve_wd = np.mean(errors_wd_matrix, axis=0)
avg_curve_kl = np.mean(errors_kl_matrix, axis=0)
avg_curve_nll = np.mean(errors_nll_matrix, axis=0)

best_n_wd = range_N[np.argmin(avg_curve_wd)]
best_n_kl = range_N[np.argmin(avg_curve_kl)]
best_n_nll = range_N[np.argmin(avg_curve_nll)]

# Setup GT lines (invariato)
if scelta_dist == 'k':
    asse_x_generale = np.linspace(max(0.0001, global_min_x), min(0.9999, global_max_x), num_points)
else:
    asse_x_generale = np.linspace(global_min_x, global_max_x, num_points)

cdf_vera = distribuzione.cdf(asse_x_generale)
pdf_vera = distribuzione.pdf(asse_x_generale)


def plot_spaghetti(ax, x_list, y_list, y_true, label_true, title, color_true='k'):
    for x, y in zip(x_list, y_list):
        ax.plot(x, y, 'k-', linewidth=0.5, alpha=0.3)
    ax.plot(asse_x_generale, y_true, color=color_true, linewidth=2, linestyle='-', label=label_true)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

# Funzione helper aggiornata per disegnare Media e Mediana
def add_stat_lines(ax, mean_val, median_val, std_val, label, color='darkorange', text_x_offset=1.02):
    text_x_offset = 1.02
    # 1. Disegna le linee orizzontali
    # Linea MEDIA (più marcata)
    ax.axhline(mean_val, color=color, linestyle='--', linewidth=1.5, alpha=0.9)
    # Linea MEDIANA (più leggera)
    ax.axhline(median_val, color=color, linestyle='--', linewidth=1.5, alpha=0.6)

    # 2. Ottieni i limiti per calcolare la scala relativa
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    if y_range == 0:
        y_range = 1.0

    # 3. Preparazione Contenuti e Stili
    # Definiamo i dati per Mean e Median separatamente
    # Mean: Bold
    mean_text = rf"{label} Mean: {mean_val:.4f} $\pm$ {std_val:.4f}"
    mean_style = 'bold'

    # Median: Normal (non bold)
    median_text = f"{label} Median: {median_val:.4f}"
    median_style = 'normal'  # o 'regular'

    # 4. Logica di Ordinamento (Chi sta sopra?)
    # Se mean >= median, Mean sta sopra. Se median > mean, Median sta sopra.
    if mean_val >= median_val:
        top_data = (mean_val, mean_text, mean_style)
        bot_data = (median_val, median_text, median_style)
    else:
        top_data = (median_val, median_text, median_style)
        bot_data = (mean_val, mean_text, mean_style)

    # Estraiamo i dati ordinati
    val_top, txt_top, style_top = top_data
    val_bot, txt_bot, style_bot = bot_data

    # 5. Calcolo Sovrapposizione
    dist = abs(mean_val - median_val)
    overlap_threshold = 0.07 * y_range

    if dist < overlap_threshold:
        # === CASO SOVRAPPOSIZIONE ===
        # Ancoriamo entrambi al punto medio
        mid_point = (mean_val + median_val) / 2

        # Testo Superiore: ancorato al mid_point, ma spinto verso l'alto (va='bottom')
        ax.text(text_x_offset, mid_point, txt_top,
                transform=ax.get_yaxis_transform(),
                color=color, fontsize=9, va='bottom', fontweight=style_top)

        # Testo Inferiore: ancorato al mid_point, ma spinto verso il basso (va='top')
        ax.text(text_x_offset, mid_point, txt_bot,
                transform=ax.get_yaxis_transform(),
                color=color, fontsize=9, va='top', fontweight=style_bot)

    else:
        # === CASO NORMALE (Separati) ===
        # Per evitare che i testi si scontrino nel mezzo se le linee sono vicine ma non troppo,
        # applichiamo la logica: il valore più alto ha il testo SOPRA la sua linea,
        # il valore più basso ha il testo SOTTO la sua linea.

        # Testo per il valore più alto (va='bottom' -> sopra la linea)
        ax.text(text_x_offset, val_top, txt_top,
                transform=ax.get_yaxis_transform(),
                color=color, fontsize=9, va='center', fontweight=style_top)

        # Testo per il valore più basso (va='top' -> sotto la linea)
        ax.text(text_x_offset, val_bot, txt_bot,
                transform=ax.get_yaxis_transform(),
                color=color, fontsize=9, va='center', fontweight=style_bot)


# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# Parametri Booleani richiesti
SAVE_VERTICAL = True  # Formato Originale (3x2)
SAVE_HORIZONTAL = True  # Formato Nuovo (2x3)

today_str = datetime.now().strftime("%Y%m%d")

# Setup Cartelle
dir_vert = f"img/{today_str}"
dir_horz = f"img/{today_str}_h"

if SAVE_VERTICAL:
    os.makedirs(dir_vert, exist_ok=True)
if SAVE_HORIZONTAL:
    os.makedirs(dir_horz, exist_ok=True)


# Funzione Helper per mappare gli assi
# Restituisce un dizionario con chiavi logiche ('m_cdf', 'n_cdf', etc.)
# mappate sugli assi fisici in base all'orientamento.
def get_axes_mapping_fig1_2(ax_matrix, is_horizontal):
    targets = {}
    if not is_horizontal:
        # VERTICAL (3 rows, 2 cols)
        # 00 => 00 (M CDF)
        # 01 => 01 (N CDF)  <- Nota: nel tuo mapping originale era diretto
        # Manteniamo la struttura logica: Col 0 = M, Col 1 = N
        targets['m_row1'] = ax_matrix[0, 0]
        targets['n_row1'] = ax_matrix[0, 1]
        targets['m_row2'] = ax_matrix[1, 0]
        targets['n_row2'] = ax_matrix[1, 1]
        targets['m_row3'] = ax_matrix[2, 0]
        targets['n_row3'] = ax_matrix[2, 1]
    else:
        # HORIZONTAL (2 rows, 3 cols) - TRANSPOSED MAPPING
        # Richiesta:
        # 00 => 00 (M Row1) -> diventa Row0 Col0
        # 01 => 10 (N Row1) -> diventa Row1 Col0
        # 10 => 01 (M Row2) -> diventa Row0 Col1
        # 11 => 11 (N Row2) -> diventa Row1 Col1
        # 20 => 02 (M Row3) -> diventa Row0 Col2
        # 21 => 12 (N Row3) -> diventa Row1 Col2

        targets['m_row1'] = ax_matrix[0, 0]
        targets['n_row1'] = ax_matrix[1, 0]  # Transposed
        targets['m_row2'] = ax_matrix[0, 1]  # Transposed
        targets['n_row2'] = ax_matrix[1, 1]
        targets['m_row3'] = ax_matrix[0, 2]  # Transposed
        targets['n_row3'] = ax_matrix[1, 2]
    return targets


# =============================================================================
# 5. VISUALIZATION - FIGURE 1: FOCUS CDF
# =============================================================================

def draw_fig1_content(ax_map):
    # CDF Plots (Row 1 logical)
    plot_spaghetti(ax_map['m_row1'], sim_data['x_grids'], sim_data['cdf_M'], cdf_vera, 'True CDF', f"CDF (N=M={M})")
    plot_spaghetti(ax_map['n_row1'], sim_data['x_grids'], sim_data['cdf_N_cdf'], cdf_vera, 'True CDF',
                   f"CDF (N={int(N_cdf)})")

    # PDF Derivative Plots (Row 2 logical)
    plot_spaghetti(ax_map['m_row2'], sim_data['x_grids'], sim_data['pdf_M'], pdf_vera, 'True PDF',
                   f"Derivative PDF (N=M={M})")
    plot_spaghetti(ax_map['n_row2'], sim_data['x_grids'], sim_data['pdf_conn_to_cdf'], pdf_vera, 'True PDF',
                   f"Derivative PDF (N={int(N_cdf)})")

    # Boxplot WD (Row 3 logical)
    # M
    ax_map['m_row3'].boxplot(scalar_metrics['wd_emp_M'], medianprops=dict(color='k', linewidth=1.5))
    add_stat_lines(ax_map['m_row3'], mean_wd_emp_M, med_wd_emp_M, std_wd_emp_M, "Emp", color='darkorange',
                   text_x_offset=1.02)
    add_stat_lines(ax_map['m_row3'], mean_wd_true_M, med_wd_true_M, std_wd_true_M, "True", color='firebrick',
                   text_x_offset=1.35)
    ax_map['m_row3'].set_title(f"Est CDF vs ECDF (N=M={M})", fontsize=10, fontweight='bold')
    ax_map['m_row3'].set_ylabel("Wasserstein distance")
    ax_map['m_row3'].grid(True, alpha=0.3)

    # N
    ax_map['n_row3'].boxplot(scalar_metrics['wd_emp_N'], medianprops=dict(color='k', linewidth=1.5))
    add_stat_lines(ax_map['n_row3'], mean_wd_emp_N, med_wd_emp_N, std_wd_emp_N, "Emp", color='darkorange',
                   text_x_offset=1.02)
    add_stat_lines(ax_map['n_row3'], mean_wd_true_N, med_wd_true_N, std_wd_true_N, "True", color='firebrick',
                   text_x_offset=1.35)
    ax_map['n_row3'].set_title(f"Est CDF vs ECDF (N={int(N_cdf)})", fontsize=10, fontweight='bold')
    ax_map['n_row3'].set_ylabel("Wasserstein distance")
    ax_map['n_row3'].grid(True, alpha=0.3)

    # Tick params for all
    for k, ax in ax_map.items():
        ax.tick_params(labelleft=True)


# --- GENERAZIONE FIGURA 1 ---
file_name_1 = f"{dist_string}_1cdf.jpg"

# 1. Verticale
if SAVE_VERTICAL:
    fig1_v, ax1_v = plt.subplots(3, 2, sharey='row', figsize=(12, 18))
    fig1_v.suptitle(f"CDF Analysis: {nome_dist} (M={M}) - {NUM_SIMULATIONS} runs", fontsize=14)
    mapping_v = get_axes_mapping_fig1_2(ax1_v, is_horizontal=False)
    draw_fig1_content(mapping_v)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig1_v.savefig(os.path.join(dir_vert, file_name_1), dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[VERT] Fig 1 Saved: {os.path.join(dir_vert, file_name_1)}")
    plt.close(fig1_v)

# 2. Orizzontale (Sharey='col' è cruciale qui perché le colonne contengono grandezze omogenee)
if SAVE_HORIZONTAL:
    fig1_h, ax1_h = plt.subplots(2, 3, sharey='col', figsize=(18, 12))
    fig1_h.suptitle(f"CDF Analysis: {nome_dist} (M={M}) - {NUM_SIMULATIONS} runs", fontsize=14)
    mapping_h = get_axes_mapping_fig1_2(ax1_h, is_horizontal=True)
    draw_fig1_content(mapping_h)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig1_h.savefig(os.path.join(dir_horz, file_name_1), dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[HORZ] Fig 1 Saved: {os.path.join(dir_horz, file_name_1)}")
    plt.close(fig1_h)


# =============================================================================
# 6. VISUALIZATION - FIGURE 2: FOCUS PDF
# =============================================================================

def draw_fig2_content(ax_map):
    # Spaghetti Plots (Row 1 logical)
    plot_spaghetti(ax_map['m_row1'], sim_data['x_grids'], sim_data['pdf_M'], pdf_vera,
                   'True PDF', f"PDF Estimator (N=M={M})")
    plot_spaghetti(ax_map['n_row1'], sim_data['x_grids'], sim_data['pdf_N_pdf'], pdf_vera,
                   'True PDF', f"PDF Estimator (N={int(N_pdf)})")

    # NLL Boxplots (Row 2 logical)
    ax_map['m_row2'].boxplot(scalar_metrics['nll_M'], medianprops=dict(color='k', linewidth=1.5))
    add_stat_lines(ax_map['m_row2'], mean_nll_M, med_nll_M, std_nll_M, "NLL", color='darkorange')
    ax_map['m_row2'].set_title(f"Est PDF vs Samples (N=M={M})", fontsize=10, fontweight='bold')
    ax_map['m_row2'].set_ylabel("Negative log-likelihood")
    ax_map['m_row2'].grid(True, alpha=0.3)

    ax_map['n_row2'].boxplot(scalar_metrics['nll_N'], medianprops=dict(color='k', linewidth=1.5))
    add_stat_lines(ax_map['n_row2'], mean_nll_N, med_nll_N, std_nll_N, "NLL", color='darkorange')
    ax_map['n_row2'].set_title(f"Est PDF vs Samples (N={int(N_pdf)})", fontsize=10, fontweight='bold')
    ax_map['n_row2'].set_ylabel("Negative log-likelihood")
    ax_map['n_row2'].grid(True, alpha=0.3)

    # KL Boxplots (Row 3 logical)
    ax_map['m_row3'].boxplot(scalar_metrics['kl_M'], medianprops=dict(color='k', linewidth=1.5))
    add_stat_lines(ax_map['m_row3'], mean_kl_M, med_kl_M, std_kl_M, "KL", color='darkorange')
    ax_map['m_row3'].set_title(f"Est PDF vs True (N=M={M})", fontsize=10, fontweight='bold')
    ax_map['m_row3'].set_ylabel("Kullback–Leibler divergence")
    ax_map['m_row3'].grid(True, alpha=0.3)

    ax_map['n_row3'].boxplot(scalar_metrics['kl_N'], medianprops=dict(color='k', linewidth=1.5))
    add_stat_lines(ax_map['n_row3'], mean_kl_N, med_kl_N, std_kl_N, "KL", color='darkorange')
    ax_map['n_row3'].set_title(f"Est PDF vs True (N={N_pdf})", fontsize=10, fontweight='bold')
    ax_map['n_row3'].set_ylabel("Kullback–Leibler divergence")
    ax_map['n_row3'].grid(True, alpha=0.3)

    for k, ax in ax_map.items():
        ax.tick_params(labelleft=True)


# --- GENERAZIONE FIGURA 2 ---
file_name_2 = f"{dist_string}_2pdf.jpg"

# 1. Verticale
if SAVE_VERTICAL:
    fig2_v, ax2_v = plt.subplots(3, 2, sharey='row', figsize=(12, 18))
    fig2_v.suptitle(f"PDF Analysis: {nome_dist} (M={M}) - {NUM_SIMULATIONS} runs", fontsize=14)
    mapping_v = get_axes_mapping_fig1_2(ax2_v, is_horizontal=False)
    draw_fig2_content(mapping_v)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig2_v.savefig(os.path.join(dir_vert, file_name_2), dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[VERT] Fig 2 Saved: {os.path.join(dir_vert, file_name_2)}")
    plt.close(fig2_v)

# 2. Orizzontale
if SAVE_HORIZONTAL:
    fig2_h, ax2_h = plt.subplots(2, 3, sharey='col', figsize=(18, 12))
    fig2_h.suptitle(f"PDF Analysis: {nome_dist} (M={M}) - {NUM_SIMULATIONS} runs", fontsize=14)
    mapping_h = get_axes_mapping_fig1_2(ax2_h, is_horizontal=True)
    draw_fig2_content(mapping_h)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig2_h.savefig(os.path.join(dir_horz, file_name_2), dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[HORZ] Fig 2 Saved: {os.path.join(dir_horz, file_name_2)}")
    plt.close(fig2_h)


# =============================================================================
# 7. VISUALIZATION - FIGURE 3: BIAS-VARIANCE
# =============================================================================

# Helper per mappare Fig 3 (1x3 vs 3x1)
def get_axes_mapping_fig3(ax_array, is_horizontal_mode):
    # Logical Names: 'wd', 'kl', 'nll'
    # Originale (Vertical Mode set): 1 Row x 3 Cols (Wide strip)
    # Nuovo (Horizontal Mode set): Trasposto -> 3 Rows x 1 Col (Vertical strip)
    # Nota: L'utente ha chiesto di trasporre le logiche.
    # Se il set principale è Verticale (3x2), Fig3 è (1x3).
    # Se il set principale è Orizzontale (2x3), Fig3 diventa (3x1).

    t = {}
    if not is_horizontal_mode:
        # Array shape (3,) flat or (1,3)
        # Assumiamo flatten access
        ax_flat = ax_array.flatten()
        t['wd'] = ax_flat[0]
        t['kl'] = ax_flat[1]
        t['nll'] = ax_flat[2]
    else:
        # Anche qui flatten, ma l'aspetto della figura è verticale
        ax_flat = ax_array.flatten()
        t['wd'] = ax_flat[0]
        t['kl'] = ax_flat[1]
        t['nll'] = ax_flat[2]
    return t


def draw_fig3_content(ax_map):
    # Wasserstein
    ax_map['wd'].plot(range_N, avg_curve_wd, 'b-o', label='Mean WD')
    ax_map['wd'].axvline(best_n_wd, color='r', linestyle='--', label=f'Best N={best_n_wd}')
    ax_map['wd'].axvline(N_cdf, color='orange', linestyle=':', linewidth=2, label=f'Heur N={int(N_cdf)}')
    ax_map['wd'].set_title("CDF Distance (Wasserstein)")
    ax_map['wd'].set_xlabel("Degree N")
    ax_map['wd'].grid(True, alpha=0.3)
    ax_map['wd'].legend()

    # KL Divergence
    ax_map['kl'].plot(range_N, avg_curve_kl, 'g-o', label='Mean KL')
    ax_map['kl'].axvline(best_n_kl, color='r', linestyle='--', label=f'Best N={best_n_kl}')
    ax_map['kl'].axvline(N_pdf, color='orange', linestyle=':', linewidth=2, label=f'Heur N={int(N_pdf)}')
    ax_map['kl'].set_title("PDF Distance (KL Divergence)")
    ax_map['kl'].set_xlabel("Degree N")
    ax_map['kl'].grid(True, alpha=0.3)
    ax_map['kl'].legend()

    # NLL
    ax_map['nll'].plot(range_N, avg_curve_nll, 'm-o', label='Mean NLL')
    ax_map['nll'].axvline(best_n_nll, color='r', linestyle='--', label=f'Best N={best_n_nll}')
    ax_map['nll'].axvline(N_pdf, color='orange', linestyle=':', linewidth=2, label=f'Heur N={int(N_pdf)}')
    ax_map['nll'].set_title("Sample Fit (Neg Log Likelihood)")
    ax_map['nll'].set_xlabel("Degree N")
    ax_map['nll'].grid(True, alpha=0.3)
    ax_map['nll'].legend()


# --- GENERAZIONE FIGURA 3 ---
file_name_3 = f"{dist_string}_3bias_tradeoff.jpg"

# 1. Verticale (Standard: 1 riga, 3 colonne)
if SAVE_HORIZONTAL or SAVE_VERTICAL:
    fig3, ax3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle(f"Metric Sensitivity vs Degree N (Avg over {NUM_SIMULATIONS} runs) - {nome_dist}", fontsize=14)
    map_v = get_axes_mapping_fig3(ax3, is_horizontal_mode=False)
    draw_fig3_content(map_v)
    plt.tight_layout()
    if SAVE_VERTICAL:
        fig3.savefig(os.path.join(dir_vert, file_name_3), dpi=150, bbox_inches='tight', facecolor='white')
        print(f"[VERT] Fig 3 Saved: {os.path.join(dir_vert, file_name_3)}")
    if SAVE_HORIZONTAL:
        fig3.savefig(os.path.join(dir_horz, file_name_3), dpi=150, bbox_inches='tight', facecolor='white')
        print(f"[HORZ] Fig 3 Saved: {os.path.join(dir_horz, file_name_3)}")
    plt.close(fig3)
