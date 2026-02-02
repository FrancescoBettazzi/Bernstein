import os
from datetime import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy
from scipy.integrate import trapezoid

# Modulo specifico per Bernstein Exponentials (domino [0, inf))
from bernstein_exp import create_ecdf, calculate_bernstein_exp_cdf, calculate_bernstein_exp_pdf

# =============================================================================
# 1. CONFIGURATION AND PARAMETERS
# =============================================================================

# Scegli la distribuzione in base alle immagini fornite:
# 'erlang', 'weibull', 'lognormal'
scelta_dist = 'lognormal'

M = 100  # Numero di campioni
NUM_SIMULATIONS = 10  # Numero di simulazioni Monte Carlo
num_points = 500  # Risoluzione grafici

# --- HEURISTIC N CALCULATION ---
# Calcolo gradi N euristici in base a M
N_pdf = math.ceil(M / math.log(M, 2))
N_cdf = math.ceil(M / math.log(M, 2)) ** 2

# Inizializzazione Variabili Distribuzione
distribuzione = None
nome_dist = ""
dist_string = ""

# --- CONFIGURAZIONE DISTRIBUZIONI (da immagini) ---

if scelta_dist == 'erlang':
    # Erlang(n, lambda). Immagine: media 1 => scala = 1/n
    n_erlang = 5  # Esempio: n=5 (puoi cambiare questo valore)
    # In scipy: Gamma(a=n, scale=1/lambda_rate).
    # Se media=1, scale=1/n.
    distribuzione = stats.gamma(a=n_erlang, scale=1 / n_erlang)
    nome_dist = f"Erlang(n={n_erlang}, mean=1)"
    dist_string = f"erlang_n{n_erlang}_mu1"

elif scelta_dist == 'weibull':
    # Immagine Weibull: (1, 1.5) o (1, 0.5). Assumiamo (Scale=1, Shape=k)
    '''
    w_shape = 1.5
    
    w_shape = 0.5
    '''

    w_shape = 1.5  # k
    w_scale = 1.0  # lambda
    # Scipy weibull_min: c=shape, scale=scale
    distribuzione = stats.weibull_min(c=w_shape, scale=w_scale)
    nome_dist = f"Weibull(scale={w_scale}, shape={w_shape})"
    dist_string = f"weibull_scale{w_scale}_shape{w_shape}"

elif scelta_dist == 'lognormal':
    # Immagine Lognormal: (1, 1.8). Assumiamo (Scale=1, Shape=sigma)
    # Parametri (Scale, s). Scale = exp(mu). Se Scale=1 => mu=0.
    '''
    ln_shape = 1.8
    ln_shape = 0.8
    ln_shape = 0.2
    '''
    ln_shape = 1.8  # sigma (s)
    ln_scale = 1.0  # exp(mu)
    distribuzione = stats.lognorm(s=ln_shape, scale=ln_scale)
    nome_dist = f"Lognormal(s={ln_shape}, scale={ln_scale})"
    dist_string = f"lognormal_scale{ln_scale}_shape{ln_shape}"

dist_string += f"_M{M}_runs{NUM_SIMULATIONS}"

visual_xlim = distribuzione.ppf(0.999) * 1.5
# visual_xlim = 10
# print(f"Limite visuale asse X impostato a: {visual_xlim:.2f}")
# =============================================================================
# 2. DATA STRUCTURES
# =============================================================================

sim_data = {
    'x_grids': [],
    'cdf_N_cdf': [], 'pdf_N_pdf': [],
    'cdf_M': [], 'pdf_M': [],
    'pdf_conn_to_cdf': []
}

# Scalar Metrics
scalar_metrics = {
    'wd_true_M': [], 'wd_emp_M': [],
    'wd_true_N': [], 'wd_emp_N': [],
    'kl_M': [], 'nll_M': [],
    'kl_N': [], 'nll_N': []
}

# Metrics for Bias-Variance Curves
max_n_test = int(N_cdf * 1.5)
range_N = np.unique(np.arange(5, max_n_test, 5).astype(int))
errors_wd_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))
errors_kl_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))
errors_nll_matrix = np.zeros((NUM_SIMULATIONS, len(range_N)))

# Definiamo i limiti globali per i plot (dominio positivo)
# Usiamo la ppf per trovare dove finisce la massa significativa
true_max_x = distribuzione.ppf(0.999)
global_max_x = true_max_x

# =============================================================================
# 3. SIMULATION LOOP
# =============================================================================

for i in range(NUM_SIMULATIONS):
    print(f"Cycle {i + 1}/{NUM_SIMULATIONS}")

    # A. Generazione Campioni
    campioni = distribuzione.rvs(size=M)
    # Assicuriamo che siano positivi (dovrebbero esserlo per queste distribuzioni)
    campioni = np.abs(campioni)
    campioni_ordinati = np.sort(campioni)

    # ECDF function creation (Step function)
    ecdf = create_ecdf(campioni)

    # B. Definizione Griglia Locale
    # Per Bernstein Exp il dominio è [0, inf).
    # Usiamo un range ragionevole basato sui dati correnti e sulla verità
    max_curr = max(campioni[-1], true_max_x)
    curr_asse_x = np.linspace(1e-9, max_curr, num_points)
    sim_data['x_grids'].append(curr_asse_x)

    # Aggiorniamo il max globale per i plot finali
    if max_curr > global_max_x:
        global_max_x = max_curr

    # C. Parametro di Trasformazione (Euristica Adattiva)
    # λ_trans = 1 / mean(samples) è una buona scelta standard per mappare i dati in [0,1]
    scale_param = 1.0 / np.mean(campioni)

    # True Values su griglia locale
    cdf_true_loc = distribuzione.cdf(curr_asse_x)
    pdf_true_loc = distribuzione.pdf(curr_asse_x)
    # Opzionale: limita i valori infiniti per la stabilità dei calcoli successivi
    # pdf_true_loc = np.clip(pdf_true_loc, 0, 1e10)
    ecdf_vals_grid = ecdf(curr_asse_x)

    # -----------------------------------------------------------
    # PART 1: Estimations (Using Bernstein EXP)
    # -----------------------------------------------------------

    # === Case N = M ===
    cdf_stima_M = calculate_bernstein_exp_cdf(ecdf, M, curr_asse_x, scale=scale_param)
    pdf_stima_M = calculate_bernstein_exp_pdf(ecdf, M, curr_asse_x, scale=scale_param)

    # PDF sui campioni per NLL
    pdf_stima_M_samples = calculate_bernstein_exp_pdf(ecdf, M, campioni, scale=scale_param)

    sim_data['cdf_M'].append(cdf_stima_M)
    sim_data['pdf_M'].append(pdf_stima_M)

    # --- METRICS CALCULATION (N=M) ---
    diff_cdf_M = np.abs(ecdf_vals_grid - cdf_stima_M)
    nll_samples_M = -np.log(pdf_stima_M_samples + 1e-12)

    wd_true_M = trapezoid(np.abs(cdf_true_loc - cdf_stima_M), curr_asse_x)
    wd_emp_M = trapezoid(diff_cdf_M, curr_asse_x)
    kl_M = entropy(pk=pdf_true_loc, qk=pdf_stima_M + 1e-12)
    nll_M = np.mean(nll_samples_M)

    scalar_metrics['wd_true_M'].append(wd_true_M)
    scalar_metrics['wd_emp_M'].append(wd_emp_M)
    scalar_metrics['kl_M'].append(kl_M)
    scalar_metrics['nll_M'].append(nll_M)

    # === Case N Heuristic ===

    # CDF (N_cdf)
    cdf_stima_N = calculate_bernstein_exp_cdf(ecdf, int(N_cdf), curr_asse_x, scale=scale_param)
    pdf_conn_to_cdf = calculate_bernstein_exp_pdf(ecdf, int(N_cdf), curr_asse_x, scale=scale_param)

    sim_data['cdf_N_cdf'].append(cdf_stima_N)
    sim_data['pdf_conn_to_cdf'].append(pdf_conn_to_cdf)

    diff_cdf_N = np.abs(ecdf_vals_grid - cdf_stima_N)
    wd_true_N = trapezoid(np.abs(cdf_true_loc - cdf_stima_N), curr_asse_x)
    wd_emp_N = trapezoid(diff_cdf_N, curr_asse_x)

    scalar_metrics['wd_true_N'].append(wd_true_N)
    scalar_metrics['wd_emp_N'].append(wd_emp_N)

    # PDF (N_pdf)
    pdf_stima_N = calculate_bernstein_exp_pdf(ecdf, int(N_pdf), curr_asse_x, scale=scale_param)
    pdf_stima_N_samples = calculate_bernstein_exp_pdf(ecdf, int(N_pdf), campioni, scale=scale_param)

    sim_data['pdf_N_pdf'].append(pdf_stima_N)

    nll_samples_N = -np.log(pdf_stima_N_samples + 1e-12)
    kl_N = entropy(pk=pdf_true_loc, qk=pdf_stima_N + 1e-12)
    nll_N = np.mean(nll_samples_N)

    scalar_metrics['kl_N'].append(kl_N)
    scalar_metrics['nll_N'].append(nll_N)

    # -----------------------------------------------------------
    # PART 2: Bias-Variance Analysis (Loop over N)
    # -----------------------------------------------------------
    for idx_n, n_val in enumerate(range_N):
        cdf_temp = calculate_bernstein_exp_cdf(ecdf, int(n_val), curr_asse_x, scale=scale_param)
        pdf_temp = calculate_bernstein_exp_pdf(ecdf, int(n_val), curr_asse_x, scale=scale_param)
        pdf_temp_samples = calculate_bernstein_exp_pdf(ecdf, int(n_val), campioni, scale=scale_param)

        errors_wd_matrix[i, idx_n] = trapezoid(np.abs(cdf_true_loc - cdf_temp), curr_asse_x)
        errors_kl_matrix[i, idx_n] = entropy(pk=pdf_true_loc, qk=pdf_temp + 1e-12)
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

# Curve medie (invariato)
avg_curve_wd = np.mean(errors_wd_matrix, axis=0)
avg_curve_kl = np.mean(errors_kl_matrix, axis=0)
avg_curve_nll = np.mean(errors_nll_matrix, axis=0)

best_n_wd = range_N[np.argmin(avg_curve_wd)]
best_n_kl = range_N[np.argmin(avg_curve_kl)]
best_n_nll = range_N[np.argmin(avg_curve_nll)]

# Asse X Generale per il plot Ground Truth (invariato)
asse_x_generale = np.linspace(1e-9, global_max_x, num_points)
cdf_vera = distribuzione.cdf(asse_x_generale)
pdf_vera = distribuzione.pdf(asse_x_generale)

# Helper Functions per il Plotting
def plot_spaghetti(ax, x_list, y_list, y_true, label_true, title, color_true='k', x_max=None):
    ax.plot(asse_x_generale, y_true, color=color_true, linewidth=2, linestyle='-', label=label_true)
    for x, y in zip(x_list, y_list):
        ax.plot(x, y, 'k-', linewidth=0.5, alpha=0.3)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

    if x_max is not None:
        ax.set_xlim(0, x_max)


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

today_str = datetime.now().strftime("%Y%m%d")
output_dir = f"img/{today_str}"
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# Parametri Booleani
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
def get_axes_mapping_fig1_2(ax_matrix, is_horizontal):
    """
    Restituisce un dizionario con chiavi logiche mappate sugli assi fisici.
    Logica Orizzontale richiesta:
    00 -> 00 (M Row1)
    01 -> 10 (N Row1)
    10 -> 01 (M Row2)
    11 -> 11 (N Row2)
    20 -> 02 (M Row3)
    21 -> 12 (N Row3)
    """
    targets = {}
    if not is_horizontal:
        # VERTICAL (3 rows, 2 cols)
        # Col 0 = M, Col 1 = N
        targets['m_row1'] = ax_matrix[0, 0]
        targets['n_row1'] = ax_matrix[0, 1]
        targets['m_row2'] = ax_matrix[1, 0]
        targets['n_row2'] = ax_matrix[1, 1]
        targets['m_row3'] = ax_matrix[2, 0]
        targets['n_row3'] = ax_matrix[2, 1]
    else:
        # HORIZONTAL (2 rows, 3 cols) - TRANSPOSED
        # Row 0 = M, Row 1 = N (Columns represent the original rows)
        targets['m_row1'] = ax_matrix[0, 0]
        targets['n_row1'] = ax_matrix[1, 0]
        targets['m_row2'] = ax_matrix[0, 1]
        targets['n_row2'] = ax_matrix[1, 1]
        targets['m_row3'] = ax_matrix[0, 2]
        targets['n_row3'] = ax_matrix[1, 2]
    return targets


def sync_ylim(ax1, ax2):
    """Sincronizza i limiti Y di due assi."""
    ylim_1 = ax1.get_ylim()
    ylim_2 = ax2.get_ylim()
    global_min = min(ylim_1[0], ylim_2[0])
    global_max = max(ylim_1[1], ylim_2[1])
    ax1.set_ylim(global_min, global_max)
    ax2.set_ylim(global_min, global_max)


# =============================================================================
# 5. VISUALIZATION - FIGURE 1: FOCUS CDF
# =============================================================================

def draw_fig1_content(ax_map):
    # --- Row 1 Logical: Spaghetti CDF ---
    plot_spaghetti(ax_map['m_row1'], sim_data['x_grids'], sim_data['cdf_M'], cdf_vera,
                   'True CDF', f"CDF (N=M={M})", x_max=visual_xlim)
    plot_spaghetti(ax_map['n_row1'], sim_data['x_grids'], sim_data['cdf_N_cdf'], cdf_vera,
                   'True CDF', f"CDF (N={int(N_cdf)})", x_max=visual_xlim)

    # --- Row 2 Logical: Derivative PDF ---
    plot_spaghetti(ax_map['m_row2'], sim_data['x_grids'], sim_data['pdf_M'], pdf_vera,
                   'True PDF', f"Derivative PDF (N=M={M})", x_max=visual_xlim)
    plot_spaghetti(ax_map['n_row2'], sim_data['x_grids'], sim_data['pdf_conn_to_cdf'], pdf_vera,
                   'True PDF', f"Derivative PDF (N={int(N_cdf)})", x_max=visual_xlim)

    # --- Row 3 Logical: Boxplots WD ---
    # M Case
    ax_map['m_row3'].boxplot(scalar_metrics['wd_emp_M'], medianprops=dict(color='k', linewidth=1.5))
    add_stat_lines(ax_map['m_row3'], mean_wd_emp_M, med_wd_emp_M, std_wd_emp_M, "Emp", color='darkorange',
                   text_x_offset=1.02)
    add_stat_lines(ax_map['m_row3'], mean_wd_true_M, med_wd_true_M, std_wd_true_M, "True", color='firebrick',
                   text_x_offset=1.35)
    ax_map['m_row3'].set_title(f"Est CDF vs ECDF (N=M={M})", fontsize=10, fontweight='bold')
    ax_map['m_row3'].set_ylabel("Wasserstein distance")
    ax_map['m_row3'].grid(True, alpha=0.3)

    # N Case
    ax_map['n_row3'].boxplot(scalar_metrics['wd_emp_N'], medianprops=dict(color='k', linewidth=1.5))
    add_stat_lines(ax_map['n_row3'], mean_wd_emp_N, med_wd_emp_N, std_wd_emp_N, "Emp", color='darkorange',
                   text_x_offset=1.02)
    add_stat_lines(ax_map['n_row3'], mean_wd_true_N, med_wd_true_N, std_wd_true_N, "True", color='firebrick',
                   text_x_offset=1.35)
    ax_map['n_row3'].set_title(f"Est CDF vs ECDF (N={int(N_cdf)})", fontsize=10, fontweight='bold')
    ax_map['n_row3'].set_ylabel("Wasserstein distance")
    ax_map['n_row3'].grid(True, alpha=0.3)

    # Fix Scale shared manually for Boxplots
    sync_ylim(ax_map['m_row3'], ax_map['n_row3'])

    for k, ax in ax_map.items():
        ax.tick_params(labelleft=True)


# --- SAVE FIG 1 ---
file_name_1 = f"{dist_string}_1cdf.jpg"

if SAVE_VERTICAL:
    fig1_v, ax1_v = plt.subplots(3, 2, sharey='row', figsize=(12, 18))
    fig1_v.suptitle(f"CDF Analysis: {nome_dist} (M={M}) - {NUM_SIMULATIONS} runs", fontsize=14)
    mapping_v = get_axes_mapping_fig1_2(ax1_v, is_horizontal=False)
    draw_fig1_content(mapping_v)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig1_v.savefig(os.path.join(dir_vert, file_name_1), dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[VERT] Fig 1 Saved: {os.path.join(dir_vert, file_name_1)}")
    plt.close(fig1_v)

if SAVE_HORIZONTAL:
    # sharey='col' perché ora le colonne contengono grandezze simili (es. Col 0 = Spaghetti, Col 2 = Boxplots)
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
    # --- Row 1 Logical: Spaghetti PDF ---
    plot_spaghetti(ax_map['m_row1'], sim_data['x_grids'], sim_data['pdf_M'], pdf_vera,
                   'True PDF', f"PDF Estimator (N=M={M})", x_max=visual_xlim)
    plot_spaghetti(ax_map['n_row1'], sim_data['x_grids'], sim_data['pdf_N_pdf'], pdf_vera,
                   'True PDF', f"PDF Estimator (N={int(N_pdf)})", x_max=visual_xlim)

    # --- Row 2 Logical: NLL ---
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

    # Fix Scale shared NLL
    sync_ylim(ax_map['m_row2'], ax_map['n_row2'])

    # --- Row 3 Logical: KL Divergence ---
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

    # Fix Scale shared KL
    sync_ylim(ax_map['m_row3'], ax_map['n_row3'])

    for k, ax in ax_map.items():
        ax.tick_params(labelleft=True)


# --- SAVE FIG 2 ---
file_name_2 = f"{dist_string}_2pdf.jpg"

if SAVE_VERTICAL:
    fig2_v, ax2_v = plt.subplots(3, 2, sharey='row', figsize=(12, 18))
    fig2_v.suptitle(f"PDF Analysis: {nome_dist} (M={M}) - {NUM_SIMULATIONS} runs", fontsize=14)
    mapping_v = get_axes_mapping_fig1_2(ax2_v, is_horizontal=False)
    draw_fig2_content(mapping_v)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig2_v.savefig(os.path.join(dir_vert, file_name_2), dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[VERT] Fig 2 Saved: {os.path.join(dir_vert, file_name_2)}")
    plt.close(fig2_v)

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

def get_axes_mapping_fig3(ax_array, is_horizontal_mode):
    # Se is_horizontal_mode=True, la figura risultante è "verticale" (3x1)
    # Se is_horizontal_mode=False, la figura risultante è "orizzontale" (1x3)
    t = {}
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


# --- SAVE FIG 3 ---
file_name_3 = f"{dist_string}_3bias_tradeoff.jpg"

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

'''if SAVE_HORIZONTAL or SAVE_VERTICAL:
    # Qui invertiamo: diventa 3 righe x 1 colonna
    fig3_h, ax3_h = plt.subplots(3, 1, figsize=(6, 18))
    fig3_h.suptitle(f"Metric Sensitivity - {nome_dist}", fontsize=14)
    map_h = get_axes_mapping_fig3(ax3_h, is_horizontal_mode=True)
    draw_fig3_content(map_h)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig3_h.savefig(os.path.join(dir_horz, file_name_3), dpi=300, bbox_inches='tight')
    print(f"Fig 3 Saved")
    plt.close(fig3_h)'''
