import os
from datetime import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import entropy, gaussian_kde

# Import custom (assicurati che i file siano nella stessa directory o path)
from bernstein import create_ecdf, calculate_bernstein_pdf
from bernstein_exp import calculate_bernstein_exp_pdf
from KumaraswamyDist import KumaraswamyDist

# =============================================================================
# 1. CONFIGURAZIONE
# =============================================================================

# Scegli la distribuzione qui:
# 'u', 'n', 'k', 'k_d', 'k_u', 'erlang',
# 'weibull_1_5', 'weibull_0_5',
# 'lognormal_1_8', 'lognormal_0_8', 'lognormal_0_2'

# DIST_KEY = 'lognormal_0_2'
DIST_KEYS = ['u', 'n', 'k', 'k_d', 'k_u', 'erlang', 'weibull_1_5', 'weibull_0_5', 'lognormal_1_8', 'lognormal_0_8', 'lognormal_0_2']

M = 200  # Numero campioni
NUM_SIMULATIONS = 100  # Totale simulazioni
NUM_POINTS = 500  # Risoluzione griglia
N_PLOT_LINES = 50  # Quante linee disegnare nel grafico "Spaghetti"

for DIST_KEY in DIST_KEYS:
    # Dizionario configurazione distribuzioni
    dist_config = {}

    # Variabili di supporto
    dist_obj = None  # Oggetto distribuzione (scipy o custom)
    dist_name = ""  # Nome per il plot
    dist_string = ""
    use_exp_bernstein = False  # Flag per cambiare metodo
    support_type = "bounded"  # 'bounded' o 'semi-infinite' (per la griglia x)

    # --- LOGICA SELEZIONE DISTRIBUZIONE ---
    if DIST_KEY == 'u':
        # Uniforme [5, 15] -> loc=5, scale=10
        dist_obj = stats.uniform(loc=5, scale=10)
        dist_name = "Uniforme [5, 15]"
        support_type = "bounded"
        dist_string = "uniform"

    elif DIST_KEY == 'n':
        # Gaussiana mu=0 sigma=1
        dist_obj = stats.norm(loc=0, scale=1)
        dist_name = "Normale (mu=0, sigma=1)"
        support_type = "unbounded"  # Tratteremo come bounded sui min/max campionari per Bernstein STD
        dist_string = "gaussian"

    elif DIST_KEY == 'k':
        dist_obj = KumaraswamyDist(a=2, b=5)
        dist_name = "Kumaraswamy (a=2, b=5)"
        support_type = "bounded"
        dist_string = "kumaraswamy_2_5"

    elif DIST_KEY == 'k_d':
        dist_obj = KumaraswamyDist(a=1, b=3)
        dist_name = "Kumaraswamy (a=1, b=3)"
        support_type = "bounded"
        dist_string = "kumaraswamy_1_3"

    elif DIST_KEY == 'k_u':
        dist_obj = KumaraswamyDist(a=0.5, b=0.5)
        dist_name = "Kumaraswamy (a=0.5, b=0.5)"
        support_type = "bounded"
        dist_string = "kumaraswamy_05_05"

    # --- DISTRIBUZIONI CHE USANO BERNSTEIN EXP ---
    elif DIST_KEY == 'erlang':
        # Erlang n=5, scale=1/n (media=1) -> In scipy Erlang è Gamma con a intero
        # scale = 1/5 = 0.2
        dist_obj = stats.gamma(a=5, scale=0.2)
        dist_name = "Erlang (n=5, mu=1)"
        use_exp_bernstein = True
        support_type = "semi-infinite"
        dist_string = "erlang"

    elif DIST_KEY == 'weibull_1_5':
        # Shape (c) = 1.5, Scale = 1
        dist_obj = stats.weibull_min(c=1.5, scale=1)
        dist_name = "Weibull (shape=1.5, scale=1)"
        use_exp_bernstein = True
        support_type = "semi-infinite"
        dist_string = "weibull_1_15"

    elif DIST_KEY == 'weibull_0_5':
        dist_obj = stats.weibull_min(c=0.5, scale=1)
        dist_name = "Weibull (shape=0.5, scale=1)"
        use_exp_bernstein = True
        support_type = "semi-infinite"
        dist_string = "weibull_1_05"

    elif DIST_KEY == 'lognormal_1_8':
        # s = shape parameter (sigma), scale = exp(mu). Se scale=1 -> mu=0
        dist_obj = stats.lognorm(s=1.8, scale=1)
        dist_name = "Lognormal (s=1.8, scale=1)"
        use_exp_bernstein = True
        support_type = "semi-infinite"
        dist_string = "lognormal_1_18"

    elif DIST_KEY == 'lognormal_0_8':
        dist_obj = stats.lognorm(s=0.8, scale=1)
        dist_name = "Lognormal (s=0.8, scale=1)"
        use_exp_bernstein = True
        support_type = "semi-infinite"
        dist_string = "lognormal_1_08"

    elif DIST_KEY == 'lognormal_0_2':
        dist_obj = stats.lognorm(s=0.2, scale=1)
        dist_name = "Lognormal (s=0.2, scale=1)"
        use_exp_bernstein = True
        support_type = "semi-infinite"
        dist_string = "lognormal_1_02"

    else:
        raise ValueError(f"Distribuzione '{DIST_KEY}' non riconosciuta.")

    # Euristica N
    N_pdf = math.ceil(M / math.log(M, 2))

    # Storage
    kl_bernstein_list = []
    kl_kde_list = []

    # Salviamo i dati dei primi N run per il plot "spaghetti"
    plot_runs = []

    # =============================================================================
    # 2. LOOP DI CONFRONTO
    # =============================================================================

    print(f"Analisi Distribuzione: {dist_name}")
    print(f"Metodo Bernstein: {'EXP (Semi-Infinite)' if use_exp_bernstein else 'STANDARD (Bounded)'}")
    print(f"Campioni M={M} | N_Bernstein={N_pdf}")
    print("-" * 60)

    for i in range(NUM_SIMULATIONS):
        # 1. Generazione
        campioni = dist_obj.rvs(size=M)

        # 2. Definizione Griglia X di valutazione
        # La griglia serve sia per calcolare l'errore che per il plot

        if support_type == "bounded":
            # Per Kuma o Uniforme, stiamo strettamente nei bound teorici
            if DIST_KEY == 'u':
                lower, upper = 5, 15
            else:  # Kumaraswamy
                lower, upper = 0.0001, 0.9999
            x_eval = np.linspace(lower, upper, NUM_POINTS)

        elif support_type == "unbounded":
            # Caso specifico per la NORMALE (copre valori negativi)
            # Usiamo i percentili teorici per coprire il 99.8% della distribuzione
            lower_lim = dist_obj.ppf(0.001)
            upper_lim = dist_obj.ppf(0.999)
            x_eval = np.linspace(lower_lim, upper_lim, NUM_POINTS)

        else:  # semi-infinite (Erlang, Weibull, Lognormal)
            # Parte da 0 (o quasi) e va fino al percentile alto
            upper_lim = dist_obj.ppf(0.999)
            x_eval = np.linspace(1e-9, upper_lim, NUM_POINTS)

        # 3. Ground Truth
        pdf_true = dist_obj.pdf(x_eval)

        # 4. BERNSTEIN ESTIMATION
        ecdf = create_ecdf(campioni)

        if use_exp_bernstein:
            # Usa la variante esponenziale
            # Nota: calculate_bernstein_exp_pdf solitamente non richiede a,b bounds rigidi come la standard
            # o se li richiede, sono impliciti nel dominio [0, inf).
            # Assumiamo la firma: (ecdf, N, x_eval) o simile.
            # Adatto in base alla tua richiesta precedente.
            pdf_bern = calculate_bernstein_exp_pdf(ecdf, N_pdf, x_eval)
        else:
            # Usa la variante standard
            # Serve definire a, b dai campioni per il supporto locale
            a_samp, b_samp = np.min(campioni), np.max(campioni)
            pdf_bern = calculate_bernstein_pdf(ecdf, N_pdf, a_samp, b_samp, x_eval)

        # 5. KDE ESTIMATION
        kde_func = gaussian_kde(campioni)
        pdf_kde = kde_func(x_eval)

        # 6. CALCOLO KL DIVERGENCE
        # Aggiungiamo eps per stabilità numerica
        kl_b = entropy(pk=pdf_true, qk=pdf_bern + 1e-12)
        kl_k = entropy(pk=pdf_true, qk=pdf_kde + 1e-12)

        kl_bernstein_list.append(kl_b)
        kl_kde_list.append(kl_k)

        # Salvataggio per plot (solo primi N_PLOT_LINES)
        if i < N_PLOT_LINES:
            plot_runs.append({
                'x': x_eval,
                'pdf_bern': pdf_bern,
                'pdf_kde': pdf_kde,
                'pdf_true': pdf_true
            })

        if (i + 1) % 10 == 0:
            print(f"Run {i + 1}/{NUM_SIMULATIONS} | KL Bern: {kl_b:.4f} | KL KDE: {kl_k:.4f}")

    # =============================================================================
    # 3. VISUALIZZAZIONE
    # =============================================================================

    # Creiamo una figura
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Comparison: {dist_name} (num. samples M={M}) - {NUM_SIMULATIONS} runs", fontsize=16)

    # Definizione griglia 2x2
    # height_ratios=[1.2, 1]: La riga sopra (curve) è leggermente più alta
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

    # Creazione assi (4 grafici distinti)
    ax_bern_curve = fig.add_subplot(gs[0, 0])  # Top Left: Curve Bernstein
    ax_kde_curve = fig.add_subplot(gs[0, 1])  # Top Right: Curve KDE
    ax_box_bern = fig.add_subplot(gs[1, 0])  # Bottom Left: Boxplot Bernstein
    ax_box_kde = fig.add_subplot(gs[1, 1])  # Bottom Right: Boxplot KDE

    # ---------------------------------------------------------
    # RIGA 1: PLOT DELLE CURVE (Spaghetti Plot)
    # ---------------------------------------------------------

    base_x = plot_runs[0]['x']
    base_y = plot_runs[0]['pdf_true']
    max_pdf_val = np.max(base_y)

    # 1.1 Bernstein Curves
    for run in plot_runs:
        ax_bern_curve.plot(run['x'], run['pdf_bern'], color='blue', alpha=0.15, lw=1)
        max_pdf_val = max(max_pdf_val, np.max(run['pdf_bern']))

    ax_bern_curve.plot(base_x, base_y, 'k-', lw=2.5, label='Ground Truth', zorder=10)
    ax_bern_curve.set_title(f"Bernstein Estimations (N={N_pdf})")
    ax_bern_curve.set_ylabel("PDF")
    ax_bern_curve.legend(loc='upper right')
    ax_bern_curve.grid(True, alpha=0.3)

    # 1.2 KDE Curves
    for run in plot_runs:
        ax_kde_curve.plot(run['x'], run['pdf_kde'], color='red', alpha=0.15, lw=1)
        max_pdf_val = max(max_pdf_val, np.max(run['pdf_kde']))

    ax_kde_curve.plot(base_x, base_y, 'k-', lw=2.5, label='Ground Truth', zorder=10)
    ax_kde_curve.set_title(f"Standard KDE Estimations")
    ax_kde_curve.set_ylabel("PDF")
    ax_kde_curve.legend(loc='upper right')
    ax_kde_curve.grid(True, alpha=0.3)

    # Sincronizzazione RIGA 1 (PDF)
    common_pdf_ylim = (0, max_pdf_val * 1.05)
    ax_bern_curve.set_ylim(common_pdf_ylim)
    ax_kde_curve.set_ylim(common_pdf_ylim)
    ax_bern_curve.set_xlim(base_x[0], base_x[-1])
    ax_kde_curve.set_xlim(base_x[0], base_x[-1])

    # ---------------------------------------------------------
    # RIGA 2: BOXPLOT ERRORI (Separati ma sincronizzati)
    # ---------------------------------------------------------

    # Calcolo limiti asse Y comuni per i boxplot (Min/Max globale degli errori)
    all_errors = kl_bernstein_list + kl_kde_list
    min_err, max_err = min(all_errors), max(all_errors)
    # Aggiungiamo margine (10% sopra e sotto)
    y_box_margin = (max_err - min_err) * 0.1
    common_box_ylim = (max(0, min_err - y_box_margin), max_err + y_box_margin)


    # Funzione helper per disegnare boxplot personalizzati
    def draw_custom_boxplot(ax, data, color, title, label_y=True):
        # Calcolo statistiche
        mediana = np.median(data)
        dev_std = np.std(data)

        # Disegno Boxplot
        bplot = ax.boxplot(data)

        # Stile Box
        # for patch in bplot['boxes']:
        #    patch.set_facecolor(color)
        #    patch.set_alpha(0.6)

        # Stile Mediana del boxplot (linea solida dentro il box)
        for median_line in bplot['medians']:
            median_line.set(color='black', linewidth=1.5)

        # --- LINEA MEDIANA ESTESA E LABEL ---
        # Disegna linea orizzontale rossa tratteggiata su tutto il grafico
        ax.axhline(mediana, color=color, linestyle='--', linewidth=1.5, alpha=0.8)

        # Etichetta con valore
        text_str = f"Median: {mediana:.4f}\nStd Dev: {dev_std:.4f}"
        # Posizioniamo il testo leggermente sopra la linea, a destra
        ax.text(1.02, mediana, text_str, transform=ax.get_yaxis_transform(),
                color=color, fontsize=9, va='center')
                # ,bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec='red'))

        ax.set_title(title)
        if label_y:
            ax.set_ylabel("KL Divergence")

        ax.set_ylim(common_box_ylim)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # Rimuoviamo i tick x inutili (c'è solo 1 box)
        ax.set_xticks([])


    # 2.1 Boxplot Bernstein (Blu)
    draw_custom_boxplot(ax_box_bern, kl_bernstein_list, 'blue',
                        "Bernstein KL Error Distribution", label_y=True)

    # 2.2 Boxplot KDE (Rosso)
    draw_custom_boxplot(ax_box_kde, kl_kde_list, 'red',
                        "Standard KDE KL Error Distribution", label_y=True)

    plt.tight_layout()

    output_dir = "img"
    os.makedirs(output_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y%m%d")
    file_name = f"{today_str}_kde_vs_bernstein_M{M}_SIMUL{NUM_SIMULATIONS}_{dist_string}.png"
    full_path = os.path.join(output_dir, file_name)
    fig.savefig(full_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)
