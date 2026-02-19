import numpy as np
from scipy.interpolate import interp1d
from scipy.special import comb, gammaln


def create_ecdf(campioni):
    """
    Crea la funzione ECDF (Empirical CDF) interpolata.
    Restituisce un oggetto chiamabile vettorizzato.
    """
    M = len(campioni)
    campioni_ordinati = np.sort(campioni)
    y_gradino = np.arange(1, M + 1) / M

    # interp1d gestisce nativamente input vettoriali (array numpy)
    ecdf = interp1d(
        campioni_ordinati,
        y_gradino,
        kind='previous',
        bounds_error=False,
        fill_value=(0.0, 1.0)
    )

    return ecdf


'''def get_bernstein_basis(z, N):
    """
    Funzione helper per calcolare la matrice delle basi.
    Restituisce una matrice di shape (len(z), N+1)
    """
    # Assicuriamo che z sia colonna (M, 1) e n sia riga (1, N+1) per il broadcasting
    z = z[:, np.newaxis]
    n = np.arange(N + 1)

    # Calcolo vettorializzato dei coefficienti binomiali e delle potenze
    # (N su n) * z^n * (1-z)^(N-n)
    coeffs = comb(N, n)
    basis = coeffs * (z ** n) * ((1 - z) ** (N - n))

    return basis'''


def get_bernstein_basis(z, N):
    """
    Funzione helper per calcolare la matrice delle basi.
    VERSIONE STABILE (Log-Sum-Exp) per N grandi.
    Restituisce una matrice di shape (len(z), N+1)
    """
    # Assicuriamo che z sia colonna (M, 1) e n sia riga (1, N+1) per il broadcasting
    z = z[:, np.newaxis]
    n = np.arange(N + 1)

    # --- INIZIO MODIFICA PER STABILITÀ NUMERICA ---

    # 1. Calcolo Logaritmo del Coefficiente Binomiale
    # log(N!) - log(n!) - log((N-n)!)
    log_coeffs = gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1)

    # 2. Gestione sicura dei logaritmi per le potenze (evita log(0))
    eps = 1e-16
    z_safe = np.clip(z, eps, 1.0 - eps)

    # 3. Calcolo esponenti nel dominio log
    # log(z^n) -> n * log(z)
    log_pow_z = n * np.log(z_safe)
    # log((1-z)^(N-n)) -> (N-n) * log(1-z)
    log_pow_1_z = (N - n) * np.log(1.0 - z_safe)

    # 4. Somma logaritmica ed esponenziale finale
    log_basis = log_coeffs + log_pow_z + log_pow_1_z
    basis = np.exp(log_basis)

    # --- FINE MODIFICA ---

    return basis


def calculate_bernstein_cdf(ecdf, N, a, b, asse_x):
    """
    Calcola la CDF di Bernstein in modo vettorializzato.
    """
    # 1. Normalizzazione x -> z in [0, 1]
    asse_x = np.asarray(asse_x)
    z = np.clip((asse_x - a) / (b - a), 0.0, 1.0)

    # 2. Calcolo dei pesi w_n = F(a + (b-a) * n/N)
    # n va da 0 a N
    n_range = np.arange(N + 1)
    eval_points = a + (b - a) * (n_range / N)
    weights = ecdf(eval_points)  # shape (N+1,)

    # 3. Calcolo Base e Prodotto Scalare
    # Basis shape: (len(x), N+1)
    basis = get_bernstein_basis(z, N)

    # Prodotto matriciale: (M, N+1) @ (N+1,) -> (M,)
    return basis @ weights


def calculate_bernstein_pdf(ecdf, N, a, b, asse_x):
    """
    Calcola la PDF come derivata analitica della CDF di Bernstein.
    """
    asse_x = np.asarray(asse_x)

    # 1. Calcoliamo z (normalizzato)
    # Rimuovi il clip immediato per poter creare una maschera
    z_raw = (asse_x - a) / (b - a)

    # Creiamo una maschera per i valori validi (dentro il supporto)
    mask_inside = (z_raw >= 0.0) & (z_raw <= 1.0)

    # Ora clippiamo per il calcolo della base (per evitare errori numerici sui bordi esatti)
    z = np.clip(z_raw, 0.0, 1.0)

    # ... (Step 2 e 3: Calcolo diffs e basis_deriv rimangono uguali) ...
    # Ricopia il codice esistente per diffs e basis_deriv
    k_range = np.arange(N + 1)
    eval_points = a + (b - a) * (k_range / N)
    F_vals = ecdf(eval_points)
    diffs = np.diff(F_vals)

    basis_deriv = get_bernstein_basis(z, N - 1)
    scale_factor = N / (b - a)

    # 4. Risultato finale
    pdf_values = (basis_deriv @ diffs) * scale_factor

    # APPLICA LA MASCHERA: Fuori dal supporto [a, b], la densità è 0
    pdf_values[~mask_inside] = 0.0

    return pdf_values


def calculate_bernstein_exponential_cdf(ecdf, N, asse_y):
    """
    Versione vettorializzata per la trasformata esponenziale.
    """
    asse_y = np.asarray(asse_y)
    x = np.exp(-asse_y)  # x gioca il ruolo di z qui

    # Pesi
    n_range = np.arange(N + 1)

    # Gestione log(0): n=0 -> cdf_val=1.0, else cdf(-log(n/N))
    # Creiamo array pesi inizializzato a 0
    weights = np.zeros(N + 1)

    # n=0
    weights[0] = 1.0

    # n > 0
    if N > 0:
        n_vals = n_range[1:]
        args = -np.log(n_vals / N)
        weights[1:] = ecdf(args)

    # Calcolo base (su x)
    basis = get_bernstein_basis(x, N)

    return basis @ weights
