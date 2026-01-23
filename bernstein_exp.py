import numpy as np
from scipy.interpolate import interp1d
from scipy.special import comb, gammaln


# --- Funzioni Helper Esistenti (invariate) ---
def create_ecdf(campioni):
    M = len(campioni)
    campioni_ordinati = np.sort(campioni)
    y_gradino = np.arange(1, M + 1) / M
    ecdf = interp1d(
        campioni_ordinati,
        y_gradino,
        kind='previous',
        bounds_error=False,
        fill_value=(0.0, 1.0)
    )
    return ecdf


'''def get_bernstein_basis(z, N):
    z = z[:, np.newaxis]
    n = np.arange(N + 1)
    coeffs = comb(N, n)
    basis = coeffs * (z ** n) * ((1 - z) ** (N - n))
    return basis'''


def get_bernstein_basis(z, N):
    """
    Calcola la base di Bernstein usando il log-sum-exp trick per stabilità numerica
    con N grandi (evita overflow/underflow dei coefficienti binomiali).
    """
    z = z[:, np.newaxis]
    n = np.arange(N + 1)

    # 1. Calcolo Logaritmo del Coefficiente Binomiale: log(N!) - log(n!) - log((N-n)!)
    # gammaln(x) calcola il log della funzione Gamma, che equivale a log((x-1)!)
    # Quindi usiamo n+1 per ottenere log(n!)
    log_coeffs = gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1)

    # 2. Gestione sicura dei logaritmi per le potenze
    # Aggiungiamo un epsilon minuscolo per evitare log(0) -> -inf
    eps = 1e-16
    z_safe = np.clip(z, eps, 1.0 - eps)

    # log(z^n) = n * log(z)
    log_pow_z = n * np.log(z_safe)

    # log((1-z)^(N-n)) = (N-n) * log(1-z)
    log_pow_1_z = (N - n) * np.log(1.0 - z_safe)

    # 3. Somma tutto nel dominio log e poi esponenzia
    log_basis = log_coeffs + log_pow_z + log_pow_1_z
    basis = np.exp(log_basis)

    return basis

# --- Nuove Funzioni per Dominio [0, inf) ---

def calculate_bernstein_exp_cdf(ecdf, N, asse_x, scale=1.0):
    """
    Calcola la CDF di Bernstein su dominio [0, inf) usando la trasformazione:
    z = 1 - exp(-scale * x)
    """
    asse_x = np.asarray(asse_x)

    # 1. Trasformazione x -> z in [0, 1]
    # Mappiamo [0, inf) -> [0, 1]
    z = 1.0 - np.exp(-scale * asse_x)

    # Clip per sicurezza numerica
    z = np.clip(z, 0.0, 1.0)

    # 2. Calcolo dei pesi w_n
    # I punti di valutazione nel dominio z sono k/N.
    # Dobbiamo trasformarli indietro nel dominio x per interrogare la ECDF originale.
    # z = 1 - e^(-sx) => e^(-sx) = 1 - z => -sx = ln(1-z) => x = -ln(1-z)/s

    k_range = np.arange(N + 1)
    z_nodes = k_range / N

    # Gestione singolarità: per k=N, z_nodes=1, ln(0) = -inf.
    # Tuttavia, sappiamo che ECDF(inf) = 1.0.
    weights = np.zeros(N + 1)

    # Calcoliamo per k < N
    mask_finite = (z_nodes < 1.0)
    if np.any(mask_finite):
        x_nodes = -np.log(1.0 - z_nodes[mask_finite]) / scale
        weights[mask_finite] = ecdf(x_nodes)

    # Per k = N (cioè x -> infinito), il peso è 1.0
    weights[~mask_finite] = 1.0

    # 3. Calcolo Base e Prodotto Scalare
    basis = get_bernstein_basis(z, N)  # Shape (M, N+1)

    return basis @ weights


def calculate_bernstein_exp_pdf(ecdf, N, asse_x, scale=1.0):
    """
    Calcola la PDF su [0, inf) usando la regola della catena.
    PDF_x = PDF_z * |dz/dx|
    """
    asse_x = np.asarray(asse_x)

    # 1. Trasformazione x -> z
    z = 1.0 - np.exp(-scale * asse_x)
    z = np.clip(z, 0.0, 1.0)

    # 2. Calcolo differenze dei pesi (come nel caso limitato)
    k_range = np.arange(N + 1)
    z_nodes = k_range / N

    weights = np.zeros(N + 1)
    mask_finite = (z_nodes < 1.0)
    if np.any(mask_finite):
        x_nodes = -np.log(1.0 - z_nodes[mask_finite]) / scale
        weights[mask_finite] = ecdf(x_nodes)
    weights[~mask_finite] = 1.0

    diffs = np.diff(weights)  # Shape (N,)

    # 3. Base di grado N-1
    basis_deriv = get_bernstein_basis(z, N - 1)

    # 4. Calcolo derivata rispetto a z
    # d/dz (Bernstein) = N * sum(diffs * basis_N-1)
    pdf_z = N * (basis_deriv @ diffs)

    # 5. Regola della catena (Jacobiano)
    # z = 1 - e^(-sx)  -> dz/dx = s * e^(-sx)
    # Nota: e^(-sx) è esattamente (1 - z)
    jacobian = scale * np.exp(-scale * asse_x)

    return pdf_z * jacobian
