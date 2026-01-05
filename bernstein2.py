import numpy as np
from scipy.interpolate import interp1d
from scipy.special import comb


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


def get_bernstein_basis(z, N):
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
    Formula: PDF_N(x) = (N / (b-a)) * SOMMA_{0}^{N-1} [ (F((k+1)/N) - F(k/N)) * b_{N-1, k}(z) ]
    """
    # 1. Normalizzazione
    asse_x = np.asarray(asse_x)
    z = np.clip((asse_x - a) / (b - a), 0.0, 1.0)

    # 2. Calcolo differenze dei pesi (coefficienti della derivata)
    # Valutiamo la CDF in 0/N, ..., N/N
    k_range = np.arange(N + 1)
    eval_points = a + (b - a) * (k_range / N)
    F_vals = ecdf(eval_points)

    # diffs[k] = F((k+1)/N) - F(k/N). Shape: (N,)
    diffs = np.diff(F_vals)

    # 3. Calcolo Base di grado N-1
    # Nota: la derivata di un polinomio di Bernstein di grado N è espressa
    # usando basi di grado N-1
    basis_deriv = get_bernstein_basis(z, N - 1)

    # 4. Somma pesata e fattore di scala (regola della catena dz/dx)
    # Scale factor: N * dz/dx = N * (1/(b-a))
    scale_factor = N / (b - a)

    return (basis_deriv @ diffs) * scale_factor


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
