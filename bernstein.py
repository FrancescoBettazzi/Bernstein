import numpy as np
from scipy.interpolate import interp1d
from scipy.special import comb


def create_cdf_gradino(campioni):
    M = len(campioni)
    campioni_ordinati = np.sort(campioni)
    y_gradino = np.arange(1, M + 1) / M

    cdf_gradino = interp1d(campioni_ordinati, y_gradino, kind='previous', bounds_error=False, fill_value=(0.0, 1.0))

    return cdf_gradino


# B_N(cdf, x) = SOMMA(n=0,..,N){cdf(n/N)*binomiale(N n)*(x^n)*(1-x)^(N-n)}
def calculate_bernstein_cdf(cdf_gradino, N, a, b, asse_x):
    B_N_cdf = []
    for x in asse_x:
        z = (x - a) / (b - a)  # perché supporto finito [a, b]
        z = np.clip(z, 0.0, 1.0)
        somma = 0.0
        for n in range(N + 1):  # n = 0, ... , N
            # somma += cdf_gradino(n / N) * comb(N, n) * (z ** n) * ((1 - z) ** (N - n))
            # NB: la x deve essere tra a e b, quindi va usato a + (b-a)*(n/N)
            somma += cdf_gradino(a + (b - a) * (n / N)) * comb(N, n) * (z ** n) * ((1 - z) ** (N - n))

        B_N_cdf.append(somma)

    return B_N_cdf

# b_N(cdf, x) = N * SOMMA(n=0,...,N-1){(cdf[(k+1)/n]-cdf[k/n])*binomiale(N n)*(x^n)*(1-x)^(N-n)}
def calculate_bernstein_pdf(cdf_gradino, N, a, b, asse_x):
    B_N_pdf = []
    for x in asse_x:
        z = (x - a) / (b - a)  # perché supporto finito [a, b]
        z = np.clip(z, 0.0, 1.0)
        somma = 0.0
        for n in range(N):  # n = 0, ... , N - 1
            somma += (cdf_gradino(a + (b - a) * ((n + 1) / N)) - cdf_gradino(a + (b - a) * (n / N))) * comb(N, n) * (z ** n) * ((1 - z) ** (N - n))

        fattore_di_scala = (N / (b - a))
        # NB: da formula sarebbe N, se z fosse definito in [0,1]. Ma siccome ho che z = (x - a) / (b - a) => devo moltiplicare N per la derivata di z [1/(b-a)]
        B_N_pdf.append(fattore_di_scala * somma)

    return B_N_pdf
