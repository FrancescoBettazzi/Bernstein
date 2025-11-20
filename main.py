import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from bernstein import create_cdf_gradino, calculate_bernstein_cdf, calculate_bernstein_pdf

scelta_dist = 'n'
#scelta_dist = 'u'
#scelta_dist = 'e'

M = 100  # n. campioni
N = math.ceil(M / math.log(M, 2))  # grado polinomio

distribuzione = None
nome_dist = ""

if scelta_dist == 'n':
    distribuzione = stats.norm(loc=0, scale=1)
    nome_dist = "Normale (0, 1)"
elif scelta_dist == 'u':
    distribuzione = stats.uniform(loc=5, scale=10)
    nome_dist = "Uniforme [5, 15]"
elif scelta_dist == 'e':
    distribuzione = stats.expon(loc=0, scale=1/0.5)
    nome_dist = "Esponenziale (lambda=0.5)"


array_campioni = []
array_cdf_stima = []
array_pdf_stima = []
array_cdf_stima_BM = []
array_pdf_stima_BM = []

array_asse_x = []

a_min = 0
b_max = 1
num_points = 500

for i in range(10):
    print("starting cycle n.", (i+1))
    campioni = distribuzione.rvs(size=M)
    campioni_ordinati = np.sort(campioni)
    array_campioni.append(campioni_ordinati)

    cdf_gradino = create_cdf_gradino(campioni)

    a = campioni_ordinati.min()
    b = campioni_ordinati.max()

    if a < a_min:
        a_min = a
    if b > b_max:
        b_max = b

    curr_asse_x = np.linspace(a, b, num_points)
    array_asse_x.append(curr_asse_x)

    cdf_stima = calculate_bernstein_cdf(cdf_gradino, N, a, b, curr_asse_x)
    pdf_stima = calculate_bernstein_pdf(cdf_gradino, N, a, b, curr_asse_x)

    array_cdf_stima.append(cdf_stima)
    array_pdf_stima.append(pdf_stima)

    # print("pdf_stima: ", pdf_stima)

    cdf_stima_BM = calculate_bernstein_cdf(cdf_gradino, M, a, b, curr_asse_x)
    pdf_stima_BM = calculate_bernstein_pdf(cdf_gradino, M, a, b, curr_asse_x)

    # print("pdf_stima_BM: ", pdf_stima_BM)

    array_cdf_stima_BM.append(cdf_stima_BM)
    array_pdf_stima_BM.append(pdf_stima_BM)

asse_x_generale = np.linspace(a_min, b_max, num_points)

cdf_vera = distribuzione.cdf(asse_x_generale)
pdf_vera = distribuzione.pdf(asse_x_generale)

print("Plotting...")
plt.figure(figsize=(20, 12))
nrows = 2
ncols = 2

# Primo plot: CDF N = M
index = 1
plt.subplot(nrows, ncols, index)
plt.title(f"Confronto CDF - {nome_dist} - M={M}, N={M}")
plt.plot(asse_x_generale, cdf_vera, 'k-', linewidth=3, label='CDF Teorica')
for i, y_cdf_stima_BM in enumerate(array_cdf_stima_BM):
    plt.plot(array_asse_x[i], y_cdf_stima_BM, 'k-', linewidth=0.5)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlabel("x")
plt.ylabel("y")

# Secondo plot: CDF N = M / log(M)
index = 2
plt.subplot(nrows, ncols, index)
plt.title(f"Confronto CDF - {nome_dist} - M={M}, N={N}")
plt.plot(asse_x_generale, cdf_vera, 'k-', linewidth=3, label='CDF Teorica')
for i, y_cdf_stima in enumerate(array_cdf_stima):
    plt.plot(array_asse_x[i], y_cdf_stima, 'k-', linewidth=0.5)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlabel("x")
plt.ylabel("y")

# Terzo plot: PDF N = M
index = 3
plt.subplot(nrows, ncols, index)
plt.title(f"Confronto PDF - {nome_dist} - M={M}, N={M}")
plt.plot(asse_x_generale, pdf_vera, 'k-', linewidth=3, label='PDF Teorica')
for i, y_pdf_stima_BM in enumerate(array_pdf_stima_BM):
    plt.plot(array_asse_x[i], y_pdf_stima_BM, 'k-', linewidth=0.5)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlabel("x")
plt.ylabel("y")

# Terzo plot: PDF N = M / log(M)
index = 4
plt.subplot(nrows, ncols, index)
plt.title(f"Confronto PDF - {nome_dist} - M={M}, N={N}")
plt.plot(asse_x_generale, pdf_vera, 'k-', linewidth=3, label='PDF Teorica')
for i, y_pdf_stima in enumerate(array_pdf_stima):
    plt.plot(array_asse_x[i], y_pdf_stima, 'k-', linewidth=0.5)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlabel("x")
plt.ylabel("y")

# SHOW PLOT
plt.tight_layout()
plt.show()
