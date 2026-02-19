import numpy as np
import matplotlib.pyplot as plt


# 1. Definizione della PDF di Kumaraswamy
def kumaraswamy_pdf(x, a, b):
    # Aggiungo epsilon per evitare divisioni per zero ai bordi se a<1 o b<1
    return a * b * (x ** (a - 1)) * ((1 - x ** a) ** (b - 1))


# 2. Setup dati
# Evitiamo 0 e 1 esatti per non avere infiniti numerici con a,b < 1
x = np.linspace(0.0001, 0.9999, 1000)

# Lista parametri: (a, b, label_descrizione, colore)
# Nota: Lascio i colori automatici per i primi 3, forzo il rosso per 1,1
parametri = [
    (2, 5, "Campana"),
    (1, 3, "Decrescente"),
    (0.5, 0.5, "Forma a U"),
    (1, 1, "Uniforme")  # Caso richiesto in rosso
]

plt.figure(figsize=(10, 6))

# 3. Plotting
for a, b, desc in parametri:
    y = kumaraswamy_pdf(x, a, b)

    # Gestione colori: Rosso per 1,1, default per gli altri
    if a == 1 and b == 1:
        c = 'red'
        lw = 2.5  # Lo faccio leggermente più spesso per risaltare
    else:
        c = None  # Lascia fare a matplotlib
        lw = 2

    plt.plot(x, y, label=f'a={a}, b={b}', color=c, linewidth=lw)

# 4. Styling
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.ylim(0, 2.5)

# Nessun titolo come richiesto
plt.tight_layout()
plt.show()
