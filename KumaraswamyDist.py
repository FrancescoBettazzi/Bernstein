import numpy as np

class KumaraswamyDist:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def rvs(self, size=1):
        """
        Genera campioni usando l'Inversion Sampling Method.
        F(x) = 1 - (1 - x^a)^b
        Inverse: x = (1 - (1 - u)^(1/b))^(1/a)
        """
        u = np.random.uniform(0, 1, size)
        return (1 - (1 - u) ** (1 / self.b)) ** (1 / self.a)

    def pdf(self, x):
        x = np.asarray(x)
        # La PDF è definita in (0,1), gestiamo i bordi per evitare errori numerici
        val = self.a * self.b * (x ** (self.a - 1)) * ((1 - x ** self.a) ** (self.b - 1))
        return val

    def cdf(self, x):
        x = np.asarray(x)
        return 1 - (1 - x ** self.a) ** self.b
