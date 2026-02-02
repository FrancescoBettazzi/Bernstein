# Bernstein Polynomial Density Estimation Framework

This project provides a comprehensive Python toolkit for the analysis, simulation, and benchmarking of **Density Estimators based on Bernstein Polynomials (BP)**. The framework is designed to transform discrete sample data into smooth, continuous functions for both bounded and semi-infinite domains.


---

## 🎯 Project Scope

The main goal of this work is to provide a robust method for smoothing sample distributions. Traditionally, this is achieved using **Kernel Density Estimation (KDE)**, which can be computationally expensive and sensitive to bandwidth selection.

This framework implements an alternative approach based on **Bernstein Operators**, as proposed by _Babu et al. (2002)_.

1.  **Non-Parametric Estimation:** Approximating the Probability Density Function (PDF) and Cumulative Distribution Function (CDF) without assuming a specific prior functional form.
2.  **Domain Analysis:** Handling different support types:
    * **Bounded:** Intervals like $[0, 1]$ (Standard Bernstein).
    * **Semi-Infinite:** Intervals like $[0, \infty)$ (Exponential Bernstein).
3.  **Benchmarking:** Evaluating the **Bias-Variance trade-off** by varying the polynomial degree $N$ and comparing accuracy metrics (KL Divergence, Wasserstein Distance) against state-of-the-art estimators like KDE.

---

## 📂 Key Files Documentation

The project focuses on three main execution scripts, each serving a specific analytical purpose:

### 1. `main.py` (Standard Analysis - Bounded)
This is the entry point for analyzing distributions defined on limited intervals (typically normalized to $[0, 1]$).

* **Supported Distributions:** Kumaraswamy, Uniform, and Normal (via support adaptation).
* **Core Logic:**
    1.  Generates Monte Carlo samples.
    2.  Calculates the Empirical CDF (ECDF).
    3.  Estimates the Bernstein CDF and derives the PDF.
* **Heuristic $N$:** It automatically calculates a starting polynomial degree based on sample size: $N \approx \lceil M / \log_2(M) \rceil$.
* **Outputs:**
    * **Fig 1 (CDF):** Visual comparison of the estimated CDF vs. the True CDF.
    * **Fig 2 (PDF):** Derivative analysis and Negative Log-Likelihood (NLL) boxplots.
    * **Fig 3 (Trade-off):** Curves showing how error metrics (WD, KL, NLL) change as degree $N$ increases.

### 2. `main_exp.py` (Exponential Analysis - Semi-Bounded)
A specialized version designed for distributions defined on the positive semi-axis $[0, \infty)$, utilizing **Exponential Bernstein Polynomials**.

* **Supported Distributions:** Erlang (Gamma), Weibull, Lognormal.
* **Adaptive Scaling:** The script implements a crucial transformation parameter, `scale_param = 1 / mean(samples)`. This adapts the basis functions to the specific scale of the input data.
* **Domain Management:** Dynamically creates evaluation grids and visual limits based on percentiles (e.g., 99.9%) to handle heavy tails.

### 3. `kde_vs_bernstein.py` (Comparative Benchmark)
A "Battle Royale" script that executes a direct comparison between Bernstein estimators and **Gaussian Kernel Density Estimation (KDE)** across a wide range of distributions.



* **Hybrid Estimator Selection:** The script is context-aware. It iterates through a configuration list (`DIST_KEYS`) and automatically selects the correct mathematical framework:
    * **Standard Bernstein:** Applied to bounded distributions (Uniform, Kumaraswamy).
    * **Exponential Bernstein:** Applied to semi-infinite distributions (Erlang, Weibull, Lognormal).
    * **Normal Distribution Handling:** Specifically manages the Gaussian distribution by defining an effective support based on percentiles (0.1% to 99.9%).

* **Batch Processing:** It processes multiple distribution scenarios in a single execution loop. Currently configured to test:
    * *Uniform* (`u`)
    * *Normal* (`n`)
    * *Kumaraswamy* (various shapes: `k`, `k_d`, `k_u`)
    * *Erlang* (`erlang`)
    * *Weibull* (`weibull_1_5`, `weibull_0_5`)
    * *Lognormal* (various sigmas: `lognormal_1_8`, `lognormal_0_2`, etc.)

* **Advanced Visualization (2x2 Grid):**
    * **Top Row (Spaghetti Plots):** Overlays 50 Monte Carlo runs of Bernstein curves (Left) and KDE curves (Right) against the Ground Truth (Black line). This visually demonstrates the variance of each method.
    * **Bottom Row (Error Boxplots):** Compares the distribution of Kullback-Leibler (KL) Divergence for both methods.
    * **Smart Annotation Logic:** The script employs a custom plotting function (`draw_custom_boxplot`) that detects when Mean and Median values are too close. It dynamically offsets text labels to prevent overlap, ensuring legible statistical reporting (Mean $\pm$ Std vs Median).



---

## ⚙️ Configuration and Parameters

Each `main` file contains a **"CONFIGURATION AND PARAMETERS"** section at the top of the script.

**For `kde_vs_bernstein.py` specifically:**

| Variable | Description | Example |
| :--- | :--- | :--- |
| `DIST_KEYS` | List of distribution keys to process sequentially. | `['u', 'n', 'k', 'lognormal_1_8']` |
| `M` | Number of samples generated per simulation. | `200` |
| `NUM_SIMULATIONS` | Total Monte Carlo iterations. | `100` |
| `N_PLOT_LINES` | Number of "spaghetti" lines to draw (subset of total runs). | `50` |
| `NUM_POINTS` | Resolution of the evaluation grid. | `500` |

---

## 🚀 How to Run

### Prerequisites & Environment
This project was developed using **Python 3.12** and the **PyCharm IDE**.

To set up the environment, please install the necessary dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

### Execution
The project includes three distinct scripts, each serving a specific testing purpose. You can execute them directly from your terminal.

NB: All results are saved automatically in the `img/YYYYMMDD/` directory.

### 1. Test Standard Bernstein Polynomials (BP)
Use this script to analyze distributions defined on bounded intervals (e.g., Uniform, Kumaraswamy).
```bash
python main.py
```

### 2. Test Exponential Bernstein (BE)
Use this script to analyze distributions defined on semi-infinite intervals $[0, \infty)$ (e.g., Exponential, Weibull, Lognormal).
```bash
python main_exp.py
```

### 3. Comparative Benchmark (Bernstein vs. KDE)
Run this script to perform a direct comparison between Bernstein estimators (both BP and BE) and Gaussian Kernel Density Estimation across multiple distributions.
```bash
python kde_vs_bernstein.py
```
