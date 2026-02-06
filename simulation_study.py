import numpy as np
import pandas as pd
import math
from scipy.special import digamma, polygamma
from scipy.optimize import newton

R = 2000
sample_sizes = [20, 50, 100]

scenarios = [
    {"name": "Scenario 1 (k=1, theta=2)", "k": 1.0, "theta": 2.0},
    {"name": "Scenario 2 (k=5, theta=1)", "k": 5.0, "theta": 1.0},
]

np.random.seed(250)

def mom_estimates(x):
    xbar = x.mean()
    s2 = x.var(ddof=1)
    k_hat = (xbar**2) / s2
    theta_hat = s2 / xbar
    return k_hat, theta_hat

def mle_estimates(x):
    xbar = x.mean()
    logx_bar = np.log(x).mean()
    log_xbar = math.log(xbar)

    def eq_t(t):
        k = math.exp(t)
        return math.log(k) - digamma(k) - log_xbar + logx_bar

    def eq_t_prime(t):
        k = math.exp(t)
        return 1.0 - k * polygamma(1, k)

    k0, _ = mom_estimates(x)
    if not np.isfinite(k0) or k0 <= 0:
        k0 = 1.0

    t0 = math.log(k0)
    t_hat = newton(eq_t, x0=t0, fprime=eq_t_prime, tol=1e-10, maxiter=100)
    k_hat = math.exp(t_hat)
    theta_hat = xbar / k_hat
    return k_hat, theta_hat

def performance(estimates, true_value):
    arr = np.array(estimates, dtype=float)
    bias = arr.mean() - true_value
    var = arr.var(ddof=1)
    mse = ((arr - true_value) ** 2).mean()
    return bias, var, mse

rows = []

for sc in scenarios:
    k_true = sc["k"]
    th_true = sc["theta"]

    for n in sample_sizes:
        k_mom_list, th_mom_list = [], []
        k_mle_list, th_mle_list = [], []

        for _ in range(R):
            x = np.random.gamma(shape=k_true, scale=th_true, size=n)

            k_mom, th_mom = mom_estimates(x)
            k_mle, th_mle = mle_estimates(x)

            k_mom_list.append(k_mom); th_mom_list.append(th_mom)
            k_mle_list.append(k_mle); th_mle_list.append(th_mle)

        b, v, m = performance(k_mom_list, k_true)
        rows.append([sc["name"], n, "MoM", "k", b, v, m])

        b, v, m = performance(k_mle_list, k_true)
        rows.append([sc["name"], n, "MLE", "k", b, v, m])

        b, v, m = performance(th_mom_list, th_true)
        rows.append([sc["name"], n, "MoM", "theta", b, v, m])

        b, v, m = performance(th_mle_list, th_true)
        rows.append([sc["name"], n, "MLE", "theta", b, v, m])

df = pd.DataFrame(rows, columns=["scenario", "n", "estimator", "parameter", "bias", "variance", "mse"])
df.to_csv("simulation_results.csv", index=False)

print("Saved: simulation_results.csv")
print(df.head(12))
