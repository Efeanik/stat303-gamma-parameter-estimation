import numpy as np
import pandas as pd
import math

from scipy.special import digamma, polygamma
from scipy.optimize import newton

data = pd.read_csv("observed_gamma_data.csv")
x = data["x"].to_numpy()
n = len(x)

xbar = x.mean()
s2 = x.var(ddof=1)

k_mom = (xbar**2) / s2
theta_mom = s2 / xbar

logx_bar = np.log(x).mean()
log_xbar = math.log(xbar)

def score_eq(k):
    return math.log(k) - digamma(k) - log_xbar + logx_bar

def score_prime(k):
    return 1.0 / k - polygamma(1, k)

k0 = k_mom
k_mle = newton(score_eq, x0=k0, fprime=score_prime, tol=1e-10, maxiter=100)
theta_mle = xbar / k_mle

out = {
    "n": n,
    "sample_mean": xbar,
    "sample_variance": s2,
    "k_MoM": k_mom,
    "theta_MoM": theta_mom,
    "k_MLE": k_mle,
    "theta_MLE": theta_mle,
}

print("SUMMARY")
for k, v in out.items():
    print(f"{k}: {v}")

pd.DataFrame([out]).to_csv("observed_estimates_summary.csv", index=False)
print("Saved: observed_estimates_summary.csv")
