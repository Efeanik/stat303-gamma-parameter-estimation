import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma

data = pd.read_csv("observed_gamma_data.csv")
x = data["x"].to_numpy()

summ = pd.read_csv("observed_estimates_summary.csv").iloc[0]
k_mom = float(summ["k_MoM"])
th_mom = float(summ["theta_MoM"])
k_mle = float(summ["k_MLE"])
th_mle = float(summ["theta_MLE"])

xmin, xmax = 0.0, np.max(x) * 1.05
grid = np.linspace(xmin, xmax, 500)

pdf_mom = gamma(a=k_mom, scale=th_mom).pdf(grid)
pdf_mle = gamma(a=k_mle, scale=th_mle).pdf(grid)

plt.figure()
plt.hist(x, bins=20, density=True, edgecolor="black")
plt.plot(grid, pdf_mom, label="Fitted Gamma (MoM)")
plt.plot(grid, pdf_mle, label="Fitted Gamma (MLE)")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Observed Data with Fitted Gamma Densities")
plt.legend()
plt.tight_layout()
plt.savefig("observed_fit.png", dpi=200)
print("Saved: observed_fit.png")
