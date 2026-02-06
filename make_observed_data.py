import numpy as np
import pandas as pd

np.random.seed(250)

k = 2.0
theta = 18.0
n = 100

x = np.random.gamma(shape=k, scale=theta, size=n)
pd.DataFrame({"x": x}).to_csv("observed_gamma_data.csv", index=False)

print("Saved: observed_gamma_data.csv")
print("n =", n)
print("mean =", x.mean())
print("var =", x.var(ddof=1))
