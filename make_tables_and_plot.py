import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("simulation_results.csv")

def pivot_table(param):
    sub = df[df["parameter"] == param].copy()
    tab = sub.pivot_table(
        index=["scenario", "n"],
        columns="estimator",
        values=["bias", "variance", "mse"]
    )
    tab = tab.reset_index()
    tab.to_csv(f"table_{param}.csv", index=False)
    return tab

tab_k = pivot_table("k")
tab_theta = pivot_table("theta")

print("Saved: table_k.csv, table_theta.csv")
print("Preview table_k.csv:")
print(tab_k.head(6))


for sc in df["scenario"].unique():
    for param in ["k", "theta"]:
        sub = df[(df["scenario"] == sc) & (df["parameter"] == param)].copy()
        mom = sub[sub["estimator"] == "MoM"].sort_values("n")
        mle = sub[sub["estimator"] == "MLE"].sort_values("n")

        plt.figure()
        plt.plot(mom["n"], mom["mse"], marker="o", label="MoM")
        plt.plot(mle["n"], mle["mse"], marker="o", label="MLE")
        plt.xlabel("Sample size n")
        plt.ylabel(f"MSE of {param} estimator")
        plt.title(f"{sc}: MSE vs n for {param}")
        plt.legend()
        plt.tight_layout()
        fname = f"mse_{param}_{sc.split()[1].lower()}.png"
        plt.savefig(fname, dpi=200)
        print("Saved:", fname)
