import pandas as pd

def fix_table(csv_path, tex_path):
    df = pd.read_csv(csv_path)

   
    if "scenario" in df.columns:
        df = df[df["scenario"].notna()]
        df = df[df["scenario"].astype(str).str.strip() != ""]

   
    rename_map = {
        "bias": "Bias (MLE)",
        "bias.1": "Bias (MoM)",
        "mse": "MSE (MLE)",
        "mse.1": "MSE (MoM)",
        "variance": "Var (MLE)",
        "variance.1": "Var (MoM)",
        "n": "$n$",
        "scenario": "Scenario"
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    
    if "$n$" in df.columns:
        df["$n$"] = pd.to_numeric(df["$n$"], errors="coerce").astype("Int64")

   
    for col in df.columns:
        if col in ["Scenario", "$n$"]:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").round(6)

    tex = df.to_latex(index=False, escape=False)

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"Saved: {tex_path}")

fix_table("table_k.csv", "table_k.tex")
fix_table("table_theta.csv", "table_theta.tex")
