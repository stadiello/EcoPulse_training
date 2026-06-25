import pandas as pd

df = pd.read_csv("data/ecopulse_esc50_yamnet_soft.csv")

cols = ["soft_bird", "soft_human", "soft_motor", "soft_nature"]

print(df[["category", "eco_label"] + cols].head(30))

print("\nMoyennes par classe réelle :")
print(df.groupby("eco_label")[cols].mean())

print("\nClasse teacher dominante :")
df["teacher_pred"] = df[cols].idxmax(axis=1).str.replace("soft_", "")
print(pd.crosstab(df["eco_label"], df["teacher_pred"]))