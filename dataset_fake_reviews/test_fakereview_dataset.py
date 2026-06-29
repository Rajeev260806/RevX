import pandas as pd

df = pd.read_csv("fake reviews dataset.csv")

print(df.shape)
print(df.columns.tolist())
print(df.head(10))
print(df['label'].value_counts())   # confirm CG vs OR balance