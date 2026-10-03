import numpy as np
import pandas as pd

df = pd.read_csv("traffic_data.csv")
segments = ["Segment 1", "Segment 2", "Segment 3", "Segment 4", "Segment 5"]

WINDOW = 6
HORIZON = 3

X, y = [], []

for i in range(WINDOW, len(df) - HORIZON):
    row = []
    for seg in segments:
        row.extend(df[seg].iloc[i-WINDOW:i].values)
    row.append(df["hour"].iloc[i])
    label = [df[seg].iloc[i + HORIZON] for seg in segments]
    X.append(row)
    y.append(label)

X = np.array(X)
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)
print(f"X: {X.shape}, y: {y.shape}")
print("X.npy ve y.npy olusturuldu.")