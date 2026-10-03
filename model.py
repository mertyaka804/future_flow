import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

X = np.load("X.npy")
y = np.load("y.npy")

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Model
model = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Sonuclar
segments = ["Segment 1", "Segment 2", "Segment 3", "Segment 4", "Segment 5"]
print("\n--- Sonuçlar ---")
for i, seg in enumerate(segments):
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    print(f"{seg} → MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Grafik
fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
for i, (seg, ax) in enumerate(zip(segments, axes)):
    ax.plot(y_test[:, i], label="Gerçek", color="steelblue")
    ax.plot(y_pred[:, i], label="Tahmin", color="red", linestyle="--")
    ax.set_title(seg)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("tahmin_sonuc.png", dpi=150, bbox_inches="tight")
print("\ntahmin_sonuc.png kaydedildi.")
import os
print(os.path.abspath("tahmin_sonuc.png"))