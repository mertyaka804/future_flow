import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

time_steps = 288
times = np.arange(time_steps)
hours = times * 5 / 60

segments = {
    "Segment 1": {"base": 20, "avm": False},
    "Segment 2": {"base": 25, "avm": False},
    "Segment 3": {"base": 30, "avm": True},
    "Segment 4": {"base": 20, "avm": False},
    "Segment 5": {"base": 15, "avm": False},
}

def traffic_pattern(hours, base, avm=False):
    traffic = np.zeros(len(hours))
    for i, h in enumerate(hours):
        if h < 6:
            val = base * 0.3
        elif 7.5 <= h <= 9.5:
            val = base * 3.5
        elif 12 <= h <= 13.5:
            val = base * 1.8
        elif 17 <= h <= 19.5:
            val = base * 4.0
        elif avm and 10 <= h <= 22:
            val = base * 2.5
        else:
            val = base * 0.8
        noise = np.random.normal(0, base * 0.1)
        traffic[i] = np.clip(val + noise, 0, 100)
    return traffic

data = {}
for seg_name, props in segments.items():
    data[seg_name] = traffic_pattern(hours, props["base"], props["avm"])

df = pd.DataFrame(data)
df["time_step"] = times
df["hour"] = hours
df.to_csv("traffic_data.csv", index=False)
print("traffic_data.csv olusturuldu.")