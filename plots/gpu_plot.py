import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("gpu_metrics.csv")

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Plot GPU Utilization
plt.figure(figsize=(10, 6))
for gpu_id in data['gpu_id'].unique():
    gpu_data = data[data['gpu_id'] == gpu_id]
    plt.plot(gpu_data['timestamp'], gpu_data['utilization'], label=f"GPU {gpu_id}")

plt.title("GPU Utilization Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Utilization (%)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
