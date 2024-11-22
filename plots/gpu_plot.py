import pandas as pd
import matplotlib.pyplot as plt

# Load the GPU metrics log into a pandas DataFrame
df = pd.read_csv('gpu_usage_442.log')

# Display the first few rows of the DataFrame to inspect the data
print("First few rows of the data:")
print(df.head())

# Display the column names to check for any issues
print("\nColumn names in the data:")
print(df.columns)

# Strip any leading/trailing whitespace from column headers
df.columns = df.columns.str.strip()

# Check again if 'power.draw [W]' is in the columns
if 'power.draw [W]' not in df.columns:
    print("\n'power.draw [W]' column not found! Available columns:")
    print(df.columns)
else:
    print("\n'power.draw [W]' column found successfully!")

# Convert the timestamp column to a datetime type
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract start and end times
start_time = df['timestamp'].min()
end_time = df['timestamp'].max()

# Calculate total duration
duration = end_time - start_time
duration_minutes = duration.total_seconds() // 60
duration_seconds = duration.total_seconds() % 60

# Format duration for the legend
duration_label = f"Total duration: {int(duration_minutes)} min {int(duration_seconds)} s"

# Plot GPU utilization over time
plt.figure(figsize=(12, 7))
plt.plot(df['timestamp'], df['utilization.gpu [%]'], label='GPU Utilization (%)', color='b')

# Ensure the 'power.draw [W]' column exists and plot it
if 'power.draw [W]' in df.columns:
    plt.plot(df['timestamp'], df['power.draw [W]'], label='Power Draw (W)', color='r')

# Add labels and title
plt.xlabel('Timestamp')
plt.ylabel('Usage')
plt.title('A100: Large Grid GPU Utilization and Power Draw Over Time')

# Add legend
plt.legend(loc='upper left')

# Add annotation for total duration in the upper right corner
plt.gca().text(0.98, 0.98, duration_label, transform=plt.gca().transAxes,
               fontsize=12, color='black',
               ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

# Rotate the timestamp labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()

