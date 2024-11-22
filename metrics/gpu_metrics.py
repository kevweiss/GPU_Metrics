import subprocess
import csv
import time
from datetime import datetime

def monitor_gpu(output_file, interval=5, duration=60):
    """
    Monitors GPU metrics using nvidia-smi and saves to a CSV file.

    :param output_file: Path to the CSV file to save data.
    :param interval: Time (in seconds) between each monitoring snapshot.
    :param duration: Total time (in seconds) to monitor GPU usage.
    """
    end_time = time.time() + duration
    fields = ["timestamp", "gpu_id", "utilization", "memory_usage", "temperature"]

    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fields)

        while time.time() < end_time:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                # Run nvidia-smi command
                result = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used,temperature.gpu",
                        "--format=csv,noheader,nounits"
                    ],
                    encoding="utf-8"
                )
                for line in result.strip().split("\n"):
                    gpu_data = [timestamp] + line.split(", ")
                    writer.writerow(gpu_data)
            except subprocess.CalledProcessError as e:
                print(f"Error executing nvidia-smi: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")

            time.sleep(interval)

if __name__ == "__main__":
    monitor_gpu(output_file="gpu_metrics.csv", interval=5, duration=300)  # Monitor for 5 minutes
