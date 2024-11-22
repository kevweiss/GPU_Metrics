import subprocess
import os
import time
import threading

# Global stop event
stop_monitoring = threading.Event()

def monitor_gpu_performance():
    """
    Monitors GPU performance using nvidia-smi and logs data to specified files.

    Logs:
    - GPU power usage and other metrics to `./GPU_Metrics/plots/gpu_power.log`
    - GPU utilization and memory metrics to `./GPU_Metrics/plots/gpu_usage.log`
    """
    # Define the output directory relative to the current script
    output_dir = os.path.join(os.getcwd(), "GPU_Metrics", "plots")
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    # File paths for logs
    output_files = [os.path.join(output_dir, "gpu_power.log"), os.path.join(output_dir, "gpu_usage.log")]

    commands = [
        "nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,power.draw --format=csv,nounits -l 1",
        "nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,nounits -l 1"
    ]

    processes = []
    for cmd, file in zip(commands, output_files):
        output = open(file, "w")
        proc = subprocess.Popen(cmd, shell=True, stdout=output, stderr=subprocess.PIPE)
        processes.append((proc, output))

    try:
        print("Monitoring GPU performance... Press Ctrl+C to stop.")
        while not stop_monitoring.is_set():
            time.sleep(1)  # Keep the script running to monitor performance
    finally:
        for proc, output in processes:
            proc.terminate()
            output.close()
        print(f"Logs saved to {output_files[0]} and {output_files[1]}.")

def stop_monitoring_after_delay(delay):
    """
    Stops GPU monitoring after a given delay.
    This is useful for testing or limiting the monitoring duration.
    """
    time.sleep(delay)
    stop_monitoring.set()
