from matplotlib import pyplot as plt
import numpy as np

def plot_anomalies(original_data, noisy_data, anomalies, scores, title="Anomaly Detection Results"):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.scatter(np.arange(len(noisy_data)), noisy_data[:, 0], c=scores, cmap='coolwarm', alpha=0.6)
    plt.scatter(np.where(anomalies)[0], noisy_data[anomalies, 0], 
                facecolors='none', edgecolors='black', s=100, linewidths=1.5, label='Detected Anomalies')
    plt.colorbar(label='Anomaly Scores')
    plt.xlabel("Sample Index")
    plt.ylabel("Feature Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_repair_comparison(original_data, repaired_data, anomalies, title="Repair Comparison"):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.scatter(np.arange(len(original_data)), original_data[:, 0], c='blue', alpha=0.4, label='Original Data')
    plt.scatter(np.arange(len(repaired_data)), repaired_data[:, 0], c='green', alpha=0.6, label='Repaired Data')
    plt.scatter(np.where(anomalies)[0], original_data[anomalies, 0], 
                facecolors='none', edgecolors='red', s=100, linewidths=1.5, label='Anomalies')
    plt.xlabel("Sample Index")
    plt.ylabel("Feature Value")
    plt.legend()
    plt.grid(True)
    plt.show()