import os
import pandas as pd
import matplotlib.pyplot as plt

# === Configuration ===
DATA_DIR = "dtu_mesh_eval"   # folder containing CSVs
X_COLUMN = "iterations"       # x-axis
Y_COLUMNS = ["accuracy", "completeness", "overall"]  # the 3 metrics you want to plot
OUTPUT_FILE = "combined_plot.png"
# ======================

def main():
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("No CSV files found.")
        return

    n_metrics = len(Y_COLUMNS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5), sharex=True)

    if n_metrics == 1:
        axes = [axes]  # ensure axes is iterable

    for i, metric in enumerate(Y_COLUMNS):
        ax = axes[i]
        for file in csv_files:
            filepath = os.path.join(DATA_DIR, file)
            df = pd.read_csv(filepath)
            ax.plot(df[metric].values, label=file.replace(".csv", ""))
    
        # Set custom xticks just like your original code
        df = pd.read_csv(os.path.join(DATA_DIR, csv_files[0]))
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df[X_COLUMN].astype(str), rotation=45)
    
        ax.set_title(metric)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.set_ylim(bottom=0)
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
