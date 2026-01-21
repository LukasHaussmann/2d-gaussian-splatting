import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

def plot_metrics_from_runs(experiment_root, output_file):
    csv_files = glob.glob(os.path.join(experiment_root, "**", "metrics.csv"), recursive=True)
    
    if not csv_files:
        print("No metrics.csv files found to plot.")
        return

    metrics_to_plot = ["accuracy", "completeness", "overall"]
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        found_data = False
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if metric not in df.columns:
                    continue
                
                found_data = True
                exp_dir = os.path.dirname(csv_file)
                label = os.path.basename(exp_dir)
                
                # --- NEW: RUNTIME READING ---
                runtime_file = os.path.join(exp_dir, "runtime.txt")
                if os.path.exists(runtime_file):
                    with open(runtime_file, "r") as f:
                        # Append time to legend label (e.g. "experiment (45s)")
                        content = f.read().strip()
                        try:
                            seconds = float(content)
                            label += f" ({seconds:.0f}s)"
                        except ValueError:
                            pass
                # ----------------------------

                if "iterations" in df.columns:
                    ax.plot(df["iterations"], df[metric], marker='o', label=label)
                    ax.set_xlabel("Iterations")
                else:
                    ax.plot(df[metric].values, marker='o', label=label)
                    ax.set_xlabel("Checkpoints")

            except Exception as e:
                print(f"Error plotting {csv_file}: {e}")

        ax.set_title(metric.capitalize())
        ax.grid(True)
        if found_data:
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")
    plt.close()