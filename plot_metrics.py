import os 
import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics_from_runs(
    base_dir,
    output_file,
    x_column="iterations",
    y_columns=("accuracy", "completeness", "overall"),
):
    """
    Plot evaluation metrics from CSV files stored in subdirectories.

    Parameters
    ----------
    base_dir : str
        Root directory containing experiment subdirectories.
    output_file : str
        Path to save the output plot.
    x_column : str
        Column name for x-axis values.
    y_columns : tuple[str]
        Metric columns to plot.
    """

    csv_paths = []
    labels = []

    # --- collect CSVs from all subdirectories ---
    for name in sorted(os.listdir(base_dir)):
        subdir = os.path.join(base_dir, name)
        if not os.path.isdir(subdir):
            continue

        csv_files = [f for f in os.listdir(subdir) if f.endswith(".csv")]
        if not csv_files:
            continue

        # assume exactly one CSV per subdirectory
        csv_paths.append(os.path.join(subdir, csv_files[0]))
        labels.append(name)

    if not csv_paths:
        print("No CSV files found in subdirectories.")
        return

    # --- create subplots ---
    n_metrics = len(y_columns)
    fig, axes = plt.subplots(
        1, n_metrics, figsize=(6 * n_metrics, 5), sharex=True
    )

    if n_metrics == 1:
        axes = [axes]

    # --- reference CSV for x-axis ---
    reference_df = pd.read_csv(csv_paths[0])

    # --- plotting ---
    for i, metric in enumerate(y_columns):
        ax = axes[i]

        for csv_path, label in zip(csv_paths, labels):
            df = pd.read_csv(csv_path)
            ax.plot(df[metric].values, label=label)

        ax.set_xticks(range(len(reference_df)))
        ax.set_xticklabels(
            reference_df[x_column].astype(str), rotation=45
        )

        ax.set_title(metric)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.set_ylim(bottom=0)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()
