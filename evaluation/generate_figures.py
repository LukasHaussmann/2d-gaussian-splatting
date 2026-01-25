#!/usr/bin/env python3
"""
Generate publication-ready figures from evaluation results.

Creates figures for:
1. Convergence curves: Quality vs Time with std shading
2. Multi-metric plot: 3-panel (Accuracy, Completeness, Overall)
3. Speedup bar chart: Time to reach quality targets
4. Per-scene comparison: Grouped bar chart
5. Mesh comparison grid: Visual quality comparison

Usage:
    python -m evaluation.generate_figures [--results-dir RESULTS_DIR]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.utils.io_utils import (
    load_config,
    ensure_dir,
    get_default_config_path,
    format_time_budget,
)

# Publication-quality matplotlib settings
def setup_publication_style(config: Optional[Dict] = None):
    """Configure matplotlib for publication-quality figures."""
    if config is None:
        config = {}

    style = config.get('figure_style', {})

    plt.rcParams.update({
        'font.family': style.get('font_family', 'serif'),
        'font.size': style.get('font_size', 12),
        'axes.titlesize': style.get('title_size', 14),
        'axes.labelsize': style.get('font_size', 12),
        'xtick.labelsize': style.get('font_size', 12) - 2,
        'ytick.labelsize': style.get('font_size', 12) - 2,
        'legend.fontsize': style.get('legend_size', 10),
        'figure.dpi': style.get('dpi', 300),
        'savefig.dpi': style.get('dpi', 300),
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.grid': True,
        'grid.alpha': style.get('grid_alpha', 0.3),
        'axes.axisbelow': True,
        'lines.linewidth': style.get('line_width', 2.0),
        'lines.markersize': style.get('marker_size', 8),
    })


# Color scheme
COLORS = {
    'ours': '#1f77b4',  # Blue
    'baseline_2dgs': '#ff7f0e',  # Orange
    '2DGS': '#ff7f0e',
}

MARKERS = {
    'ours': 'o',
    'baseline_2dgs': 's',
    '2DGS': 's',
}

METHOD_NAMES = {
    'ours': 'Ours (Fast Init)',
    'baseline_2dgs': '2DGS Baseline',
}


def plot_convergence_curves(
    results_dir: Path,
    output_path: Path,
    metric: str = 'overall'
) -> None:
    """
    Plot convergence curves: Quality vs Time with std shading.

    Shows how quality improves with training time for each method.
    """
    summary_file = results_dir / "results" / "time_comparison_summary.csv"

    if not summary_file.exists():
        print(f"Warning: Summary file not found: {summary_file}")
        return

    df = pd.read_csv(summary_file)

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    methods = sorted(df['method'].unique())

    for method in methods:
        method_df = df[df['method'] == method].sort_values('time_budget')

        x = method_df['time_budget'].values / 60  # Convert to minutes
        y = method_df[f'{metric}_mean'].values
        #yerr = method_df[f'{metric}_std'].values

        color = COLORS.get(method, '#333333')
        marker = MARKERS.get(method, 'o')
        label = METHOD_NAMES.get(method, method)

        # Plot line with markers
        ax.plot(x, y, color=color, marker=marker, label=label, linewidth=2, markersize=8)

        # Add shaded std region
        #ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

    ax.set_xlabel('Training Time (minutes)')
    ax.set_ylabel(f'{metric.capitalize()} (mm)')
    ax.set_title(f'Convergence: {metric.capitalize()} vs Training Time')
    ax.legend(loc='upper right')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Use integer x-axis ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ensure_dir(output_path.parent)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved convergence plot to: {output_path}")


def plot_multi_metric_convergence(
    results_dir: Path,
    output_path: Path
) -> None:
    """
    Create 3-panel plot showing all metrics (Accuracy, Completeness, Overall).
    """
    summary_file = results_dir / "results" / "time_comparison_summary.csv"

    if not summary_file.exists():
        print(f"Warning: Summary file not found: {summary_file}")
        return

    df = pd.read_csv(summary_file)

    if df.empty:
        return

    metrics = ['accuracy', 'completeness', 'overall']
    metric_titles = ['Accuracy', 'Completeness', 'Overall']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = sorted(df['method'].unique())

    for ax, metric, title in zip(axes, metrics, metric_titles):
        for method in methods:
            method_df = df[df['method'] == method].sort_values('time_budget')

            x = method_df['time_budget'].values / 60
            y = method_df[f'{metric}_mean'].values
            #yerr = method_df[f'{metric}_std'].values

            color = COLORS.get(method, '#333333')
            marker = MARKERS.get(method, 'o')
            label = METHOD_NAMES.get(method, method)

            ax.plot(x, y, color=color, marker=marker, label=label, linewidth=2, markersize=6)
            #ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)

        ax.set_xlabel('Time (min)')
        ax.set_ylabel(f'{title} (mm)')
        ax.set_title(title)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(methods), bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    ensure_dir(output_path.parent)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved multi-metric plot to: {output_path}")


def plot_speedup_bars(
    results_dir: Path,
    output_path: Path,
    quality_thresholds: List[float] = [0.60, 0.55, 0.50, 0.45]
) -> None:
    """
    Create bar chart showing speedup at different quality targets.
    """
    time_results_file = results_dir / "results" / "time_based_results.csv"

    if not time_results_file.exists():
        print(f"Warning: Time results file not found: {time_results_file}")
        return

    df = pd.read_csv(time_results_file)

    if df.empty:
        return

    time_budgets = sorted(df['time_budget'].unique())

    speedups = []
    labels = []

    for threshold in quality_thresholds:
        times = {}
        for method in ['ours', 'baseline_2dgs']:
            method_df = df[df['method'] == method]

            time_to_reach = None
            for t in time_budgets:
                t_df = method_df[method_df['time_budget'] == t]
                if not t_df.empty:
                    mean_overall = t_df['overall'].mean()
                    if mean_overall <= threshold:
                        time_to_reach = t
                        break

            times[method] = time_to_reach

        if times.get('ours') and times.get('baseline_2dgs'):
            speedup = times['baseline_2dgs'] / times['ours']
            speedups.append(speedup)
            labels.append(f'≤{threshold:.2f}')
        elif times.get('ours') and not times.get('baseline_2dgs'):
            speedups.append(float('inf'))
            labels.append(f'≤{threshold:.2f}')

    if not speedups:
        print("No speedup data available")
        return

    # Filter out infinite speedups for plotting
    finite_mask = [s != float('inf') for s in speedups]
    plot_speedups = [s if s != float('inf') else 0 for s in speedups]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(labels))
    bars = ax.bar(x, plot_speedups, color=COLORS['ours'], edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bar, speedup, finite in zip(bars, speedups, finite_mask):
        height = bar.get_height()
        if finite:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{speedup:.1f}×',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., 0.1,
                    '∞',
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Target Quality (Overall Chamfer Distance, mm)')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('Speedup: Our Method vs 2DGS Baseline')

    # Add horizontal line at 1x
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Baseline (1×)')

    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right')

    ensure_dir(output_path.parent)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved speedup bar chart to: {output_path}")


def plot_per_scene_comparison(
    results_dir: Path,
    output_path: Path
) -> None:
    """
    Create grouped bar chart comparing methods per scene.
    """
    time_results_file = results_dir / "results" / "time_based_results.csv"

    if not time_results_file.exists():
        print(f"Warning: Time results file not found: {time_results_file}")
        return

    df = pd.read_csv(time_results_file)

    if df.empty:
        return

    # Get final time budget results
    max_time = df['time_budget'].max()
    final_df = df[df['time_budget'] == max_time]

    scenes = sorted(final_df['scan'].unique())
    methods = ['ours', 'baseline_2dgs']

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(scenes))
    width = 0.35

    for i, method in enumerate(methods):
        method_df = final_df[final_df['method'] == method]
        values = []
        for scene in scenes:
            scene_df = method_df[method_df['scan'] == scene]
            if not scene_df.empty:
                values.append(scene_df['overall'].values[0])
            else:
                values.append(np.nan)

        offset = (i - 0.5) * width
        color = COLORS.get(method, '#333333')
        label = METHOD_NAMES.get(method, method)
        ax.bar(x + offset, values, width, label=label, color=color, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=45, ha='right')
    ax.set_xlabel('Scene')
    ax.set_ylabel('Overall Chamfer Distance (mm)')
    ax.set_title(f'Per-Scene Comparison (Training Time: {format_time_budget(max_time)})')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    ensure_dir(output_path.parent)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved per-scene comparison to: {output_path}")


def plot_improvement_distribution(
    results_dir: Path,
    output_path: Path
) -> None:
    """
    Create histogram showing distribution of improvements across scenes.
    """
    time_results_file = results_dir / "results" / "time_based_results.csv"

    if not time_results_file.exists():
        print(f"Warning: Time results file not found: {time_results_file}")
        return

    df = pd.read_csv(time_results_file)

    if df.empty:
        return

    # Get final time budget results
    max_time = df['time_budget'].max()
    final_df = df[df['time_budget'] == max_time]

    # Compute per-scene improvement
    ours_df = final_df[final_df['method'] == 'ours'].set_index('scan')
    baseline_df = final_df[final_df['method'] == 'baseline_2dgs'].set_index('scan')

    common_scenes = ours_df.index.intersection(baseline_df.index)

    if len(common_scenes) == 0:
        print("No common scenes found for improvement calculation")
        return

    improvements = []
    for scene in common_scenes:
        ours_val = ours_df.loc[scene, 'overall']
        baseline_val = baseline_df.loc[scene, 'overall']
        # Positive improvement means we're better (lower is better)
        improvement = (baseline_val - ours_val) / baseline_val * 100
        improvements.append(improvement)

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(min(improvements) - 5, max(improvements) + 5, 15)
    ax.hist(improvements, bins=bins, color=COLORS['ours'], edgecolor='black', alpha=0.8)

    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Improvement')
    ax.axvline(x=np.mean(improvements), color='green', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(improvements):.1f}%')

    ax.set_xlabel('Improvement over 2DGS Baseline (%)')
    ax.set_ylabel('Number of Scenes')
    ax.set_title('Distribution of Improvements Across Scenes')
    ax.legend(loc='upper left')

    ensure_dir(output_path.parent)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved improvement distribution to: {output_path}")


def create_mesh_comparison_grid(
    results_dir: Path,
    output_path: Path,
    scenes: List[str] = None,
    max_scenes: int = 6
) -> None:
    """
    Create a grid of mesh snapshots comparing methods.

    Looks for snapshot images in the experiment directories.
    """
    time_dir = results_dir / "time_based"

    if not time_dir.exists():
        print(f"Warning: Time-based directory not found: {time_dir}")
        return

    # Find latest time budget
    time_dirs = sorted([d for d in time_dir.iterdir() if d.is_dir()])
    if not time_dirs:
        print("No time budget directories found")
        return

    latest_time_dir = time_dirs[-1]

    # Find scene snapshots
    ours_dir = latest_time_dir / "ours"
    baseline_dir = latest_time_dir / "baseline_2dgs"

    if scenes is None:
        scenes = sorted([d.name for d in ours_dir.iterdir() if d.is_dir()])[:max_scenes]

    if not scenes:
        print("No scene directories found")
        return

    n_scenes = len(scenes)
    fig, axes = plt.subplots(n_scenes, 2, figsize=(8, 3 * n_scenes))

    if n_scenes == 1:
        axes = axes.reshape(1, -1)

    for row, scene in enumerate(scenes):
        for col, (method_dir, method_name) in enumerate([
            (ours_dir, 'Ours'),
            (baseline_dir, '2DGS Baseline')
        ]):
            ax = axes[row, col]

            # Look for snapshot image
            snapshot_paths = [
                method_dir / scene / "train" / "snapshot.png",
                method_dir / scene / "snapshot.png",
            ]

            snapshot_found = False
            for snapshot_path in snapshot_paths:
                if snapshot_path.exists():
                    img = plt.imread(str(snapshot_path))
                    ax.imshow(img)
                    snapshot_found = True
                    break

            if not snapshot_found:
                ax.text(0.5, 0.5, 'No snapshot', ha='center', va='center', transform=ax.transAxes)

            ax.axis('off')

            if row == 0:
                ax.set_title(method_name, fontsize=14, fontweight='bold')

            if col == 0:
                ax.text(-0.1, 0.5, scene, ha='right', va='center',
                        transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()

    ensure_dir(output_path.parent)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved mesh comparison grid to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate figures from evaluation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="evaluation_results/fast_init_convergence_study",
        help="Path to evaluation results directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: results_dir/figures)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file for styling"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    ensure_dir(output_dir)

    # Load config for styling
    config_path = args.config or get_default_config_path()
    if Path(config_path).exists():
        config = load_config(config_path)
    else:
        config = {}

    setup_publication_style(config)

    print(f"Generating figures from: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Generate all figures
    print("\n--- Generating Convergence Curves ---")
    plot_convergence_curves(results_dir, output_dir / "convergence_time.png")
    plot_convergence_curves(results_dir, output_dir / "convergence_time.pdf")

    print("\n--- Generating Multi-Metric Plot ---")
    plot_multi_metric_convergence(results_dir, output_dir / "multi_metric_convergence.png")
    plot_multi_metric_convergence(results_dir, output_dir / "multi_metric_convergence.pdf")

    print("\n--- Generating Speedup Chart ---")
    plot_speedup_bars(results_dir, output_dir / "speedup_analysis.png")
    plot_speedup_bars(results_dir, output_dir / "speedup_analysis.pdf")

    print("\n--- Generating Per-Scene Comparison ---")
    plot_per_scene_comparison(results_dir, output_dir / "per_scene_comparison.png")
    plot_per_scene_comparison(results_dir, output_dir / "per_scene_comparison.pdf")

    print("\n--- Generating Improvement Distribution ---")
    plot_improvement_distribution(results_dir, output_dir / "improvement_distribution.png")
    plot_improvement_distribution(results_dir, output_dir / "improvement_distribution.pdf")

    print("\n--- Generating Mesh Comparison Grid ---")
    create_mesh_comparison_grid(results_dir, output_dir / "mesh_comparison_grid.png")
    create_mesh_comparison_grid(results_dir, output_dir / "mesh_comparison_grid.pdf")

    print(f"\n{'='*60}")
    print(f"Figure generation complete. Files saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
