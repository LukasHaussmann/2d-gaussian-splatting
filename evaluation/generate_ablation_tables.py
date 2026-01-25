#!/usr/bin/env python3
"""
Generate LaTeX tables and figures for ablation study results.

Creates:
1. Grouped ablation tables (by category)
2. Per-component contribution analysis
3. Parameter sensitivity plots

Usage:
    python -m evaluation.generate_ablation_tables [--results-dir RESULTS_DIR]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.utils.io_utils import (
    load_config,
    ensure_dir,
    format_time_budget,
)
from evaluation.utils.latex_utils import (
    format_metric,
    format_metric_with_std,
    bold,
    underline,
    make_table_header,
    make_table_footer,
    make_table_row,
    escape_latex,
    find_best_and_second,
)


def load_ablation_results(results_dir: Path) -> pd.DataFrame:
    """Load ablation results from CSV file."""
    results_file = results_dir / "ablation_results.csv"

    if not results_file.exists():
        print(f"Warning: Results file not found: {results_file}")
        return pd.DataFrame()

    df = pd.read_csv(results_file)
    return df


def load_ablation_config(results_dir: Path) -> Dict:
    """Load ablation configuration."""
    config_file = results_dir / "ablation_config.yaml"

    if config_file.exists():
        return load_config(config_file)

    # Fall back to default config
    default_config = Path(__file__).parent / "config" / "ablation_config.yaml"
    if default_config.exists():
        return load_config(default_config)

    return {}


def compute_ablation_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std for each ablation across scenes."""
    metrics = ['accuracy', 'completeness', 'overall']

    stats = df.groupby('ablation').agg({
        metric: ['mean', 'std', 'count'] for metric in metrics
    }).reset_index()

    # Flatten column names
    stats.columns = ['ablation'] + [
        f"{col[0]}_{col[1]}" for col in stats.columns[1:]
    ]

    return stats


def compute_relative_change(stats: pd.DataFrame, baseline: str = 'base') -> pd.DataFrame:
    """Compute relative change from baseline for each ablation."""
    if baseline not in stats['ablation'].values:
        print(f"Warning: Baseline '{baseline}' not found in results")
        return stats

    baseline_row = stats[stats['ablation'] == baseline].iloc[0]

    for metric in ['accuracy', 'completeness', 'overall']:
        baseline_val = baseline_row[f'{metric}_mean']
        stats[f'{metric}_change'] = (
            (stats[f'{metric}_mean'] - baseline_val) / baseline_val * 100
        )

    return stats


def generate_ablation_table(
    stats: pd.DataFrame,
    config: Dict,
    output_path: Path,
    group: Optional[str] = None
) -> str:
    """
    Generate LaTeX ablation table.

    Parameters
    ----------
    stats : DataFrame
        Ablation statistics
    config : dict
        Ablation configuration
    output_path : Path
        Output file path
    group : str, optional
        Specific ablation group to include
    """
    ablations_config = config.get('ablations', {})
    groups_config = config.get('ablation_groups', {})

    # Determine which ablations to include
    if group and group in groups_config:
        ablation_names = ['base'] + groups_config[group]['experiments']
        group_title = groups_config[group]['name']
    else:
        ablation_names = ['base'] + list(ablations_config.keys())
        group_title = "All Ablations"

    # Filter stats to included ablations
    stats_filtered = stats[stats['ablation'].isin(ablation_names)].copy()

    if stats_filtered.empty:
        print(f"No data for ablations: {ablation_names}")
        return ""

    # Get display names
    def get_display_name(ablation):
        if ablation == 'base':
            return 'Ours (Full Method)'
        return ablations_config.get(ablation, {}).get('name', ablation)

    columns = [
        "Configuration",
        "Acc. $\\downarrow$",
        "Comp. $\\downarrow$",
        "Overall $\\downarrow$",
        "$\\Delta$ Overall"
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{Ablation Study: {group_title}. "
        r"Chamfer distance (mm). $\Delta$ shows relative change from full method.}",
        f"\\label{{tab:ablation_{group or 'all'}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]

    # Find best values for highlighting
    metrics = ['accuracy', 'completeness', 'overall']
    best_indices = {}
    for metric in metrics:
        values = stats_filtered[f'{metric}_mean'].tolist()
        best_idx, _ = find_best_and_second(values, lower_is_better=True)
        best_indices[metric] = best_idx

    # Add rows
    for idx, (_, row) in enumerate(stats_filtered.iterrows()):
        ablation = row['ablation']
        display_name = get_display_name(ablation)

        row_values = [escape_latex(display_name)]

        for metric in metrics:
            mean = row[f'{metric}_mean']
            std = row[f'{metric}_std']
            is_best = (idx == best_indices[metric])

            if is_best:
                row_values.append(bold(f"{mean:.2f}") + f" $\\pm$ {std:.2f}")
            else:
                row_values.append(f"{mean:.2f} $\\pm$ {std:.2f}")

        # Add delta column
        if 'overall_change' in row:
            change = row['overall_change']
            if ablation == 'base':
                row_values.append("-")
            elif change > 0:
                row_values.append(f"\\textcolor{{red}}{{+{change:.1f}\\%}}")
            else:
                row_values.append(f"\\textcolor{{green}}{{{change:.1f}\\%}}")
        else:
            row_values.append("-")

        # Bold the base row
        if ablation == 'base':
            lines.append(r"\rowcolor{gray!15}")

        lines.append(" & ".join(row_values) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    table_str = "\n".join(lines)

    ensure_dir(output_path.parent)
    output_path.write_text(table_str)
    print(f"Saved ablation table to: {output_path}")

    return table_str


def generate_combined_ablation_table(
    stats: pd.DataFrame,
    config: Dict,
    output_path: Path
) -> str:
    """Generate a single table with all ablations organized by group."""
    ablations_config = config.get('ablations', {})
    groups_config = config.get('ablation_groups', {})

    def get_display_name(ablation):
        if ablation == 'base':
            return 'Ours (Full Method)'
        return ablations_config.get(ablation, {}).get('name', ablation)

    columns = [
        "Configuration",
        "Accuracy $\\downarrow$",
        "Completeness $\\downarrow$",
        "Overall $\\downarrow$"
    ]

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Ablation study results on DTU benchmark. "
        r"Chamfer distance metrics (mm). Lower is better. "
        r"Best result in each group shown in \textbf{bold}.}",
        r"\label{tab:ablation_full}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]

    # Add base configuration first
    base_row = stats[stats['ablation'] == 'base']
    if not base_row.empty:
        row = base_row.iloc[0]
        lines.append(r"\rowcolor{gray!15}")
        row_values = [
            bold(escape_latex(get_display_name('base'))),
            bold(f"{row['accuracy_mean']:.2f}") + f" $\\pm$ {row['accuracy_std']:.2f}",
            bold(f"{row['completeness_mean']:.2f}") + f" $\\pm$ {row['completeness_std']:.2f}",
            bold(f"{row['overall_mean']:.2f}") + f" $\\pm$ {row['overall_std']:.2f}",
        ]
        lines.append(" & ".join(row_values) + r" \\")
        lines.append(r"\midrule")

    # Add groups
    for group_name, group_info in groups_config.items():
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{group_info['name']}}}}} \\\\")

        group_ablations = group_info['experiments']
        group_stats = stats[stats['ablation'].isin(group_ablations)]

        if group_stats.empty:
            lines.append("\\multicolumn{4}{c}{\\textit{No data}} \\\\")
            continue

        # Find best in group
        best_idx = group_stats['overall_mean'].idxmin()

        for _, row in group_stats.iterrows():
            ablation = row['ablation']
            is_best = (row.name == best_idx)

            row_values = [escape_latex(get_display_name(ablation))]

            for metric in ['accuracy', 'completeness', 'overall']:
                mean = row[f'{metric}_mean']
                std = row[f'{metric}_std']
                formatted = f"{mean:.2f} $\\pm$ {std:.2f}"
                if is_best:
                    formatted = bold(f"{mean:.2f}") + f" $\\pm$ {std:.2f}"
                row_values.append(formatted)

            lines.append(" & ".join(row_values) + r" \\")

        lines.append(r"\midrule")

    # Remove last midrule and add bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")

    lines.extend([
        r"\end{tabular}",
        r"\end{table*}",
    ])

    table_str = "\n".join(lines)

    ensure_dir(output_path.parent)
    output_path.write_text(table_str)
    print(f"Saved combined ablation table to: {output_path}")

    return table_str


def plot_ablation_comparison(
    stats: pd.DataFrame,
    config: Dict,
    output_path: Path
) -> None:
    """Create bar chart comparing ablations to baseline."""
    ablations_config = config.get('ablations', {})

    # Exclude base from comparison
    stats_no_base = stats[stats['ablation'] != 'base'].copy()

    if stats_no_base.empty:
        print("No ablation data to plot")
        return

    # Get base values
    base_row = stats[stats['ablation'] == 'base']
    if base_row.empty:
        print("No baseline data found")
        return
    base_overall = base_row.iloc[0]['overall_mean']

    # Sort by overall change
    if 'overall_change' in stats_no_base.columns:
        stats_no_base = stats_no_base.sort_values('overall_change')

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    ablations = stats_no_base['ablation'].tolist()
    changes = stats_no_base['overall_change'].tolist() if 'overall_change' in stats_no_base.columns else [0] * len(ablations)

    # Color based on positive/negative change
    colors = ['#d62728' if c > 0 else '#2ca02c' for c in changes]

    # Get display names
    def get_display_name(ablation):
        return ablations_config.get(ablation, {}).get('name', ablation)

    display_names = [get_display_name(a) for a in ablations]

    y_pos = np.arange(len(ablations))
    bars = ax.barh(y_pos, changes, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)
    ax.set_xlabel('Change in Overall Chamfer Distance (%)')
    ax.set_title('Ablation Study: Impact of Each Component')
    ax.axvline(x=0, color='black', linewidth=1)

    # Add value labels
    for bar, change in zip(bars, changes):
        width = bar.get_width()
        label_x = width + 0.5 if width >= 0 else width - 0.5
        ha = 'left' if width >= 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{change:+.1f}%', ha=ha, va='center', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', label='Worse than full method'),
        Patch(facecolor='#2ca02c', label='Better than full method'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"Saved ablation comparison plot to: {output_path}")


def plot_parameter_sensitivity(
    df: pd.DataFrame,
    config: Dict,
    output_dir: Path
) -> None:
    """Create parameter sensitivity plots for numeric parameters."""
    ablations_config = config.get('ablations', {})

    # Group ablations by parameter type
    param_groups = {
        'matches_per_ref': ['sparse_matches', 'dense_matches'],
        'normal_estimate_knn': ['small_normal_knn', 'large_normal_knn'],
        'nns_per_ref': ['nns_1', 'nns_5'],
        'scaling_factor': ['small_scale', 'large_scale'],
    }

    # Get base config values
    base_config = config.get('base_config', {})

    for param, ablations in param_groups.items():
        # Collect data points
        points = []

        # Add base value
        if param in base_config:
            base_val = base_config[param]
            base_df = df[df['ablation'] == 'base']
            if not base_df.empty:
                points.append({
                    'value': base_val,
                    'overall': base_df['overall'].mean(),
                    'std': base_df['overall'].std(),
                    'label': 'base'
                })

        # Add ablation values
        for ablation in ablations:
            if ablation in ablations_config:
                changes = ablations_config[ablation].get('changes', {})
                if param in changes:
                    ablation_df = df[df['ablation'] == ablation]
                    if not ablation_df.empty:
                        points.append({
                            'value': changes[param],
                            'overall': ablation_df['overall'].mean(),
                            'std': ablation_df['overall'].std(),
                            'label': ablation
                        })

        if len(points) < 2:
            continue

        # Sort by parameter value
        points.sort(key=lambda x: x['value'])

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 5))

        x = [p['value'] for p in points]
        y = [p['overall'] for p in points]
        yerr = [p['std'] for p in points]

        ax.errorbar(x, y, yerr=yerr, marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)

        # Highlight base value
        base_point = next((p for p in points if p['label'] == 'base'), None)
        if base_point:
            ax.axvline(x=base_point['value'], color='green', linestyle='--', alpha=0.7, label='Selected value')

        ax.set_xlabel(param.replace('_', ' ').title())
        ax.set_ylabel('Overall Chamfer Distance (mm)')
        ax.set_title(f'Parameter Sensitivity: {param.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_path = output_dir / f"sensitivity_{param}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
        plt.close()

        print(f"Saved sensitivity plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ablation study tables and figures")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="evaluation_results/ablation_study",
        help="Path to ablation results directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results_dir/tables and results_dir/figures)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    tables_dir = Path(args.output_dir) / "tables" if args.output_dir else results_dir / "tables"
    figures_dir = Path(args.output_dir) / "figures" if args.output_dir else results_dir / "figures"

    ensure_dir(tables_dir)
    ensure_dir(figures_dir)

    print(f"Loading results from: {results_dir}")

    # Load data
    df = load_ablation_results(results_dir)
    config = load_ablation_config(results_dir)

    if df.empty:
        print("No ablation results found. Run ablation study first.")
        return

    # Compute statistics
    stats = compute_ablation_statistics(df)
    stats = compute_relative_change(stats)

    # Save statistics
    stats.to_csv(results_dir / "ablation_statistics.csv", index=False)
    print(f"Saved statistics to: {results_dir / 'ablation_statistics.csv'}")

    # Generate tables
    print("\n--- Generating Ablation Tables ---")

    # Combined table
    generate_combined_ablation_table(stats, config, tables_dir / "ablation_full.tex")

    # Per-group tables
    groups = config.get('ablation_groups', {})
    for group_name in groups:
        generate_ablation_table(
            stats, config,
            tables_dir / f"ablation_{group_name}.tex",
            group=group_name
        )

    # Generate figures
    print("\n--- Generating Ablation Figures ---")
    plot_ablation_comparison(stats, config, figures_dir / "ablation_comparison.png")
    plot_parameter_sensitivity(df, config, figures_dir)

    print(f"\n{'='*60}")
    print(f"Ablation analysis complete.")
    print(f"Tables saved to: {tables_dir}")
    print(f"Figures saved to: {figures_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
