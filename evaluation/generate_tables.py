#!/usr/bin/env python3
"""
Generate publication-ready LaTeX tables from evaluation results.

Creates tables for:
1. Time comparison table: Quality at fixed time budgets
2. SOTA comparison table: Final quality vs published methods
3. Per-scene breakdown: Individual scene results
4. Speedup summary: Time to reach quality thresholds

Usage:
    python -m evaluation.generate_tables [--results-dir RESULTS_DIR]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.utils.io_utils import (
    load_config,
    ensure_dir,
    get_sota_baselines_path,
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


def generate_time_comparison_table(
    results_dir: Path,
    output_path: Path
) -> str:
    """
    Generate LaTeX table comparing methods across time budgets.

    Columns: Method | 1min | 3min | 5min | 10min | 15min
    Rows for each metric (Accuracy, Completeness, Overall)
    """
    summary_file = results_dir / "results" / "time_comparison_summary.csv"

    if not summary_file.exists():
        print(f"Warning: Summary file not found: {summary_file}")
        return ""

    df = pd.read_csv(summary_file)

    if df.empty:
        return ""

    # Get unique time budgets and methods
    time_budgets = sorted(df['time_budget'].unique())
    methods = sorted(df['method'].unique())

    # Method display names
    method_names = {
        'ours': 'Ours (Fast Init)',
        'baseline_2dgs': '2DGS Baseline',
    }

    # Generate table
    time_labels = [format_time_budget(t) for t in time_budgets]
    columns = ["Method", "Metric"] + time_labels

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Reconstruction quality at different training time budgets on DTU benchmark. "
        r"Lower is better for all metrics. Best results in \textbf{bold}, second-best \underline{underlined}.}",
        r"\label{tab:time_comparison}",
        r"\begin{tabular}{ll" + "c" * len(time_budgets) + "}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]

    metrics = ['accuracy', 'completeness', 'overall']
    metric_display = {
        'accuracy': 'Accuracy $\\downarrow$',
        'completeness': 'Completeness $\\downarrow$',
        'overall': 'Overall $\\downarrow$',
    }

    for metric in metrics:
        for method in methods:
            method_df = df[df['method'] == method]

            row_values = [
                method_names.get(method, method),
                metric_display[metric]
            ]

            # Collect values for ranking
            values_for_ranking = []
            all_values = []
            for t in time_budgets:
                t_row = method_df[method_df['time_budget'] == t]
                if not t_row.empty:
                    mean = t_row[f'{metric}_mean'].values[0]
                    std = t_row[f'{metric}_std'].values[0]
                    all_values.append((mean, std))
                    values_for_ranking.append(mean)
                else:
                    all_values.append((np.nan, np.nan))
                    values_for_ranking.append(np.nan)

            # Find best across methods for each time budget
            for i, (mean, std) in enumerate(all_values):
                # Get all method values for this time budget
                t = time_budgets[i]
                all_method_values = []
                for m in methods:
                    m_df = df[(df['method'] == m) & (df['time_budget'] == t)]
                    if not m_df.empty:
                        all_method_values.append(m_df[f'{metric}_mean'].values[0])
                    else:
                        all_method_values.append(np.nan)

                best_idx, second_idx = find_best_and_second(all_method_values, lower_is_better=True)
                method_idx = methods.index(method)

                is_best = (method_idx == best_idx)
                is_second = (method_idx == second_idx)

                if np.isnan(mean):
                    row_values.append("-")
                else:
                    row_values.append(format_metric_with_std(mean, std, 2, is_best, is_second))

            lines.append(" & ".join(row_values) + r" \\")

        if metric != metrics[-1]:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    table_str = "\n".join(lines)

    # Save to file
    ensure_dir(output_path.parent)
    output_path.write_text(table_str)
    print(f"Saved time comparison table to: {output_path}")

    return table_str


def generate_sota_comparison_table(
    results_dir: Path,
    output_path: Path
) -> str:
    """
    Generate LaTeX table comparing with state-of-the-art methods.

    Includes neural implicit methods, gaussian splatting methods, and MVS methods.
    """
    sota_file = results_dir / "results" / "sota_comparison.csv"
    sota_baselines_csv = results_dir / "results" / "sota_baselines.csv"

    if sota_file.exists():
        df = pd.read_csv(sota_file)
    elif sota_baselines_csv.exists():
        df = pd.read_csv(sota_baselines_csv)
    else:
        # Load from YAML config directly
        sota_baselines_path = get_sota_baselines_path()
        if not sota_baselines_path.exists():
            print("Warning: No SOTA data found")
            return ""

        sota_config = load_config(sota_baselines_path)
        records = []
        for method_name, data in sota_config.get('methods', {}).items():
            records.append({
                'method': method_name,
                'accuracy': data.get('accuracy', np.nan),
                'completeness': data.get('completeness', np.nan),
                'overall': data.get('overall', np.nan),
                'category': data.get('category', 'other'),
            })
        df = pd.DataFrame(records)

    if df.empty:
        return ""

    # Sort by overall score
    df = df.sort_values('overall')

    columns = ["Method", "Accuracy $\\downarrow$", "Completeness $\\downarrow$", "Overall $\\downarrow$"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison with state-of-the-art methods on DTU benchmark (15 scenes). "
        r"Chamfer distance metrics (mm). Lower is better.}",
        r"\label{tab:sota_comparison}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]

    metrics = ['accuracy', 'completeness', 'overall']

    # Find best and second best for each metric
    best_indices = {}
    second_indices = {}
    for metric in metrics:
        values = df[metric].tolist()
        best_indices[metric], second_indices[metric] = find_best_and_second(values, lower_is_better=True)

    # Group by category if available
    if 'category' in df.columns:
        categories = ['neural_implicit', 'gaussian_splatting', 'mvs']
        category_names = {
            'neural_implicit': 'Neural Implicit',
            'gaussian_splatting': 'Gaussian Splatting',
            'mvs': 'Multi-View Stereo',
        }

        for cat in categories:
            cat_df = df[df['category'] == cat]
            if cat_df.empty:
                continue

            lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{category_names.get(cat, cat)}}}}} \\\\")

            for idx, row in cat_df.iterrows():
                row_values = [escape_latex(row['method'])]
                global_idx = df.index.get_loc(idx)

                for metric in metrics:
                    is_best = (global_idx == best_indices[metric])
                    is_second = (global_idx == second_indices[metric])
                    row_values.append(format_metric(row[metric], 2, is_best, is_second))

                lines.append(" & ".join(row_values) + r" \\")

            lines.append(r"\midrule")

        # Remove last midrule
        lines[-1] = r"\bottomrule"
    else:
        # Simple table without categories
        for idx, row in df.iterrows():
            row_values = [escape_latex(row['method'])]
            global_idx = df.index.get_loc(idx)

            for metric in metrics:
                is_best = (global_idx == best_indices[metric])
                is_second = (global_idx == second_indices[metric])
                row_values.append(format_metric(row[metric], 2, is_best, is_second))

            lines.append(" & ".join(row_values) + r" \\")

        lines.append(r"\bottomrule")

    lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])

    table_str = "\n".join(lines)

    ensure_dir(output_path.parent)
    output_path.write_text(table_str)
    print(f"Saved SOTA comparison table to: {output_path}")

    return table_str


def generate_per_scene_table(
    results_dir: Path,
    output_path: Path
) -> str:
    """
    Generate LaTeX table with per-scene breakdown.
    """
    per_scene_file = results_dir / "results" / "per_scene_comparison.csv"

    if not per_scene_file.exists():
        print(f"Warning: Per-scene file not found: {per_scene_file}")
        return ""

    df = pd.read_csv(per_scene_file)

    if df.empty:
        return ""

    # Determine methods from columns
    methods = []
    for col in df.columns:
        if 'ours' in col.lower():
            methods.append('ours')
        elif 'baseline' in col.lower():
            methods.append('baseline_2dgs')
    methods = list(set(methods))

    method_names = {
        'ours': 'Ours',
        'baseline_2dgs': '2DGS',
    }

    # Create table with scene rows and method columns for overall metric
    columns = ["Scene"] + [method_names.get(m, m) for m in methods]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-scene reconstruction quality (Overall Chamfer distance, mm). Lower is better.}",
        r"\label{tab:per_scene}",
        r"\begin{tabular}{l" + "c" * len(methods) + "}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]

    # Find overall columns for each method
    overall_cols = {m: None for m in methods}
    for col in df.columns:
        for m in methods:
            if 'overall' in col.lower() and m in col.lower():
                overall_cols[m] = col

    for _, row in df.iterrows():
        scene = row.get('scan', row.iloc[0])
        row_values = [escape_latex(str(scene))]

        method_vals = []
        for m in methods:
            col = overall_cols.get(m)
            if col and col in df.columns:
                method_vals.append(row[col])
            else:
                method_vals.append(np.nan)

        best_idx, second_idx = find_best_and_second(method_vals, lower_is_better=True)

        for i, val in enumerate(method_vals):
            is_best = (i == best_idx)
            is_second = (i == second_idx)
            row_values.append(format_metric(val, 2, is_best, is_second))

        lines.append(" & ".join(row_values) + r" \\")

    # Add mean row
    lines.append(r"\midrule")
    mean_values = [escape_latex("Mean")]
    for m in methods:
        col = overall_cols.get(m)
        if col and col in df.columns:
            mean_values.append(format_metric(df[col].mean(), 2))
        else:
            mean_values.append("-")
    lines.append(" & ".join(mean_values) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    table_str = "\n".join(lines)

    ensure_dir(output_path.parent)
    output_path.write_text(table_str)
    print(f"Saved per-scene table to: {output_path}")

    return table_str


def generate_speedup_table(
    results_dir: Path,
    output_path: Path,
    quality_thresholds: List[float] = [0.60, 0.55, 0.50, 0.45]
) -> str:
    """
    Generate table showing time to reach quality thresholds.

    Shows speedup factor of our method vs baseline.
    """
    time_results_file = results_dir / "results" / "time_based_results.csv"

    if not time_results_file.exists():
        print(f"Warning: Time results file not found: {time_results_file}")
        return ""

    df = pd.read_csv(time_results_file)

    if df.empty:
        return ""

    methods = df['method'].unique()
    time_budgets = sorted(df['time_budget'].unique())

    columns = ["Target Quality"] + ["Ours", "2DGS Baseline", "Speedup"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Time to reach quality thresholds (Overall Chamfer distance). "
        r"Speedup shows how much faster our method reaches the target quality.}",
        r"\label{tab:speedup}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]

    for threshold in quality_thresholds:
        row_values = [f"$\\leq$ {threshold:.2f}"]

        times = {}
        for method in ['ours', 'baseline_2dgs']:
            method_df = df[df['method'] == method]

            # Find first time budget that achieves threshold
            time_to_reach = None
            for t in time_budgets:
                t_df = method_df[method_df['time_budget'] == t]
                if not t_df.empty:
                    mean_overall = t_df['overall'].mean()
                    if mean_overall <= threshold:
                        time_to_reach = t
                        break

            times[method] = time_to_reach

        # Format times
        for method in ['ours', 'baseline_2dgs']:
            t = times.get(method)
            if t is not None:
                row_values.append(format_time_budget(t))
            else:
                row_values.append(">15min")

        # Compute speedup
        if times.get('ours') and times.get('baseline_2dgs'):
            speedup = times['baseline_2dgs'] / times['ours']
            row_values.append(f"{speedup:.1f}$\\times$")
        elif times.get('ours') and not times.get('baseline_2dgs'):
            row_values.append("$\\infty$")
        else:
            row_values.append("-")

        lines.append(" & ".join(row_values) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    table_str = "\n".join(lines)

    ensure_dir(output_path.parent)
    output_path.write_text(table_str)
    print(f"Saved speedup table to: {output_path}")

    return table_str


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from evaluation results")
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
        help="Output directory for tables (default: results_dir/tables)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "tables"
    ensure_dir(output_dir)

    print(f"Generating tables from: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Generate all tables
    print("\n--- Generating Time Comparison Table ---")
    generate_time_comparison_table(results_dir, output_dir / "time_comparison.tex")

    print("\n--- Generating SOTA Comparison Table ---")
    generate_sota_comparison_table(results_dir, output_dir / "sota_comparison.tex")

    print("\n--- Generating Per-Scene Table ---")
    generate_per_scene_table(results_dir, output_dir / "per_scene.tex")

    print("\n--- Generating Speedup Table ---")
    generate_speedup_table(results_dir, output_dir / "speedup.tex")

    print(f"\n{'='*60}")
    print(f"Table generation complete. Files saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
