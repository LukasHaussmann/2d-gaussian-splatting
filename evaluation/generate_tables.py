#!/usr/bin/env python3
"""
Generate publication-ready LaTeX tables for DTU benchmark comparison and ablation studies.

Usage:
    # Generate DTU comparison table
    python -m evaluation.generate_tables [--results-dir RESULTS_DIR] [--output-dir OUTPUT_DIR]

    # Generate ablation study table
    python -m evaluation.generate_tables --ablation-dir ablation_results/normal_estimation
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

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
    find_best_and_second,
)


def generate_ablations_table(
    ablation_dir: Path,
    output_path: Path,
    time_budget: int,
) -> str:
    """
    Generate LaTeX table for ablation study results.

    Rows are experiments, columns are accuracy, completeness, and overall.
    Values are the mean across all scenes for the specified time budget.

    Args:
        ablation_dir: Path to ablation results directory (e.g., ablation_results/normal_estimation)
        output_path: Path to save the LaTeX table
        time_budget: Time budget in seconds to use for the table

    Returns:
        LaTeX table string
    """
    # Load results CSV
    results_file = ablation_dir / "results.csv"
    if not results_file.exists():
        print(f"Warning: Results file not found: {results_file}")
        return ""

    results_df = pd.read_csv(results_file)
    if results_df.empty:
        return ""

    # Derive experiments from the data
    ablation_name = ablation_dir.name
    experiments = results_df['experiment'].unique().tolist()

    if not experiments:
        print(f"Warning: No experiments found in results")
        return ""

    # Filter to the specified time budget
    tb_df = results_df[results_df['time_budget'] == time_budget]
    if tb_df.empty:
        available = sorted(results_df['time_budget'].unique().tolist())
        print(f"Warning: No results for time budget {time_budget}s. Available: {available}")
        return ""

    # Compute mean metrics for each experiment
    metrics = ['accuracy', 'completeness', 'overall']
    table_data = []
    for exp_name in experiments:
        exp_df = tb_df[tb_df['experiment'] == exp_name]
        row_data = {'experiment': exp_name}
        for metric in metrics:
            if not exp_df.empty:
                row_data[metric] = exp_df[metric].mean()
            else:
                row_data[metric] = None
        table_data.append(row_data)

    # Find best and second best for each metric column
    col_rankings = {}
    for metric in metrics:
        col_values = [row.get(metric) for row in table_data]
        col_rankings[metric] = find_best_and_second(col_values, lower_is_better=True)

    # Generate LaTeX table
    col_headers = ["Experiment", "Accuracy", "Completeness", "Overall"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{Ablation study: {ablation_name.replace('_', ' ')}. Chamfer distance (mm) $\\downarrow$ at {time_budget}s.}}",
        f"\\label{{tab:ablation_{ablation_name}}}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        " & ".join(col_headers) + r" \\",
        r"\midrule",
    ]

    # Generate rows
    for i, row in enumerate(table_data):
        row_values = [row['experiment'].replace('_', r'\_')]

        for metric in metrics:
            val = row.get(metric)
            best_idx, second_idx = col_rankings[metric]
            is_best = (i == best_idx)
            is_second = (i == second_idx)
            row_values.append(format_metric(val, 3, is_best, is_second))

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


def load_2dgs_reproduced_metrics(
    experiments_dir: Path,
    iteration: int = 30000,
) -> dict:
    """
    Load per-scene metrics from 2DGS experiments directory.

    Args:
        experiments_dir: Path to 2DGS experiments directory (e.g., ~/2d-gaussian-splatting/experiments)
        iteration: Iteration number to load metrics for (default: 30000)

    Returns:
        Dict mapping scan names to metric dicts with 'accuracy', 'completeness', 'overall'
    """
    experiments_dir = Path(experiments_dir).expanduser()
    if not experiments_dir.exists():
        print(f"Warning: 2DGS experiments directory not found: {experiments_dir}")
        return {}

    results = {}
    for scene_dir in experiments_dir.iterdir():
        if not scene_dir.is_dir() or not scene_dir.name.startswith("scan"):
            continue

        metrics_file = scene_dir / "metrics.csv"
        if not metrics_file.exists():
            continue

        try:
            df = pd.read_csv(metrics_file)
            row = df[df['iterations'] == iteration]
            if not row.empty:
                results[scene_dir.name] = row['overall'].iloc[0]
        except Exception as e:
            print(f"Warning: Could not read {metrics_file}: {e}")

    return results


def load_metrics_from_evaluation(
    results_dir: Path,
    method: str,
    time_budget: int,
) -> dict:
    """
    Load per-scene metrics from the evaluation pipeline results.

    Args:
        results_dir: Root evaluation results directory
        method: Method name (e.g., 'ours', 'baseline_2dgs')
        time_budget: Time budget in seconds

    Returns:
        Dict mapping scan names to overall chamfer distance values
    """
    metrics_file = results_dir / "time_based" / method / f"metrics_{time_budget}s.csv"
    if not metrics_file.exists():
        print(f"Warning: Metrics file not found: {metrics_file}")
        return {}

    try:
        df = pd.read_csv(metrics_file)
        if 'overall' not in df.columns or 'scan' not in df.columns:
            print(f"Warning: Missing required columns in {metrics_file}")
            return {}
        return {row['scan']: row['overall'] for _, row in df.iterrows()}
    except Exception as e:
        print(f"Warning: Could not read {metrics_file}: {e}")
        return {}


def generate_dtu_comparison_table(
    results_dir: Path,
    output_path: Path,
    time_budget: int = 900,
    methods_to_include: List[str] = None,
) -> str:
    """
    Generate paper-style DTU benchmark comparison table.

    Methods as rows, scans as columns, with Mean and Time at the end.
    Standard format used in papers like 2DGS, GOF, NeuS, etc.

    Args:
        results_dir: Path to evaluation results directory
        output_path: Path to save the LaTeX table
        time_budget: Time budget in seconds for comparing methods
        methods_to_include: List of methods to include (default: SOTA + Ours + 2DGS*)
    """
    if methods_to_include is None:
        methods_to_include = ["3DGS", "SuGaR", "2DGS", "2DGS*", "Ours"]

    # Load SOTA baselines config
    sota_baselines_path = get_sota_baselines_path()
    if not sota_baselines_path.exists():
        print("Warning: SOTA baselines config not found")
        return ""

    sota = load_config(sota_baselines_path)
    sota_methods = sota.get('methods', {})

    # Helper to load per-scene data from config
    def load_perscene_from_config(key):
        data = sota.get(key, {})
        return {f"scan{k.replace('scan', '')}": v['overall']
                for k, v in data.items() if v.get('overall') is not None}

    # Load per-scene data for SOTA methods from config
    perscene_data = {
        "3DGS": load_perscene_from_config('per_scene_3dgs'),
        "SuGaR": load_perscene_from_config('per_scene_sugar'),
        "2DGS": load_perscene_from_config('per_scene_2dgs'),
    }

    # Load 2DGS 30k reproduced results
    dgs_30k_data = load_2dgs_reproduced_metrics(
        Path("~/2d-gaussian-splatting/experiments").expanduser()
    )
    if dgs_30k_data:
        perscene_data["2DGS 30k"] = dgs_30k_data
    else:
        print("Warning: No '2DGS 30k' reproduced metrics found")
        perscene_data["2DGS 30k"] = {}

    # Load "Ours" results from evaluation pipeline
    ours_data = load_metrics_from_evaluation(results_dir, "ours", time_budget)
    if ours_data:
        perscene_data["Ours"] = ours_data
    else:
        print(f"Warning: No 'ours' metrics found for time budget {time_budget}s")
        perscene_data["Ours"] = {}

    # Load 2DGS baseline with same runtime from evaluation (2DGS*)
    baseline_data = load_metrics_from_evaluation(results_dir, "baseline_2dgs", time_budget)
    if baseline_data:
        perscene_data["2DGS*"] = baseline_data
    else:
        print(f"Warning: No 'baseline_2dgs' metrics found for time budget {time_budget}s")
        perscene_data["2DGS*"] = {}

    # Format runtime string
    runtime_str = format_time_budget(time_budget)

    # Load runtime from config for SOTA methods
    runtimes = {
        "Ours": runtime_str,
        "2DGS*": runtime_str,  # Same runtime as Ours
        "2DGS 30k": "13min",  # Reproduced 2DGS at 30k iterations
    }
    for method_name in methods_to_include:
        if method_name not in runtimes:
            method_info = sota_methods.get(method_name, {})
            runtimes[method_name] = method_info.get('runtime', '-')

    # Determine available scans (from Ours data, fallback to standard DTU scans)
    if perscene_data["Ours"]:
        available_scans = sorted([int(s.replace('scan', '')) for s in perscene_data["Ours"].keys()])
    else:
        # Fallback to standard DTU scans
        available_scans = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

    scan_cols = [str(s) for s in available_scans]

    # Build table data
    table_data = []
    for method_name in methods_to_include:
        row_data = {'name': method_name, 'time': runtimes.get(method_name, '-')}
        values = []

        method_perscene = perscene_data.get(method_name, {})
        for scan_id in available_scans:
            scan_key = f"scan{scan_id}"
            val = method_perscene.get(scan_key)
            row_data[str(scan_id)] = val
            if val is not None:
                values.append(val)

        # Calculate mean from available data, or use reported mean
        if values:
            row_data['mean'] = sum(values) / len(values)
        else:
            # Use reported mean from config
            row_data['mean'] = sota_methods.get(method_name, {}).get('overall')

        table_data.append(row_data)

    # Generate LaTeX table
    n_scans = len(scan_cols)
    col_headers = ["Method"] + scan_cols + ["Mean", "Time"]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Quantitative results on DTU dataset. We report the Chamfer distance (mm) $\downarrow$.}",
        r"\label{tab:dtu_comparison}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * n_scans + "cc}",
        r"\toprule",
        " & ".join(col_headers) + r" \\",
        r"\midrule",
    ]

    # Find best and second best for each column
    col_rankings = {}
    for col in scan_cols + ['mean']:
        col_values = []
        for row in table_data:
            if col == 'mean':
                col_values.append(row['mean'])
            else:
                col_values.append(row.get(col))
        col_rankings[col] = find_best_and_second(col_values, lower_is_better=True)

    # Generate rows
    for i, row in enumerate(table_data):
        row_values = [row['name']]

        # Per-scene values
        for scan_id in scan_cols:
            val = row.get(scan_id)
            best_idx, second_idx = col_rankings[scan_id]
            is_best = (i == best_idx)
            is_second = (i == second_idx)
            row_values.append(format_metric(val, 2, is_best, is_second))

        # Mean
        best_idx, second_idx = col_rankings['mean']
        is_best = (i == best_idx)
        is_second = (i == second_idx)
        row_values.append(format_metric(row['mean'], 2, is_best, is_second))

        # Time
        row_values.append(row['time'])

        lines.append(" & ".join(row_values) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    table_str = "\n".join(lines)

    ensure_dir(output_path.parent)
    output_path.write_text(table_str)
    print(f"Saved DTU comparison table to: {output_path}")

    return table_str


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX table from evaluation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="evaluation_results/estimated_normals_convergence_study",
        help="Path to evaluation results directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for table (default: results_dir/tables)"
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=900,
        help="Time budget in seconds for method comparison (default: 900)"
    )
    parser.add_argument(
        "--ablation-dir",
        type=str,
        default=None,
        help="Path to ablation results directory (e.g., ablation_results/normal_estimation)"
    )
    args = parser.parse_args()

    # Generate ablation table if ablation-dir is specified
    if args.ablation_dir:
        ablation_dir = Path(args.ablation_dir)
        output_dir = Path(args.output_dir) if args.output_dir else ablation_dir / "tables"
        ensure_dir(output_dir)

        print(f"Ablation directory: {ablation_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Time budget: {args.time_budget}s")
        print(f"\n--- Generating Ablation Table for {ablation_dir.name} ---")
        generate_ablations_table(
            ablation_dir,
            output_dir / f"ablation_{ablation_dir.name}.tex",
            time_budget=args.time_budget,
        )
    else:
        # Generate DTU comparison table
        results_dir = Path(args.results_dir)
        output_dir = Path(args.output_dir) if args.output_dir else results_dir / "tables"
        ensure_dir(output_dir)

        print(f"Results directory: {results_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Time budget: {args.time_budget}s")
        print("\n--- Generating DTU Comparison Table ---")
        generate_dtu_comparison_table(
            results_dir,
            output_dir / "dtu_comparison.tex",
            time_budget=args.time_budget,
        )

    print(f"\nTable generation complete. File saved to: {output_dir}")


if __name__ == "__main__":
    main()
