#!/usr/bin/env python3
"""
Aggregate evaluation results and compute statistics.

Collects results from all experiments, computes mean and std across scenes,
and merges with state-of-the-art baselines for comparison.

Usage:
    python -m evaluation.aggregate_results [--results-dir RESULTS_DIR]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.utils.io_utils import (
    load_config,
    save_results_csv,
    get_default_config_path,
    get_sota_baselines_path,
    ensure_dir,
    format_time_budget,
)


def load_metrics_csv(csv_path: Path) -> pd.DataFrame:
    """Load metrics CSV and standardize column names."""
    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Rename columns if needed
    rename_map = {
        'mean_d2s': 'accuracy',
        'mean_s2d': 'completeness',
    }
    df = df.rename(columns=rename_map)

    return df


def aggregate_time_based_results(results_dir: Path) -> pd.DataFrame:
    """
    Aggregate time-based evaluation results.

    Returns DataFrame with columns:
        method, time_budget, scene, accuracy, completeness, overall
    """
    time_dir = results_dir / "time_based"

    if not time_dir.exists():
        print(f"Warning: Time-based results directory not found: {time_dir}")
        return pd.DataFrame()

    all_results = []

    # Find all time budget directories
    for time_budget_dir in sorted(time_dir.iterdir()):
        if not time_budget_dir.is_dir():
            continue

        time_budget_str = time_budget_dir.name
        time_budget = int(time_budget_str.replace("s", ""))

        # Load results for each method
        for method in ["ours", "baseline_2dgs"]:
            metrics_file = time_budget_dir / f"metrics_{method.replace('_2dgs', '')}.csv"

            if not metrics_file.exists():
                continue

            try:
                df = load_metrics_csv(metrics_file)
                df['method'] = method
                df['time_budget'] = time_budget
                all_results.append(df)
            except Exception as e:
                print(f"Warning: Could not load {metrics_file}: {e}")

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # Ensure required columns exist
    required_cols = ['method', 'time_budget', 'scan', 'accuracy', 'completeness', 'overall']
    for col in required_cols:
        if col not in combined.columns:
            combined[col] = np.nan

    return combined[required_cols]


def aggregate_iteration_based_results(results_dir: Path) -> pd.DataFrame:
    """
    Aggregate iteration-based evaluation results.

    Returns DataFrame with columns:
        method, iterations, scene, accuracy, completeness, overall
    """
    iter_dir = results_dir / "iteration_based"

    if not iter_dir.exists():
        print(f"Warning: Iteration-based results directory not found: {iter_dir}")
        return pd.DataFrame()

    all_results = []

    for method_dir in sorted(iter_dir.iterdir()):
        if not method_dir.is_dir():
            continue

        method = method_dir.name

        # Look for metrics files with iteration info
        for metrics_file in method_dir.glob("**/metrics*.csv"):
            try:
                df = load_metrics_csv(metrics_file)
                df['method'] = method

                # Try to extract iteration count from path or file
                if 'iterations' not in df.columns:
                    # Default to final iteration
                    df['iterations'] = 30000

                all_results.append(df)
            except Exception as e:
                print(f"Warning: Could not load {metrics_file}: {e}")

    if not all_results:
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # Ensure required columns exist
    required_cols = ['method', 'iterations', 'scan', 'accuracy', 'completeness', 'overall']
    for col in required_cols:
        if col not in combined.columns:
            combined[col] = np.nan

    return combined[required_cols]


def compute_statistics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Compute mean and std for metrics grouped by specified columns.

    Returns DataFrame with mean and std columns for each metric.
    """
    metrics = ['accuracy', 'completeness', 'overall']

    agg_funcs = {
        metric: ['mean','count'] for metric in metrics
    }

    stats = df.groupby(group_cols).agg(agg_funcs).reset_index()

    # Flatten column names
    stats.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in stats.columns
    ]

    return stats


def compute_relative_improvement(
    df: pd.DataFrame,
    baseline_method: str = "baseline_2dgs"
) -> pd.DataFrame:
    """
    Compute relative improvement over baseline.

    Returns DataFrame with improvement columns.
    """
    metrics = ['accuracy', 'completeness', 'overall']

    # Separate baseline and other methods
    baseline = df[df['method'] == baseline_method].copy()
    others = df[df['method'] != baseline_method].copy()

    if baseline.empty or others.empty:
        return df

    # Merge to compute improvements
    # This is a simplified version - would need more logic for proper merging
    # based on time_budget or iterations

    return df


def load_sota_baselines() -> pd.DataFrame:
    """Load state-of-the-art baseline results."""
    sota_path = get_sota_baselines_path()

    if not sota_path.exists():
        print(f"Warning: SOTA baselines file not found: {sota_path}")
        return pd.DataFrame()

    sota_config = load_config(sota_path)
    methods_data = sota_config.get('methods', {})

    records = []
    for method_name, data in methods_data.items():
        records.append({
            'method': method_name,
            'accuracy': data.get('accuracy', np.nan),
            'completeness': data.get('completeness', np.nan),
            'overall': data.get('overall', np.nan),
            'source': data.get('source', ''),
            'category': data.get('category', 'other'),
        })

    return pd.DataFrame(records)


def merge_with_sota(
    our_results: pd.DataFrame,
    sota_baselines: pd.DataFrame
) -> pd.DataFrame:
    """Merge our results with SOTA baselines for comparison table."""
    # Compute mean stats for our methods
    metrics = ['accuracy', 'completeness', 'overall']

    if 'time_budget' in our_results.columns:
        # Get final time budget results
        max_time = our_results['time_budget'].max()
        our_final = our_results[our_results['time_budget'] == max_time]
    else:
        our_final = our_results

    our_stats = our_final.groupby('method')[metrics].agg(['mean']).reset_index()
    our_stats.columns = ['method'] + [f"{m}_{s}" for m in metrics for s in ['mean']]

    # Add our methods to SOTA table
    for _, row in our_stats.iterrows():
        method = "Ours (Fast Init)" if row['method'] == 'ours' else row['method']
        sota_baselines = pd.concat([sota_baselines, pd.DataFrame([{
            'method': method,
            'accuracy': row['accuracy_mean'],
            'completeness': row['completeness_mean'],
            'overall': row['overall_mean'],
            'source': 'This work',
            'category': 'gaussian_splatting',
        }])], ignore_index=True)

    return sota_baselines


def create_time_comparison_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary table comparing methods across time budgets.

    Returns DataFrame suitable for LaTeX table generation.
    """
    if df.empty:
        return pd.DataFrame()

    time_budgets = sorted(df['time_budget'].unique())
    methods = df['method'].unique()

    records = []
    for method in methods:
        method_df = df[df['method'] == method]
        for time_budget in time_budgets:
            time_df = method_df[method_df['time_budget'] == time_budget]
            if time_df.empty:
                continue

            records.append({
                'method': method,
                'time_budget': time_budget,
                'time_label': format_time_budget(time_budget),
                'accuracy_mean': time_df['accuracy'].mean(),
                'completeness_mean': time_df['completeness'].mean(),
                'overall_mean': time_df['overall'].mean(),
                'n_scenes': len(time_df),
            })

    return pd.DataFrame(records)


def create_per_scene_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create per-scene comparison between methods.

    Returns DataFrame with columns: scene, method, metrics...
    """
    if df.empty:
        return pd.DataFrame()

    # Get final time budget or max iterations
    if 'time_budget' in df.columns:
        max_val = df['time_budget'].max()
        final_df = df[df['time_budget'] == max_val]
    elif 'iterations' in df.columns:
        max_val = df['iterations'].max()
        final_df = df[df['iterations'] == max_val]
    else:
        final_df = df

    return final_df.pivot_table(
        index='scan',
        columns='method',
        values=['accuracy', 'completeness', 'overall'],
        aggfunc='mean'
    ).reset_index()


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
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
        help="Output directory for aggregated results (default: results_dir/results)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "results"
    ensure_dir(output_dir)

    print(f"Aggregating results from: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Aggregate time-based results
    print("\n--- Time-based Results ---")
    time_results = aggregate_time_based_results(results_dir)
    if not time_results.empty:
        save_results_csv(time_results, output_dir / "time_based_results.csv")
        print(f"Saved {len(time_results)} time-based results")

        # Compute statistics
        time_stats = compute_statistics(time_results, ['method', 'time_budget'])
        save_results_csv(time_stats, output_dir / "time_based_stats.csv")

        # Create summary table
        time_summary = create_time_comparison_summary(time_results)
        save_results_csv(time_summary, output_dir / "time_comparison_summary.csv")

        print("\nTime-based summary:")
        print(time_summary.to_string(index=False))
    else:
        print("No time-based results found")

    # Aggregate iteration-based results
    print("\n--- Iteration-based Results ---")
    iter_results = aggregate_iteration_based_results(results_dir)
    if not iter_results.empty:
        save_results_csv(iter_results, output_dir / "iteration_based_results.csv")
        print(f"Saved {len(iter_results)} iteration-based results")
    else:
        print("No iteration-based results found")

    # Load SOTA baselines
    print("\n--- SOTA Baselines ---")
    sota_baselines = load_sota_baselines()
    if not sota_baselines.empty:
        save_results_csv(sota_baselines, output_dir / "sota_baselines.csv")
        print(f"Loaded {len(sota_baselines)} SOTA methods")

        # Merge with our results
        if not time_results.empty:
            comparison = merge_with_sota(time_results, sota_baselines.copy())
            save_results_csv(comparison, output_dir / "sota_comparison.csv")
            print("\nSOTA comparison table:")
            print(comparison[['method', 'accuracy', 'completeness', 'overall']].to_string(index=False))

    # Per-scene comparison
    print("\n--- Per-scene Comparison ---")
    if not time_results.empty:
        per_scene = create_per_scene_comparison(time_results)
        if not per_scene.empty:
            # Flatten multi-index columns for CSV
            per_scene.columns = ['_'.join(map(str, col)).strip('_') for col in per_scene.columns]
            save_results_csv(per_scene, output_dir / "per_scene_comparison.csv")
            print(f"Saved per-scene comparison for {len(per_scene)} scenes")

    print(f"\n{'='*60}")
    print(f"Aggregation complete. Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
