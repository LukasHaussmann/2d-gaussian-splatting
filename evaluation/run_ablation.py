#!/usr/bin/env python3
"""
Ablation study runner for 2DGS fast convergence experiments.

Systematically evaluates the contribution of each component/parameter
by running experiments with individual modifications from the base config.

Usage:
    python -m evaluation.run_ablation [--config CONFIG] [--ablations NAME1 NAME2 ...]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from copy import deepcopy

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.utils.io_utils import (
    ensure_dir,
    load_config,
    save_config,
    create_marker_file,
    marker_exists,
    parse_scan_id,
)

try:
    from mesh_snapshot import mesh_snapshot_from_file
except ImportError:
    mesh_snapshot_from_file = None


def get_ablation_config_path() -> Path:
    """Get path to ablation configuration file."""
    return Path(__file__).parent / "config" / "ablation_config.yaml"


def merge_configs(base: Dict, changes: Dict) -> Dict:
    """Merge changes into base config."""
    result = deepcopy(base)
    result.update(changes)
    return result


def build_train_args(config: Dict[str, Any]) -> str:
    """Build training argument string from config dict."""
    args = []
    for key, value in config.items():
        if value is not None:
            args.append(f"--{key} {value}")
    return " ".join(args)


def build_render_args(config: Dict[str, Any]) -> str:
    """Build rendering argument string from config."""
    args = []
    if config.get('skip_train', True):
        args.append("--skip_train")
    for key in ['depth_ratio', 'num_cluster', 'voxel_size', 'sdf_trunc', 'depth_trunc']:
        if key in config:
            args.append(f"--{key} {config[key]}")
    return " ".join(args)


def run_training(
    source: str,
    exp_dir: str,
    train_args: str,
    time_limit: Optional[int] = None,
    iterations: Optional[int] = None,
) -> bool:
    """Run training for a single experiment."""
    cmd = f"python train.py -s {source} -m {exp_dir} {train_args}"

    if time_limit is not None:
        cmd += f" --time_limit {time_limit}"
    if iterations is not None:
        cmd += f" --iterations {iterations}"

    print(f"\n{'='*80}")
    print(f"Running: {cmd}")
    print(f"{'='*80}\n")

    result = os.system(cmd)
    return result == 0


def run_rendering(source: str, exp_dir: str, render_args: str) -> bool:
    """Run mesh extraction/rendering."""
    cmd = f"python render.py -s {source} -m {exp_dir} {render_args}"
    print(f"Rendering: {cmd}")
    result = os.system(cmd)
    return result == 0


def run_evaluation(
    mesh_file: str,
    scan_id: int,
    mask_dir: str,
    dtu_dir: str,
    metrics_file: str,
    scene_name: str,
    ablation_name: str
) -> bool:
    """Run mesh evaluation and append results to CSV."""
    subprocess.run(f'echo -n "{ablation_name},{scene_name}," >> {metrics_file}', shell=True, check=True)

    script_dir = Path(__file__).parent.parent / "scripts" / "eval_dtu"
    cmd = (
        f'python {script_dir}/evaluate_single_scene.py '
        f'--input_mesh {mesh_file} '
        f'--scan_id {scan_id} '
        f'--mask_dir {mask_dir} '
        f'--DTU {dtu_dir} '
        f'| sed "s/ /,/g" | grep ^[^cull] >> {metrics_file}'
    )

    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def find_mesh_file(exp_dir: str) -> Optional[str]:
    """Find the generated mesh file in experiment directory."""
    train_dir = Path(exp_dir) / "train"
    if not train_dir.exists():
        return None

    subdirs = list(train_dir.iterdir())
    if not subdirs:
        return None

    mesh_file = subdirs[0] / "fuse_post.ply"
    if mesh_file.exists():
        return str(mesh_file)
    return None


def run_ablation_study(
    config: Dict[str, Any],
    ablations_to_run: Optional[List[str]] = None,
    resume: bool = True
) -> None:
    """
    Run ablation study experiments.

    Parameters
    ----------
    config : dict
        Ablation configuration
    ablations_to_run : list, optional
        Specific ablations to run (default: all)
    resume : bool
        Whether to resume from previous runs
    """
    output_dir = Path(config['experiment']['output_dir'])
    dtu_source = config['experiment']['dtu_source']
    scenes = config['scenes']
    time_budget = config['time_budget']
    base_config = config['base_config']
    ablations = config['ablations']
    render_config = config['rendering']

    ensure_dir(output_dir)
    save_config(config, output_dir / "ablation_config.yaml")

    render_args = build_render_args(render_config)

    # Initialize metrics file
    metrics_file = output_dir / "ablation_results.csv"
    if not metrics_file.exists():
        header = "ablation,scene,accuracy,completeness,overall"
        metrics_file.write_text(header + "\n")

    # Determine which ablations to run
    if ablations_to_run:
        ablations_to_run = set(ablations_to_run)
        # Always include base
        ablations_to_run.add("base")
    else:
        ablations_to_run = {"base"} | set(ablations.keys())

    # Run base configuration first
    if "base" in ablations_to_run:
        print(f"\n{'#'*80}")
        print("# Running BASE configuration (our full method)")
        print(f"{'#'*80}")

        base_dir = output_dir / "base"
        ensure_dir(base_dir)

        train_args = build_train_args(base_config)

        for scene in scenes:
            scan_id = parse_scan_id(scene)
            source = f"{dtu_source}/{scene}"
            exp_dir = base_dir / scene
            marker = exp_dir / ".complete"

            if resume and marker_exists(marker):
                print(f"Skipping {scene} (base) - already complete")
                continue

            print(f"\n--- Base: {scene} ---")
            run_training(source, str(exp_dir), train_args, time_limit=time_budget)
            run_rendering(source, str(exp_dir), render_args)

            mesh_file = find_mesh_file(str(exp_dir))
            if mesh_file:
                if mesh_snapshot_from_file:
                    try:
                        mesh_snapshot_from_file(mesh_file, str(exp_dir / "snapshot.png"))
                    except:
                        pass
                run_evaluation(
                    mesh_file, scan_id, dtu_source, dtu_source,
                    str(metrics_file), scene, "base"
                )
                create_marker_file(marker)

    # Run each ablation
    for ablation_name, ablation_config in ablations.items():
        if ablation_name not in ablations_to_run:
            continue

        print(f"\n{'#'*80}")
        print(f"# Ablation: {ablation_config['name']}")
        print(f"# {ablation_config['description']}")
        print(f"{'#'*80}")

        ablation_dir = output_dir / ablation_name
        ensure_dir(ablation_dir)

        # Merge changes with base config
        modified_config = merge_configs(base_config, ablation_config['changes'])
        train_args = build_train_args(modified_config)

        # Save ablation-specific config
        save_config({
            'name': ablation_config['name'],
            'description': ablation_config['description'],
            'changes': ablation_config['changes'],
            'full_config': modified_config
        }, ablation_dir / "config.yaml")

        for scene in scenes:
            scan_id = parse_scan_id(scene)
            source = f"{dtu_source}/{scene}"
            exp_dir = ablation_dir / scene
            marker = exp_dir / ".complete"

            if resume and marker_exists(marker):
                print(f"Skipping {scene} ({ablation_name}) - already complete")
                continue

            print(f"\n--- {ablation_name}: {scene} ---")
            run_training(source, str(exp_dir), train_args, time_limit=time_budget)
            run_rendering(source, str(exp_dir), render_args)

            mesh_file = find_mesh_file(str(exp_dir))
            if mesh_file:
                if mesh_snapshot_from_file:
                    try:
                        mesh_snapshot_from_file(mesh_file, str(exp_dir / "snapshot.png"))
                    except:
                        pass
                run_evaluation(
                    mesh_file, scan_id, dtu_source, dtu_source,
                    str(metrics_file), scene, ablation_name
                )
                create_marker_file(marker)

    print(f"\n{'='*80}")
    print(f"Ablation study complete. Results saved to: {metrics_file}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation study experiments")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to ablation configuration file"
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=None,
        help="Specific ablations to run (default: all)"
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Specific scenes to evaluate (overrides config)"
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=None,
        help="Time budget in seconds (overrides config)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from previous runs"
    )
    parser.add_argument(
        "--list-ablations",
        action="store_true",
        help="List available ablations and exit"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing"
    )
    args = parser.parse_args()

    # Load configuration
    config_path = args.config or get_ablation_config_path()
    config = load_config(config_path)

    # List ablations if requested
    if args.list_ablations:
        print("\nAvailable ablations:")
        print("-" * 60)
        for name, ablation in config['ablations'].items():
            print(f"  {name:25s} - {ablation['name']}")
            print(f"    {' '*25}   {ablation['description']}")
        print("\nAblation groups:")
        print("-" * 60)
        for group_name, group in config['ablation_groups'].items():
            print(f"  {group['name']}: {', '.join(group['experiments'])}")
        return

    # Override config if specified
    if args.scenes:
        config['scenes'] = args.scenes
    if args.time_budget:
        config['time_budget'] = args.time_budget

    print(f"Output directory: {config['experiment']['output_dir']}")
    print(f"Scenes: {config['scenes']}")
    print(f"Time budget: {config['time_budget']}s")

    if args.ablations:
        print(f"Running ablations: {args.ablations}")
    else:
        print(f"Running all {len(config['ablations'])} ablations + base")

    if args.dry_run:
        print("\n[DRY RUN] Would run ablation study with above settings")
        return

    run_ablation_study(
        config,
        ablations_to_run=args.ablations,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
