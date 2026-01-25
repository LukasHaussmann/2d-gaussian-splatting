#!/usr/bin/env python3
"""
Main evaluation runner for 2DGS fast convergence experiments.

Runs time-based and iteration-based evaluations on DTU benchmark,
comparing our fast initialization method against vanilla 2DGS baseline.

Usage:
    python -m evaluation.run_evaluation [--config CONFIG_PATH] [--scenes SCENE1 SCENE2 ...]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.utils.io_utils import (
    ensure_dir,
    load_config,
    save_config,
    get_default_config_path,
    check_experiment_complete,
    create_marker_file,
    marker_exists,
    format_time_budget,
    parse_scan_id,
)

try:
    from mesh_snapshot import mesh_snapshot_from_file
except ImportError:
    mesh_snapshot_from_file = None


def build_train_args(config: Dict[str, Any], fast_init: int) -> str:
    """Build training argument string from config."""
    training_cfg = config['training']
    args = []

    for key, value in training_cfg.items():
        if value is not None:
            args.append(f"--{key} {value}")

    args.append(f"--fast_init {fast_init}")
    return " ".join(args)


def build_render_args(config: Dict[str, Any]) -> str:
    """Build rendering argument string from config."""
    render_cfg = config['rendering']
    args = []

    if render_cfg.get('skip_train', True):
        args.append("--skip_train")

    for key in ['depth_ratio', 'num_cluster', 'voxel_size', 'sdf_trunc', 'depth_trunc']:
        if key in render_cfg:
            args.append(f"--{key} {render_cfg[key]}")

    return " ".join(args)


def build_2dgs_baseline_args(config: Dict[str, Any]) -> str:
    """Build training arguments for 2DGS baseline."""
    baseline_cfg = config['baseline_2dgs_training']
    args = []

    for key, value in baseline_cfg.items():
        if value is not None:
            args.append(f"--{key} {value}")

    return " ".join(args)


def run_training(
    source: str,
    exp_dir: str,
    train_args: str,
    time_limit: Optional[int] = None,
    iterations: Optional[int] = None,
    use_2dgs_baseline: bool = False,
    repo_dir_2dgs: str = "~/2d-gaussian-splatting"
) -> bool:
    """
    Run training for a single experiment.

    Returns True if training completed successfully.
    """
    if use_2dgs_baseline:
        train_script = f"{repo_dir_2dgs}/train.py"
    else:
        train_script = "train.py"

    cmd = f"python {train_script} -s {source} -m {exp_dir} {train_args}"

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

    print(f"\n{'='*80}")
    print(f"Rendering: {cmd}")
    print(f"{'='*80}\n")

    result = os.system(cmd)
    return result == 0


def run_evaluation(
    mesh_file: str,
    scan_id: int,
    mask_dir: str,
    dtu_dir: str,
    metrics_file: str,
    scene_name: str
) -> bool:
    """
    Run mesh evaluation using Chamfer distance.

    Appends results to metrics_file CSV.
    """
    # Write scene name to metrics file
    subprocess.run(f'echo -n "{scene_name}," >> {metrics_file}', shell=True, check=True)

    # Run evaluation script
    script_dir = Path(__file__).parent.parent / "scripts" / "eval_dtu"
    cmd = (
        f'python {script_dir}/evaluate_single_scene.py '
        f'--input_mesh {mesh_file} '
        f'--scan_id {scan_id} '
        f'--mask_dir {mask_dir} '
        f'--DTU {dtu_dir} '
        f'| sed "s/ /,/g" | grep ^[^cull] >> {metrics_file}'
    )

    print(f"Evaluating: {cmd}")
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


def create_mesh_snapshot(mesh_file: str, snapshot_file: str) -> None:
    """Create a snapshot image of the mesh."""
    if mesh_snapshot_from_file is not None:
        try:
            mesh_snapshot_from_file(mesh_file, snapshot_file)
        except Exception as e:
            print(f"Warning: Could not create mesh snapshot: {e}")


def run_time_based_evaluation(
    config: Dict[str, Any],
    scenes: List[str],
    time_budgets: List[int],
    output_dir: Path,
    resume: bool = True
) -> None:
    """
    Run time-based evaluation across all scenes and time budgets.

    For each time budget, trains both our method and baseline 2DGS,
    then evaluates the resulting meshes.
    """
    dtu_source = config['experiment']['dtu_source']
    repo_dir_2dgs = config['experiment']['repo_dir_2dgs']

    train_args_ours = build_train_args(config, fast_init=1)
    train_args_baseline = build_2dgs_baseline_args(config)
    render_args = build_render_args(config)

    time_dir = output_dir / "time_based"

    for time_limit in time_budgets:
        time_label = format_time_budget(time_limit)
        print(f"\n{'#'*80}")
        print(f"# Time budget: {time_label} ({time_limit}s)")
        print(f"{'#'*80}")

        # Setup directories
        ours_dir = time_dir / f"{time_limit}s" / "ours"
        baseline_dir = time_dir / f"{time_limit}s" / "baseline_2dgs"
        ensure_dir(ours_dir)
        ensure_dir(baseline_dir)

        # Metrics files
        metrics_ours = time_dir / f"{time_limit}s" / "metrics_ours.csv"
        metrics_baseline = time_dir / f"{time_limit}s" / "metrics_baseline.csv"

        # Initialize metrics files
        header = "scan,accuracy,completeness,overall"
        for mf in [metrics_ours, metrics_baseline]:
            if not mf.exists():
                mf.write_text(header + "\n")

        for scene in scenes:
            scan_id = parse_scan_id(scene)
            source = f"{dtu_source}/{scene}"

            # --- Our method ---
            exp_dir_ours = ours_dir / scene
            marker_ours = exp_dir_ours / ".complete"

            if resume and marker_exists(marker_ours):
                print(f"Skipping {scene} (ours) - already complete")
            else:
                print(f"\n--- Training {scene} with our method ({time_label}) ---")
                run_training(
                    source=source,
                    exp_dir=str(exp_dir_ours),
                    train_args=train_args_ours,
                    time_limit=time_limit
                )
                run_rendering(source, str(exp_dir_ours), render_args)

                mesh_file = find_mesh_file(str(exp_dir_ours))
                if mesh_file:
                    create_mesh_snapshot(
                        mesh_file,
                        str(exp_dir_ours / "train" / "snapshot.png")
                    )
                    run_evaluation(
                        mesh_file, scan_id, dtu_source, dtu_source,
                        str(metrics_ours), scene
                    )
                    create_marker_file(marker_ours)

            # --- Baseline 2DGS ---
            exp_dir_baseline = baseline_dir / scene
            marker_baseline = exp_dir_baseline / ".complete"

            if resume and marker_exists(marker_baseline):
                print(f"Skipping {scene} (baseline) - already complete")
            else:
                print(f"\n--- Training {scene} with 2DGS baseline ({time_label}) ---")
                run_training(
                    source=source,
                    exp_dir=str(exp_dir_baseline),
                    train_args=train_args_baseline,
                    time_limit=time_limit,
                    use_2dgs_baseline=True,
                    repo_dir_2dgs=repo_dir_2dgs
                )
                run_rendering(source, str(exp_dir_baseline), render_args)

                mesh_file = find_mesh_file(str(exp_dir_baseline))
                if mesh_file:
                    create_mesh_snapshot(
                        mesh_file,
                        str(exp_dir_baseline / "train" / "snapshot.png")
                    )
                    run_evaluation(
                        mesh_file, scan_id, dtu_source, dtu_source,
                        str(metrics_baseline), scene
                    )
                    create_marker_file(marker_baseline)


def run_iteration_based_evaluation(
    config: Dict[str, Any],
    scenes: List[str],
    iteration_checkpoints: List[int],
    output_dir: Path,
    resume: bool = True
) -> None:
    """
    Run iteration-based evaluation.

    Trains our method to maximum iterations and evaluates at checkpoints.
    """
    dtu_source = config['experiment']['dtu_source']
    repo_dir_2dgs = config['experiment']['repo_dir_2dgs']

    train_args_ours = build_train_args(config, fast_init=1)
    train_args_baseline = build_2dgs_baseline_args(config)
    render_args = build_render_args(config)

    max_iterations = max(iteration_checkpoints)
    iter_dir = output_dir / "iteration_based"

    for method_name, train_args, use_2dgs in [
        ("ours", train_args_ours, False),
        ("baseline_2dgs", train_args_baseline, True)
    ]:
        method_dir = iter_dir / method_name
        ensure_dir(method_dir)

        for scene in scenes:
            scan_id = parse_scan_id(scene)
            source = f"{dtu_source}/{scene}"
            exp_dir = method_dir / scene
            marker = exp_dir / ".complete"

            if resume and marker_exists(marker):
                print(f"Skipping {scene} ({method_name}) - already complete")
                continue

            print(f"\n--- Training {scene} with {method_name} ({max_iterations} iters) ---")

            # Train to max iterations with checkpoints
            checkpoint_args = "--save_iterations" + "".join([f" {it}" for it in iteration_checkpoints])

            run_training(
                source=source,
                exp_dir=str(exp_dir),
                train_args=f"{train_args} {checkpoint_args}",
                iterations=max_iterations,
                use_2dgs_baseline=use_2dgs,
                repo_dir_2dgs=repo_dir_2dgs
            )

            # Evaluate at each checkpoint
            for checkpoint in iteration_checkpoints:
                checkpoint_dir = exp_dir / f"point_cloud" / f"iteration_{checkpoint}"
                if checkpoint_dir.exists():
                    # Render mesh at this checkpoint
                    render_args_checkpoint = f"{render_args} --iteration {checkpoint}"
                    run_rendering(source, str(exp_dir), render_args_checkpoint)

            run_rendering(source, str(exp_dir), render_args)

            mesh_file = find_mesh_file(str(exp_dir))
            if mesh_file:
                create_marker_file(marker)


def main():
    parser = argparse.ArgumentParser(description="Run 2DGS fast convergence evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Specific scenes to evaluate (overrides config)"
    )
    parser.add_argument(
        "--time-budgets",
        nargs="+",
        type=int,
        default=None,
        help="Time budgets in seconds (overrides config)"
    )
    parser.add_argument(
        "--skip-time-based",
        action="store_true",
        help="Skip time-based evaluation"
    )
    parser.add_argument(
        "--skip-iteration-based",
        action="store_true",
        help="Skip iteration-based evaluation"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from previous runs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    args = parser.parse_args()

    # Load configuration
    config_path = args.config or get_default_config_path()
    config = load_config(config_path)

    # Override scenes if specified
    if args.scenes:
        config['scenes'] = args.scenes

    # Override time budgets if specified
    if args.time_budgets:
        config['time_budgets'] = args.time_budgets

    scenes = config['scenes']
    time_budgets = config['time_budgets']
    iteration_checkpoints = config['iteration_checkpoints']

    # Setup output directory
    output_dir = Path(config['experiment']['output_dir'])
    ensure_dir(output_dir)

    # Save config to output directory
    save_config(config, output_dir / "config.yaml")

    print(f"Output directory: {output_dir}")
    print(f"Scenes: {scenes}")
    print(f"Time budgets: {time_budgets}")
    print(f"Iteration checkpoints: {iteration_checkpoints}")

    if args.dry_run:
        print("\n[DRY RUN] Would run evaluation with above settings")
        return

    # Run evaluations
    resume = not args.no_resume

    if not args.skip_time_based:
        print("\n" + "="*80)
        print("TIME-BASED EVALUATION")
        print("="*80)
        run_time_based_evaluation(
            config, scenes, time_budgets, output_dir, resume
        )

    if not args.skip_iteration_based:
        print("\n" + "="*80)
        print("ITERATION-BASED EVALUATION")
        print("="*80)
        run_iteration_based_evaluation(
            config, scenes, iteration_checkpoints, output_dir, resume
        )

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
