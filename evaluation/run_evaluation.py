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
    time_checkpoints: Optional[List[int]] = None,
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

    cmd = f"python {train_script} -s {source} -m {exp_dir} -r 2 {train_args}"

    if time_limit is not None:
        cmd += f" --time_limit {time_limit}"
    if iterations is not None:
        cmd += f" --iterations {iterations}"
    if time_checkpoints is not None and len(time_checkpoints) > 0:
        cmd += " --time_checkpoints " + " ".join(str(t) for t in time_checkpoints)

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


def find_mesh_file(exp_dir: str, iteration: Optional[int] = None) -> Optional[str]:
    """Find the generated mesh file in experiment directory.

    Args:
        exp_dir: Experiment directory path
        iteration: Specific iteration to look for (e.g., for ours_1234/)
    """
    train_dir = Path(exp_dir) / "train"
    if not train_dir.exists():
        return None

    if iteration is not None:
        # Look for specific iteration directory
        iter_dir = train_dir / f"ours_{iteration}"
        if iter_dir.exists():
            mesh_file = iter_dir / "fuse_post.ply"
            if mesh_file.exists():
                return str(mesh_file)
        return None

    # Default: find any mesh
    subdirs = list(train_dir.iterdir())
    if not subdirs:
        return None

    mesh_file = subdirs[0] / "fuse_post.ply"
    if mesh_file.exists():
        return str(mesh_file)
    return None


def load_time_checkpoints_mapping(exp_dir: str) -> Dict[int, Dict[str, Any]]:
    """Load the time checkpoints mapping from a training run.

    Returns dict mapping time (seconds) -> {"iteration": int, "actual_time": float}
    """
    mapping_file = Path(exp_dir) / "time_checkpoints.json"
    if not mapping_file.exists():
        return {}

    with open(mapping_file) as f:
        # JSON keys are strings like "60.0", convert to int via float
        raw_mapping = yaml.safe_load(f)
        return {int(float(k)): v for k, v in raw_mapping.items()}


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
    resume: bool = True,
    time_budgets_per_method: Optional[Dict[str, List[int]]] = None,
) -> None:
    """
    Run time-based evaluation across all scenes and time budgets.

    Runs a single training per scene that saves at all time checkpoints,
    then evaluates the meshes at each checkpoint for convergence analysis.

    Args:
        config: Configuration dict
        scenes: List of scene names to evaluate
        time_budgets: Default time budgets (used if time_budgets_per_method not specified)
        output_dir: Output directory for results
        resume: Whether to resume from previous runs
        time_budgets_per_method: Optional dict mapping method name to list of time budgets
                                 e.g., {"ours": [60, 120, 180], "baseline_2dgs": [60, 180, 360, 720]}
    """
    dtu_source = config['experiment']['dtu_source']
    repo_dir_2dgs = config['experiment']['repo_dir_2dgs']

    train_args_ours = build_train_args(config, fast_init=1)
    train_args_baseline = build_2dgs_baseline_args(config)
    render_args = build_render_args(config)

    time_dir = output_dir / "time_based"

    # Determine time budgets for each method
    if time_budgets_per_method is None:
        time_budgets_per_method = {
            "ours": time_budgets,
            "baseline_2dgs": time_budgets,
        }

    time_budgets_ours = time_budgets_per_method.get("ours", time_budgets)
    time_budgets_baseline = time_budgets_per_method.get("baseline_2dgs", time_budgets)

    max_time_ours = max(time_budgets_ours)
    max_time_baseline = max(time_budgets_baseline)

    # Setup directories for each method
    ours_dir = time_dir / "ours"
    baseline_dir = time_dir / "baseline_2dgs"
    ensure_dir(ours_dir)
    ensure_dir(baseline_dir)

    # Initialize metrics files for each time checkpoint
    header = "scan,accuracy,completeness,overall"
    for time_budget in time_budgets_ours:
        metrics_file = ours_dir / f"metrics_{time_budget}s.csv"
        if not metrics_file.exists():
            metrics_file.write_text(header + "\n")
    for time_budget in time_budgets_baseline:
        metrics_file = baseline_dir / f"metrics_{time_budget}s.csv"
        if not metrics_file.exists():
            metrics_file.write_text(header + "\n")

    for scene in scenes:
        scan_id = parse_scan_id(scene)
        source = f"{dtu_source}/{scene}"

        # --- Our method ---
        exp_dir_ours = ours_dir / scene
        marker_ours = exp_dir_ours / ".train_complete"

        # Train once with all time checkpoints
        if resume and marker_exists(marker_ours):
            print(f"Skipping training {scene} (ours) - already complete")
        else:
            print(f"\n--- Training {scene} with our method (max {max_time_ours}s, checkpoints: {time_budgets_ours}) ---")
            run_training(
                source=source,
                exp_dir=str(exp_dir_ours),
                train_args=train_args_ours,
                time_limit=max_time_ours,
                time_checkpoints=time_budgets_ours
            )
            create_marker_file(marker_ours)

        # Evaluate at each time checkpoint
        time_mapping = load_time_checkpoints_mapping(str(exp_dir_ours))
        for time_budget in time_budgets_ours:
            marker_eval = exp_dir_ours / f".eval_{time_budget}s_complete"
            if resume and marker_exists(marker_eval):
                print(f"Skipping evaluation {scene} (ours) at {time_budget}s - already complete")
                continue

            if time_budget not in time_mapping:
                print(f"Warning: No checkpoint found for {time_budget}s in {scene} (ours)")
                continue

            iteration = time_mapping[time_budget]["iteration"]
            print(f"\n--- Evaluating {scene} (ours) at {time_budget}s (iter {iteration}) ---")

            # Render mesh at this iteration
            render_args_iter = f"{render_args} --iteration {iteration}"
            run_rendering(source, str(exp_dir_ours), render_args_iter)

            mesh_file = find_mesh_file(str(exp_dir_ours), iteration=iteration)
            if mesh_file:
                metrics_file = ours_dir / f"metrics_{time_budget}s.csv"
                run_evaluation(
                    mesh_file, scan_id, dtu_source, dtu_source,
                    str(metrics_file), scene
                )
                create_marker_file(marker_eval)

        # --- Baseline 2DGS ---
        exp_dir_baseline = baseline_dir / scene
        marker_baseline = exp_dir_baseline / ".train_complete"

        # Train once with all time checkpoints (if baseline supports it)
        if resume and marker_exists(marker_baseline):
            print(f"Skipping training {scene} (baseline) - already complete")
        else:
            print(f"\n--- Training {scene} with 2DGS baseline (max {max_time_baseline}s, checkpoints: {time_budgets_baseline}) ---")
            run_training(
                source=source,
                exp_dir=str(exp_dir_baseline),
                train_args=train_args_baseline,
                time_limit=max_time_baseline,
                time_checkpoints=time_budgets_baseline,
                use_2dgs_baseline=True,
                repo_dir_2dgs=repo_dir_2dgs
            )
            create_marker_file(marker_baseline)

        # Evaluate at each time checkpoint
        time_mapping = load_time_checkpoints_mapping(str(exp_dir_baseline))
        for time_budget in time_budgets_baseline:
            marker_eval = exp_dir_baseline / f".eval_{time_budget}s_complete"
            if resume and marker_exists(marker_eval):
                print(f"Skipping evaluation {scene} (baseline) at {time_budget}s - already complete")
                continue

            if time_budget not in time_mapping:
                print(f"Warning: No checkpoint found for {time_budget}s in {scene} (baseline)")
                continue

            iteration = time_mapping[time_budget]["iteration"]
            print(f"\n--- Evaluating {scene} (baseline) at {time_budget}s (iter {iteration}) ---")

            # Render mesh at this iteration
            render_args_iter = f"{render_args} --iteration {iteration}"
            run_rendering(source, str(exp_dir_baseline), render_args_iter)

            mesh_file = find_mesh_file(str(exp_dir_baseline), iteration=iteration)
            if mesh_file:
                metrics_file = baseline_dir / f"metrics_{time_budget}s.csv"
                run_evaluation(
                    mesh_file, scan_id, dtu_source, dtu_source,
                    str(metrics_file), scene
                )
                create_marker_file(marker_eval)


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
        help="Time budgets in seconds for both methods (overrides config)"
    )
    parser.add_argument(
        "--time-budgets-ours",
        nargs="+",
        type=int,
        default=None,
        help="Time budgets in seconds for 'ours' method only"
    )
    parser.add_argument(
        "--time-budgets-baseline",
        nargs="+",
        type=int,
        default=None,
        help="Time budgets in seconds for '2DGS baseline' method only"
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

    # Build per-method time budgets from config
    time_budgets_per_method = {}
    methods_config = config.get('methods', {})
    for method_name, method_cfg in methods_config.items():
        if 'time_budgets' in method_cfg:
            time_budgets_per_method[method_name] = method_cfg['time_budgets']

    # If no per-method budgets found in config, use default for all methods
    if not time_budgets_per_method:
        time_budgets_per_method = None

    # Command-line arguments override config values
    if args.time_budgets_ours or args.time_budgets_baseline:
        if time_budgets_per_method is None:
            time_budgets_per_method = {}
        if args.time_budgets_ours:
            time_budgets_per_method["ours"] = args.time_budgets_ours
        if args.time_budgets_baseline:
            time_budgets_per_method["baseline_2dgs"] = args.time_budgets_baseline

    # Setup output directory
    output_dir = Path(config['experiment']['output_dir'])
    ensure_dir(output_dir)

    # Save config to output directory
    save_config(config, output_dir / "config.yaml")

    print(f"Output directory: {output_dir}")
    print(f"Scenes: {scenes}")
    if time_budgets_per_method:
        print(f"Time budgets (ours): {time_budgets_per_method['ours']}")
        print(f"Time budgets (baseline): {time_budgets_per_method['baseline_2dgs']}")
    else:
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
            config, scenes, time_budgets, output_dir, resume,
            time_budgets_per_method=time_budgets_per_method
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
