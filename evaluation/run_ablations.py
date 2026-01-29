#!/usr/bin/env python3
"""
Ablation study for surface reconstruction.

Evaluates the effect of:
- Normal estimation vs random normals
- Normal consistency regularization strength
- Depth filtering during initialization
- Matches per reference image count
- Densification enabling/disabling

Usage:
    python -m evaluation.run_ablations --ablation normal_estimation
    python -m evaluation.run_ablations --ablation normal_regularization
    python -m evaluation.run_ablations --ablation depth_filtering
    python -m evaluation.run_ablations --ablation matches_per_ref
    python -m evaluation.run_ablations --ablation densification
    python -m evaluation.run_ablations --ablation all
"""

import json
import os
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import yaml

# Optional mesh snapshot support
try:
    from evaluation.generate_figures import render_mesh_snapshot
    HAS_MESH_SNAPSHOT = True
except ImportError:
    try:
        from generate_figures import render_mesh_snapshot
        HAS_MESH_SNAPSHOT = True
    except ImportError:
        HAS_MESH_SNAPSHOT = False
        print("Warning: mesh_snapshot not available, skipping snapshots")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AblationConfig:
    """Configuration for ablation experiments."""
    output_dir: str = "ablation_results"
    dtu_source: str = "../DTU"

    # Time budgets to evaluate (in seconds)
    time_budgets: List[int] = field(default_factory=lambda: [30, 60, 90, 120, 150, 180])

    # Scenes to evaluate
    scenes: List[str] = field(default_factory=lambda: [
        'scan24', 'scan37', 'scan40', 'scan55', 'scan63'
    ])

    # Base training parameters (best configuration)
    base_params: Dict[str, Any] = field(default_factory=lambda: {
        'fast_init': 1,
        'matches_per_ref': 15000,
        'nns_per_ref': 3,
        'scaling_factor': 0.001,
        'proj_err_tolerance': 0.01,
        'normal_estimate_knn': 20,
        'estimate_normals': 1,
        'initial_depth_pruning': 1,
        'roma_model': 'Indoor',
        'iterations': 5000,
        'position_lr_init': 0.000016,
        'position_lr_final': 0.0000016,
        'position_lr_delay_mult': 0.01,
        'position_lr_max_steps': 30000,
        'feature_lr': 0.0025,
        'opacity_lr': 0.025,
        'scaling_lr': 0.005,
        'rotation_lr': 0.001,
        'percent_dense': 0.01,
        'lambda_dssim': 0.2,
        'lambda_dist': 0.0,
        'lambda_normal': 0.01,
        'densify_from_iter': 0,
        'dist_reg_from': 0,
        'normal_reg_from': 0,
        'opacity_cull': 0.005,
        'densification_interval': 100,
        'opacity_reset_interval': 30000,
        'densify_until_iter': 0,
        'densify_grad_threshold': 0.0002,
    })

    # Rendering parameters for mesh extraction
    render_params: Dict[str, Any] = field(default_factory=lambda: {
        'skip_train': True,
        'depth_ratio': 1.0,
        'num_cluster': 1,
        'voxel_size': 0.004,
        'sdf_trunc': 0.016,
        'depth_trunc': 3.0,
    })


# =============================================================================
# Helper Functions
# =============================================================================

def build_train_command(source: str, exp_dir: str, params: Dict, time_limit: int) -> str:
    """Build the training command string."""
    args = ' '.join(f"--{k} {v}" for k, v in params.items())
    return f"python train.py -s {source} -m {exp_dir} -r 2 {args} --time_limit {time_limit}"


def build_render_command(source: str, exp_dir: str, params: Dict, iteration: int = -1) -> str:
    """Build the render/mesh extraction command."""
    args_list = []
    for k, v in params.items():
        if v is True:
            args_list.append(f"--{k}")
        else:
            args_list.append(f"--{k} {v}")
    args = ' '.join(args_list)
    return f"python render.py -s {source} -m {exp_dir} --iteration {iteration} {args}"


def run_command(cmd: str) -> bool:
    """Run a shell command and return success status."""
    print(f"Running: {cmd}")
    return os.system(cmd) == 0


def evaluate_mesh(mesh_file: str, scan_id: str, dtu_dir: str) -> Optional[Dict[str, float]]:
    """
    Evaluate mesh quality and return metrics.

    Returns dict with 'accuracy', 'completeness', 'overall' or None on failure.
    """
    cmd = (
        f'python scripts/eval_dtu/evaluate_single_scene.py '
        f'--input_mesh {mesh_file} '
        f'--scan_id {scan_id} '
        f'--mask_dir {dtu_dir} '
        f'--DTU {dtu_dir}'
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Evaluation failed: {result.stderr}")
        return None

    # Parse output: "accuracy completeness overall"
    for line in result.stdout.strip().split('\n'):
        if not line.startswith('cull'):
            parts = line.split()
            if len(parts) >= 3:
                return {
                    'accuracy': float(parts[0]),
                    'completeness': float(parts[1]),
                    'overall': float(parts[2])
                }
    return None


def find_mesh_file(exp_dir: str) -> Optional[str]:
    """Find the extracted mesh file in experiment directory."""
    train_dir = Path(exp_dir) / "train"
    if not train_dir.exists():
        return None
    subdirs = list(train_dir.iterdir())
    if subdirs:
        mesh_file = subdirs[0] / "fuse_post.ply"
        if mesh_file.exists():
            return str(mesh_file)
    return None


def save_mesh_snapshot(mesh_file: str, output_path: str) -> None:
    """Save snapshots of the mesh with white and grey backgrounds."""
    if not HAS_MESH_SNAPSHOT:
        return

    output_path = Path(output_path)

    # Save white background version
    try:
        render_mesh_snapshot(
            Path(mesh_file),
            output_path,
            background_color=[1, 1, 1, 1]
        )
        print(f"Snapshot saved: {output_path}")
    except Exception as e:
        print(f"Failed to save snapshot: {e}")

    # Save grey background version
    try:
        grey_path = output_path.parent / f"{output_path.stem}_grey{output_path.suffix}"
        render_mesh_snapshot(
            Path(mesh_file),
            grey_path,
            background_color=[0.5, 0.5, 0.5, 1]
        )
        print(f"Snapshot saved: {grey_path}")
    except Exception as e:
        print(f"Failed to save grey snapshot: {e}")


def load_completed_experiments(csv_file: Path) -> set:
    """Load set of (experiment, scene, time_budget) tuples already completed."""
    completed = set()
    if csv_file.exists():
        with open(csv_file) as f:
            for line in f.readlines()[1:]:  # Skip header
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    completed.add((parts[0], parts[1], parts[2]))
    return completed


def load_time_checkpoints(exp_dir: str) -> Dict[int, int]:
    """
    Load time checkpoint mapping from a training run.

    Returns dict mapping time_budget (seconds) -> iteration number.
    """
    mapping_file = Path(exp_dir) / "time_checkpoints.json"
    if not mapping_file.exists():
        return {}

    with open(mapping_file) as f:
        raw = json.load(f)
        # Keys are strings like "60.0", convert to int
        return {int(float(k)): v['iteration'] for k, v in raw.items()}


def append_result(csv_file: Path, exp_name: str, scene: str, time_budget: int,
                  metrics: Dict[str, float]) -> None:
    """Append a result row to the CSV file."""
    with open(csv_file, 'a') as f:
        f.write(f"{exp_name},{scene},{time_budget},"
                f"{metrics['accuracy']},{metrics['completeness']},{metrics['overall']}\n")


# =============================================================================
# Core Ablation Runner
# =============================================================================

def run_ablation_experiments(
    config: AblationConfig,
    ablation_name: str,
    experiments: Dict[str, Dict[str, Any]],
    resume: bool = True
) -> str:
    """
    Run a set of ablation experiments.

    Parameters
    ----------
    config : AblationConfig
        Global configuration
    ablation_name : str
        Name of this ablation study (used for output directory)
    experiments : dict
        Mapping of experiment_name -> parameter overrides
    resume : bool
        Whether to skip already-completed experiments

    Returns
    -------
    str
        Path to results CSV file
    """
    # Setup output directory
    ablation_dir = Path(config.output_dir) / ablation_name
    ablation_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_file = ablation_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.safe_dump({
            'ablation_name': ablation_name,
            'experiments': experiments,
            'time_budgets': config.time_budgets,
            'scenes': config.scenes,
            'base_params': config.base_params,
        }, f)

    # Initialize CSV file
    csv_file = ablation_dir / "results.csv"
    if not csv_file.exists():
        csv_file.write_text("experiment,scene,time_budget,accuracy,completeness,overall\n")

    # Load completed experiments for resume
    completed = load_completed_experiments(csv_file) if resume else set()

    # Run each experiment
    for exp_name, param_changes in experiments.items():
        print(f"\n{'='*70}")
        print(f"Experiment: {exp_name}")
        print(f"Parameter changes: {param_changes}")
        print(f"{'='*70}")

        # Merge with base params
        params = {**config.base_params, **param_changes}

        for scene in config.scenes:
            scan_id = scene.replace('scan', '')
            source = f"{config.dtu_source}/{scene}"

            # Use max time budget for training (checkpoints will be saved)
            max_time = max(config.time_budgets)
            exp_dir = str(ablation_dir / exp_name / scene)

            # Check if we need to run training (any time budget incomplete)
            needs_training = any(
                (exp_name, scene, str(tb)) not in completed
                for tb in config.time_budgets
            )

            if not needs_training:
                print(f"Skipping {exp_name}/{scene} - all time budgets complete")
                continue

            print(f"\n--- {exp_name}: {scene} (training up to {max_time}s) ---")

            # Train with time checkpoints
            time_checkpoints_arg = ' '.join(str(t) for t in config.time_budgets)
            train_cmd = build_train_command(source, exp_dir, params, max_time)
            train_cmd += f" --time_checkpoints {time_checkpoints_arg}"

            if not run_command(train_cmd):
                print(f"Training failed for {scene}")
                continue

            # Load time checkpoint -> iteration mapping
            time_mapping = load_time_checkpoints(exp_dir)
            if not time_mapping:
                print(f"Warning: No time checkpoints found for {scene}, using final iteration")

            # Evaluate at each time budget
            for time_budget in config.time_budgets:
                if (exp_name, scene, str(time_budget)) in completed:
                    print(f"Skipping evaluation at {time_budget}s - already complete")
                    continue

                # Get iteration for this time budget
                iteration = time_mapping.get(time_budget, -1)
                if iteration == -1:
                    print(f"Warning: No checkpoint for {time_budget}s, using final iteration")

                print(f"\n  Evaluating at {time_budget}s (iteration {iteration})...")

                # Extract mesh for this time checkpoint
                render_cmd = build_render_command(source, exp_dir, config.render_params, iteration)

                if not run_command(render_cmd):
                    print(f"Rendering failed for {scene} at {time_budget}s")
                    continue

                # Find and evaluate mesh
                mesh_file = find_mesh_file(exp_dir)
                if not mesh_file:
                    print(f"No mesh found for {scene}")
                    continue

                # Save snapshot
                snapshot_dir = ablation_dir / exp_name / scene / "snapshots"
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                save_mesh_snapshot(mesh_file, str(snapshot_dir / f"mesh_{time_budget}s.png"))

                # Evaluate
                metrics = evaluate_mesh(mesh_file, scan_id, config.dtu_source)
                if metrics:
                    append_result(csv_file, exp_name, scene, time_budget, metrics)
                    print(f"  Results ({time_budget}s): acc={metrics['accuracy']:.4f}, "
                          f"comp={metrics['completeness']:.4f}, overall={metrics['overall']:.4f}")

    print(f"\n{'='*70}")
    print(f"Ablation '{ablation_name}' complete. Results saved to: {csv_file}")
    print(f"{'='*70}")

    return str(csv_file)


# =============================================================================
# Ablation Functions
# =============================================================================

def run_normal_estimation_ablation(config: AblationConfig, resume: bool = True) -> str:
    """
    Compare normal estimation (PCA-based) vs random normals.

    Experiments:
    - estimated_normals: estimate_normals=1 (PCA normal estimation)
    - random_normals: estimate_normals=0 (random initialization)
    """
    experiments = {
        'estimated_normals': {'estimate_normals': 1},
        'random_normals': {'estimate_normals': 0},
    }

    return run_ablation_experiments(
        config=config,
        ablation_name='normal_estimation',
        experiments=experiments,
        resume=resume
    )


def run_normal_regularization_ablation(
    config: AblationConfig,
    lambda_values: List[float] = None,
    resume: bool = True
) -> str:
    """
    Evaluate effect of normal consistency regularization strength.

    Parameters
    ----------
    lambda_values : list of float, optional
        Values of lambda_normal to test. Default: [0.0, 0.005, 0.01, 0.025, 0.05]
        0.0 means no regularization.
    """
    if lambda_values is None:
        lambda_values = [0.0, 0.01, 0.05]

    experiments = {
        f'lambda_{v}'.replace('.', 'p'): {'lambda_normal': v}
        for v in lambda_values
    }

    return run_ablation_experiments(
        config=config,
        ablation_name='normal_regularization',
        experiments=experiments,
        resume=resume
    )


def run_depth_filtering_ablation(config: AblationConfig, resume: bool = True) -> str:
    """
    Compare depth filtering vs no depth filtering during initialization.

    Experiments:
    - with_depth_filter: initial_depth_pruning=1
    - no_depth_filter: initial_depth_pruning=0
    """
    experiments = {
        'with_depth_filter': {'initial_depth_pruning': 1},
        'no_depth_filter': {'initial_depth_pruning': 0},
        'no_opacity_tuning': {'density_dependent_opacities': 0},
        'no_depth_filter_no_opacity_tuning': {'initial_depth_pruning': 0,'density_dependent_opacities': 0},
    }

    return run_ablation_experiments(
        config=config,
        ablation_name='depth_filtering',
        experiments=experiments,
        resume=resume
    )


def run_matches_per_ref_ablation(
    config: AblationConfig,
    matches_values: List[int] = None,
    resume: bool = True
) -> str:
    """
    Evaluate effect of matches_per_ref on reconstruction quality.

    Parameters
    ----------
    matches_values : list of int, optional
        Values of matches_per_ref to test. Default: [15000, 25000]
    """
    if matches_values is None:
        matches_values = [15000, 20000, 25000]

    experiments = {
        f'matches_{v}': {'matches_per_ref': v}
        for v in matches_values
    }

    return run_ablation_experiments(
        config=config,
        ablation_name='matches_per_ref',
        experiments=experiments,
        resume=resume
    )


def run_densification_ablation(config: AblationConfig, resume: bool = True) -> str:
    """
    Compare enabling vs disabling densification.

    Experiments:
    - with_densification: densify_from_iter=0 (enable densification)
    - no_densification: densify_from_iter=0 (disable densification)
    """
    experiments = {
        'with_densification': {'densify_from_iter': 0, 'densify_until_iter': 15000},
        'no_densification': {'densify_from_iter': 0, 'densify_until_iter': 0},
    }

    return run_ablation_experiments(
        config=config,
        ablation_name='densification',
        experiments=experiments,
        resume=resume
    )


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument(
        '--ablation',
        type=str,
        choices=['normal_estimation', 'normal_regularization', 'depth_filtering', 'matches_per_ref', 'densification', 'all'],
        default='all',
        help='Which ablation to run'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='ablation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--dtu-source',
        type=str,
        default='../DTU',
        help='Path to DTU dataset'
    )
    parser.add_argument(
        '--time-budgets',
        type=int,
        nargs='+',
        default=[30,60,90,120,150,180],
        help='Time budgets in seconds'
    )
    parser.add_argument(
        '--scenes',
        type=str,
        nargs='+',
        default=['scan24', 'scan37', 'scan40', 'scan55', 'scan63','scan65','scan69','scan83','scan97','scan105','scan106','scan110','scan114','scan118','scan122'],
        help='Scenes to evaluate'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from previous runs'
    )
    parser.add_argument(
        '--lambda-values',
        type=float,
        nargs='+',
        default=None,
        help='Lambda values for normal regularization ablation'
    )
    parser.add_argument(
        '--matches-values',
        type=int,
        nargs='+',
        default=None,
        help='Matches per ref values for matches_per_ref ablation'
    )

    args = parser.parse_args()

    # Create configuration
    config = AblationConfig(
        output_dir=args.output_dir,
        dtu_source=args.dtu_source,
        time_budgets=args.time_budgets,
        scenes=args.scenes,
    )

    resume = not args.no_resume

    print(f"Output directory: {config.output_dir}")
    print(f"DTU source: {config.dtu_source}")
    print(f"Time budgets: {config.time_budgets}s")
    print(f"Scenes: {config.scenes}")
    print(f"Resume: {resume}")

    # Run requested ablation(s)
    if args.ablation in ['normal_estimation', 'all']:
        print("\n" + "#"*70)
        print("# Normal Estimation Ablation")
        print("#"*70)
        run_normal_estimation_ablation(config, resume=resume)

    if args.ablation in ['normal_regularization', 'all']:
        print("\n" + "#"*70)
        print("# Normal Regularization Ablation")
        print("#"*70)
        run_normal_regularization_ablation(config, lambda_values=args.lambda_values, resume=resume)

    if args.ablation in ['depth_filtering', 'all']:
        print("\n" + "#"*70)
        print("# Depth Filtering Ablation")
        print("#"*70)
        run_depth_filtering_ablation(config, resume=resume)

    if args.ablation in ['matches_per_ref', 'all']:
        print("\n" + "#"*70)
        print("# Matches Per Ref Ablation")
        print("#"*70)
        run_matches_per_ref_ablation(config, matches_values=args.matches_values, resume=resume)

    if args.ablation in ['densification', 'all']:
        print("\n" + "#"*70)
        print("# Densification Ablation")
        print("#"*70)
        run_densification_ablation(config, resume=resume)


if __name__ == "__main__":
    main()
