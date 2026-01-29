"""
I/O utility functions for the evaluation pipeline.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import yaml
import pandas as pd


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist and return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def load_results_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load results from CSV file."""
    return pd.read_csv(csv_path)


def save_results_csv(df: pd.DataFrame, csv_path: Union[str, Path]) -> None:
    """Save DataFrame to CSV file."""
    ensure_dir(Path(csv_path).parent)
    df.to_csv(csv_path, index=False)


def load_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], json_path: Union[str, Path]) -> None:
    """Save data to JSON file."""
    ensure_dir(Path(json_path).parent)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def get_experiment_dirs(
    base_dir: Union[str, Path],
    pattern: Optional[str] = None
) -> List[Path]:
    """
    Get list of experiment directories.

    Parameters
    ----------
    base_dir : str or Path
        Base directory containing experiments
    pattern : str, optional
        Glob pattern for filtering directories

    Returns
    -------
    List[Path]
        Sorted list of experiment directories
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    if pattern:
        dirs = list(base_dir.glob(pattern))
    else:
        dirs = [d for d in base_dir.iterdir() if d.is_dir()]

    return sorted(dirs)


def find_mesh_file(exp_dir: Union[str, Path]) -> Optional[Path]:
    """
    Find the mesh file in an experiment directory.

    Looks for fuse_post.ply in the train subdirectory structure.
    """
    exp_dir = Path(exp_dir)
    train_dir = exp_dir / "train"

    if not train_dir.exists():
        return None

    # Find the iteration subdirectory
    subdirs = list(train_dir.iterdir())
    if not subdirs:
        return None

    mesh_dir = subdirs[0]
    mesh_file = mesh_dir / "fuse_post.ply"

    if mesh_file.exists():
        return mesh_file
    return None


def load_evaluation_results(results_json: Union[str, Path]) -> Dict[str, float]:
    """
    Load evaluation results from JSON file.

    Returns dict with keys: accuracy (mean_d2s), completeness (mean_s2d), overall
    """
    data = load_json(results_json)
    return {
        'accuracy': data.get('mean_d2s', float('nan')),
        'completeness': data.get('mean_s2d', float('nan')),
        'overall': data.get('overall', float('nan'))
    }


def check_experiment_complete(exp_dir: Union[str, Path]) -> bool:
    """Check if an experiment has completed successfully."""
    exp_dir = Path(exp_dir)
    mesh_file = find_mesh_file(exp_dir)
    return mesh_file is not None and mesh_file.exists()


def create_marker_file(path: Union[str, Path], content: str = "") -> None:
    """Create a marker file to indicate completion."""
    with open(path, 'w') as f:
        f.write(content)


def marker_exists(path: Union[str, Path]) -> bool:
    """Check if a marker file exists."""
    return Path(path).exists()


def get_default_config_path() -> Path:
    """Get path to default configuration file."""
    return Path(__file__).parent.parent / "config" / "default_config.yaml"


def get_sota_baselines_path() -> Path:
    """Get path to SOTA baselines configuration file."""
    return Path(__file__).parent.parent / "config" / "sota_baselines.yaml"


def format_time_budget(seconds: int) -> str:
    """Format time budget as human-readable string."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}min"
    else:
        hours = seconds // 3600
        return f"{hours}h"


def parse_scan_id(scene_name: str) -> int:
    """Extract numeric scan ID from scene name (e.g., 'scan24' -> 24)."""
    return int(scene_name.replace("scan", ""))
