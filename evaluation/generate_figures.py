#!/usr/bin/env python3
"""
Generate mesh comparison figures from evaluation results.

Usage:
    # Generate all default figures (snapshots, comparison grid, convergence graphs)
    python -m evaluation.generate_figures [--results-dir RESULTS_DIR]

    # Generate method/time comparison figure (Ours@T vs 2DGS@T vs 2DGS@converged)
    python -m evaluation.generate_figures --method-comparison --scenes scan24 scan37 scan40 --comparison-time 180
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.utils.io_utils import ensure_dir

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. Mesh snapshot functions will be disabled.")


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def render_mesh_snapshot(
    mesh_path: Path,
    output_path: Path,
    width: int = 1080,
    height: int = 1080,
    camera_distance_factor: float = 1.2,
    background_color: list = None,
) -> bool:
    """
    Render a mesh snapshot using Open3D offscreen renderer.

    Args:
        mesh_path: Path to the mesh file (.ply)
        output_path: Path to save the rendered image
        width: Image width in pixels
        height: Image height in pixels
        camera_distance_factor: Factor to adjust camera distance (smaller = closer)
        background_color: RGBA background color [r, g, b, a], default [1,1,1,1] (white)

    Returns:
        True if successful, False otherwise
    """
    if background_color is None:
        background_color = [1, 1, 1, 1]  # white
    if not OPEN3D_AVAILABLE:
        print("Open3D not available, cannot render mesh snapshot")
        return False

    if not mesh_path.exists():
        print(f"Mesh file not found: {mesh_path}")
        return False

    try:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.full((len(mesh.vertices), 3), 0.5)
        )

        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"

        scene = renderer.scene
        scene.add_geometry("mesh", mesh, material)

        # Keep indirect light for soft shadows, but disable skybox
        # Post-processing will fix the background color
        scene.show_skybox(False)

        # Camera orientation matrix (from DTU dataset coordinate system)
        R = np.array([
            [0.195078, -0.83833,  0.509064],
            [0.000000, -0.641533, -0.501670],
            [-0.580313,  0.741876, -0.213375]
        ])
        up = R.T @ np.array([0, 1, 0])

        center = mesh.get_center()

        eye = np.array([1.8416653703527446, -0.10236946700075061, -1.4249754471438827])
        direction = center - eye
        direction /= np.linalg.norm(direction)
        eye = eye + direction * camera_distance_factor

        scene.camera.look_at(center, eye, up)
        scene.set_background(background_color)

        scene.camera.set_projection(
            60.0,
            width / height,
            0.1,
            1000.0,
            o3d.visualization.rendering.Camera.FovType.Vertical
        )

        view_dir = center - eye
        view_dir /= np.linalg.norm(view_dir)

        # Sun light (original values)
        scene.scene.set_sun_light(
            view_dir.astype(np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
            9000.0
        )
        scene.scene.enable_sun_light(True)

        # Fill light from above-right (original values)
        scene.scene.add_point_light(
            "fill",
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
            np.array(center + np.array([2.0, 2.0, 2.0]), dtype=np.float32),
            50000.0,
            0.0,
            False
        )

        img = renderer.render_to_image()
        ensure_dir(output_path.parent)

        # Post-process to make background pure white if requested
        # Open3D's tone mapping prevents pure white backgrounds
        if background_color == [1, 1, 1, 1]:
            img_arr = np.asarray(img)
            # Find pixels close to the Open3D "white" (~235) and make them pure white
            grey_mask = np.all(img_arr > 230, axis=2)
            img_arr = img_arr.copy()
            img_arr[grey_mask] = [255, 255, 255]
            from PIL import Image
            Image.fromarray(img_arr).save(str(output_path))
        else:
            o3d.io.write_image(str(output_path), img)
        return True

    except Exception as e:
        print(f"Error rendering mesh {mesh_path}: {e}")
        return False


def find_mesh_for_time_budget(
    scene_dir: Path,
    time_budget_seconds: float,
) -> Optional[Path]:
    """
    Find the mesh file for a given time budget using time_checkpoints.json.

    Args:
        scene_dir: Directory containing the scene data
        time_budget_seconds: Target time budget in seconds

    Returns:
        Path to mesh file, or None if not found
    """
    checkpoints_file = scene_dir / "time_checkpoints.json"
    if not checkpoints_file.exists():
        return None

    with open(checkpoints_file, 'r') as f:
        checkpoints = json.load(f)

    time_key = str(float(time_budget_seconds))
    if time_key not in checkpoints:
        available_times = sorted([float(k) for k in checkpoints.keys()])
        closest = min(available_times, key=lambda x: abs(x - time_budget_seconds))
        time_key = str(closest)

    iteration = checkpoints[time_key]["iteration"]
    mesh_path = scene_dir / "train" / f"ours_{iteration}" / "fuse_post.ply"

    if mesh_path.exists():
        return mesh_path

    mesh_path_alt = scene_dir / "train" / f"ours_{iteration}" / "fuse.ply"
    if mesh_path_alt.exists():
        return mesh_path_alt

    return None


def create_mesh_snapshot_comparison(
    results_dir: Path,
    output_path: Path,
    methods: Optional[Dict[str, str]] = None,
    scenes: Optional[List[str]] = None,
    time_budget_seconds: float = 900.0,
    snapshot_size: Tuple[int, int] = (1080, 1080),
    force_regenerate: bool = False,
) -> None:
    """
    Create a comparison figure with mesh snapshots from different methods.

    This function finds meshes from different methods in the evaluation results,
    renders snapshots of each mesh, and creates a side-by-side comparison grid.

    Args:
        results_dir: Directory containing evaluation results
        output_path: Path to save the comparison figure
        methods: Dict mapping method directory names to display labels.
                 Default: {"ours": "Ours", "baseline_2dgs": "2DGS"}
        scenes: List of scene names to include. Default: all found scenes
        time_budget_seconds: Time budget to use for selecting meshes
        snapshot_size: (width, height) for each rendered snapshot
        force_regenerate: If True, regenerate snapshots even if they exist

    Example:
        create_mesh_snapshot_comparison(
            results_dir=Path("evaluation_results/fast_init_convergence_study"),
            output_path=Path("figures/mesh_comparison.png"),
            methods={"ours": "Ours (Fast Init)", "baseline_2dgs": "2DGS"},
            scenes=["scan24", "scan37", "scan40"],
            time_budget_seconds=600.0,
        )
    """
    if methods is None:
        methods = {
            "ours": "Ours",
            "baseline_2dgs": "2DGS",
        }

    time_dir = results_dir / "time_based"
    if not time_dir.exists():
        print(f"Error: Time-based directory not found: {time_dir}")
        return

    if scenes is None:
        first_method_dir = time_dir / list(methods.keys())[0]
        if first_method_dir.exists():
            scenes = sorted([
                d.name for d in first_method_dir.iterdir()
                if d.is_dir() and d.name.startswith("scan")
            ])
        else:
            print(f"Error: Method directory not found: {first_method_dir}")
            return

    if not scenes:
        print("No scenes found")
        return

    print(f"Creating mesh snapshot comparison for {len(scenes)} scenes: {scenes}")
    print(f"Methods: {list(methods.keys())}")
    print(f"Time budget: {time_budget_seconds}s")

    n_methods = len(methods)
    n_scenes = len(scenes)

    snapshots: Dict[str, Dict[str, Optional[Path]]] = {
        method: {} for method in methods
    }

    snapshot_dir = results_dir / "snapshots" / f"{int(time_budget_seconds)}s"
    ensure_dir(snapshot_dir)

    for method_name in methods:
        method_dir = time_dir / method_name
        if not method_dir.exists():
            print(f"Warning: Method directory not found: {method_dir}")
            continue

        for scene in scenes:
            scene_dir = method_dir / scene
            if not scene_dir.exists():
                print(f"Warning: Scene directory not found: {scene_dir}")
                snapshots[method_name][scene] = None
                continue

            snapshot_path = snapshot_dir / method_name / f"{scene}.png"

            if snapshot_path.exists() and not force_regenerate:
                print(f"  Using existing snapshot: {snapshot_path}")
                snapshots[method_name][scene] = snapshot_path
                continue

            mesh_path = find_mesh_for_time_budget(scene_dir, time_budget_seconds)
            if mesh_path is None:
                print(f"  No mesh found for {method_name}/{scene}")
                snapshots[method_name][scene] = None
                continue

            print(f"  Rendering {method_name}/{scene} from {mesh_path.name}...")
            # Render with white background (default)
            success = render_mesh_snapshot(
                mesh_path,
                snapshot_path,
                width=snapshot_size[0],
                height=snapshot_size[1],
                background_color=[1, 1, 1, 1],
            )

            # Also render with grey background
            grey_snapshot_path = snapshot_path.parent / f"{snapshot_path.stem}_grey{snapshot_path.suffix}"
            render_mesh_snapshot(
                mesh_path,
                grey_snapshot_path,
                width=snapshot_size[0],
                height=snapshot_size[1],
                background_color=[0.5, 0.5, 0.5, 1],
            )

            if success:
                snapshots[method_name][scene] = snapshot_path
            else:
                snapshots[method_name][scene] = None

    fig, axes = plt.subplots(
        n_methods, n_scenes,
        figsize=(2.5 * n_scenes, 2.5 * n_methods),
        squeeze=False
    )

    for row, (method_name, method_label) in enumerate(methods.items()):
        for col, scene in enumerate(scenes):
            ax = axes[row, col]
            snapshot_path = snapshots.get(method_name, {}).get(scene)

            if snapshot_path and snapshot_path.exists():
                img = plt.imread(str(snapshot_path))
                ax.imshow(img)
            else:
                ax.text(
                    0.5, 0.5, 'No snapshot',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=10, color='gray'
                )
                ax.set_facecolor('#f0f0f0')

            ax.axis('off')

            if row == 0:
                scan_num = scene.replace('scan', '')
                ax.set_title(f'Scan {scan_num}', fontsize=11, fontweight='bold')

            if col == 0:
                ax.text(
                    -0.1, 0.5, method_label,
                    ha='right', va='center',
                    transform=ax.transAxes,
                    fontsize=12, fontweight='bold',
                    rotation=0
                )

    time_label = f"{int(time_budget_seconds)}s"
    if time_budget_seconds >= 60:
        time_label = f"{int(time_budget_seconds // 60)}min"
    fig.suptitle(
        f"Mesh Comparison @ {time_label}",
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()

    ensure_dir(output_path.parent)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved mesh snapshot comparison to: {output_path}")


def create_mesh_comparison_grid(
    results_dir: Path,
    output_path: Path,
    scenes: Optional[List[str]] = None,
) -> None:
    """
    Create a grid of mesh snapshots comparing methods.

    Looks for snapshot images in the experiment directories.
    Supports both directory structures:
    - time_based/ours/, time_based/baseline_2dgs/ (methods directly under time_based)
    - time_based/360s/ours/, time_based/360s/baseline_2dgs/ (methods under time budget dirs)
    """
    time_dir = results_dir / "time_based"

    if not time_dir.exists():
        print(f"Warning: Time-based directory not found: {time_dir}")
        return

    # Check directory structure - methods might be directly under time_based
    # or under a time budget subdirectory
    ours_dir = time_dir / "ours"
    baseline_dir = time_dir / "baseline_2dgs"
    time_budget_name = None

    if ours_dir.exists():
        # Methods are directly under time_based
        print("Using directory structure: time_based/{method}/")
    else:
        # Look for time budget directories (e.g., 360s)
        time_dirs = sorted([
            d for d in time_dir.iterdir()
            if d.is_dir() and d.name not in ['ours', 'baseline_2dgs']
        ])
        if not time_dirs:
            print("No method or time budget directories found")
            return

        latest_time_dir = time_dirs[-1]
        time_budget_name = latest_time_dir.name
        print(f"Using time budget: {time_budget_name}")

        ours_dir = latest_time_dir / "ours"
        baseline_dir = latest_time_dir / "baseline_2dgs"

    if not ours_dir.exists():
        print(f"Warning: 'ours' directory not found: {ours_dir}")
        return

    if scenes is None:
        scenes = sorted([
            d.name for d in ours_dir.iterdir()
            if d.is_dir() and d.name.startswith("scan")
        ])

    if not scenes:
        print("No scene directories found")
        return

    n_scenes = len(scenes)
    print(f"Creating mesh comparison for {n_scenes} scenes: {scenes}")

    # Two rows (methods) x n_scenes columns
    fig, axes = plt.subplots(2, n_scenes, figsize=(2 * n_scenes, 5))

    if n_scenes == 1:
        axes = axes.reshape(-1, 1)

    methods = [
        (ours_dir, 'Ours'),
        (baseline_dir, '2DGS')
    ]

    for row, (method_dir, method_name) in enumerate(methods):
        for col, scene in enumerate(scenes):
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

            # Scene labels on top row
            if row == 0:
                scan_num = scene.replace('scan', '')
                ax.set_title(f'{scan_num}', fontsize=10)

            # Method labels on the left
            if col == 0:
                ax.text(-0.1, 0.5, method_name, ha='right', va='center',
                        transform=ax.transAxes, fontsize=12, fontweight='bold')

    # Add figure title with runtime
    if time_budget_name:
        if time_budget_name.endswith('s'):
            try:
                seconds = int(time_budget_name[:-1])
                minutes = seconds // 60
                title = f"Mesh Comparison (Runtime: {minutes}min)"
            except ValueError:
                title = f"Mesh Comparison (Runtime: {time_budget_name})"
        else:
            title = f"Mesh Comparison (Runtime: {time_budget_name})"
    else:
        title = "Mesh Comparison"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    ensure_dir(output_path.parent)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved mesh comparison grid to: {output_path}")


def create_method_time_comparison(
    results_dir: Path,
    output_path: Path,
    scenes: List[str],
    time_budget: int = 180,
    converged_time: int = 600,
    snapshot_size: Tuple[int, int] = (1080, 1080),
    force_regenerate: bool = False,
    zoom_regions: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    zoom_size: float = 0.35,
    zoom_position: str = "bottom-right",
) -> None:
    """
    Create a comparison figure with three columns:
    1. Ours at specified time budget
    2. 2DGS at the same time budget
    3. 2DGS fully converged (at converged_time)

    Args:
        results_dir: Directory containing evaluation results
        output_path: Path to save the comparison figure
        scenes: List of scene names to include (rows)
        time_budget: Time budget in seconds for ours and 2DGS comparison columns
        converged_time: Time budget in seconds for fully converged 2DGS column
        snapshot_size: (width, height) for each rendered snapshot
        force_regenerate: If True, regenerate snapshots even if they exist
        zoom_regions: Dict mapping scene names to zoom box as (x, y, width, height)
                      where values are fractions (0-1) of image dimensions.
                      Example: {"scan24": (0.3, 0.4, 0.2, 0.2)} zooms into a region
                      starting at 30% from left, 40% from top, with 20% width/height.
        zoom_size: Size of the inset box as fraction of the subplot (default: 0.35)
        zoom_position: Position of inset: "bottom-right", "bottom-left",
                       "top-right", or "top-left" (default: "bottom-right")
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available, cannot create mesh comparison")
        return

    time_dir = results_dir / "time_based"
    if not time_dir.exists():
        print(f"Error: Time-based directory not found: {time_dir}")
        return

    # Define the three columns
    columns = [
        {"method": "ours", "time": time_budget, "label": f"Ours ({time_budget}s)"},
        {"method": "baseline_2dgs", "time": time_budget, "label": f"2DGS ({time_budget}s)"},
        {"method": "baseline_2dgs", "time": converged_time, "label": f"2DGS ({converged_time}s)"},
    ]

    n_cols = len(columns)
    n_rows = len(scenes)

    print(f"Creating method/time comparison figure")
    print(f"  Scenes: {scenes}")
    print(f"  Columns: {[c['label'] for c in columns]}")

    # Render/collect snapshots
    snapshots = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    for col_idx, col_info in enumerate(columns):
        method_name = col_info["method"]
        time_seconds = col_info["time"]

        method_dir = time_dir / method_name
        if not method_dir.exists():
            print(f"Warning: Method directory not found: {method_dir}")
            continue

        snapshot_dir = results_dir / "snapshots" / f"{time_seconds}s" / method_name
        ensure_dir(snapshot_dir)

        for row_idx, scene in enumerate(scenes):
            scene_dir = method_dir / scene
            if not scene_dir.exists():
                print(f"Warning: Scene directory not found: {scene_dir}")
                continue

            snapshot_path = snapshot_dir / f"{scene}.png"

            if snapshot_path.exists() and not force_regenerate:
                print(f"  Using existing snapshot: {snapshot_path}")
                snapshots[row_idx][col_idx] = snapshot_path
                continue

            mesh_path = find_mesh_for_time_budget(scene_dir, float(time_seconds))
            if mesh_path is None:
                print(f"  No mesh found for {method_name}/{scene} at {time_seconds}s")
                continue

            print(f"  Rendering {method_name}/{scene} at {time_seconds}s...")
            success = render_mesh_snapshot(
                mesh_path,
                snapshot_path,
                width=snapshot_size[0],
                height=snapshot_size[1],
                background_color=[1, 1, 1, 1],
            )

            if success:
                snapshots[row_idx][col_idx] = snapshot_path

    # Create the figure
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
        squeeze=False
    )

    # Determine inset position
    inset_positions = {
        "bottom-right": (1 - zoom_size - 0.02, 0.02),
        "bottom-left": (0.02, 0.02),
        "top-right": (1 - zoom_size - 0.02, 1 - zoom_size - 0.02),
        "top-left": (0.02, 1 - zoom_size - 0.02),
    }
    inset_anchor = inset_positions.get(zoom_position, inset_positions["bottom-right"])

    for row_idx, scene in enumerate(scenes):
        # Get zoom region for this scene (if defined)
        zoom_box = zoom_regions.get(scene) if zoom_regions else None

        for col_idx, col_info in enumerate(columns):
            ax = axes[row_idx, col_idx]
            snapshot_path = snapshots[row_idx][col_idx]

            if snapshot_path and snapshot_path.exists():
                img = plt.imread(str(snapshot_path))
                ax.imshow(img)

                # Add zoom inset only for Ours (col 0) and 2DGS converged (col 2)
                if zoom_box is not None and col_idx in [0, 2]:
                    img_h, img_w = img.shape[:2]
                    zx, zy, zw, zh = zoom_box  # fractions

                    # Calculate pixel coordinates for the zoom region
                    x_start = int(zx * img_w)
                    y_start = int(zy * img_h)
                    x_end = int((zx + zw) * img_w)
                    y_end = int((zy + zh) * img_h)

                    # Draw rectangle on main image showing zoom area
                    from matplotlib.patches import Rectangle
                    rect = Rectangle(
                        (x_start, y_start), x_end - x_start, y_end - y_start,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)

                    # Create inset axes
                    inset_ax = ax.inset_axes(
                        [inset_anchor[0], inset_anchor[1], zoom_size, zoom_size]
                    )

                    # Extract and display zoomed region
                    zoomed_img = img[y_start:y_end, x_start:x_end]
                    inset_ax.imshow(zoomed_img)
                    inset_ax.axis('off')

                    # Add border to inset
                    for spine in inset_ax.spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(2)
                        spine.set_visible(True)
            else:
                ax.text(
                    0.5, 0.5, 'No snapshot',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=10, color='gray'
                )
                ax.set_facecolor('#f0f0f0')

            ax.axis('off')

            # Column headers on top row
            if row_idx == 0:
                ax.set_title(col_info["label"], fontsize=12, fontweight='bold')

    plt.tight_layout()

    ensure_dir(output_path.parent)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved method/time comparison figure to: {output_path}")


def create_convergence_graph(
    results_dir: Path,
    output_path: Path,
    metric: str = "overall",
    y_label: str = None,
    time_limits: Optional[Dict[str, int]] = None,
) -> None:
    """
    Create convergence graph showing chamfer distance vs. optimization time.

    Args:
        results_dir: Directory containing evaluation results
        output_path: Path to save the figure
        metric: Which metric to plot ('accuracy', 'completeness', or 'overall')
        y_label: Custom y-axis label (default: auto-generated from metric name)
        time_limits: Dict mapping method names to max time in seconds to display
                     e.g., {"ours": 180, "baseline_2dgs": 720}
    """
    time_dir = results_dir / "time_based"

    if not time_dir.exists():
        print(f"Warning: Time-based directory not found: {time_dir}")
        return

    # Method styling
    methods = {
        "ours": {"label": "Ours", "color": "#ff7f0e", "marker": "o", "linestyle": "-"},
        "baseline_2dgs": {"label": "2DGS", "color": "#2ca02c", "marker": None, "linestyle": "--"},
    }

    fig, ax = plt.subplots(figsize=(5, 4))

    all_values = []
    method_data = {}

    for method_name, style in methods.items():
        method_dir = time_dir / method_name
        if not method_dir.exists():
            print(f"Warning: Method directory not found: {method_dir}")
            continue

        # Find all metrics files for different time checkpoints
        metrics_files = sorted(method_dir.glob("metrics_*s.csv"))
        if not metrics_files:
            print(f"Warning: No metrics files found for {method_name}")
            continue

        # Get time limit for this method
        max_time = time_limits.get(method_name) if time_limits else None

        times = []
        mean_values = []

        for metrics_file in metrics_files:
            # Extract time from filename (e.g., metrics_60s.csv -> 60)
            time_str = metrics_file.stem.replace("metrics_", "").replace("s", "")
            try:
                time_seconds = int(time_str)
            except ValueError:
                continue

            # Skip if beyond time limit for this method
            if max_time is not None and time_seconds > max_time:
                continue

            try:
                df = pd.read_csv(metrics_file)
                if metric in df.columns and len(df) > 0:
                    times.append(time_seconds)
                    mean_values.append(df[metric].mean())
            except Exception as e:
                print(f"Warning: Could not read {metrics_file}: {e}")
                continue

        if times:
            # Sort by time
            sorted_indices = np.argsort(times)
            times = np.array(times)[sorted_indices]
            mean_values = np.array(mean_values)[sorted_indices]

            method_data[method_name] = {
                "times": times,
                "values": mean_values,
                "style": style,
                "final_time": int(times[-1]),
            }
            all_values.extend(mean_values)

    # Plot each method
    for method_name, data in method_data.items():
        style = data["style"]
        label = f"{style['label']}"

        ax.plot(
            data["times"],
            data["values"],
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=8,
            linewidth=2.5,
        )

    ax.set_title("Chamfer distance on DTU dataset", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)
    else:
        ax.set_ylabel(f"Chamfer Distance ({metric.capitalize()})", fontsize=12)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)

    # Set y-axis limits to emphasize differences (don't start at 0)
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min
        # Add some padding
        ax.set_ylim(bottom=max(0, y_min - 0.1 * y_range), top=y_max + 0.1 * y_range)

    ax.set_xlim(left=0)

    # Light grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()

    ensure_dir(output_path.parent)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved convergence graph to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate mesh comparison figure from evaluation results")
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
        "--scenes",
        type=str,
        nargs='+',
        default=None,
        help="Specific scenes to include (default: all)"
    )
    parser.add_argument(
        "--time-budgets",
        type=int,
        nargs='+',
        default=[60, 180, 300, 600, 900],
        help="Time budgets in seconds for mesh snapshot comparison (default: 60 180 300 600 900)"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs='+',
        default=None,
        help="Methods to compare (default: ours baseline_2dgs)"
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of mesh snapshots even if they exist"
    )
    parser.add_argument(
        "--skip-snapshots",
        action="store_true",
        help="Skip mesh snapshot generation (useful if Open3D not available)"
    )
    parser.add_argument(
        "--method-comparison",
        action="store_true",
        help="Generate method/time comparison figure (Ours@T, 2DGS@T, 2DGS@converged)"
    )
    parser.add_argument(
        "--comparison-time",
        type=int,
        default=180,
        help="Time budget in seconds for method comparison (default: 180)"
    )
    parser.add_argument(
        "--converged-time",
        type=int,
        default=600,
        help="Time budget in seconds for fully converged 2DGS (default: 600)"
    )
    parser.add_argument(
        "--zoom-regions",
        type=str,
        nargs='+',
        default=None,
        help="Per-scene zoom regions as 'scene:x,y,w,h' fractions (0-1). Example: --zoom-regions scan24:0.3,0.4,0.2,0.2 scan37:0.5,0.3,0.15,0.15"
    )
    parser.add_argument(
        "--zoom-size",
        type=float,
        default=0.35,
        help="Size of zoom inset as fraction of subplot (default: 0.35)"
    )
    parser.add_argument(
        "--zoom-position",
        type=str,
        default="bottom-right",
        choices=["bottom-right", "bottom-left", "top-right", "top-left"],
        help="Position of zoom inset (default: bottom-right)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    ensure_dir(output_dir)

    setup_publication_style()

    print(f"Generating figures from: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Parse methods argument
    methods_dict = None
    if args.methods:
        methods_dict = {m: m.replace("_", " ").title() for m in args.methods}

    # Generate method/time comparison figure if requested
    if args.method_comparison:
        if not args.scenes:
            print("Error: --scenes is required for --method-comparison")
            return

        # Parse per-scene zoom regions if provided
        zoom_regions = None
        if args.zoom_regions:
            zoom_regions = {}
            for item in args.zoom_regions:
                try:
                    scene, coords = item.split(':')
                    parts = [float(x) for x in coords.split(',')]
                    if len(parts) == 4:
                        zoom_regions[scene] = tuple(parts)
                    else:
                        print(f"Warning: zoom region for {scene} must have 4 values (x,y,w,h)")
                except ValueError:
                    print(f"Warning: Invalid zoom region format: {item}")

        print("\n--- Generating Method/Time Comparison Figure ---")
        create_method_time_comparison(
            results_dir,
            output_path=output_dir / f"method_comparison_{args.comparison_time}s.png",
            scenes=args.scenes,
            time_budget=args.comparison_time,
            converged_time=args.converged_time,
            force_regenerate=args.force_regenerate,
            zoom_regions=zoom_regions,
            zoom_size=args.zoom_size,
            zoom_position=args.zoom_position,
        )
        print(f"\nFigure generation complete. File saved to: {output_dir}")
        return

    # Generate mesh snapshot comparisons
    if not args.skip_snapshots and OPEN3D_AVAILABLE:
        print("\n--- Generating Mesh Snapshot Comparisons ---")
        for time_budget in args.time_budgets:
            print(f"\nTime budget: {time_budget}s")
            create_mesh_snapshot_comparison(
                results_dir,
                output_dir / f"mesh_snapshots_{time_budget}s.png",
                methods=methods_dict,
                scenes=args.scenes,
                time_budget_seconds=float(time_budget),
                force_regenerate=args.force_regenerate,
            )
            """
            create_mesh_snapshot_comparison(
                results_dir,
                output_dir / f"mesh_snapshots_{time_budget}s.pdf",
                methods=methods_dict,
                scenes=args.scenes,
                time_budget_seconds=float(time_budget),
                force_regenerate=args.force_regenerate,
            )
            """
    elif args.skip_snapshots:
        print("\n--- Skipping Mesh Snapshot Generation (--skip-snapshots) ---")
    else:
        print("\n--- Skipping Mesh Snapshot Generation (Open3D not available) ---")

    print("\n--- Generating Mesh Comparison Grid ---")
    create_mesh_comparison_grid(results_dir, output_dir / "mesh_comparison.png", scenes=args.scenes)
    #create_mesh_comparison_grid(results_dir, output_dir / "mesh_comparison.pdf", scenes=args.scenes)

    print("\n--- Generating Convergence Graphs ---")
    # Time limits: ours up to 180s, 2DGS up to 12min (720s)
    time_limits = {"ours": 180, "baseline_2dgs": 720}
    for metric in ["overall"]: #, "accuracy", "completeness"]:
        create_convergence_graph(results_dir, output_dir / f"convergence_{metric}.png", metric=metric, time_limits=time_limits)
        #create_convergence_graph(results_dir, output_dir / f"convergence_{metric}.pdf", metric=metric)

    print(f"\n{'='*60}")
    print(f"Figure generation complete. Files saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
