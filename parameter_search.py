import os
import yaml
from copy import deepcopy
from itertools import product
from mesh_snapshot import mesh_snapshot, mesh_snapshot_from_file
from plot_metrics import plot_metrics_from_runs
import subprocess

BASE_CONFIG = dict(
    matches_per_ref = 15_000,
    nns_per_ref = 3,
    scaling_factor = 0.001,
    proj_err_tolerance = 0.01,
    normal_estimate_knn = 20,
    gaussians_init = '2dgs',
    fast_init = 0,
    estimate_normals = 1,
    roma_model="Indoor",

    iterations=1_000,
    position_lr_init=0.00016,
    position_lr_final=0.0000016,
    position_lr_delay_mult=0.01,
    position_lr_max_steps=30_000,
    feature_lr=0.0025,
    opacity_lr=0.025,
    scaling_lr=0.005,
    rotation_lr=0.001,
    percent_dense=0.01,
    lambda_dssim=0.2,
    lambda_dist=0.0,
    lambda_normal=0.025,
    densify_from_iter=0,
    dist_reg_from=0,
    normal_reg_from=0,
    opacity_cull=0.005,
    densification_interval=100,
    opacity_reset_interval=30_000,
    densify_until_iter=15_000,
    densify_grad_threshold=0.0002,
    #depth_ratio=1.0
)

SWEEPS = [dict(
    #feature_lr=[0.0025, 0.005],
    #lambda_normal=[0.05],
    #matches_per_ref = [20_000],
    #densify_until_iter=[1000],
    #dist_reg_from=[1000],
    #normal_reg_from=[1000],
    #lambda_dist=[1000],
    #estimate_normals=[0,1],
    old_init = [1],

    #gaussians_init =['edgs'],
    #opacity_cull=[0.05]

    #lambda_dist=[0, 250, 500],
    #densify_grad_threshold=[1e-4, 2e-4],
),]

EXPERIMENT_ROOT = "experiments_presentation"

def format_value(v):
    if isinstance(v, float):
        return f"{v:.2e}".replace("+", "")
    return str(v)

def experiment_name(sweep_keys: list, sweep_values: tuple) -> str:
    parts = [
        f"{k}={format_value(v)}"
        for k, v in zip(sweep_keys, sweep_values)
    ]
    return "_".join(parts)

def main():
    scenes = ['scan105']
    for scene in scenes:
        for SWEEP in SWEEPS:
            keys = list(SWEEP.keys())
            values = list(SWEEP.values())
            experiment_root = os.path.join(EXPERIMENT_ROOT, '_'.join(keys) + '_' + scene)
            os.makedirs(experiment_root, exist_ok=True)
            dtu_source = "../DTU"
            #save_iters = [500, 1000]
            save_iters = [1, 500, 1000]


            for combo in product(*values):
                config = deepcopy(BASE_CONFIG)
                config.update(dict(zip(keys, combo)))

                exp_name = experiment_name(keys, combo)
                exp_dir = os.path.join(experiment_root, exp_name)

                source = dtu_source + "/" + scene
                train_args = ' '.join([" --" + key + " " + (str(value) if value is not None else "") for key, value in config.items()])
                render_args = " --skip_train \
                        --depth_ratio 1.0 \
                        --num_cluster 1 \
                        --voxel_size 0.004 \
                        --sdf_trunc 0.016 \
                        --depth_trunc 3.0"

                if os.path.exists(exp_dir):
                    print(f"Skipping existing {exp_name}")
                    continue

                os.makedirs(exp_dir, exist_ok=True)

                # Save config
                with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
                    yaml.safe_dump(config, f)

                # Run experiment
                print("python train.py -s " + source + " -m " + exp_dir + train_args + " --save_iterations " + ' '.join([str(it) for it in save_iters]))
                os.system("python train.py -s " + source + " -m " + exp_dir + train_args + " --save_iterations " + ' '.join([str(it) for it in save_iters]))

                metrics_file = exp_dir+'/metrics.csv'
                metrics_header = "iterations,accuracy,completeness,overall"
                subprocess.run(f'echo "{metrics_header}" > {metrics_file}', shell=True, check=True)

                for iter in save_iters:
                    print("python render.py --iteration " + str(iter) + " -s " + source + " -m " + exp_dir + render_args)
                    os.system("python render.py --iteration " + str(iter) + " -s " + source + " -m " + exp_dir + render_args)
                    mesh_file = exp_dir + "/train/ours_" + str(iter) + "/fuse_post.ply"
                    snapshot_file = exp_dir + "/train/ours_" + str(iter) + "/snapshot.png"
                    mesh_snapshot_from_file(mesh_file, snapshot_file)

                    subprocess.run(f'echo -n "{iter}," >> {metrics_file}', shell=True, check=True)
                    cmd = (
                        f'python scripts/eval_dtu/evaluate_single_scene.py '
                        f'--input_mesh {exp_dir}/train/ours_{iter}/fuse_post.ply '
                        f'--scan_id {scene.replace("scan","",1)} '
                        f'--mask_dir ../DTU/ '
                        f'--DTU ../DTU/ '
                        f'| sed "s/ /,/g" | grep ^[^cull] >> {metrics_file}'
                    )
                    subprocess.run(cmd, shell=True, check=True)

        plot_metrics_from_runs(experiment_root, os.path.join(experiment_root, 'metrics.png'))

if __name__ == "__main__":
    main()

