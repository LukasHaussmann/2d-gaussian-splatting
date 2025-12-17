import os
import json
import yaml
import csv
import hashlib
from copy import deepcopy
from itertools import product
from mesh_snapshot import mesh_snapshot
from plot_metrics import plot_metrics_from_runs
import subprocess

BASE_CONFIG = dict(
    matches_per_ref = 25_000,
    nns_per_ref = 3,
    proj_err_tolerance = 0.01,
    normal_estimate_knn = 20,

    iterations=15_000,
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
    lambda_normal=0.05,
    densify_from_iter=0,
    dist_reg_from = 0,
    normal_reg_from = 0,
    opacity_cull=0.005,
    densification_interval=100,
    opacity_reset_interval=30_000,
    densify_until_iter=15_000,
    densify_grad_threshold=0.0002,
)

SWEEP = dict(
    #position_lr_init=[1.6e-4, 3.2e-4],
    #feature_lr=[0.0025, 0.005],
    lambda_normal=[0.025],
    
    #lambda_dist=[0, 250, 500],
    #densify_grad_threshold=[1e-4, 2e-4],

)

EXPERIMENT_ROOT = "experiments_regularization"

def config_hash(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:8]

def run_training(config: dict, output_dir: str) -> dict:
    print(dict)
    #os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

def main():
    os.makedirs(EXPERIMENT_ROOT, exist_ok=True)

    keys = list(SWEEP.keys())
    values = list(SWEEP.values())
    dtu_source = "../DTU"
    scene = "scan105"
    save_iters = [1, 1000, 3000, 5000, 7000, 10000, 15000]

    index_rows = []
    exp_id = 0

    for combo in product(*values):
        config = deepcopy(BASE_CONFIG)
        config.update(dict(zip(keys, combo)))

        exp_hash = config_hash(config)
        exp_name = f"exp_{exp_id:04d}_{exp_hash}"
        exp_dir = os.path.join(EXPERIMENT_ROOT, exp_name)

        source = dtu_source + "/" + scene
        train_args = ' '.join([" --" + key + " " + str(value) for key, value in config.items()])
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
            mesh_snapshot(mesh_file, snapshot_file)
            
            subprocess.run(f'echo -n "{iter}," >> {metrics_file}', shell=True, check=True)
            cmd = (
                f'python scripts/eval_dtu/evaluate_single_scene.py '
                f'--input_mesh {exp_dir}/train/ours_{iter}/fuse_post.ply '
                f'--scan_id 105 '
                f'--mask_dir ../DTU/ '
                f'--DTU ../DTU/ '
                f'| sed "s/ /,/g" | grep ^[^cull] >> {metrics_file}'
            )
            subprocess.run(cmd, shell=True, check=True)

        
        exp_id += 1

    plot_metrics_from_runs(EXPERIMENT_ROOT, os.path.join(EXPERIMENT_ROOT, 'metrics.png'))
    """
    with open(os.path.join(EXPERIMENT_ROOT, "index.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=index_rows[0].keys())
        writer.writeheader()
        writer.writerows(index_rows)
    """

if __name__ == "__main__":
    main()

