import os
import yaml
import subprocess


CONFIG = dict(
    matches_per_ref = 15_000,
    nns_per_ref = 3,
    scaling_factor = 0.001,
    proj_err_tolerance = 0.01,
    normal_estimate_knn = 20,
    fast_init = 1,
    estimate_normals = 1,
    #per_view_normals = 0,
    roma_model="Indoor",

    iterations=5_000,
    position_lr_init=0.000016,
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
    lambda_normal=0.01,
    densify_from_iter=0,
    dist_reg_from=0,
    normal_reg_from=0,
    opacity_cull=0.005,
    densification_interval=100,
    opacity_reset_interval=30_000,
    densify_until_iter=0,
    densify_grad_threshold=0.0002,
    #depth_ratio=1.0
)
EXPERIMENT_ROOT = "dtu_scans_evaluation/"
if __name__ == "__main__":
    exp_root = EXPERIMENT_ROOT
    os.makedirs(exp_root, exist_ok=True)

    time_limit = 300
    dtu_source = "../DTU"
    repo_dir_2dgs = '~/2d-gaussian-splatting'
    metrics_header = "scan,accuracy,completeness,overall" + f" ({time_limit}s)"
    scenes = ['scan106', 'scan24']
    keys = list(CONFIG.keys())
    values = list(CONFIG.values())
    # Save config
    with open(os.path.join(exp_root, "config.yaml"), "w") as f:
        yaml.safe_dump(CONFIG, f)

    train_args = ' '.join([" --" + key + " " + (str(value) if value is not None else "") for key, value in CONFIG.items()]) + ' --time_limit ' + str(time_limit)
    render_args = " --skip_train \
                        --depth_ratio 1.0 \
                        --num_cluster 1 \
                        --voxel_size 0.004 \
                        --sdf_trunc 0.016 \
                        --depth_trunc 3.0"
    metrics_file_ours = exp_root + '/metrics_ours.csv'
    metrics_file_2dgs = exp_root + '/metrics_2dgs.csv'
    subprocess.run(f'echo "{metrics_header}" > {metrics_file_ours}', shell=True, check=True)
    subprocess.run(f'echo "{metrics_header}" > {metrics_file_2dgs}', shell=True, check=True)
    for scene in scenes:
        exp_dir = os.path.join(exp_root, scene)

        source = dtu_source + "/" + scene
        train_command = "python train.py -s " + source + " -m " + exp_dir + train_args
        print(train_command)
        os.system(train_command)

        render_command = "python render.py -s " + source + " -m " + exp_dir + render_args
        print(render_command)
        os.system(render_command)

        train_dir = os.path.join(exp_dir, "train")
        mesh_dir = os.listdir(train_dir)[0]
        mesh_file = os.path.join(train_dir, mesh_dir) + "/fuse_post.ply"
        #snapshot_file = exp_dir + "/train/ours_" + str(iter) + "/snapshot.png"
        #mesh_snapshot_from_file(mesh_file, snapshot_file)

        subprocess.run(f'echo -n "{scene}," >> {metrics_file_ours}', shell=True, check=True)
        cmd = (
            f'python scripts/eval_dtu/evaluate_single_scene.py '
            f'--input_mesh {mesh_file} '
            f'--scan_id {scene.replace("scan","",1)} '
            f'--mask_dir ../DTU/ '
            f'--DTU ../DTU/ '
            f'| sed "s/ /,/g" | grep ^[^cull] >> {metrics_file_ours}'
        )
        subprocess.run(cmd, shell=True, check=True)

        common_args = " --test_iterations -1 --depth_ratio 1.0 --lambda_dist 1000"
        exp_dir_2dgs = exp_dir + "_2dgs"
        train_cmd_2dgs = f"python {repo_dir_2dgs}/train.py -s " + source + " -m " + exp_dir_2dgs + common_args + " --time_limit " + str(time_limit)
        print(train_cmd_2dgs)
        os.system(train_cmd_2dgs)

        render_command = "python render.py -s " + source + " -m " + exp_dir_2dgs + render_args
        print(render_command)
        os.system(render_command)

        train_dir = os.path.join(exp_dir_2dgs, "train")
        mesh_dir = os.listdir(train_dir)[0]
        mesh_file = os.path.join(train_dir, mesh_dir) + "/fuse_post.ply"
        #snapshot_file = exp_dir + "/train/ours_" + str(iter) + "/snapshot.png"
        #mesh_snapshot_from_file(mesh_file, snapshot_file)

        subprocess.run(f'echo -n "{scene}," >> {metrics_file_2dgs}', shell=True, check=True)
        cmd = (
            f'python scripts/eval_dtu/evaluate_single_scene.py '
            f'--input_mesh {mesh_file} '
            f'--scan_id {scene.replace("scan","",1)} '
            f'--mask_dir ../DTU/ '
            f'--DTU ../DTU/ '
            f'| sed "s/ /,/g" | grep ^[^cull] >> {metrics_file_2dgs}'
        )
        subprocess.run(cmd, shell=True, check=True)
