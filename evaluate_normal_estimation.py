import os
import subprocess

from mesh_snapshot import mesh_snapshot_from_file

EXPERIMENT_ROOT = "dtu_normal_estimation_evaluation/"
if __name__ == "__main__":
    exp_root = EXPERIMENT_ROOT
    os.makedirs(exp_root, exist_ok=True)

    time_limit = 200
    dtu_source = "../DTU"
    metrics_header = "parameter_value,accuracy,completeness,overall" + f" ({time_limit}s)"
    scenes = ['scan106', 'scan24']
    train_args = '--time_limit ' + str(time_limit)
    render_args = " --skip_train \
                        --depth_ratio 1.0 \
                        --num_cluster 1 \
                        --voxel_size 0.004 \
                        --sdf_trunc 0.016 \
                        --depth_trunc 3.0"
    for scene in scenes:
        metrics_file = exp_root + f'/metrics_{scene}.csv'
        subprocess.run(f'echo "{metrics_header}" > {metrics_file}', shell=True, check=True)
        source = dtu_source + "/" + scene
        param_values = [0,1]
        for param_value in param_values:
            exp_dir = os.path.join(exp_root, scene + f'_{param_value}')

            train_command = "python train.py -s " + source + " -m " + exp_dir + f' --estimate_normals {param_value} --time_limit {time_limit}'
            print(train_command)
            os.system(train_command)

            render_command = "python render.py -s " + source + " -m " + exp_dir + render_args
            print(render_command)
            os.system(render_command)

            train_dir = os.path.join(exp_dir, "train")
            mesh_dir = os.path.join(train_dir, os.listdir(train_dir)[0])
            mesh_file = mesh_dir + "/fuse_post.ply"
            snapshot_file = os.path.join(mesh_dir, "snapshot.png")
            mesh_snapshot_from_file(mesh_file, snapshot_file)

            subprocess.run(f'echo -n "{param_value}," >> {metrics_file}', shell=True, check=True)
            cmd = (
                f'python scripts/eval_dtu/evaluate_single_scene.py '
                f'--input_mesh {mesh_file} '
                f'--scan_id {scene.replace("scan","",1)} '
                f'--mask_dir ../DTU/ '
                f'--DTU ../DTU/ '
                f'| sed "s/ /,/g" | grep ^[^cull] >> {metrics_file}'
            )
            subprocess.run(cmd, shell=True, check=True)

