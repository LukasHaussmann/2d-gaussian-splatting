#!/bin/bash

FILE_HEADER="iterations,accuracy,completeness,overall"

SKIP_TRAIN=false
SKIP_RENDER=false

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip_train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip_render)
            SKIP_RENDER=true
            shift
            ;;
        *)
            # First non-option argument is treated as NAME
            if [ -z "$MODEL_NAME" ]; then
                MODEL_NAME="$1"
            else
                echo "Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Require NAME
if [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 <model_name> [--skip_train] [--skip_render]"
    exit 1
fi


# Check if argument was provided
if [ -z "$MODEL_NAME" ]; then
	echo "Usage: $0 <model_name>"
    exit 1
fi

if [ "$SKIP_TRAIN" = false ]; then
	python train.py -s ../DTU/scan105/ -m output/"$MODEL_NAME" --save_iterations 1000 3000 5000 7000 10000 15000 30000 --test_iterations -1 --depth_ratio 1.0 -r 2 --lambda_dist 1000
fi

if [ "$SKIP_RENDER" = false ]; then
	for iter in 1000 3000 5000 7000 10000 15000 30000; do
	    python render.py \
		--iteration "$iter" \
		-m output/"$MODEL_NAME" \
		--skip_train \
		--depth_ratio 1.0 \
		--num_cluster 1 \
		--voxel_size 0.004 \
		--sdf_trunc 0.016 \
		--depth_trunc 3.0
	done
fi

FILE_NAME="dtu_mesh_eval/${MODEL_NAME}.csv"
touch "$FILE_NAME"

echo "$FILE_HEADER" > "$FILE_NAME" 

for iter in 1000 3000 5000 7000 10000 15000 30000; do
echo -n "$iter," >> "$FILE_NAME"
python scripts/eval_dtu/evaluate_single_scene.py --input_mesh output/"$MODEL_NAME"/train/ours_"$iter"/fuse_post.ply --scan_id 105 --mask_dir ../DTU/ --DTU ../DTU/ | sed "s/ /,/g" >> "$FILE_NAME"
done
