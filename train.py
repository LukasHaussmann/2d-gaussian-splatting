#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision
from scene.corr_init import init_gaussians_with_corr, init_gaussians_with_corr_fast
from utils.render_utils import save_img_f32, save_img_u8
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from mesh_snapshot import mesh_snapshot
import open3d as o3d
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# -- New function for automated rendering of images at certain checkpoints --
# def render_checkpoint(model_path, iteration, views, gaussians, pipeline, background):
#     """
#     Automated evaluation: Renders specific views and saves RGB + Normals.
#     Calculates PSNR, SSIM, and LPIPS.
#     """
#     render_path = os.path.join(model_path, "progress_renders", f"iteration_{iteration}")
#     os.makedirs(render_path, exist_ok=True)
    
#     # Initialize Metrics
#     psnr_accum = 0.0
#     ssim_accum = 0.0
    
#     # Render first 5 views for speed
#     subset_views = views[:5] 

#     for idx, view in enumerate(subset_views):
#         # Render
#         render_pkg = render(view, gaussians, pipeline, background)
#         image = render_pkg["render"]
        
#         # --- ROBUST FIX: Force device and Clamp ---
#         image = torch.clamp(image, 0.0, 1.0).to("cuda")
#         gt = torch.clamp(view.original_image[0:3, :, :], 0.0, 1.0).to("cuda")
#         # ------------------------------------------

#         # Save Normal Maps (visualize as RGB)
#         if "rend_normal" in render_pkg:
#             # rend_normal is [-1, 1], convert to [0, 1] for saving
#             normal_vis = (render_pkg["rend_normal"] * 0.5 + 0.5).to("cuda")
#             torchvision.utils.save_image(normal_vis, os.path.join(render_path, f"{view.image_name}_normal.png"))

#         # Save RGB
#         torchvision.utils.save_image(image, os.path.join(render_path, f"{view.image_name}_rgb.png"))
        
#         # --- Metrics Calculation ---
#         psnr_accum += psnr(image, gt).mean().double()
#         ssim_accum += ssim(image, gt).mean().double()

#     # Average metrics
#     avg_psnr = psnr_accum / len(subset_views)
#     avg_ssim = ssim_accum / len(subset_views)

#     # Log metrics
#     log_file = os.path.join(model_path, "progress_log.txt")
#     with open(log_file, "a") as f:
#         f.write(f"Iter {iteration}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}\n")
    
#     print(f"\n[ITER {iteration}] Checkpoint rendered. PSNR: {avg_psnr:.4f}")
    
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    #start_time = time.time()
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    # # --- ZERO-SHOT EVALUATION (ITERATION 0) ---
    # # This renders the scene immediately after initialization, before any training
    # print("\n[ITER 0] Rendering initialization state (Zero-Shot)...")
    # test_cameras = scene.getTestCameras()
    # render_checkpoint(scene.model_path, 0, test_cameras, gaussians, pipe, background)
    
    # # [NEW CODE] Explicitly save Iteration 0 if requested
    # if 0 in saving_iterations:
    #     print("\n[ITER 0] Saving Gaussians (Initialization)")
    #     scene.save(0)
    # # ------------------------------------------

    """ for convergence frame capture
    test_cameras = scene.getTestCameras()
    fixed_cam = test_cameras[1]     # always same camera angle
    progress_dir = os.path.join(scene.model_path, "progress")
    os.makedirs(progress_dir, exist_ok=True)
    frame_idx = 0
    snapshot_interval = 50
    """
    if dataset.render_snapshots == 1:
        snapshot_dir = os.path.join(scene.model_path, "normal_snapshots/")
        os.makedirs(snapshot_dir, exist_ok=True)
    if dataset.mesh_snapshots == 1:
        mesh_snapshot_dir = os.path.join(scene.model_path, "mesh_snapshots/")
        os.makedirs(mesh_snapshot_dir, exist_ok=True)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > opt.normal_reg_from else 0.0
        lambda_dist = opt.lambda_dist if iteration > opt.dist_reg_from else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss
        
        total_loss.backward()

        # =================== GRADIENT VISUALIZATION CODE ===================
        if iteration == 7000:
            print(f"\n[ITER {iteration}] Exporting Gradient Heatmap...")
            try:
                grads = viewspace_point_tensor.grad
                if grads is not None:
                    grad_norms = torch.norm(grads, dim=-1)
                    
                    # Normalize for visualization
                    max_grad = torch.quantile(grad_norms, 0.99)
                    normalized_grads = torch.clamp(grad_norms / max_grad, 0, 1)
                    
                    # Apply Colormap
                    grad_np = normalized_grads.detach().cpu().numpy()
                    cmap = plt.get_cmap('jet')
                    colors_rgba = cmap(grad_np) 
                    colors_rgb = colors_rgba[:, :3] 

                    # Construct PLY
                    xyz = gaussians.get_xyz.detach().cpu().numpy()
                    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
                    elements = np.empty(xyz.shape[0], dtype=dtype)
                    
                    elements['x'] = xyz[:, 0]
                    elements['y'] = xyz[:, 1]
                    elements['z'] = xyz[:, 2]
                    elements['red']   = (colors_rgb[:, 0] * 255).astype(np.uint8)
                    elements['green'] = (colors_rgb[:, 1] * 255).astype(np.uint8)
                    elements['blue']  = (colors_rgb[:, 2] * 255).astype(np.uint8)
                    
                    # Save
                    heatmap_path = os.path.join(scene.model_path, "point_cloud", f"iteration_{iteration}", "gradient_heatmap.ply")
                    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
                    
                    # Use 'plyfile' directly (assuming it is installed via pip)
                    from plyfile import PlyData, PlyElement
                    el = PlyElement.describe(elements, 'vertex')
                    PlyData([el]).write(heatmap_path)
                    print(f"Saved Gradient Heatmap to: {heatmap_path}")
            except Exception as e:
                print(f"Visualization failed: {e}")
        # ===================================================================
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            # # -- New: running automated eval at custom interval --
            # if iteration % 1000 == 0: # Or define a custom interval
            #     print(f"\n[ITER {iteration}] Running automated evaluation...")
            #     test_cameras = scene.getTestCameras()
            #     render_checkpoint(scene.model_path, iteration, test_cameras, gaussians, pipe, background)
            # # -- End Change --

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            #"""
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                #if iteration < opt.densify_until_iter and iteration % 10 == 0:
                #    opacities_new = torch.log(torch.exp(gaussians._opacity.data) * 0.99)
                #    gaussians._opacity.data = opacities_new
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            #"""

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if dataset.render_snapshots == 1 and (iteration-1) % (dataset.snapshot_frequency) == 0:
                snapshot_cam = scene.getTrainCameras()[dataset.snapshot_camera_id]
                snapshot_render = render(snapshot_cam, gaussians, pipe, background)
                #snapshot_depth = snapshot_render['surf_depth'].cpu()
                snapshot_depthnormal = snapshot_render['surf_normal'].cpu()
                snapshot_normal = torch.nn.functional.normalize(snapshot_render['rend_normal'], dim=0)

                save_img_u8(snapshot_depthnormal.permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(snapshot_dir, 'depth_normal_{0:05d}'.format(iteration) + ".png"))
                save_img_u8(snapshot_normal.permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(snapshot_dir, 'normal_{0:05d}'.format(iteration) + ".png"))

            if dataset.mesh_snapshots == 1 and (iteration-1) % (dataset.mesh_snapshot_frequency) == 0:
                gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)
                save_sh_degreee = gaussians.active_sh_degree
                gaussExtractor.gaussians.active_sh_degree = 0
                gaussExtractor.reconstruction(scene.getTrainCameras())
                depth_trunc = 3.0
                voxel_size = 0.004
                sdf_trunc = 0.016
                mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
                mesh_post = post_process_mesh(mesh, cluster_to_keep=1)
                mesh_snapshot(mesh_post, os.path.join(mesh_snapshot_dir, 'mesh_snapshot_{0:05d}'.format(iteration) + ".png"))
                gaussians.active_sh_degree = save_sh_degreee
                o3d.io.write_triangle_mesh(os.path.join(scene.model_path, 'mesh_{0:05d}'.format(iteration) + ".ply"), mesh_post)

            """
            if iteration % snapshot_interval == 0:
                test_render = render(fixed_cam, gaussians, pipe, background)["render"]
                save_path = os.path.join(progress_dir, f"{frame_idx:05d}.png")
                
                # convert to uint8 and save
                img = (torch.clamp(test_render, 0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
                
                import cv2
                img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                text = f"Iter: {iteration:d}"
                 # Compute elapsed seconds
                #elapsed_sec = time.time() - start_time
                # or: text = f"Iter: {iteration:d}   Time: {iteration * iter_ms / 1000:.1f}s"
                
                cv2.putText(
                    img_cv, text,
                    (20, 60),                 # position
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 3,  # white text, bold
                    cv2.LINE_AA
                )
                
                cv2.putText(
                    img_cv, f"Time: {elapsed_sec:.1f}s",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (255, 255, 255), 3, cv2.LINE_AA
                )
                


                # Convert back to RGB for saving with imageio
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                
                import imageio
                imageio.imwrite(save_path, img_cv)
                frame_idx += 1
            """
        
        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    import numpy as np
    import torch
    from utils.general_utils import colormap

    # Scalars
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {
                'name': 'train',
                'cameras': [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)

                    # Rendered RGB in [0,1], ground truth
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if tb_writer and (idx < 5):
                        # ---------- DEPTH IMAGE ----------
                        depth = render_pkg["surf_depth"]  # torch tensor
                        norm = depth.max()
                        depth = depth / (norm + 1e-8)

                        # depth: (1, H, W) -> numpy HWC via colormap -> torch NCHW
                        depth_np = depth.detach().cpu().numpy()[0]   # (H, W)
                        depth_rgb = colormap(depth_np, cmap='turbo') # (H, W, 3), uint8
                        depth_rgb = np.array(depth_rgb, copy=True)
                        depth_t = torch.from_numpy(depth_rgb).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
                        depth_t = depth_t.float() / 255.0

                        # ---------- RENDERED IMAGE ----------
                        img_np = image.detach().cpu().numpy()[0]     # (3, H, W) or (H, W, 3) depending on code
                        if img_np.ndim == 3 and img_np.shape[0] == 3:
                            # (C, H, W) -> (H, W, C)
                            img_np = np.transpose(img_np, (1, 2, 0))
                        img_rgb = colormap(img_np)                   # (H, W, 3)
                        img_rgb = np.array(img_rgb, copy=True)
                        image_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
                        image_t = image_t.float() / 255.0

                        tb_writer.add_images(
                            config['name'] + "_view_{}/depth".format(viewpoint.image_name),
                            depth_t,
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            config['name'] + "_view_{}/render".format(viewpoint.image_name),
                            image_t,
                            global_step=iteration,
                        )

                        # ---------- NORMALS / ALPHA / DIST ----------
                        try:
                            rend_alpha = render_pkg['rend_alpha']          # expect (1,1,H,W) or similar
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5

                            tb_writer.add_images(
                                config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name),
                                rend_normal[None],
                                global_step=iteration,
                            )
                            tb_writer.add_images(
                                config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name),
                                surf_normal[None],
                                global_step=iteration,
                            )
                            tb_writer.add_images(
                                config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name),
                                rend_alpha[None],
                                global_step=iteration,
                            )

                            # rend_dist: tensor -> numpy -> colormap -> torch NCHW
                            rend_dist = render_pkg["rend_dist"]            # torch tensor
                            dist_np = rend_dist.detach().cpu().numpy()[0]  # (H, W)
                            dist_rgb = colormap(dist_np)                   # (H, W, 3)
                            dist_rgb = np.array(dist_rgb, copy=True)
                            dist_t = torch.from_numpy(dist_rgb).permute(2, 0, 1).unsqueeze(0)
                            dist_t = dist_t.float() / 255.0

                            tb_writer.add_images(
                                config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name),
                                dist_t,
                                global_step=iteration,
                            )
                        except Exception:
                            pass

                        # ---------- GROUND TRUTH ----------
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )

                    # accumulate metrics
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
