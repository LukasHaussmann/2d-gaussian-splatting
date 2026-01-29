import cv2
from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer
from pytorch3d.structures import Pointclouds
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.ops import knn_points, estimate_pointcloud_normals
from romatch import roma_outdoor, roma_indoor
import torch
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from PIL import Image
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch.nn.functional as F
from romatch.utils import get_tuple_transform_ops

from depth_anything_v2.dpt import DepthAnythingV2
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from arguments import ModelParams
import time
from collections import defaultdict


def estimate_normals_gpu(points: torch.Tensor, k: int = 20, batch_size: int = 20000) -> torch.Tensor:
    """
    Fast GPU-based normal estimation using pytorch3d's knn_points and batched PCA.
    Processes points in batches to avoid OOM.

    Args:
        points: (N, 3) tensor of 3D points on GPU
        k: number of neighbors for normal estimation
        batch_size: number of query points to process at once

    Returns:
        normals: (N, 3) tensor of unit normals on GPU
    """
    device = points.device
    N = points.shape[0]

    # Full point cloud as reference (1, N, 3)
    points_ref = points.unsqueeze(0)

    all_normals = []

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        points_query = points[start_idx:end_idx].unsqueeze(0)  # (1, batch, 3)

        # Find k nearest neighbors from full point cloud
        knn_result = knn_points(points_query, points_ref, K=k, return_nn=True)
        neighbors = knn_result.knn[0]  # (batch, K, 3)

        # Center the neighborhoods
        centroids = neighbors.mean(dim=1, keepdim=True)  # (batch, 1, 3)
        centered = neighbors - centroids  # (batch, K, 3)

        # Compute covariance matrices: (batch, 3, 3)
        cov = torch.bmm(centered.transpose(1, 2), centered) / k

        # Eigen decomposition - normals are eigenvectors with smallest eigenvalue
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Normal is the eigenvector corresponding to smallest eigenvalue (first column)
        normals_batch = eigenvectors[:, :, 0]  # (batch, 3)
        normals_batch = F.normalize(normals_batch, dim=1)

        all_normals.append(normals_batch)

    return torch.cat(all_normals, dim=0)


def orient_normals_towards_cameras(
    points: torch.Tensor,
    normals: torch.Tensor,
    camera_centers: torch.Tensor
) -> torch.Tensor:
    """
    Orient normals to point towards the nearest camera.

    Args:
        points: (N, 3) tensor of 3D points
        normals: (N, 3) tensor of normals
        camera_centers: (C, 3) tensor of camera centers

    Returns:
        oriented_normals: (N, 3) tensor with consistently oriented normals
    """
    # Find nearest camera for each point
    # points: (N, 3), camera_centers: (C, 3)
    points_batch = points.unsqueeze(0)  # (1, N, 3)
    cameras_batch = camera_centers.unsqueeze(0)  # (1, C, 3)

    knn_result = knn_points(points_batch, cameras_batch, K=1, return_nn=True)
    nearest_cameras = knn_result.knn[0, :, 0, :]  # (N, 3)

    # Vector from point to camera
    to_camera = nearest_cameras - points  # (N, 3)

    # Flip normals that point away from camera
    dot_product = (normals * to_camera).sum(dim=1, keepdim=True)  # (N, 1)
    flip_mask = dot_product < 0

    oriented_normals = torch.where(flip_mask, -normals, normals)

    return oriented_normals


def pairwise_distances(matrix):
    """
    Computes the pairwise Euclidean distances between all vectors in the input matrix.

    Args:
        matrix (torch.Tensor): Input matrix of shape [N, D], where N is the number of vectors and D is the dimensionality.

    Returns:
        torch.Tensor: Pairwise distance matrix of shape [N, N].
    """
    # Compute squared pairwise distances
    squared_diff = torch.cdist(matrix, matrix, p=2)
    return squared_diff

def k_closest_vectors(matrix, k):
    """
    Finds the k-closest vectors for each vector in the input matrix based on Euclidean distance.

    Args:
        matrix (torch.Tensor): Input matrix of shape [N, D], where N is the number of vectors and D is the dimensionality.
        k (int): Number of closest vectors to return for each vector.

    Returns:
        torch.Tensor: Indices of the k-closest vectors for each vector, excluding the vector itself.
    """
    # Compute pairwise distances
    distances = pairwise_distances(matrix)

    # For each vector, sort distances and get the indices of the k-closest vectors (excluding itself)
    # Set diagonal distances to infinity to exclude the vector itself from the nearest neighbors
    distances.fill_diagonal_(float('inf'))

    # Get the indices of the k smallest distances (k-closest vectors)
    _, indices = torch.topk(distances, k, largest=False, dim=1)

    return indices

def select_cameras_kmeans(cameras, K):
    """
    Selects K cameras from a set using K-means clustering.

    Args:
        cameras: NumPy array of shape (N, 16), representing N cameras with their 4x4 homogeneous matrices flattened.
        K: Number of clusters (cameras to select).

    Returns:
        selected_indices: List of indices of the cameras closest to the cluster centers.
    """
    # Ensure input is a NumPy array
    if not isinstance(cameras, np.ndarray):
        cameras = np.asarray(cameras)

    if cameras.shape[1] != 16:
        raise ValueError("Each camera must have 16 values corresponding to a flattened 4x4 matrix.")

    # Perform K-means clustering
    cluster_centers, _ = kmeans(cameras, K)

    # Assign each camera to a cluster and find distances to cluster centers
    cluster_assignments, _ = vq(cameras, cluster_centers)

    # Find the camera nearest to each cluster center
    selected_indices = []
    for k in range(K):
        cluster_members = cameras[cluster_assignments == k]
        distances = cdist([cluster_centers[k]], cluster_members)[0]
        nearest_camera_idx = np.where(cluster_assignments == k)[0][np.argmin(distances)]
        selected_indices.append(nearest_camera_idx)

    return selected_indices


def compute_warp_and_confidence(viewpoint_cam1, viewpoint_cam2, roma_model, device="cuda", verbose=False, output_dict={}):
    """
    Computes the warp and confidence between two viewpoint cameras using the roma_model.

    Args:
        viewpoint_cam1: Source viewpoint camera.
        viewpoint_cam2: Target viewpoint camera.
        roma_model: Pre-trained Roma model for correspondence matching.
        device: Device to run the computation on.
        verbose: If True, displays the images.

    Returns:
        certainty: Confidence tensor.
        warp: Warp tensor.
        imB: Processed image B as numpy array.
    """
    # Prepare images
    imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    imB = viewpoint_cam2.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    imA = Image.fromarray(np.clip(imA * 255, 0, 255).astype(np.uint8))
    imB = Image.fromarray(np.clip(imB * 255, 0, 255).astype(np.uint8))

    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        cax1 = ax[0].imshow(imA)
        ax[0].set_title("Image 1")
        cax2 = ax[1].imshow(imB)
        ax[1].set_title("Image 2")
        fig.colorbar(cax1, ax=ax[0])
        fig.colorbar(cax2, ax=ax[1])

        for axis in ax:
            axis.axis('off')
        # Save the figure into the dictionary
        output_dict[f'image_pair'] = fig

    # Transform images
    ws, hs = roma_model.w_resized, roma_model.h_resized
    test_transform = get_tuple_transform_ops(resize=(hs, ws), normalize=True)
    im_A, im_B = test_transform((imA, imB))
    batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}

    # Forward pass through Roma model
    corresps = roma_model.forward(batch) if not roma_model.symmetric else roma_model.forward_symmetric(batch)
    finest_scale = 1
    hs, ws = roma_model.upsample_res if roma_model.upsample_preds else (hs, ws)

    # Process certainty and warp
    certainty = corresps[finest_scale]["certainty"]
    im_A_to_im_B = corresps[finest_scale]["flow"]
    if roma_model.attenuate_cert:
        low_res_certainty = F.interpolate(
            corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
        )
        certainty -= 0.5 * low_res_certainty * (low_res_certainty < 0)

    # Upsample predictions if needed
    if roma_model.upsample_preds:
        im_A_to_im_B = F.interpolate(
            im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
        )
        certainty = F.interpolate(
            certainty, size=(hs, ws), align_corners=False, mode="bilinear"
        )

    # Convert predictions to final format
    im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)
    im_A_coords = torch.stack(torch.meshgrid(
        torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
        torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
        indexing='ij'
    ), dim=0).permute(1, 2, 0).unsqueeze(0).expand(im_A_to_im_B.size(0), -1, -1, -1)

    warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
    certainty = certainty.sigmoid()

    return certainty[0, 0], warp[0], np.array(imB)


def resize_batch(tensors_3d, tensors_4d, target_shape):
    """
    Resizes a batch of tensors with shapes [B, H, W] and [B, H, W, 4] to the target spatial dimensions.

    Args:
        tensors_3d: Tensor of shape [B, H, W].
        tensors_4d: Tensor of shape [B, H, W, 4].
        target_shape: Tuple (target_H, target_W) specifying the target spatial dimensions.

    Returns:
        resized_tensors_3d: Tensor of shape [B, target_H, target_W].
        resized_tensors_4d: Tensor of shape [B, target_H, target_W, 4].
    """
    target_H, target_W = target_shape

    # Resize [B, H, W] tensor
    resized_tensors_3d = F.interpolate(
        tensors_3d.unsqueeze(1), size=(target_H, target_W), mode="bilinear", align_corners=False
    ).squeeze(1)

    # Resize [B, H, W, 4] tensor
    B, _, _, C = tensors_4d.shape
    resized_tensors_4d = F.interpolate(
        tensors_4d.permute(0, 3, 1, 2), size=(target_H, target_W), mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)

    return resized_tensors_3d, resized_tensors_4d

def aggregate_confidences_and_warps(viewpoint_stack, closest_indices, roma_model, source_idx, verbose=False, output_dict={}):
    """
    Aggregates confidences and warps by iterating over the nearest neighbors of the source viewpoint.

    Args:
        viewpoint_stack: Stack of viewpoint cameras.
        closest_indices: Indices of the nearest neighbors for each viewpoint.
        roma_model: Pre-trained Roma model.
        source_idx: Index of the source viewpoint.
        verbose: If True, displays intermediate results.

    Returns:
        certainties_max: Aggregated maximum confidences.
        warps_max: Aggregated warps corresponding to maximum confidences.
        certainties_max_idcs: Pixel-wise index of the image  from which we taken the best matching.
        imB_compound: List of the neighboring images.
    """
    certainties_all, warps_all, imB_compound = [], [], []

    for nn in tqdm(closest_indices[source_idx]):

        viewpoint_cam1 = viewpoint_stack[source_idx]
        viewpoint_cam2 = viewpoint_stack[nn]

        certainty, warp, imB = compute_warp_and_confidence(viewpoint_cam1, viewpoint_cam2, roma_model, verbose=verbose, output_dict=output_dict)
        certainties_all.append(certainty)
        warps_all.append(warp)
        imB_compound.append(imB)

    certainties_all = torch.stack(certainties_all, dim=0)
    target_shape = imB_compound[0].shape[:2]
    if verbose:
        print("certainties_all.shape:", certainties_all.shape)
        print("torch.stack(warps_all, dim=0).shape:", torch.stack(warps_all, dim=0).shape)
        print("target_shape:", target_shape)

    certainties_all_resized, warps_all_resized = resize_batch(certainties_all,
                                                              torch.stack(warps_all, dim=0),
                                                              target_shape
                                                              )

    if verbose:
        print("warps_all_resized.shape:", warps_all_resized.shape)
        for n, cert in enumerate(certainties_all):
            fig, ax = plt.subplots()
            cax = ax.imshow(cert.cpu().numpy(), cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise Confidence")
            output_dict[f'certainty_{n}'] = fig

        for n, warp in enumerate(warps_all):
            fig, ax = plt.subplots()
            cax = ax.imshow(warp.cpu().numpy()[:, :, :3], cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise warp")
            output_dict[f'warp_resized_{n}'] = fig

        for n, cert in enumerate(certainties_all_resized):
            fig, ax = plt.subplots()
            cax = ax.imshow(cert.cpu().numpy(), cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise Confidence resized")
            output_dict[f'certainty_resized_{n}'] = fig

        for n, warp in enumerate(warps_all_resized):
            fig, ax = plt.subplots()
            cax = ax.imshow(warp.cpu().numpy()[:, :, :3], cmap='viridis')
            fig.colorbar(cax, ax=ax)
            ax.set_title("Pixel-wise warp resized")
            output_dict[f'warp_resized_{n}'] = fig

    certainties_max, certainties_max_idcs = torch.max(certainties_all_resized, dim=0)
    H, W = certainties_max.shape

    warps_max = warps_all_resized[certainties_max_idcs, torch.arange(H).unsqueeze(1), torch.arange(W)]

    imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
    imA = np.clip(imA * 255, 0, 255).astype(np.uint8)

    return certainties_max, warps_max, certainties_max_idcs, imA, imB_compound, certainties_all_resized, warps_all_resized

def extract_keypoints_and_colors(imA, imB_compound, certainties_max, certainties_max_idcs, matches, roma_model,
                                 verbose=False, output_dict={}):
    """
    Extracts keypoints and corresponding colors from the source image (imA) and multiple target images (imB_compound).

    Args:
        imA: Source image as a NumPy array (H_A, W_A, C).
        imB_compound: List of target images as NumPy arrays [(H_B, W_B, C), ...].
        certainties_max: Tensor of pixel-wise maximum confidences.
        certainties_max_idcs: Tensor of pixel-wise indices for the best matches.
        matches: Matches in normalized coordinates.
        roma_model: Roma model instance for keypoint operations.
        verbose: if to show intermediate outputs and visualize results

    Returns:
        kptsA_np: Keypoints in imA in normalized coordinates.
        kptsB_np: Keypoints in imB in normalized coordinates.
        kptsA_color: Colors of keypoints in imA.
        kptsB_color: Colors of keypoints in imB based on certainties_max_idcs.
    """
    H_A, W_A, _ = imA.shape
    H, W = certainties_max.shape

    # Convert matches to pixel coordinates
    kptsA, kptsB = roma_model.to_pixel_coordinates(
        matches, W_A, H_A, H, W  # W, H
    )

    kptsA_np = kptsA.detach().cpu().numpy()
    kptsB_np = kptsB.detach().cpu().numpy()
    kptsA_np = kptsA_np[:, [1, 0]]

    if verbose:
        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(imA)
        ax.set_title("Reference image, imA")
        output_dict[f'reference_image'] = fig

        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(imB_compound[0])
        ax.set_title("Image to compare to image, imB_compound")
        output_dict[f'imB_compound'] = fig

        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(np.flipud(imA))
        cax = ax.scatter(kptsA_np[:, 0], H_A - kptsA_np[:, 1], s=.03)
        ax.set_title("Keypoints in imA")
        ax.set_xlim(0, W_A)
        ax.set_ylim(0, H_A)
        output_dict[f'kptsA'] = fig

        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(np.flipud(imB_compound[0]))
        cax = ax.scatter(kptsB_np[:, 0], H_A - kptsB_np[:, 1], s=.03)
        ax.set_title("Keypoints in imB")
        ax.set_xlim(0, W_A)
        ax.set_ylim(0, H_A)
        output_dict[f'kptsB'] = fig

    # Keypoints are in format (row, column) so the first value is alwain in range [0;height] and second is in range[0;width]

    kptsA_np = kptsA.detach().cpu().numpy()
    kptsB_np = kptsB.detach().cpu().numpy()

    # Extract colors for keypoints in imA (vectorized)
    # New experimental version
    kptsA_x = np.round(kptsA_np[:, 0] / 1.).astype(int)
    kptsA_y = np.round(kptsA_np[:, 1] / 1.).astype(int)
    kptsA_color = imA[np.clip(kptsA_x, 0, H - 1), np.clip(kptsA_y, 0, W - 1)]

    # Create a composite image from imB_compound
    imB_compound_np = np.stack(imB_compound, axis=0)
    H_B, W_B, _ = imB_compound[0].shape

    # Extract colors for keypoints in imB using certainties_max_idcs
    imB_np = imB_compound_np[
        certainties_max_idcs.detach().cpu().numpy(),
        np.arange(H).reshape(-1, 1),
        np.arange(W)
    ]

    if verbose:
        print("imB_np.shape:", imB_np.shape)
        print("imB_np:", imB_np)
        fig, ax = plt.subplots(figsize=(12, 6))
        cax = ax.imshow(np.flipud(imB_np))
        cax = ax.scatter(kptsB_np[:, 0], H_A - kptsB_np[:, 1], s=.03)
        ax.set_title("np.flipud(imB_np[0]")
        ax.set_xlim(0, W_A)
        ax.set_ylim(0, H_A)
        output_dict[f'np.flipud(imB_np[0]'] = fig


    kptsB_x = np.round(kptsB_np[:, 0]).astype(int)
    kptsB_y = np.round(kptsB_np[:, 1]).astype(int)

    certainties_max_idcs_np = certainties_max_idcs.detach().cpu().numpy()
    kptsB_proj_matrices_idx = certainties_max_idcs_np[np.clip(kptsA_x, 0, H - 1), np.clip(kptsA_y, 0, W - 1)]
    kptsB_color = imB_compound_np[kptsB_proj_matrices_idx, np.clip(kptsB_y, 0, H - 1), np.clip(kptsB_x, 0, W - 1)]

    # Normalize keypoints in both images
    kptsA_np[:, 0] = kptsA_np[:, 0] / H * 2.0 - 1.0
    kptsA_np[:, 1] = kptsA_np[:, 1] / W * 2.0 - 1.0
    kptsB_np[:, 0] = kptsB_np[:, 0] / W_B * 2.0 - 1.0
    kptsB_np[:, 1] = kptsB_np[:, 1] / H_B * 2.0 - 1.0

    return kptsA_np[:, [1, 0]], kptsB_np, kptsB_proj_matrices_idx, kptsA_color, kptsB_color

def prepare_tensor(input_array, device):
    """
    Converts an input array to a torch tensor, clones it, and detaches it for safe computation.
    Args:
        input_array (array-like): The input array to convert.
        device (str or torch.device): The device to move the tensor to.
    Returns:
        torch.Tensor: A detached tensor clone of the input array on the specified device.
    """
    if not isinstance(input_array, torch.Tensor):
        return torch.tensor(input_array, dtype=torch.float32).to(device).clone().detach()
    return input_array.clone().detach().to(device).to(torch.float32)

def triangulate_points(P1, P2, k1_x, k1_y, k2_x, k2_y, device="cuda"):
    """
    Solves for a batch of 3D points given batches of projection matrices and corresponding image points.

    Parameters:
    - P1, P2: Tensors of projection matrices of size (batch_size, 4, 4) or (4, 4)
    - k1_x, k1_y: Tensors of shape (batch_size,)
    - k2_x, k2_y: Tensors of shape (batch_size,)

    Returns:
    - X: A tensor containing the 3D homogeneous coordinates, shape (batch_size, 4)
    """
    EPS = 1e-4
    # Ensure inputs are tensors

    P1 = prepare_tensor(P1, device)
    P2 = prepare_tensor(P2, device)
    k1_x = prepare_tensor(k1_x, device)
    k1_y = prepare_tensor(k1_y, device)
    k2_x = prepare_tensor(k2_x, device)
    k2_y =  prepare_tensor(k2_y, device)
    batch_size = k1_x.shape[0]

    # Expand P1 and P2 if they are not batched
    if P1.ndim == 2:
        P1 = P1.unsqueeze(0).expand(batch_size, -1, -1)
    if P2.ndim == 2:
        P2 = P2.unsqueeze(0).expand(batch_size, -1, -1)

    # Extract columns from P1 and P2
    P1_0 = P1[:, :, 0]  # Shape: (batch_size, 4)
    P1_1 = P1[:, :, 1]
    P1_2 = P1[:, :, 2]

    P2_0 = P2[:, :, 0]
    P2_1 = P2[:, :, 1]
    P2_2 = P2[:, :, 2]

    # Reshape kx and ky to (batch_size, 1)
    k1_x = k1_x.view(-1, 1)
    k1_y = k1_y.view(-1, 1)
    k2_x = k2_x.view(-1, 1)
    k2_y = k2_y.view(-1, 1)

    # Construct the equations for each batch
    # For camera 1
    A1 = P1_0 - k1_x * P1_2  # Shape: (batch_size, 4)
    A2 = P1_1 - k1_y * P1_2
    # For camera 2
    A3 = P2_0 - k2_x * P2_2
    A4 = P2_1 - k2_y * P2_2

    # Stack the equations
    A = torch.stack([A1, A2, A3, A4], dim=1)  # Shape: (batch_size, 4, 4)

    # Right-hand side (constants)
    b = -A[:, :, 3]  # Shape: (batch_size, 4)
    A_reduced = A[:, :, :3]  # Coefficients of x, y, z

    # Solve using torch.linalg.lstsq (supports batching)
    X_xyz = torch.linalg.lstsq(A_reduced, b.unsqueeze(2)).solution.squeeze(2)  # Shape: (batch_size, 3)

    # Append 1 to get homogeneous coordinates
    ones = torch.ones((batch_size, 1), dtype=torch.float32, device=X_xyz.device)
    X = torch.cat([X_xyz, ones], dim=1)  # Shape: (batch_size, 4)

    # Now compute the errors of projections.
    seeked_splats_proj1 = (X.unsqueeze(1) @ P1).squeeze(1)
    seeked_splats_proj1 = seeked_splats_proj1 / (EPS + seeked_splats_proj1[:, [3]])
    seeked_splats_proj2 = (X.unsqueeze(1) @ P2).squeeze(1)
    seeked_splats_proj2 = seeked_splats_proj2 / (EPS + seeked_splats_proj2[:, [3]])
    proj1_target = torch.concat([k1_x, k1_y], dim=1)
    proj2_target = torch.concat([k2_x, k2_y], dim=1)
    errors_proj1 = torch.abs(seeked_splats_proj1[:, :2] - proj1_target).sum(1).detach().cpu().numpy()
    errors_proj2 = torch.abs(seeked_splats_proj2[:, :2] - proj2_target).sum(1).detach().cpu().numpy()

    return X, errors_proj1, errors_proj2


def select_best_keypoints(
        NNs_triangulated_points, NNs_errors_proj1, NNs_errors_proj2, device="cuda"):
    """
    From all the points fitted to  keypoints and corresponding colors from the source image (imA) and multiple target images (imB_compound).

    Args:
        NNs_triangulated_points:  torch tensor with keypoints coordinates (num_nns, num_points, dim). dim can be arbitrary,
            usually 3 or 4(for homogeneous representation).
        NNs_errors_proj1:  numpy array with projection error of the estimated keypoint on the reference frame (num_nns, num_points).
        NNs_errors_proj2:  numpy array with projection error of the estimated keypoint on the neighbor frame (num_nns, num_points).
    Returns:
        selected_keypoints: keypoints with the best score.
    """

    NNs_errors_proj = np.maximum(NNs_errors_proj1, NNs_errors_proj2)

    # Convert indices to PyTorch tensor
    indices = torch.from_numpy(np.argmin(NNs_errors_proj, axis=0)).long().to(device)

    # Create index tensor for the second dimension
    n_indices = torch.arange(NNs_triangulated_points.shape[1]).long().to(device)

    # Use advanced indexing to select elements
    NNs_triangulated_points_selected = NNs_triangulated_points[indices, n_indices, :]  # Shape: [N, k]

    return NNs_triangulated_points_selected, np.min(NNs_errors_proj, axis=0)

def get_intrinsic_matrix(camera):
    # Calculate focal lengths in pixels
    fx = (camera.image_width / 2.0) / np.tan(camera.FoVx / 2.0)
    fy = (camera.image_height / 2.0) / np.tan(camera.FoVy / 2.0)

    # Principal point (usually center of image)
    cx = camera.image_width / 2.0
    cy = camera.image_height / 2.0

    K = np.array([
        [fx,  0, cx],
        [0,  fy, cy],
        [0,   0,  1]
    ])
    return K

def screenspace_to_camera(u, v, z_c, camera):
    device = z_c.device

    K = torch.as_tensor(
        get_intrinsic_matrix(camera),
        device=device,
        dtype=torch.float32
    )

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_c = (u - cx) * z_c / fx
    y_c = (v - cy) * z_c / fy

    return torch.stack([x_c, y_c, z_c], dim=1)

def camera_to_world(points_cam, camera):
    device = points_cam.device

    R = torch.as_tensor(camera.R, device=device, dtype=torch.float32)
    t = torch.as_tensor(camera.T, device=device, dtype=torch.float32)

    points_world = torch.matmul(points_cam - t, R.T)
    return points_world

def screenspace_to_world(u, v, z, camera):
    points_cam = screenspace_to_camera(u, v, z, camera)
    points_world = camera_to_world(points_cam, camera)
    return points_world
def world_to_screenspace(camera, points3d):
    device = points3d.device
    R = torch.as_tensor(camera.R, device=device, dtype=torch.float32)

    t = torch.as_tensor(camera.T, device=device, dtype=torch.float32)
    K = torch.as_tensor(get_intrinsic_matrix(camera), device=device, dtype=torch.float32)
    points_cam = torch.matmul(points3d.float(),R) + t

    x_c = points_cam[:, 0]
    y_c = points_cam[:, 1]
    z_c = points_cam[:, 2]

    u = (K[0, 0] * x_c / z_c) + K[0, 2]
    v = (K[1, 1] * y_c / z_c) + K[1, 2]
    return u,v,z_c

def compute_alpha_comp_depth(camera, points3d):
    DEVICE = points3d.device
    W = camera.image_width
    H = camera.image_height
    R_colmap = torch.from_numpy(camera.R.T).to(DEVICE)
    T_colmap = torch.from_numpy(camera.T).to(DEVICE)
    K = torch.from_numpy(get_intrinsic_matrix(camera)).to(DEVICE)

    cameras = cameras_from_opencv_projection(
        R=R_colmap.unsqueeze(0).float(),
        tvec=T_colmap.unsqueeze(0).float(),
        camera_matrix=K.unsqueeze(0).float(),
        image_size=torch.tensor([[H, W]])
    )
    raster_settings = PointsRasterizationSettings(
        image_size=(H,W),
        radius=0.010,     # The "splat" size; increase this to fill holes
        points_per_pixel=10, # Number of points to track per pixel (z-buffer)
    )
    point_cloud = Pointclouds(points=[points3d])
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(point_cloud)
    zbuf = fragments.zbuf          # (1, H, W, K)
    dists = fragments.dists        # (1, H, W, K)

    # Mask invalid points
    valid = zbuf > 0

    # Alpha computation (matches AlphaCompositor)
    sigma = raster_settings.radius
    alpha = torch.exp(-dists / (sigma ** 2))
    alpha = alpha * valid

    # Front-to-back transmittance
    one_minus_alpha = 1.0 - alpha
    T = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), one_minus_alpha[..., :-1]], dim=-1),
        dim=-1,
    )

    weights = alpha * T

    # Alpha-composited depth
    depth_alpha = (weights * zbuf).sum(dim=-1)

    # Normalize by total weight (important!)
    weight_sum = weights.sum(dim=-1)
    depth_alpha = depth_alpha / (weight_sum + 1e-6)

    # Mask empty pixels
    depth_alpha[weight_sum < 1e-6] = torch.nan
    #plt.figure(figsize=(18, 6))
    #plt.imshow(1/depth_alpha[0,...].cpu().numpy(), cmap='RdYlBu')
    #plt.axis('off')
    #plt.savefig('composited_depth.png',dpi=300)
    return depth_alpha

def depth_least_squares_fit(points_inv_depth, pred_inv_depth, empty_mask):
    pred_inv_depth_masked = pred_inv_depth[~empty_mask].unsqueeze(-1)
    points_inv_depth_masked = points_inv_depth[~empty_mask].unsqueeze(-1)
    X = torch.cat([pred_inv_depth_masked, torch.ones_like(pred_inv_depth_masked)], dim=1)
    y = points_inv_depth_masked
    XTX_inv = (X.T @ X).inverse()
    XTY = X.T @ y
    AB = XTX_inv @ XTY
    scale, shift = AB[0][0], AB[1][0]
    """
    ransac = RANSACRegressor(min_samples=50)
    ransac.fit(pred_inv_depth_masked.cpu().numpy(), points_inv_depth_masked.cpu().numpy())

    scale = ransac.estimator_.coef_[0][0]
    shift = ransac.estimator_.intercept_[0]
    """

    #print(f"Relationship found: PCL_Inv = {scale:.4f} * Pred + {shift:.4f}")
    return scale,shift

def align_depth_prediction(camera, points3d, rbg, model, vis_file=None):
    DEVICE = points3d.device

    composited_depth = compute_alpha_comp_depth(camera, points3d)
    empty_mask = ~(composited_depth > 0)
    points_inv_depth_map = 1/(composited_depth + 1e-6)

    raw_img = cv2.imread(camera.image_path)
    raw_img = cv2.resize(raw_img, (camera.image_width, camera.image_height))
    predicted_inv_depth_map_np = model.infer_image(raw_img) # HxW raw depth map in numpy
    predicted_inv_depth_map = torch.from_numpy(predicted_inv_depth_map_np).to(DEVICE)

    scale, shift = depth_least_squares_fit(points_inv_depth_map[0,...], predicted_inv_depth_map, empty_mask[0,...])

    if vis_file is not None:
        aligned_depth = 1/(scale * predicted_inv_depth_map + shift)
        aligned_depth[empty_mask[0,...]] = torch.nan

        concatenated_depth = torch.cat([composited_depth[~empty_mask], aligned_depth[~empty_mask[0,...]]], dim=0)
        cat_np = concatenated_depth.cpu().numpy()
        vmin = np.percentile(cat_np, 2, axis=0)
        vmax = np.percentile(cat_np, 98, axis=0)

        composited_depth_clamped = torch.clamp(composited_depth, vmin, vmax)
        aligned_depth_clamped = torch.clamp(aligned_depth, vmin, vmax)

        fig,axes = plt.subplots(1, 2,figsize=(18, 6))
        axes[0].imshow(torch.cat([composited_depth_clamped[0,...], aligned_depth_clamped], dim=1).cpu().numpy(), cmap='RdYlBu')
        axes[0].set_title('Composited point depth | aligned predicted depth')
        axes[0].axis('off')
        abs_error = axes[1].imshow((torch.abs(composited_depth_clamped[0,...] - aligned_depth_clamped)).cpu().numpy(), cmap='magma')
        axes[1].set_title('Absolute error point depth vs aligned prediction' + f' median depth {torch.nanmedian(concatenated_depth):.2f}')
        cbar = fig.colorbar(abs_error, ax=axes[1])#, fraction=0.046, pad=0.04)
        axes[1].axis('off')
        plt.savefig(vis_file,dpi=300)
    return 1/(scale * predicted_inv_depth_map + shift), scale, shift

def depth_filter_points(viewpoint_cam1, scene_info, points, model):
    device = points.device
    sfm_points = torch.from_numpy(scene_info.point_cloud.points).to(device)
    sfm_rbg = torch.from_numpy(scene_info.point_cloud.points).to(device)
    aligned_depth_map,scale,shift = align_depth_prediction(viewpoint_cam1, sfm_points, sfm_rbg,model)#, f'depth_visuals/sfm_align_{source_idx}.png')

    W = viewpoint_cam1.image_width
    H = viewpoint_cam1.image_height
    R_colmap = torch.from_numpy(viewpoint_cam1.R.T).to(device)
    T_colmap = torch.from_numpy(viewpoint_cam1.T).to(device)
    K = torch.from_numpy(get_intrinsic_matrix(viewpoint_cam1)).to(device)

    cameras = cameras_from_opencv_projection(
        R=R_colmap.unsqueeze(0).float(),
        tvec=T_colmap.unsqueeze(0).float(),
        camera_matrix=K.unsqueeze(0).float(),
        image_size=torch.tensor([[H, W]])
    )
    raster_settings = PointsRasterizationSettings(
        image_size=(H,W),
        radius=0.010,     # The "splat" size; increase this to fill holes
        points_per_pixel=10, # Number of points to track per pixel (z-buffer)
    )
    point_cloud = Pointclouds(points=[points])
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(point_cloud)
    #zbuf = fragments.zbuf          # (1, H, W, K)
    depth_map = fragments.zbuf[0,..., 0]
    empty_mask = fragments.idx[0,..., 0] == -1
    depth_map[empty_mask] = torch.nan
    aligned_corr_depth = aligned_depth_map.clone()
    aligned_corr_depth[empty_mask] = torch.nan
    #visualize_point_vs_aligned_depth(depth_map, aligned_corr_depth, out_file=f'depth_visuals/corr_depth_aligned_{source_idx}.png')

    u,v,z = world_to_screenspace(viewpoint_cam1, points)

    valid_mask = (
            (z > 0) &           # In front of camera
            (u >= 0) & (u < W) &    # Inside image width
            (v >= 0) & (v < H)      # Inside image height
    )
    u_valid = u[valid_mask].long()
    v_valid = v[valid_mask].long()
    z_valid = z[valid_mask]
    z_predicted = aligned_depth_map[v_valid, u_valid]
    depth_error = (z_predicted - z_valid) / z_predicted
    tolerance = 0.05
    prune_mask = valid_mask.clone()
    prune_mask[valid_mask] = depth_error <= tolerance
    return prune_mask
    #xyz_corrected = screenspace_to_world(u_valid, v_valid, z_predicted, viewpoint_cam1)
def init_gaussians_with_corr(args : ModelParams, gaussians, scene, scene_info, device):
    print("init_gaussians_with_corr")
    if args.roma_model == "Outdoor":
        roma_model = roma_outdoor(device=device)
    elif args.roma_model == "Indoor":
        roma_model = roma_indoor(device=device)
    else:
        raise Exception("Unknown roma model parameter")
    roma_model.upsample_preds = False
    roma_model.symmetric = False
    M = args.matches_per_ref
    upper_thresh = roma_model.sample_thresh
    expansion_factor = 1
    keypoint_fit_error_tolerance = args.proj_err_tolerance
    visualizations = {}
    viewpoint_stack = scene.getTrainCameras().copy()
    NUM_REFERENCE_FRAMES = min(180, len(viewpoint_stack))
    NUM_NNS_PER_REFERENCE = args.nns_per_ref
    # Select cameras using K-means
    viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)

    selected_indices = select_cameras_kmeans(cameras=viewpoint_cam_all.detach().cpu().numpy(), K=NUM_REFERENCE_FRAMES)
    selected_indices = sorted(selected_indices)
    # Find the k-closest vectors for each vector
    viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)
    closest_indices = k_closest_vectors(viewpoint_cam_all, NUM_NNS_PER_REFERENCE)
    #print("Indices of k-closest vectors for each vector:\n", closest_indices)

    closest_indices_selected = closest_indices[:, :].detach().cpu().numpy()

    all_new_xyz = []
    all_new_features_dc = []
    all_new_features_rest = []
    all_new_opacities = []
    all_new_scaling = []
    all_new_rotation = []
    all_new_colors = []

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitb' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'submodules/depth_anything_v2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(device).eval()

    # Run roma_model.match once to kinda initialize the model
    with torch.no_grad():
        viewpoint_cam1 = viewpoint_stack[0]
        viewpoint_cam2 = viewpoint_stack[1]
        imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imB = viewpoint_cam2.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imA = Image.fromarray(np.clip(imA * 255, 0, 255).astype(np.uint8))
        imB = Image.fromarray(np.clip(imB * 255, 0, 255).astype(np.uint8))
        warp, certainty_warp = roma_model.match(imA, imB, device=device)
        print("Once run full roma_model.match warp.shape:", warp.shape)
        print("Once run full roma_model.match certainty_warp.shape:", certainty_warp.shape)
        del warp, certainty_warp
        torch.cuda.empty_cache()

    for source_idx in tqdm(sorted(selected_indices)):
        # 1. Compute keypoints and warping for all the neighboring views
        with torch.no_grad():
            # Call the aggregation function to get imA and imB_compound
            certainties_max, warps_max, certainties_max_idcs, imA, imB_compound, certainties_all, warps_all = aggregate_confidences_and_warps(
                viewpoint_stack=viewpoint_stack,
                closest_indices=closest_indices_selected,
                roma_model=roma_model,
                source_idx=source_idx,
                verbose=False, output_dict=visualizations
            )

        # Triangulate keypoints
        with torch.no_grad():
            matches = warps_max
            certainty = certainties_max
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
            matches, certainty = (
                matches.reshape(-1, 4),
                certainty.reshape(-1),
            )

            # Select based on certainty elements with high confidence. These are basically all of
            # kptsA_np.
            good_samples = torch.multinomial(certainty,
                                             num_samples=min(expansion_factor * M, len(certainty)),
                                             replacement=False)
        certainties_max, warps_max, certainties_max_idcs, imA, imB_compound, certainties_all, warps_all
        reference_image_dict = {
            "ref_image": imA,
            "NNs_images": imB_compound,
            "certainties_all": certainties_all,
            "warps_all": warps_all,
            "triangulated_points": [],
            "triangulated_points_errors_proj1": [],
            "triangulated_points_errors_proj2": []

        }
        with torch.no_grad():
            for NN_idx in tqdm(range(len(warps_all))):
                matches_NN = warps_all[NN_idx].reshape(-1, 4)[good_samples]

                # Extract keypoints and colors
                kptsA_np, kptsB_np, kptsB_proj_matrices_idcs, kptsA_color, kptsB_color = extract_keypoints_and_colors(
                    imA, imB_compound, certainties_max, certainties_max_idcs, matches_NN, roma_model
                )

                proj_matrices_A = viewpoint_stack[source_idx].full_proj_transform
                proj_matrices_B = viewpoint_stack[closest_indices_selected[source_idx, NN_idx]].full_proj_transform
                triangulated_points, triangulated_points_errors_proj1, triangulated_points_errors_proj2 = triangulate_points(
                    P1=torch.stack([proj_matrices_A] * M, axis=0),
                    P2=torch.stack([proj_matrices_B] * M, axis=0),
                    k1_x=kptsA_np[:M, 0], k1_y=kptsA_np[:M, 1],
                    k2_x=kptsB_np[:M, 0], k2_y=kptsB_np[:M, 1])

                reference_image_dict["triangulated_points"].append(triangulated_points)
                reference_image_dict["triangulated_points_errors_proj1"].append(triangulated_points_errors_proj1)
                reference_image_dict["triangulated_points_errors_proj2"].append(triangulated_points_errors_proj2)

        with torch.no_grad():
            NNs_triangulated_points_selected, NNs_triangulated_points_selected_proj_errors = select_best_keypoints(
                NNs_triangulated_points=torch.stack(reference_image_dict["triangulated_points"], dim=0),
                NNs_errors_proj1=np.stack(reference_image_dict["triangulated_points_errors_proj1"], axis=0),
                NNs_errors_proj2=np.stack(reference_image_dict["triangulated_points_errors_proj2"], axis=0))

        # 4. Save as gaussians
        viewpoint_cam1 = viewpoint_stack[source_idx]
        N = len(NNs_triangulated_points_selected)
        with torch.no_grad():
            new_xyz = NNs_triangulated_points_selected[:, :-1]
            if args.initial_depth_pruning == 1:
                prune_mask = depth_filter_points(viewpoint_cam1, scene_info, new_xyz, model)
            else:
                prune_mask = torch.ones(N, dtype=bool, device=device)
            new_xyz = new_xyz[prune_mask]
            all_new_xyz.append(new_xyz.cpu())  # seeked_splats
            all_new_colors.append(kptsA_color[prune_mask.cpu().numpy()] / 255.)

    all_new_xyz = np.concatenate(all_new_xyz, axis=0)
    all_new_xyz = np.asarray(all_new_xyz, dtype=np.float64)

    all_new_colors = np.concatenate(all_new_colors, axis=0)

    if args.estimate_normals == 1:
        #pc = o3d.geometry.PointCloud()
        #pc.points = o3d.utility.Vector3dVector(all_new_xyz)
        #o3d.utility.Vector3dVector()
        print("estimating normals")
        #pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=args.normal_estimate_knn))
        #pc.orient_normals_consistent_tangent_plane(args.normal_estimate_knn)

        pts = torch.tensor(all_new_xyz).to(device).unsqueeze(0)

        # Estimate normals
        # returns [Batch, Points, 3]
        normals = estimate_pointcloud_normals(pts, neighborhood_size=args.normal_estimate_knn)

        #normals = np.asarray(pc.normals)

        print("done estimating normals")
    else:
        normals = np.asarray(all_new_xyz)

    return BasicPointCloud(points=all_new_xyz, colors=all_new_colors, normals=normals)

def extract_keypoints_and_colors_single(imA, imB, matches, roma_model, verbose=False, output_dict={}):
    """
    Extracts keypoints and corresponding colors from a source image (imA) and a single target image (imB).

    Args:
        imA: Source image as a NumPy array (H_A, W_A, C).
        imB: Target image as a NumPy array (H_B, W_B, C).
        matches: Matches in normalized coordinates (torch.Tensor).
        roma_model: Roma model instance for keypoint operations.
        verbose: If True, outputs intermediate visualizations.
    Returns:
        kptsA_np: Keypoints in imA (normalized).
        kptsB_np: Keypoints in imB (normalized).
        kptsA_color: Colors of keypoints in imA.
        kptsB_color: Colors of keypoints in imB.
    """
    H_A, W_A, _ = imA.shape
    H_B, W_B, _ = imB.shape

    # Convert matches to pixel coordinates
    # Matches format: (B, 4) = (x1_norm, y1_norm, x2_norm, y2_norm)
    kptsA = matches[:, :2]  # [N, 2]
    kptsB = matches[:, 2:]  # [N, 2]

    # Scale normalized coordinates [-1,1] to pixel coordinates
    kptsA_pix = torch.zeros_like(kptsA)
    kptsB_pix = torch.zeros_like(kptsB)

    # Important! [Normalized to pixel space]
    kptsA_pix[:, 0] = (kptsA[:, 0] + 1) * (W_A - 1) / 2
    kptsA_pix[:, 1] = (kptsA[:, 1] + 1) * (H_A - 1) / 2

    kptsB_pix[:, 0] = (kptsB[:, 0] + 1) * (W_B - 1) / 2
    kptsB_pix[:, 1] = (kptsB[:, 1] + 1) * (H_B - 1) / 2

    kptsA_np = kptsA_pix.detach().cpu().numpy()
    kptsB_np = kptsB_pix.detach().cpu().numpy()

    # Extract colors
    kptsA_x = np.round(kptsA_np[:, 0]).astype(int)
    kptsA_y = np.round(kptsA_np[:, 1]).astype(int)
    kptsB_x = np.round(kptsB_np[:, 0]).astype(int)
    kptsB_y = np.round(kptsB_np[:, 1]).astype(int)

    kptsA_color = imA[np.clip(kptsA_y, 0, H_A-1), np.clip(kptsA_x, 0, W_A-1)]
    kptsB_color = imB[np.clip(kptsB_y, 0, H_B-1), np.clip(kptsB_x, 0, W_B-1)]

    # Normalize keypoints into [-1, 1] for downstream triangulation
    kptsA_np_norm = np.zeros_like(kptsA_np)
    kptsB_np_norm = np.zeros_like(kptsB_np)

    kptsA_np_norm[:, 0] = kptsA_np[:, 0] / (W_A - 1) * 2.0 - 1.0
    kptsA_np_norm[:, 1] = kptsA_np[:, 1] / (H_A - 1) * 2.0 - 1.0

    kptsB_np_norm[:, 0] = kptsB_np[:, 0] / (W_B - 1) * 2.0 - 1.0
    kptsB_np_norm[:, 1] = kptsB_np[:, 1] / (H_B - 1) * 2.0 - 1.0

    return kptsA_np_norm, kptsB_np_norm, kptsA_color, kptsB_color

def init_gaussians_with_corr_fast(args : ModelParams, gaussians, scene, scene_info, device, verbose=False):
    timings = defaultdict(list)

    print("init_gaussians_with_corr_fast")
    if args.roma_model == "Outdoor":
        roma_model = roma_outdoor(device=device)
    elif args.roma_model == "Indoor":
        roma_model = roma_indoor(device=device)
    else:
        raise Exception("Unknown roma model parameter")

    roma_model.upsample_preds = False
    roma_model.symmetric = False

    M = args.matches_per_ref
    upper_thresh = roma_model.sample_thresh
    scaling_factor = args.scaling_factor
    expansion_factor = 1
    keypoint_fit_error_tolerance = args.proj_err_tolerance
    visualizations = {}
    viewpoint_stack = scene.getTrainCameras().copy()
    NUM_REFERENCE_FRAMES = min(180, len(viewpoint_stack))
    NUM_NNS_PER_REFERENCE = 1  # Only ONE neighbor now!

    viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)

    selected_indices = select_cameras_kmeans(cameras=viewpoint_cam_all.detach().cpu().numpy(), K=NUM_REFERENCE_FRAMES)
    selected_indices = sorted(selected_indices)

    viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)
    closest_indices = k_closest_vectors(viewpoint_cam_all, NUM_NNS_PER_REFERENCE)
    closest_indices_selected = closest_indices[:, :].detach().cpu().numpy()

    all_new_xyz = []
    all_new_features_dc = []
    all_new_features_rest = []
    all_new_opacities = []
    all_new_scaling = []
    all_new_rotation = []
    all_new_colors = []
    all_new_normals = []

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitb' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'submodules/depth_anything_v2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(device).eval()

    # Dummy first pass to initialize model
    with torch.no_grad():
        viewpoint_cam1 = viewpoint_stack[0]
        viewpoint_cam2 = viewpoint_stack[1]
        imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imB = viewpoint_cam2.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imA = Image.fromarray(np.clip(imA * 255, 0, 255).astype(np.uint8))
        imB = Image.fromarray(np.clip(imB * 255, 0, 255).astype(np.uint8))
        warp, certainty_warp = roma_model.match(imA, imB, device=device)
        del warp, certainty_warp
        torch.cuda.empty_cache()

    # Main Loop over source_idx
    for source_idx in tqdm(sorted(selected_indices), desc="Profiling source frames"):

        # =================== Step 1: Compute Warp and Certainty ===================
        start = time.time()
        viewpoint_cam1 = viewpoint_stack[source_idx]
        NNs=closest_indices_selected.shape[1]
        viewpoint_cam2 = viewpoint_stack[closest_indices_selected[source_idx, np.random.randint(NNs)]]
        imA = viewpoint_cam1.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imB = viewpoint_cam2.original_image.detach().cpu().numpy().transpose(1, 2, 0)
        imA = Image.fromarray(np.clip(imA * 255, 0, 255).astype(np.uint8))
        imB = Image.fromarray(np.clip(imB * 255, 0, 255).astype(np.uint8))
        warp, certainty_warp = roma_model.match(imA, imB, device=device)

        certainties_max = certainty_warp  # New manual sampling
        timings['aggregation_warp_certainty'].append(time.time() - start)

        # =================== Step 2: Good Samples Selection ===================
        start = time.time()
        certainty = certainties_max.reshape(-1).clone()
        certainty[certainty > upper_thresh] = 1
        good_samples = torch.multinomial(certainty, num_samples=min(expansion_factor * M, len(certainty)), replacement=False)
        timings['good_samples_selection'].append(time.time() - start)

        # =================== Step 3: Triangulate Keypoints ===================
        reference_image_dict = {
            "triangulated_points": [],
            "triangulated_points_errors_proj1": [],
            "triangulated_points_errors_proj2": []
        }

        start = time.time()
        matches_NN = warp.reshape(-1, 4)[good_samples]

        # Convert matches to pixel coordinates
        kptsA_np, kptsB_np, kptsA_color, kptsB_color = extract_keypoints_and_colors_single(
            np.array(imA).astype(np.uint8),
            np.array(imB).astype(np.uint8),
            matches_NN,
            roma_model
        )

        proj_matrices_A = viewpoint_stack[source_idx].full_proj_transform
        proj_matrices_B = viewpoint_stack[closest_indices_selected[source_idx, 0]].full_proj_transform

        triangulated_points, triangulated_points_errors_proj1, triangulated_points_errors_proj2 = triangulate_points(
            P1=torch.stack([proj_matrices_A] * M, axis=0),
            P2=torch.stack([proj_matrices_B] * M, axis=0),
            k1_x=kptsA_np[:M, 0], k1_y=kptsA_np[:M, 1],
            k2_x=kptsB_np[:M, 0], k2_y=kptsB_np[:M, 1])

        reference_image_dict["triangulated_points"].append(triangulated_points)
        reference_image_dict["triangulated_points_errors_proj1"].append(triangulated_points_errors_proj1)
        reference_image_dict["triangulated_points_errors_proj2"].append(triangulated_points_errors_proj2)
        timings['triangulation_per_NN'].append(time.time() - start)

        # =================== Step 4: Select Best Triangulated Points ===================
        start = time.time()
        NNs_triangulated_points_selected, NNs_triangulated_points_selected_proj_errors = select_best_keypoints(
            NNs_triangulated_points=torch.stack(reference_image_dict["triangulated_points"], dim=0),
            NNs_errors_proj1=np.stack(reference_image_dict["triangulated_points_errors_proj1"], axis=0),
            NNs_errors_proj2=np.stack(reference_image_dict["triangulated_points_errors_proj2"], axis=0))
        timings['select_best_keypoints'].append(time.time() - start)

        # =================== Step 5: Create New Gaussians ===================
        start = time.time()
        viewpoint_cam1 = viewpoint_stack[source_idx]
        N = len(NNs_triangulated_points_selected)
        new_xyz = NNs_triangulated_points_selected[:, :-1]
        if args.initial_depth_pruning == 1:
            prune_mask = depth_filter_points(viewpoint_cam1, scene_info, new_xyz, model)
        else:
            prune_mask = torch.ones(N, dtype=bool, device=device)
        all_new_xyz.append(new_xyz[prune_mask])
        all_new_colors.append(kptsA_color[prune_mask.cpu().numpy()] / 255.)

        """
        if args.estimate_normals == 1 and args.per_view_normals == 1:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(new_xyz[prune_mask].cpu().numpy())
            o3d.utility.Vector3dVector()
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=args.normal_estimate_knn))
            pc.orient_normals_consistent_tangent_plane(args.normal_estimate_knn)
            all_new_normals.append(np.asarray(pc.normals))
        """

        #all_new_features_dc.append(RGB2SH(torch.tensor(kptsA_color.astype(np.float32) / 255.)).unsqueeze(1))
        #all_new_features_rest.append(torch.stack([gaussians._features_rest[-1].clone().detach() * 0.] * N, dim=0))

        mask_bad_points = torch.tensor(
            NNs_triangulated_points_selected_proj_errors > keypoint_fit_error_tolerance,
            dtype=torch.float32).unsqueeze(1).to(device)

        #all_new_opacities.append(torch.stack([gaussians._opacity[-1].clone().detach()] * N, dim=0) * 0. - mask_bad_points * (1e1))

        #dist_points_to_cam1 = torch.linalg.norm(viewpoint_cam1.camera_center.clone().detach() - new_xyz, dim=1, ord=2)
        #all_new_scaling.append(gaussians.scaling_inverse_activation((dist_points_to_cam1 * scaling_factor).unsqueeze(1).repeat(1, 2)))
        #all_new_rotation.append(torch.stack([gaussians._rotation[-1].clone().detach()] * N, dim=0))
        timings['save_gaussians'].append(time.time() - start)

    # =================== Final Densification Postfix ===================
    start = time.time()
    all_new_xyz = torch.cat(all_new_xyz, dim=0)
    all_new_colors = np.concatenate(all_new_colors, axis=0)
    #all_new_features_dc = torch.cat(all_new_features_dc, dim=0)
    #new_tmp_radii = torch.zeros(all_new_xyz.shape[0])
    #prune_mask = torch.ones(all_new_xyz.shape[0], dtype=torch.bool)

    all_new_xyz_np = np.asarray(all_new_xyz.cpu().numpy(), dtype=np.float64)
    normals = None
    if args.estimate_normals == 1:
        print("estimating normals (GPU)")
        pts = all_new_xyz.float()  # already on device

        # Fast GPU-based normal estimation with batching
        normals = estimate_normals_gpu(pts, k=args.normal_estimate_knn)

        # Orient normals towards cameras
        camera_centers = torch.stack([
            torch.tensor(cam.camera_center, device=device, dtype=torch.float32)
            for cam in viewpoint_stack
        ])
        normals = orient_normals_towards_cameras(pts, normals, camera_centers)
        normals = normals.cpu().numpy()

        print("done estimating normals")
    all_new_xyz = all_new_xyz_np

    timings['final_densification_postfix'].append(time.time() - start)

    # =================== Print Profiling Results ===================
    print("\n=== Profiling Summary (average per frame) ===")
    for key, times in timings.items():
        print(f"{key:35s}: {sum(times) / len(times):.4f} sec (total {sum(times):.2f} sec)")

    return BasicPointCloud(points=all_new_xyz, colors=all_new_colors, normals=normals)