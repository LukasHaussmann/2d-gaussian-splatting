from romatch import roma_outdoor, roma_indoor
import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch.nn.functional as F

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
def init_gaussians_with_corr(gaussians, scene, device):
    print("init_gaussians_with_corr")
    roma_model = roma_indoor(device=device)
    roma_model.upsample_preds = False
    roma_model.symmetric = False
    M = 15_000
    upper_thresh = roma_model.sample_thresh
    expansion_factor = 1
    keypoint_fit_error_tolerance = 0.01#cfg.proj_err_tolerance
    visualizations = {}
    viewpoint_stack = scene.getTrainCameras().copy()
    NUM_REFERENCE_FRAMES = len(viewpoint_stack)
    NUM_NNS_PER_REFERENCE = len(viewpoint_stack)
    # Select cameras using K-means
    viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)

    selected_indices = select_cameras_kmeans(cameras=viewpoint_cam_all.detach().cpu().numpy(), K=NUM_REFERENCE_FRAMES)
    selected_indices = sorted(selected_indices)
    # Find the k-closest vectors for each vector
    viewpoint_cam_all = torch.stack([x.world_view_transform.flatten() for x in viewpoint_stack], axis=0)
    closest_indices = k_closest_vectors(viewpoint_cam_all, NUM_NNS_PER_REFERENCE)
    print("Indices of k-closest vectors for each vector:\n", closest_indices)

    closest_indices_selected = closest_indices[:, :].detach().cpu().numpy()

    all_new_xyz = []
    all_new_features_dc = []
    all_new_features_rest = []
    all_new_opacities = []
    all_new_scaling = []
    all_new_rotation = []

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

