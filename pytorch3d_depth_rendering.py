import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
from scene.dataset_readers import readColmapSceneInfo
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor, NormWeightedCompositor
)
from pytorch3d.structures import Pointclouds
from scene.depth_init import get_intrinsic_matrix

source_path = "../DTU/scan105"
#source_path = "../bicycle"
images = "images"
eval = False
scene_info = readColmapSceneInfo(source_path, images, eval)

json_cams = []
camlist = []
if scene_info.test_cameras:
    camlist.extend(scene_info.test_cameras)
if scene_info.train_cameras:
    camlist.extend(scene_info.train_cameras)

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitb' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'submodules/depth_anything_v2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

camera_info = camlist[18]

W = camera_info.width
H = camera_info.height
R_colmap = torch.from_numpy(camera_info.R.T).to(DEVICE)
T_colmap = torch.from_numpy(camera_info.T).to(DEVICE)
K = torch.from_numpy(get_intrinsic_matrix(camera_info)).to(DEVICE)

cameras = cameras_from_opencv_projection(
    R=R_colmap.unsqueeze(0).float(),
    tvec=T_colmap.unsqueeze(0).float(),
    camera_matrix=K.unsqueeze(0).float(),
    image_size=torch.tensor([[H, W]])
)
raster_settings = PointsRasterizationSettings(
    image_size=(H,W),
    radius=0.01,     # The "splat" size; increase this to fill holes
    points_per_pixel=10, # Number of points to track per pixel (z-buffer)
)
rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
points = torch.from_numpy(scene_info.point_cloud.points).to(DEVICE).float()
rbg = torch.from_numpy(scene_info.point_cloud.colors).to(DEVICE).float()
point_cloud = Pointclouds(points=[points], features=[rbg])

fragments = rasterizer(point_cloud)

# get depth_map for visible points
depth_map = fragments.zbuf[..., 0]
mask = fragments.idx[..., 0] == -1
depth_map[mask] = torch.nan
depth_map_np = depth_map.cpu()[0].numpy()

inv_depth_map = 1 / (depth_map_np + 1e-6)
v_min = np.nanpercentile(inv_depth_map, 2)
v_max = np.nanpercentile(inv_depth_map, 98)

# plot depth map for masked points
fig, axes = plt.subplots(2, 3, figsize=(20, 20))
#axes[0].imshow(depth_map_normalized, cmap='RdYlBu', vmax=v_max)
axes[0][0].imshow(inv_depth_map, cmap='RdYlBu_r', vmin=v_min, vmax=v_max)
axes[0][0].set_title("sfm points Depth (rasterized) (Red=Close, Blue=Far)")
axes[0][0].axis("off")

# plot visible point selection mask
mask = fragments.idx[0, ..., 0] >= 0
mask_viz = mask.float().cpu().numpy()
axes[0][1].imshow(mask_viz, cmap='gray')
axes[0][1].set_title("Point mask")
axes[0][1].axis("off")

raw_img = cv2.imread(camera_info.image_path)
predicted_depth_map_np = model.infer_image(raw_img) # HxW raw depth map in numpy
predicted_depth_map_np[~mask.cpu().numpy()] = np.nan
axes[0][2].imshow(predicted_depth_map_np, cmap='RdYlBu_r')
axes[0][2].set_title("Predicted Depth (masked) (Red=Close, Blue=Far)")
axes[0][2].axis("off")

mask_np = mask.cpu().numpy()
y = torch.from_numpy(inv_depth_map[mask_np]).to(DEVICE).unsqueeze(-1)
predicted_depth_map = torch.from_numpy(predicted_depth_map_np[mask_np]).to(DEVICE).unsqueeze(-1)
print(predicted_depth_map.shape)
X = torch.cat([predicted_depth_map, torch.ones_like(predicted_depth_map)], dim=1)
print(X.shape, y.shape)
XTX_inv = (X.T @ X).inverse()
XTY = X.T @ y
AB = XTX_inv @ XTY
print(AB)
scale, shift = AB[0].cpu().numpy()[0], AB[1].cpu().numpy()[0]

print(f"Relationship found: PCL_Inv = {scale:.4f} * Pred + {shift:.4f}")

aligned_depth = scale * predicted_depth_map_np + shift
axes[1][0].imshow(aligned_depth, cmap='RdYlBu_r')
axes[1][0].set_title("Aligned Depth (=alpha*predicted+beta) (Red=Close, Blue=Far)")
axes[1][0].axis("off")

relative_error = np.abs(aligned_depth - inv_depth_map) / (inv_depth_map + 1e-6)

im_error = axes[1][1].imshow(relative_error, cmap='magma', vmin=0, vmax=0.2)
axes[1][1].set_title("Relative Error (sfm points vs Pred)")

# Add the colorbar to the second axes
# 'fraction' and 'pad' help keep the colorbar aligned with the plot height
cbar = fig.colorbar(im_error, ax=axes[1][1], fraction=0.046, pad=0.04)
cbar.set_label('Relative Error (%)', rotation=270, labelpad=15)

# Optional: Format colorbar ticks as percentages
cbar.ax.set_yticklabels([f'{x:.0%}' for x in cbar.get_ticks()])
axes[1][1].axis("off")
#plt.gca().invert_yaxis() # Match image coordinates


x_np = predicted_depth_map.cpu().numpy()
y_np = y.cpu().numpy()
axes[1][2].scatter(x_np,y_np, color='blue', alpha=0.6, s=1)
x_range = np.linspace(x_np.min(), x_np.max(), 100)
y_line = scale * x_range + shift

axes[1][2].plot(x_range, y_line, color='green', linewidth=3,
         label=f'Fit: y = {scale:.3f}x + {shift:.3f}')
axes[1][2].set_xlabel('predicted depth (inverse)')
axes[1][2].set_ylabel('sfm depth (inverse)')
axes[1][2].legend(loc='upper left', fontsize='medium')

plt.tight_layout()
plt.savefig('pytorch3d_renderer.png', dpi=300)

#images = renderer(point_cloud)
#plt.imshow(images[0, ..., :3].cpu().numpy())
#plt.imshow(depth_map[0, ..., :3].cpu().numpy())