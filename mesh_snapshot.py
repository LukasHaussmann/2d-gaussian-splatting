import open3d as o3d
import numpy as np

R = np.array([
    [0.195078, -0.83833 ,  0.509064],
    [0.000000, -0.641533, -0.501670],
    [-0.580313,  0.741876, -0.213375]
])

t = np.array([-1.2178, 0.268842, 0.952412])

eye = -R.T @ t
up = R.T @ np.array([0,1,0])
center = [-0.245,  0.857,  0.210]

# Input mesh and output image filename
mesh_file = "output/dtu_7knormal3kdd/train/ours_3000/fuse_post.ply"
out_image = "snapshot.png"

# Load mesh
mesh = o3d.io.read_triangle_mesh(mesh_file)
mesh.compute_vertex_normals()
mesh.vertex_colors = o3d.utility.Vector3dVector(np.full((len(mesh.vertices), 3), 0.5))  # neutral grey (or remove completely)

# Create offscreen renderer
width, height = 1920, 1080
renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

# Material settings
material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultLit"

# Add the mesh
scene = renderer.scene
scene.add_geometry("mesh", mesh, material)

# Configure camera & lighting
center = mesh.get_center()
#bbox = mesh.get_axis_aligned_bounding_box()
#extent = np.linalg.norm(bbox.get_extent())

eye = np.array([1.8416653703527446,-0.10236946700075061,-1.4249754471438827])
direction = (center - eye)
direction /= np.linalg.norm(direction)

eye = eye + direction * (1.2)  # move camera closer

scene.camera.look_at(center, eye, up)
scene.set_background([1, 1, 1, 1])     # white background

# Render and save
img = renderer.render_to_image()
o3d.io.write_image(out_image, img)
print(f"Saved snapshot to {out_image}")
