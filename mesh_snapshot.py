import open3d as o3d
import numpy as np

mesh_file = "output/dtu_7knormal3kdd/train/ours_3000/fuse_post.ply"
out_image = "snapshot.png"

def mesh_snapshot_from_file(mesh_file, out_image):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh_snapshot(mesh, out_image)
def mesh_snapshot(mesh, out_image):
    R = np.array([
        [0.195078, -0.83833 ,  0.509064],
        [0.000000, -0.641533, -0.501670],
        [-0.580313,  0.741876, -0.213375]
    ])

    t = np.array([-1.2178, 0.268842, 0.952412])

    eye = -R.T @ t
    up = R.T @ np.array([0,1,0])
    center = [-0.245,  0.857,  0.210]

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

    scene.camera.set_projection(
        60.0,                                  # MeshLab-like FOV
        width / height,
        0.1,
        1000.0,
        o3d.visualization.rendering.Camera.FovType.Vertical
    )
    view_dir = (center - eye)
    view_dir /= np.linalg.norm(view_dir)

    right = np.cross(view_dir, up)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, view_dir)
    true_up /= np.linalg.norm(true_up)
    light_dir = (
            -0.3 * np.array([1, 0, 0]) +   # left
            0.8 * np.array([0, 1, 0]) +   # up
            0.5 * view_dir                # toward camera
    )
    light_dir /= np.linalg.norm(light_dir)
    light_dir = (
            -0.4 * right      # from left
            -0.8 * true_up    # from above  (NEGATIVE is important)
            +0.4 * view_dir   # from front
    )
    light_dir /= np.linalg.norm(light_dir)
    scene.scene.set_sun_light(
        view_dir.astype(np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        9000.0
    )
    scene.scene.enable_sun_light(True)
    scene.scene.add_point_light(
        "fill",
        np.array([1.0, 1.0, 1.0], dtype=np.float32),              # color
        np.array(center + np.array([2.0, 2.0, 2.0]), dtype=np.float32),  # position
        50000.0,                                                   # intensity
        0.0,                                                       # falloff (0 = constant)
        False                                                      # cast_shadows
    )


    # Render and save
    img = renderer.render_to_image()
    o3d.io.write_image(out_image, img)
    print(f"Saved snapshot to {out_image}")

if __name__ == "__main__":
    mesh_snapshot_from_file(mesh_file, out_image)