import os
import matplotlib.pyplot as plt
from PIL import Image

base_dir = 'experiments_regularization'
for name in sorted(os.listdir(base_dir)):
    subdir = os.path.join(base_dir, name)
    if not os.path.isdir(subdir):
        continue
    meshdirs = os.listdir(os.path.join(subdir,'train'))

    images = [Image.open(os.path.join(subdir+'/train',meshdir+'/snapshot.png')) for meshdir in meshdirs]
    labels = [meshdir.strip('ours_') for meshdir in meshdirs]

    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(labels[i])
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(subdir,'test_meshes.png'), dpi=300)

