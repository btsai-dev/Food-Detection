import glob
import trimesh
import os
import numpy as np
import json
from PIL import Image
from pathlib import Path
from pycocotools.coco import COCO
import cv2
from tqdm import tqdm
try:
    from . import generic as g
except BaseException:
    import generic as g

import pyglet
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
import open3d as o3d
from collections import Counter
ROOT_DIR = str(Path(__file__).resolve().parents[1])

data_path = os.path.join(ROOT_DIR, "data")

banana_data_path = os.path.join(data_path, "banana")
banana_img_path = os.path.join(banana_data_path, "img")
banana_mask_path = os.path.join(banana_data_path, "mask")
banana_frame_path = os.path.join(banana_data_path, "frame_data")

apple_data_path = os.path.join(data_path, "apple")
apple_img_path = os.path.join(apple_data_path, "img")
apple_mask_path = os.path.join(apple_data_path, "mask")
apple_frame_path = os.path.join(apple_data_path, "frame_data")

palette = {
    '0': np.array([255, 0, 0, 255]),
    '2': np.array([0, 0, 255, 255])
}


def main():
    mesh_in = o3d.io.read_triangle_mesh(os.path.join(banana_frame_path, "export.obj"))
    print(f'Original mesh has {len(mesh_in.vertices)} vertices and {len(mesh_in.triangles)} triangles')
    voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 20
    print(f'voxel_size = {voxel_size:e}')
    mesh_smp = mesh_in.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)
    print(f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles')
    o3d.io.write_triangle_mesh(os.path.join(banana_frame_path, "export_simplified.obj"), mesh_smp)
    #o3d.visualization.draw_geometries([mesh_smp])

if __name__ == "__main__":
    main()