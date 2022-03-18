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

collection_path = ".\\model_collection\\banana"
inference_out = ".\\model_collection\\banana_infer"

palette = {
    '0': np.array([255, 0, 0, 255]),
    '2': np.array([0, 0, 255, 255])
}


def main():
    mesh_in = o3d.io.read_triangle_mesh(".\\model_collection\\banana\\export.obj")
    print(f'Original mesh has {len(mesh_in.vertices)} vertices and {len(mesh_in.triangles)} triangles')
    voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 20
    print(f'voxel_size = {voxel_size:e}')
    mesh_smp = mesh_in.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)
    print(f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles')
    o3d.io.write_triangle_mesh(os.path.join(collection_path, "export_simplified.obj"), mesh_smp)
    #o3d.visualization.draw_geometries([mesh_smp])

if __name__ == "__main__":
    main()