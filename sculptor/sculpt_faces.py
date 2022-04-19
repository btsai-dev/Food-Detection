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
import numpy as np
import warnings
warnings.filterwarnings('error')
import pyglet
import cProfile
import pstats
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import cm
import time
import subprocess
import pymeshlab
import shutil
import matplotlib.pyplot as plt

import cv2

np.set_printoptions(suppress=True)
from collections import Counter

from FramesObject import FramesObject

WEIGHT = 1

ROOT_DIR = str(Path(__file__).resolve().parents[1])

data_path = os.path.join(ROOT_DIR, "data")


palette = {
    '0': np.array([255, 0, 0, 255]),
    '1': np.array([0, 0, 255, 255]),
    '255': np.array([0, 0, 255, 255])
}

class DEBUG(object):
    DONT_PROCESS = -1
    SHOW_OBJ = 0
    SHOW_HEATMAP = 1
    SHOW_PARALLEL = 2
    SHOW_SEGMENTS = 3

MODE = DEBUG()


def compute_camera_info(_extrinsic):
    """
    Accepts a nd.array 4x4 matrix containing extrinsic information
    and a nd.array 3x3 matrix containing intrinsic information

    Formula: 0 = R_3x3 * C_3x1 + T_3x1
    If extrinisic matrix is defined as:
        [R_3x3 T_3x1]
        [0_1x3 1    ]

    Position: C = -transpose(R) * T
    Rotation: Z = Euler-Rodrigues
    :param intrinsic: Intrinisc matrix, aka intrinsics
    :param extrinsic: Extrinsic matrix, aka cameraARPoseFrame
    :return: Rotation vector, World coordinates, rotation matrix
    """
    extrinsic = np.reshape(_extrinsic, (4, 4))
    R = extrinsic[0:3, 0:3]     # Rotation matrix
    T = extrinsic[0:3, 1]       # Translation vector

    # Compute translation
    R_T = np.transpose(R)
    position = -1 * np.dot(R_T, T)
    rot_Vec, _ = cv2.Rodrigues(R)
    return np.squeeze(rot_Vec), position, extrinsic



last_time = time.time()

def time_since_last(idx):
    global last_time
    print(idx, "--- %s seconds ---" % (time.time() - last_time))
    last_time = time.time()

def export_mesh_with_texture(mesh, path):
    # export the mesh including data
    export, texture = trimesh.exchange.obj.export_obj(
        mesh, include_color=True, include_texture=True, return_texture=True)

    obj_path = os.path.join(path, 'file_name.obj')
    with open(obj_path, 'w') as f:
        f.write(export)
    for k, v in texture.items():
        with open(os.path.join('', k), 'wb') as f:
            f.write(v)
    print("Saved to", obj_path)


def _Map3DTo2D(p3d, projection_matrix, camera_pose, image_width, image_height):
    """
    @author: Elham Ravanbakhsh
    """
    #print("PARAMETERS")
    #print("p3d", p3d)
    #print("projection_matrix", projection_matrix)
    #print("camera_pose", camera_pose)
    #print("image_width", image_width)
    #print("image_height", image_height)
    view_matrix = np.linalg.inv(camera_pose)
    #print("---\nview_matrix", view_matrix)
    mvp = np.dot(projection_matrix, view_matrix)
    #print("mvp", mvp)
    p0 = np.append(p3d, [1])
    #print("p0", p0)

    e0 = np.dot(mvp, p0)
    #print("e0", e0)
    e0[:3] /= e0[3]
    #print("e0 after", e0)
    pos_x = e0[0]
    pos_y = e0[1]
    #print("pos_x", pos_x)
    #print("pos_y", pos_y)
    px = (0.5 + (pos_x) * 0.5) * image_width
    py = (1.0 - (0.5 + (pos_y) * 0.5)) * image_height
    #print("pos_x", pos_x)
    #print("pos_y", pos_y)

    if px >= 0 and px < image_width and py >= 0 and py < image_height:
        return px, py
    else:
        return None


def map3dto2d(vertex_np, frame_json, dims):
    image_width, image_height = dims
    p3d = vertex_np
    projection_matrix = np.array(frame_json["projectionMatrix"])
    camera_pose = np.array(frame_json["cameraPoseARFrame"])

    # p3d = np.reshape(p3d, (3, 3))
    projection_matrix = np.reshape(projection_matrix, (4, 4))
    camera_pose = np.reshape(camera_pose, (4, 4))

    return _Map3DTo2D(p3d, projection_matrix, camera_pose, image_width, image_height)


def heatmap_parallelism(meshFrames, meshObj):
    """
    Heatmap of regions that are parallel to camera orientation
    :param meshFrames:
    :param meshObj:
    :return:
    """
    meshFaces = meshObj.faces
    meshVertices = meshObj.vertices
    meshNormals = meshObj.face_normals

    numFaces = len(meshFaces)

    img_arr = []
    for img_idx in tqdm(range(len(meshFrames.img_list))):
        img_arr.append(np.asarray(Image.open(meshFrames.get_img(img_idx))))

    mask_arr = []
    for mask_idx in tqdm(range(len(meshFrames.mask_list))):
        mask_arr.append(np.asarray(Image.open(meshFrames.get_mask(mask_idx)).convert('L')))

    json_arr = []
    for json_idx in tqdm(range(len(meshFrames.frame_list))):
        with open(meshFrames.get_frame(json_idx)) as f:
            json_arr.append(json.load(f))


    # Build a list of face parallelism values indexed by frame
    frame_parallelism = {}
    for face_idx in tqdm(range(numFaces)):
        face = meshFaces[face_idx]
        face_norm = meshNormals[face_idx]
        # a face is made up of a size-3 array indexing the mesh.vertices
        if len(face) != 3:
            raise AttributeError("Error: Only triangle faces allowed! Should have been caught at mesh import!")
        vtx1 = face[0]
        vtx2 = face[1]
        vtx3 = face[2]

        midpoint = np.array([
            (meshVertices[vtx1][0] + meshVertices[vtx2][0] + meshVertices[vtx3][0]) / 3,
            (meshVertices[vtx1][1] + meshVertices[vtx2][1] + meshVertices[vtx3][1]) / 3,
            (meshVertices[vtx1][2] + meshVertices[vtx2][2] + meshVertices[vtx3][2]) / 3,
        ])

        for frame_idx in range(len(meshFrames)):
            # mask_np = np.asarray(Image.open(mask_path).convert('P'))
            mask_np = mask_arr[frame_idx]

            # frame_json = json.load(open(frame_path))
            frame_json = json_arr[frame_idx]

            dims = (mask_np.shape[1], mask_np.shape[0])
            mapped_coord = map3dto2d(midpoint, frame_json, dims)

            if mapped_coord:
                cam_orientation, cam_position, rot_matrix = compute_camera_info(frame_json["cameraPoseARFrame"])
                parallelism = np.abs(np.dot(cam_orientation, face_norm) / (np.linalg.norm(cam_orientation) * np.linalg.norm(face_norm)))
                frame_idx_str = str(frame_idx)

                if frame_idx_str in frame_parallelism:
                    frame_parallelism[frame_idx_str]["num_faces"] += 1
                    frame_parallelism[frame_idx_str]["coord"].append(mapped_coord)
                    frame_parallelism[frame_idx_str]["parallelism"].append(parallelism)

                else:
                    frame_parallelism[frame_idx_str] = {
                        "frame_idx": frame_idx,
                        "num_faces": 1,
                        "coord": [mapped_coord],
                        "cam_orientation": [cam_orientation],
                        "cam_position": [cam_position],
                        "rot_vec": [rot_matrix],
                        "parallelism": [parallelism]
                    }


    # Loop through each frame
    for frame_idx_str in frame_parallelism:
        frame = frame_parallelism[frame_idx_str]
        fig, ax = plt.subplots()
        # fig.canvas.restore_region()
        ax.imshow(img_arr[frame["frame_idx"]])
        norm = Normalize(vmin=-0, vmax=1)
        #print(frame["num_faces"])
        for i in range(frame["num_faces"]):
            point_color = cm.hot(frame["parallelism"][i])
            circle1 = plt.Circle(frame["coord"][i], 1, color=point_color)
            ax.add_artist(circle1)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.hot), ax=ax)

        ax.text(-100, 0, "Position: " + str(frame["cam_position"]))
        ax.text(-100, -200, "RotMatrix:" + str(frame["rot_vec"]), size=8)
        print("Position: " + str(frame["cam_position"]))
        print("Extrinsics: " + str(frame["rot_vec"]))
        print("cam_orientation: " + str(frame["cam_orientation"]))
        print()
        plt.show()


def heatmap_det(meshFrames, meshObj):
    meshFaces = meshObj.faces
    meshVertices = meshObj.vertices

    numFaces = len(meshFaces)

    ime_arr = []
    for img_idx in tqdm(range(len(meshFrames.img_list))):
        ime_arr.append(np.asarray(Image.open(meshFrames.get_img(img_idx))))

    # To minimize hard disk I/O during mask reading:
    mask_arr = []
    for mask_idx in tqdm(range(len(meshFrames.mask_list))):
        mask_arr.append(np.asarray(Image.open(meshFrames.get_mask(mask_idx)).convert('L')))

    json_arr = []
    for json_idx in tqdm(range(len(meshFrames.frame_list))):
        with open(meshFrames.get_frame(json_idx)) as f:
            json_arr.append(json.load(f))

    for face_idx in tqdm(range(numFaces)):
        face = meshFaces[face_idx]
        # a face is made up of a size-3 array indexing the mesh.vertices
        if len(face) != 3:
            raise AttributeError("Error: Only triangle faces allowed! Should have been caught at mesh import!")
        vtx1 = face[0]
        vtx2 = face[1]
        vtx3 = face[2]

        midpoint = np.array([
            (meshVertices[vtx1][0] + meshVertices[vtx2][0] + meshVertices[vtx3][0]) / 3,
            (meshVertices[vtx1][1] + meshVertices[vtx2][1] + meshVertices[vtx3][1]) / 3,
            (meshVertices[vtx1][2] + meshVertices[vtx2][2] + meshVertices[vtx3][2]) / 3,
        ])

        # Loop through frame data
        valid_frames = 0
        heatmap = plt.cm.get_cmap('hot', len(meshFrames))

        for frame_idx in range(len(meshFrames)):
            # mask_np = np.asarray(Image.open(mask_path).convert('P'))
            mask_np = mask_arr[frame_idx]

            # frame_json = json.load(open(frame_path))
            frame_json = json_arr[frame_idx]

            dims = (mask_np.shape[1], mask_np.shape[0])
            mapped_coord = map3dto2d(midpoint, frame_json, dims)

            if mapped_coord:
                valid_frames += 1

        # Reject if number of valid frames is less than 10% total
        new_facecolor = heatmap(valid_frames)
        new_facecolor = np.array([
            int(new_facecolor[0]*255),
            int(new_facecolor[1]*255),
            int(new_facecolor[2]*255),
            int(new_facecolor[3]*255)
        ])
        meshObj.visual.face_colors[face_idx] = new_facecolor


def process_model(meshFrames, meshObj, cull=True, debug=None):
    meshFaces = meshObj.faces
    meshVertices = meshObj.vertices
    meshNormals = meshObj.face_normals

    numFaces = len(meshFaces)

    ime_arr = []
    for img_idx in tqdm(range(len(meshFrames.img_list))):
        ime_arr.append(np.asarray(Image.open(meshFrames.get_img(img_idx))))

    # To minimize hard disk I/O during mask reading:
    mask_arr = []
    for mask_idx in tqdm(range(len(meshFrames.mask_list))):
        mask_arr.append(np.asarray(Image.open(meshFrames.get_mask(mask_idx)).convert('L')))

    json_arr = []
    for json_idx in tqdm(range(len(meshFrames.frame_list))):
        with open(meshFrames.get_frame(json_idx)) as f:
            json_arr.append(json.load(f))

    # Mask from which to delete faces
    face_boolmask = [False] * numFaces
    for face_idx in tqdm(range(numFaces)):
        face = meshFaces[face_idx]
        # a face is made up of a size-3 array indexing the mesh.vertices
        if len(face) != 3:
            raise AttributeError("Error: Only triangle faces allowed! Should have been caught at mesh import!")
        vtx1 = face[0]
        vtx2 = face[1]
        vtx3 = face[2]
        face_norm = meshNormals[face_idx]

        midpoint = np.array([
            (meshVertices[vtx1][0] + meshVertices[vtx2][0] + meshVertices[vtx3][0]) / 3,
            (meshVertices[vtx1][1] + meshVertices[vtx2][1] + meshVertices[vtx3][1]) / 3,
            (meshVertices[vtx1][2] + meshVertices[vtx2][2] + meshVertices[vtx3][2]) / 3,
        ])

        # Loop through frame data
        voting_list = {}

        for frame_idx in range(len(meshFrames)):

            #img_path, mask_path, frame_path = meshFrames[frame_idx]

            # mask_np = np.asarray(Image.open(mask_path).convert('P'))
            mask_np = mask_arr[frame_idx]

            # frame_json = json.load(open(frame_path))
            frame_json = json_arr[frame_idx]

            dims = (mask_np.shape[1], mask_np.shape[0])
            mapped_coord = map3dto2d(midpoint, frame_json, dims)

            if mapped_coord:
                coord = (int(mapped_coord[0]), int(mapped_coord[1]))
                # print(coord)
                class_idx = mask_np[coord[1], coord[0]]

                # Show predictions from masks
                #cam_orientation, cam_position, rot_matrix = compute_camera_info(frame_json["cameraPoseARFrame"])

                # Compute how parallel the two are. Higher = more parallel.
                #print(face_norm)
                #print(cam_orientation)
                #try:
                #    parallelism = np.abs(np.dot(cam_orientation, face_norm) / (np.linalg.norm(cam_orientation) * np.linalg.norm(face_norm)))
                #except:
                #    print("Error in parallelism calculation:")
                #    print(cam_orientation)
                #    print(face_norm)
                #    print(np.dot(cam_orientation, face_norm))
                #    print(np.linalg.norm(cam_orientation))
                #    print(np.linalg.norm(face_norm))
                #    continue

                if MODE.SHOW_SEGMENTS in debug:
                    fig, ax = plt.subplots(2)
                    #fig.canvas.restore_region()
                    ax[0].imshow(ime_arr[frame_idx], cmap=ListedColormap(['b', 'r'], N=2), vmin=0, vmax=1)
                    ax[0].text(2000, 1, str(parallelism))
                    ax[1].text(2000, 1, str(face_norm))
                    ax[1].text(2000, 625, str(cam_orientation))
                    ax[1].imshow(mask_np, cmap=ListedColormap(['b', 'r'], N=2), vmin=0, vmax=1)
                    if class_idx != 0:
                       circle1= plt.Circle(coord, 50, color='g')
                       circle2 = plt.Circle(coord, 50, color='g')
                    else:
                       circle1 = plt.Circle(coord, 50, color='c')
                       circle2 = plt.Circle(coord, 50, color='c')
                    ax[0].add_patch(circle1)
                    ax[1].add_patch(circle2)

                    plt.show()

                score = 0
                if class_idx == 0:
                    score = 1
                else:
                    score = WEIGHT
                if str(class_idx) in voting_list:
                    voting_list[str(class_idx)] += score
                else:
                    # Initilize in dictionary
                    voting_list[str(class_idx)] = 1

        # print(cat)
        # Remember that categories are strings!

        # Voting mechanism
        # print(voting_list)
        if len(voting_list) == 0:
            continue
        cat = max(voting_list, key=voting_list.get)
        if cull:
            if cat != "0":
                face_boolmask[face_idx] = True
        else:
            new_facecolor = palette[cat]
            meshObj.visual.face_colors[face_idx] = new_facecolor

    if cull:
        # Update faces with face mask
        meshObj.update_faces(face_boolmask)

        # Now we need to delete those isolated vertices
        meshObj.remove_unreferenced_vertices()

        # Merge those duplicate vertices
        meshObj.merge_vertices(merge_tex=True, merge_norm=True)


def exec_model(
        meshFrames,
        out_path="out",
        out_name="output.ply",
        cull=True,
        texture=False,
        debug=None):
    if debug is None:
        debug = []
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    tmp_path = "tmp"
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    meshObjOrig = trimesh.load(meshFrames.model_path)
    # Fix the mesh, remove degenerate faces
    meshObjOrig = trimesh.Trimesh(
        vertices=meshObjOrig.vertices,
        face_normals=meshObjOrig.face_normals,
        faces=meshObjOrig.faces,
        process=True,
        validate=True
    )
    meshObjCopy = meshObjOrig.copy()
    if MODE.DONT_PROCESS not in debug:
        process_model(meshFrames, meshObjCopy, cull=cull, debug=debug)

    if debug:
        for mode in debug:
            if mode == MODE.SHOW_OBJ:
                meshObjCopy.show()
            elif mode == MODE.SHOW_HEATMAP:
                meshHeatCopy = meshObjOrig.copy()
                heatmap_det(meshFrames, meshHeatCopy)
                meshHeatCopy.show()
            elif mode == MODE.SHOW_PARALLEL:
                heatmap_parallelism(meshFrames, meshObjOrig)


    tmp_obj_in = os.path.join(tmp_path, 'output.stl')
    tmp_obj_out = os.path.join(tmp_path, 'output.ply')

    # Export object to folder
    meshObjCopy.export(tmp_obj_in)

    # Use PolyMender to hole-fill and whatnot
    command = ["PolyMender", tmp_obj_in, "6", "0.9", tmp_obj_out]
    print("Executing PolyMender repair.")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print("PolyMender Done.")
    #if process.returncode == 0:
    #    print("No Problems. ")
    #else:
    #    print("Error while executing PolyMender.")
    print("Console Output:")
    print(process.communicate()[0].decode('ascii'))

    # Load MeshLab python API to compute volume
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(tmp_obj_out)
    out_dict = ms.get_geometric_measures()
    mesh_volume = out_dict['mesh_volume'] * 1000000

    if texture:

        # Apply texture from .obj + texture img to PLY file
        textured_obj_pth = meshFrames.tex_obj
        ms.load_new_mesh(textured_obj_pth)
        ms.set_current_mesh(0)
        ms.apply_filter('transfer_texture_to_color_per_vertex', sourcemesh=1, targetmesh=0)
        ms.save_current_mesh(os.path.join(out_path, out_name))
    else:
        shutil.move(tmp_obj_out, os.path.join(out_path, out_name))

    return os.path.join(out_path, out_name), mesh_volume



def main():
    data_id = 2
    mesh_folder = None
    mesh_masks = None
    out_folder = None
    out_file = None

    global WEIGHT
    if data_id == 0:
        mesh_folder = os.path.join(data_path, "apple", "apple_12_40_25")
        mesh_masks = os.path.join(data_path, "apple", "mask")
        out_folder = "out\\apple_WEIGHTED"
        out_file = "green_apple_textured.ply"
        WEIGHT = 0.5
    if data_id == 1:
        mesh_folder = os.path.join(data_path, "green_apple", "green_apple-very_close_16_54_52")
        mesh_masks = os.path.join(data_path, "green_apple", "mask")
        out_folder = "out\\green_WEIGHTED"
        out_file = "green_apple_textured.ply"
        WEIGHT = 4
    elif data_id == 2:
        WEIGHT = 4
        mesh_folder = os.path.join(data_path, "banana", "2022_04_07_13_39_59")
        mesh_masks = os.path.join(data_path, "banana", "masks")
        out_folder = "out\\banana_WEIGHTED"
        out_file = "banana_textured.ply"
        WEIGHT = 50
    elif data_id == 3:
        WEIGHT = 4
        mesh_folder = os.path.join(data_path, "apple_13_37_19", "2022_04_07_13_35_59")
        mesh_masks = os.path.join(data_path, "apple_13_37_19", "mask")
        out_folder = "out\\apple_13_37_19_WEIGHTED"
        out_file = "aapple_13_37_19_textured.ply"

    if not mesh_folder and not mesh_masks and not out_folder and not out_file:
        print("Error: missing one")
        return

    used_folder = mesh_folder
    used_masks = mesh_masks
    frames = FramesObject(
        model_path=glob.glob(os.path.join(used_folder, "export.obj"))[0],
        img_list=glob.glob(os.path.join(used_folder, "frame*.jpg")),
        mask_list=glob.glob(os.path.join(used_masks, "frame*.png")),
        frame_list=glob.glob(os.path.join(used_folder, "frame*.json"))
    )
    frames.add_texobj(glob.glob(os.path.join(used_folder, "textured_output.obj"))[0])

    # Bias detections against background.
    ply_textured_pth, ply_volume = exec_model(
        frames,
        out_path=out_folder,
        out_name=out_file,
        cull=False,
        texture=True,
        debug=[
            #MODE.DONT_PROCESS,
            MODE.SHOW_OBJ,
            #MODE.SHOW_PARALLEL
            #MODE.SHOW_HEATMAP
        ]
    )

    print("Volume (cm^3):", ply_volume)


if __name__ == "__main__":
    main()
