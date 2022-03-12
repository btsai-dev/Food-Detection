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
from collections import Counter

collection_path = ".\\model_collection\\banana"
inference_out = ".\\model_collection\\banana_infer"

palette = {
    '0': np.array([255, 0, 0, 255]),
    '2': np.array([0, 0, 255, 255])
}


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
    view_matrix = np.linalg.inv(camera_pose)
    mvp = np.dot(projection_matrix, view_matrix)
    p0 = np.append(p3d, [1])
    #print("projection_matrix")
    #print(projection_matrix)
    #print("camera_pose")
    #print(camera_pose)
    #print("view_matrix", view_matrix.shape)
    #print("projection_matrix", projection_matrix.shape)

    #print("p3d", p3d.shape)

    #print("mvp", mvp.shape)
    #print("p0", p0.shape)

    e0 = np.dot(mvp, p0)
    e0[:3] /= e0[3]
    pos_x = e0[0]
    pos_y = e0[1]
    px = (0.5 + (pos_x) * 0.5) * image_width
    py = (1.0 - (0.5 + (pos_y) * 0.5)) * image_height
    
    if px >= 0 and px < image_width and py >= 0 and py < image_height:
        return px, py
    else:
        return None

def map3dto2d(vertex_np, frame_json, dims):
    image_width, image_height = dims
    p3d = vertex_np
    projection_matrix = np.array(frame_json["projectionMatrix"])
    camera_pose = np.array(frame_json["cameraPoseARFrame"])

    #p3d = np.reshape(p3d, (3, 3))
    projection_matrix = np.reshape(projection_matrix, (4, 4))
    camera_pose = np.reshape(camera_pose, (4, 4))

    #print(p3d)
    #print(projection_matrix.shape)
    #print(camera_pose)

    return _Map3DTo2D(p3d, projection_matrix, camera_pose, image_width, image_height)
 
        
def get_valid_frames(vertex_np, frame_dict):
    """
    Input vertex as a size (3,) numpy array
    Input frame_dict as a dictionary mapping frame json filepath and the image filepath
    Input dims as the frame
    Returns a list of lists
    List[0] will store the frame image dimensions
    List[1] will store the image stem name (no extension)
    """
    valid_frame_list = [[], []]
    for frame_json_pth in frame_dict.keys():
        json_f = open(frame_json_pth)
        frame_json = json.load(json_f)

        frame_img_pth = frame_dict[frame_json_pth]
        img_pil = Image.open(frame_img_pth)
        dims = img_pil.size

        if map3dto2d(vertex_np, frame_json, dims) is not None:
            valid_frame_list[0].append(dims)
            valid_frame_list[1].append(Path(frame_json_pth).stem)
    return valid_frame_list

def _temp_models(frame_img_pth):
    return np.asarray(Image.open(os.path.join(inference_out, os.path.basename(frame_img_pth))))

def model_execute(frame_img_pths):
    num = len(frame_img_pths)
    
    coco = COCO(".\\model_collection\\banana_infer\\banana.json")
    cat_ids = coco.getCatIds()

    frame_results = []

    for i in tqdm(range(num)):
        frame_img_pth = frame_img_pths[i]
        img_pil = Image.open(frame_img_pth)
        img_np = np.asarray(img_pil)

        anns_ids = coco.getAnnIds(imgIds=i+1, catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)

        mask = coco.annToMask(anns[0])
        for j in range(len(anns)):
            mask += coco.annToMask(anns[j])
        frame_results.append(mask)
        cv2.imwrite(os.path.join(inference_out, Path(frame_img_pth).stem + '.png'), mask)

    # Execute model on the image
    # For now, just a place holder temp function
    print("Done generating inference!")
    return frame_results


def main():
    #im = Image.open("C:\\Users\\Godonan\\Documents\\_Computing\\Food-Detection\\sculptor\\model_collection\\banana\\textured_output.jpg")
    #mesh_tex = trimesh.load("C:\\Users\\Godonan\\Documents\\_Computing\\Food-Detection\\sculptor\\model_collection\\banana\\textured_output.obj", process=False)
    #tex = trimesh.visual.TextureVisuals(image=im)
    #mesh_tex.visual.texture = tex
    #mesh_tex.show()
    #export_mesh_with_texture(mesh_tex, "./")


    frame_json_pths = sorted(glob.glob(os.path.join(collection_path, "frame*.json")))
    frame_img_pths = sorted(glob.glob(os.path.join(collection_path, "frame*.jpg")))
    model_pth = glob.glob(os.path.join(collection_path, "*export.obj"))[0]

    # First, pre-compute the results from trained food_models and store in directory
    frame_results = model_execute(frame_img_pths)

    # Dictionary mapping json file path to image path
    frame_dict = {}
    for i in range(len(frame_json_pths)):
        frame_dict[frame_json_pths[i]] = frame_img_pths[i]


    # Dictionary mapping stem name with json filepath, image filepath, and mask numpy
    framename_dict = {}
    for i in range(len(frame_json_pths)):
        stemname = Path(frame_json_pths[i]).stem
        framename_dict[stemname] = [frame_json_pths[i], frame_img_pths[i], frame_results[i]]

        # Testing, visualize results and image paths
        #img_np = np.asarray(Image.open(frame_img_pths[i]))
        #plt.imshow(img_np)
        #plt.imshow(frame_results[i], cmap='jet', alpha=0.5)
        #plt.show()


    # Load the mesh

    mesh = trimesh.load(model_pth, process=False)

    mesh_weight_1 = trimesh.load(model_pth, process=False)
    mesh_weight_5 = trimesh.load(model_pth, process=False)
    mesh_weight_10 = trimesh.load(model_pth, process=False)
    mesh_weight_20 = trimesh.load(model_pth, process=False)
    mesh_weight_40 = trimesh.load(model_pth, process=False)
    #mesh.show()

    #mesh2 = trimesh.load("C:\\Users\\Godonan\\Documents\\!! Computing\Datasets\\OBJs\\banana-11-55-10\\2021_11_12_22_55_37\\export.obj")
    #mesh2.show()

    # Make a list of categories indexed by mesh.vertices index, default -1 invalid value
    vertex_cats = [-1] * len(mesh.vertices)

    num_faces = len(mesh.faces)
    face_mask = [False] * num_faces
    # Loop through each face
    for arridx in tqdm(range(num_faces)):
        face = mesh.faces[arridx]
        face_vertex_cats = []
        for vertex_idx in face:
            # Get vertex_np
            vertex_np = np.array(mesh.vertices[vertex_idx])
            #print(vertex_np)

            # First check if np exists in vertex_cats. if so, use that determined category
            check = vertex_cats[vertex_idx]
            if check != -1:
                face_vertex_cats.append(check)
                continue

            # Otherwise, compute category

            # Get the list of valid frames
            valid_frame_list = get_valid_frames(vertex_np, frame_dict)
            #print(valid_frame_list)

            # Get total amount of valid frames
            num_valid_frames = len(valid_frame_list[0])
            voting_list = {}

            # Check if there are any valid frames
            if num_valid_frames < 1:
                continue

            for i in range(num_valid_frames):
                # Get the dimensions and stem name
                dims = valid_frame_list[0][i]

                stemname = valid_frame_list[1][i]
                json_f = open(framename_dict[stemname][0])
                frame_json = json.load(json_f)

                # Convert vertex to numpy
                coord = map3dto2d(vertex_np, frame_json, dims)
                if coord is not None:
                    # Make int as otherwise rounding may result in out of bounds
                    # results range form 0 to 1439.
                    coord = (int(coord[0]), int(coord[1]))

                    frame_result = framename_dict[stemname][2]
                    category_id = frame_result[coord[1], coord[0]]
                    if str(category_id) in voting_list:
                        voting_list[str(category_id)] += 1
                    else:
                        voting_list[str(category_id)] = 1
                else:
                    print("Crack slipthrough!")

            # Increase weight of a detection against the background
            voting_weight_5 = {}
            for key in voting_list.keys():
                if key != "0":
                    voting_weight_5[key] = voting_list[key] * 5
                else:
                    voting_weight_5[key] = voting_list[key]


            voting_weight_10 = {}
            for key in voting_list.keys():
                if key != "0":
                    voting_weight_10[key] = voting_list[key] * 10
                else:
                    voting_weight_10[key] = voting_list[key]

            voting_weight_20 = {}
            for key in voting_list.keys():
                if key != "0":
                    voting_weight_20[key] = voting_list[key] * 20
                else:
                    voting_weight_20[key] = voting_list[key]


            voting_weight_40 = {}
            for key in voting_list.keys():
                if key != "0":
                    voting_weight_40[key] = voting_list[key] * 40
                else:
                    voting_weight_40[key] = voting_list[key]

            cat = max(voting_list, key=voting_list.get)
            vertex_cats[vertex_idx] = cat


            new_vert_color_1 = palette[str(max(voting_list, key=voting_list.get))]
            mesh_weight_1.visual.vertex_colors[vertex_idx] = new_vert_color_1

            new_vert_color_5 = palette[str(max(voting_weight_5, key=voting_weight_5.get))]
            mesh_weight_5.visual.vertex_colors[vertex_idx] = new_vert_color_5

            new_vert_color_10 = palette[str(max(voting_weight_10, key=voting_weight_10.get))]
            mesh_weight_10.visual.vertex_colors[vertex_idx] = new_vert_color_10

            new_vert_color_20 = palette[str(max(voting_weight_20, key=voting_weight_20.get))]
            mesh_weight_20.visual.vertex_colors[vertex_idx] = new_vert_color_20

            new_vert_color_40 = palette[str(max(voting_weight_40, key=voting_weight_40.get))]
            mesh_weight_40.visual.vertex_colors[vertex_idx] = new_vert_color_40

        # Now check the majority category of the three vertices, if tie the lowest matters
        #face_vertex_cats.sort()
        #num = max(face_vertex_cats, key=Counter(face_vertex_cats).get)
        #color = palette[str(num)]
        #mesh.visual.vertex_colors[vertex_idx] = color
    # We now have a bool masking for vertices
    #export_mesh_with_texture(mesh_weight_1, './mesh1')
    #export_mesh_with_texture(mesh_weight_5, './mesh5')
    #export_mesh_with_texture(mesh_weight_10, './mesh10')
    #export_mesh_with_texture(mesh_weight_20, './mesh20')
    #export_mesh_with_texture(mesh_weight_40, './mesh40')
    mesh_weight_1.show()
    mesh_weight_5.show()
    mesh_weight_10.show()
    mesh_weight_20.show()
    mesh_weight_40.show()

    mesh_weight_1.write('./mesh1/mesh1.ply')
    mesh_weight_5.write('./mesh5/mesh1.ply')
    mesh_weight_10.write('./mesh10/mesh1.ply')
    mesh_weight_20.write('./mesh20/mesh1.ply')
    mesh_weight_40.write('./mesh40/mesh1.ply')


        

if __name__ == "__main__":
    main()