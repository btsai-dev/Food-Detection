import glob
import trimesh
import os
import numpy as np
import json
from PIL import Image
from pathlib import Path
from pycocotools.coco import COCO
import cv2
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

collection_path = ".\\model_collection\\banana"
inference_out = ".\\model_collection\\banana_infer"

def _Map3DTo2D(p3d, projection_matrix, camera_pose, image_width, image_height):
    """
    @author: Elham Ravanbakhsh
    """
    view_matrix = np.linalg.inv(camera_pose)
    mvp = np.dot(projection_matrix, view_matrix)
    print(mvp.shape)
    p0 = np.append(p3d, [1])
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
    p3d = np.array(frame_json["intrinsics"])
    projection_matrix = np.array(frame_json["projectionMatrix"])
    camera_pose = np.array(frame_json["cameraPoseARFrame"])

    #p3d = np.reshape(p3d, (3, 3))
    projection_matrix = np.reshape(projection_matrix, (4, 4))
    camera_pose = np.reshape(camera_pose, (4, 4))

    #print(p3d)
    #print(projection_matrix)
    #print(camera_pose)

    return _Map3DTo2D(p3d, projection_matrix, camera_pose, image_width, image_height)
 
        
def get_valid_frames(vertex, frame_dict):
    """
    Input vertex as a size (3,) numpy array
    Input frame_dict as a dictionary mapping frame json filepath and the image filepath
    Input dims as the frame
    Returns a list of lists
    List[0] will store the frame image dimensions
    List[1] will store the image stem name (no extension)
    """
    valid_frame_list = [None] * 2
    valid_frame_list[0] = []
    valid_frame_list[1] = []
    for frame_json_pth in frame_dict.keys():
        print(frame_json_pth)
        json_f = open(frame_json_pth)
        frame_json = json.load(json_f)
        frame_img_pth = frame_dict[frame_json_pth]
        img_pil = Image.open(frame_img_pth)
        dims = img_pil.size

        if map3dto2d(vertex, frame_json, dims) is not None:
            
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

    for i in range(10):
        frame_img_pth = frame_img_pths[i]
        img_pil = Image.open(frame_img_pth)
        img_np = np.asarray(img_pil)

        anns_ids = coco.getAnnIds(imgIds=i+1, catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)

        mask = coco.annToMask(anns[0])
        for j in range(len(anns)):
            mask += coco.annToMask(anns[j])
        frame_results.append(mask)
        cv2.imwrite(os.path.join(inference_out, os.path.basename(frame_img_pth)), mask)

    # Execute model on the image
    # For now, just a place holder temp function
    print("Done generating inference!")
    return frame_results


def main():
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
    mesh = trimesh.load(model_pth)
    # mesh.show()

    # Looping through each vertex in the mesh
    for vertex in mesh.vertices:
        # Get the list of valid frames
        valid_frame_list = get_valid_frames(vertex, frame_dict)
        print(vertex)
        print(valid_frame_list)
"""
        # Get total amount of valid frames
        total = len(valid_frame_list[1])
        votes = 0
        for i in range(total):
            # Get the dimensions and stem name
            dims = valid_frame_list[0][i]
            stemname = valid_frame_list[1][i]

            # Convert vertex to numpy
            vertex_np = np.array(vertex)
            coord = map3dto2d(vertex_np, framename_dict[stemname][1])
            category = frame_results_dict[]
"""

        

if __name__ == "__main__":
    main()