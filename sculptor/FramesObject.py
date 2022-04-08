import os
import warnings
from pathlib import Path


class FramesObject:
    def __init__(self, model_path, img_list, mask_list, frame_list):
        self.base_list = [Path(os.path.basename(x)).stem for x in img_list]
        self.model_path = model_path
        self.img_list = img_list
        self.mask_list = mask_list
        self.frame_list = frame_list
        self.size = len(self.base_list)
        print("Loaded", len(img_list), "images.")
        print("Loaded", len(mask_list), "masks.")
        print("Loaded", len(frame_list), "frame data.")
        if len(img_list) == 0:
            warnings.warn("No imgs found!")
        if len(mask_list)==0:
            warnings.warn("No masks found!")
        if len(frame_list)==0:
            warnings.warn("No frame data found!")

    def add_texobj(self, obj_path):
        self.tex_obj = obj_path

    def face_bool(self, bool_list):
        self.face_bool = bool_list

    def get_model(self):
        return self.model_path

    def get_img(self, idx):
        return self.img_list[idx]

    def get_mask(self, idx):
        return self.mask_list[idx]

    def get_frame(self, idx):
        return self.frame_list[idx]

    def __getitem__(self, item):
        return self.img_list[item], self.mask_list[item], self.frame_list[item]

    def __len__(self):
        return self.size
