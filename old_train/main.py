import argparse
import sys
import os
from pathlib import Path
import random
from datetime import datetime

import json
import cv2
import matplotlib.pyplot as plt

from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import build_detection_test_loader

ROOT_DIR = str(Path(__file__).resolve().parents[0])
sys.path.append(ROOT_DIR)

DATASET_DIR = os.path.join(ROOT_DIR, 'Dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test_images')
VAL_DIR = os.path.join(DATASET_DIR, 'val')

TRAIN_JSON = os.path.join(TRAIN_DIR, 'annotations.json')
TEST_JSON = os.path.join(TEST_DIR, 'annotations.json')
VAL_JSON = os.path.join(VAL_DIR, 'annotations.json')

TRAIN_IMGS = os.path.join(TRAIN_DIR, 'images')
TEST_IMGS = os.path.join(TEST_DIR, 'images')
VAL_IMGS = os.path.join(VAL_DIR, 'images')

register_coco_instances('FoodDataset_Train', {}, TRAIN_JSON, TRAIN_IMGS)
register_coco_instances('FoodDataset_Test', {}, TEST_JSON, TEST_IMGS)
register_coco_instances('FoodDataset_Val', {}, VAL_JSON, VAL_IMGS)

def get_args():
    parser =argparse.ArgumentParser(description='Food segmentation training')
    parser.add_argument('--')


def FoodConfig(num_classes):
    cfg = get_cfg()

    # get configuration from model_zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Model
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    #cfg.MODEL.RESNETS.DEPTH = 34
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64

    # Solver
    cfg.SOLVER.BASE_LR = 0.0002
    cfg.SOLVER.MAX_ITER = 40000
    cfg.SOLVER.STEPS = (20, 10000, 20000)
    cfg.SOLVER.gamma = 0.5
    cfg.SOLVER.IMS_PER_BATCH = 2

    # Test
    cfg.TEST.DETECTIONS_PER_IMAGE = 20

    # INPUT
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)

    # DATASETS
    cfg.DATASETS.TRAIN = ('FoodDataset_Train',)
    cfg.DATASETS.TEST = ('FoodDataset_Test',)
    cfg.DATALOADER.NUM_WORKERS = 2

    # DATASETS
    dt_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    cfg.OUTPUT_DIR = os.path.join(ROOT_DIR, 'output', dt_str)
    return cfg


class COCOTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


def FoodCfg(output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("FoodDataset_Train",)
    cfg.DATASETS.TEST = ("FoodDataset_Test",)
    cfg.TEST.EVAL_PERIOD = 100
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 61

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 100
    cfg.TEST.EVAL_PERIOD = 2
    cfg.OUTPUT_DIR = output_dir
    #cfg.OUTPUT_DIR = os.path.join(ROOT_DIR, 'output', dt_str)
    #os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def train():
    dt_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    cfg = FoodCfg(os.path.join(ROOT_DIR, 'output', dt_str))
    #trainer = DefaultTrainer(cfg)
    trainer = COCOTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    print("Training completed, saved to", str(cfg.OUTPUT_DIR))

def test():
    cfg = FoodCfg(os.path.join(ROOT_DIR, 'output', '10-12-2021_12-27-24'))
    #trainer = COCOTrainer(cfg)
    #trainer.resume_or_load()
    #train_metadata = MetadataCatalog.get("FoodDataset_Train")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.03

    predictor = DefaultPredictor(cfg)
    #evaluator = COCOEvaluator("FoodDataset_Val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    #inference_on_dataset(model, val_loader, evaluator)

    val_loader = build_detection_test_loader(cfg, "FoodDataset_Val")

    val_dicts = DatasetCatalog.get("FoodDataset_Val")
    for d in random.sample(val_dicts, 3):
        img = cv2.imread(d["file_name"])
        print(img.shape)
        outputs = predictor(img)
        print(outputs)
        train_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        visualizer = Visualizer(img[:, :, ::-1], metadata=train_meta, scale=1)
        out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.figure(figsize=(20, 10))
        plt.imshow(out.get_image()[..., ::-1][..., ::-1])
        plt.show()
        #cv2.waitKey(0)

def test3():
    cfg = FoodCfg(os.path.join(ROOT_DIR, 'output', '10-12-2021_12-27-24'))
    #trainer = COCOTrainer(cfg)
    #trainer.resume_or_load()
    #train_metadata = MetadataCatalog.get("FoodDataset_Train")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.03

    predictor = DefaultPredictor(cfg)
    #evaluator = COCOEvaluator("FoodDataset_Val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    #inference_on_dataset(model, val_loader, evaluator)

    val_loader = build_detection_test_loader(cfg, "FoodDataset_Val")
    val_dicts = DatasetCatalog.get("FoodDataset_Val")
    img = cv2.imread("C:\\Users\\godonan\\Documents\\AR\\2021_11_12_22_55_37\\frame_00004.jpg")
    print(img.shape)
    outputs = predictor(img)
    train_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    visualizer = Visualizer(img[:, :, ::-1], metadata=train_meta, scale=1.2)
    out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(20, 10))
    plt.imshow(out.get_image()[..., ::-1][..., ::-1])
    plt.show()
    #cv2.imshow('window', out.get_image()[..., ::-1][..., ::-1])
    #cv2.waitKey(0)
    print("Done.")


def test2():
    cfg = get_cfg()
    #trainer = COCOTrainer(cfg)
    #trainer.resume_or_load()
    #train_metadata = MetadataCatalog.get("FoodDataset_Train")
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0

    predictor = DefaultPredictor(cfg)
    #evaluator = COCOEvaluator("FoodDataset_Val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    #inference_on_dataset(model, val_loader, evaluator)

    img = cv2.imread("C:\\Users\\godonan\\Downloads\\000000439715.jpg")
    print(img.shape)
    outputs = predictor(img)
    train_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    visualizer = Visualizer(img[:, :, ::-1], metadata=train_meta, scale=1.2)
    out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(20, 10))
    plt.imshow(out.get_image()[..., ::-1][..., ::-1])
    plt.show()
    #cv2.imshow('window', out.get_image()[..., ::-1][..., ::-1])
    #cv2.waitKey(0)
    print("Done.")

def main():
    #train()
    test3()
    # Evaluate

if __name__ == '__main__':
    main()

