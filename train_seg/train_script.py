import copy
import os
import sys
import time
from pathlib import Path

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.builder import DATASETS

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger


ROOT_DIR = str(Path(__file__).resolve().parents[1]).replace('\\', '/')
#sys.path.append(ROOT_DIR)

CONFIG = ROOT_DIR + "/configs/foodseg/Upernet_swin_256x512.py"

WORK_DIR = ROOT_DIR + "/checkpoints/seg_model_4"

def get_info():
    print("ROOT_DIR:", ROOT_DIR)
    print("CONFIG:", CONFIG)
    print("WORK_DIR:", WORK_DIR)

DETERMINISTIC = False
SEED = 42
DATA_ROOT = "C:/Users/localadmin/Documents/btsai-dev-repositories/_DATASETS/FoodSeg103/Images"
SPLITS = "C:/Users/localadmin/Documents/btsai-dev-repositories/_DATASETS/FoodSeg103/ImageSets"

@DATASETS.register_module()
class FoodDataset(CustomDataset):
    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        pass

    #CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    #           21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    #           41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    #           61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    #           81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
    #           101, 102, 103)

    # Train on ONLY apples.
    CLASSES = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0)
    
    PALETTE = [[0, 0, 0], [40, 100, 150], [80, 150, 200], [120, 200, 10], [160, 10, 60],
               [200, 60, 110], [0, 110, 160], [40, 160, 210], [80, 210, 20], [120, 20, 70],
               [160, 70, 120], [200, 120, 170], [0, 170, 220], [40, 220, 30], [80, 30, 80],
               [120, 80, 130], [160, 130, 180], [200, 180, 230], [0, 230, 40], [40, 40, 90],
               [80, 90, 140], [120, 140, 190], [160, 190, 0], [200, 0, 50], [0, 50, 100],
               [40, 100, 150], [80, 150, 200], [120, 200, 10], [160, 10, 60], [200, 60, 110],
               [0, 110, 160], [40, 160, 210], [80, 210, 20], [120, 20, 70], [160, 70, 120],
               [200, 120, 170], [0, 170, 220], [40, 220, 30], [80, 30, 80], [120, 80, 130],
               [160, 130, 180], [200, 180, 230], [0, 230, 40], [40, 40, 90], [80, 90, 140],
               [120, 140, 190], [160, 190, 0], [200, 0, 50], [0, 50, 100], [40, 100, 150],
               [80, 150, 200], [120, 200, 10], [160, 10, 60], [200, 60, 110], [0, 110, 160],
               [40, 160, 210], [80, 210, 20], [120, 20, 70], [160, 70, 120], [200, 120, 170],
               [0, 170, 220], [40, 220, 30], [80, 30, 80], [120, 80, 130], [160, 130, 180],
               [200, 180, 230], [0, 230, 40], [40, 40, 90], [80, 90, 140], [120, 140, 190],
               [160, 190, 0], [200, 0, 50], [0, 50, 100], [40, 100, 150], [80, 150, 200],
               [120, 200, 10], [160, 10, 60], [200, 60, 110], [0, 110, 160], [40, 160, 210],
               [80, 210, 20], [120, 20, 70], [160, 70, 120], [200, 120, 170], [0, 170, 220],
               [40, 220, 30], [80, 30, 80], [120, 80, 130], [160, 130, 180], [200, 180, 230],
               [0, 230, 40], [40, 40, 90], [80, 90, 140], [120, 140, 190], [160, 190, 0],
               [200, 0, 50], [0, 50, 100], [40, 100, 150], [80, 150, 200], [120, 200, 10],
               [160, 10, 60], [200, 60, 110], [0, 110, 160], [40, 160, 210]]

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
        assert os.path.exists(self.img_dir) and os.path.exists(self.ann_dir)w


def main():
    cfg = Config.fromfile(CONFIG)

    cfg.dataset_type = "FoodDataset"
    cfg.data_root = DATA_ROOT
    cfg.data = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        train=dict(
            type=cfg.dataset_type,
            data_root=cfg.data_root,
            img_dir="img_dir/train",
            ann_dir="ann_dir/train",
            pipeline=cfg.train_pipeline,
            split=SPLITS + "/train.txt"
        ),
        val=dict(
            type=cfg.dataset_type,
            data_root=cfg.data_root,
            img_dir="img_dir/test",
            ann_dir="ann_dir/test",
            pipeline=cfg.test_pipeline,
            split=SPLITS + "/test.txt"
        ),
        test=dict(
            type=cfg.dataset_type,
            data_root=cfg.data_root,
            img_dir='img_dir/test',
            ann_dir='ann_dir/test',
            pipeline=cfg.test_pipeline,
            split=SPLITS + "/test.txt"
        )
    )

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.work_dir = WORK_DIR
    cfg.gpu_ids = range(0, 1)

    distributed = False
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(CONFIG)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info

    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.seed = SEED
    meta['seed'] = SEED
    meta['exp_name'] = os.path.basename(CONFIG)

    model = build_segmentor(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))

    logger.info(model)

    datasets = [build_dataset(cfg.data["train"])]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
        if cfg.checkpoint_config is not None:
            # save mmseg version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
                config=cfg.pretty_text,
                CLASSES=datasets[0].CLASSES,
                PALETTE=datasets[0].PALETTE)

    model.CLASSES = datasets[0].CLASSES

    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=True,
        timestamp=timestamp,
        meta=meta
    )

    print("Done.")
    
if __name__ == "__main__":
    main()