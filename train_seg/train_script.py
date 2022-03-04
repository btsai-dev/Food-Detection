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