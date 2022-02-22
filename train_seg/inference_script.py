import argparse
import os
import os.path as osp
import shutil
import time
import warnings

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.builder import DATASETS
from mmcv.utils import DictAction
from pathlib import Path
from mmseg import digit_version
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import setup_multi_processes

ROOT_DIR = str(Path(__file__).resolve().parents[1]).replace('\\', '/')
#sys.path.append(ROOT_DIR)

CONFIG = ROOT_DIR + "/checkpoints/seg_model_4/Upernet_swin_256x512.py"

WORK_DIR = ROOT_DIR + "/checkpoints/seg_model_4"
SHOW_DIR = WORK_DIR + "/painted"
FORMAT_ONLY = False
CHECKPOINT = ROOT_DIR + "/checkpoints/seg_model_4/latest.pth"
OUT = None
EVAL = "mIoU"

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
    CLASSES = (0, 25)
    
    PALETTE = [[0, 0, 0], [40, 100, 150]]

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
        assert os.path.exists(self.img_dir) and os.path.exists(self.ann_dir)


def main():
    cfg = mmcv.Config.fromfile(CONFIG)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    cfg.gpu_ids = cfg.gpu_ids[0:1]

    rank, _ = get_dist_info()
    mmcv.mkdir_or_exist(osp.abspath(WORK_DIR))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    json_file = osp.join(WORK_DIR,
                         f'eval_single_scale_{timestamp}.json')
    if rank == 0:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(CONFIG))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(WORK_DIR,f'eval_single_scale_{timestamp}.json')

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, CHECKPOINT, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {}

    # Deprecated
    efficient_test = eval_kwargs.get('efficient_test', False)
    if efficient_test:
        warnings.warn(
            '``efficient_test=True`` does not have effect in tools/test.py, '
            'the evaluation and format results are CPU memory efficient by '
            'default')

    eval_on_format_results = (
        EVAL is not None and 'cityscapes' in EVAL)
    if eval_on_format_results:
        assert len(EVAL) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if FORMAT_ONLY or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None

    warnings.warn(
        'SyncBN is only supported with DDP. To be compatible with DP, '
        'we convert SyncBN to BN. Please use dist_train.sh which can '
        'avoid this error.')
    if not torch.cuda.is_available():
        assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
            'Please use MMCV >= 1.4.4 for CPU training!'
    model = revert_sync_batchnorm(model)
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    results = single_gpu_test(
        model,
        data_loader,
        True,
        SHOW_DIR,
        False,
        opacity=0.5,
        pre_eval=EVAL is not None and not eval_on_format_results,
        format_only=FORMAT_ONLY or eval_on_format_results,
        format_args=eval_kwargs)

    rank, _ = get_dist_info()
    if rank == 0:
        if OUT:
            warnings.warn(
                'The behavior of ``args.out`` has been changed since MMSeg '
                'v0.16, the pickled outputs could be seg map as type of '
                'np.array, pre-eval results or file paths for '
                '``dataset.format_results()``.')
            print(f'\nwriting results to {OUT}')
            mmcv.dump(results, OUT)
        if EVAL:
            eval_kwargs.update(metric=EVAL)
            metric = dataset.evaluate(results, **eval_kwargs)
            metric_dict = dict(config=CONFIG, metric=metric)
            mmcv.dump(metric_dict, json_file, indent=4)
            if tmpdir is not None and eval_on_format_results:
                # remove tmp dir when cityscapes evaluation
                shutil.rmtree(tmpdir)

if __name__ == "__main__":
    main()