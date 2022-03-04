from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.builder import DATASETS
import os

@DATASETS.register_module()
class FoodDataset(CustomDataset):
    CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
               21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
               41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
               61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
               81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
               101, 102, 103)

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

    
    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        pass

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
        assert os.path.exists(self.img_dir) and os.path.exists(self.ann_dir)
