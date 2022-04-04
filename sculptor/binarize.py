import glob
import os
import cv2
from pathlib import Path
from PIL import Image
img_paths = glob.glob("*.jpg")
from collections import defaultdict

for imgpath in img_paths:
    basename = Path(imgpath).stem
    im = cv2.imread(imgpath)
    th, im_th = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)

    normImage = cv2.normalize(im_th, None, 0, 1, cv2.NORM_MINMAX)
    output_path = os.path.join(basename + ".png")
    cv2.imwrite(output_path, normImage)

    im2 = Image.open(output_path)
    by_color = defaultdict(int)
    for pixel in im2.getdata():
        by_color[pixel] += 1
    print(by_color)