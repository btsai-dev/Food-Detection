import cv2
import numpy as np

px = 1117.468988560457
py = 209.76507440958204
projection_matrix = [
    [1.51982379, 0., -0.00594974, 0],
    [0., 2.0264318, -0.00313807, 0.],
    [0., 0., -0.99999976, -0.001],
    [0., 0., -1., 0.]]
camera_pose = [
    [0.00129428,  0.99995673,  0.00920632,  0.02649486],
    [-0.64781392, -0.00617494,  0.76177341,  0.29934236],
    [0.76179737, -0.00694988,  0.64777809,  0.16662851],
    [0.,  0.,  0.,  1.]
]
image_width = 1920
imag_height = 1440

pos_x = (px / image_width - 0.5)*2
pos_y = (((py / imag_height - 1) * -1) - 0.5) * 2
view_matrix = np.linalg.inv(camera_pose)
mvp = np.dot(projection_matrix, view_matrix)

e0 = np.array([pos_x, pos_y, 0.997, 0.39914286])
# Compute that ratio

print("Done.")