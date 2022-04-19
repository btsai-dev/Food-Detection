import numpy.matlib
import numpy as np

a = np.array([
    [1,2,3,],
    [4,5,6],
    [7,8,9]
])
b = np.array([11,2,1.432])
print(a.shape)
print(b.shape)
print("===")
print(np.dot(a,b))