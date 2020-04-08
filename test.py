import align2d
import numpy as np

print(dir(align2d))


aligner = align2d.ImageAlign2d()

r = aligner.align(np.zeros((480, 640, 3), np.uint8))

print(r)
# align2d.align(np.zeros((480, 640, 3), np.uint8)[:, 200:300], np.zeros((480, 640, 3), np.uint8)[:, 300:400])
