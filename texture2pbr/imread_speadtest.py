import cv2
from time import time
import scipy.misc
import skimage.io
import PIL
import numpy as np

print("cv2.misc.imread:")
for i in range(10):
    tic = time()
    im = cv2.imread("test2.png")
    print(time()-tic)

print("scipy.misc.imread:")
for i in range(10):
    tic = time()
    im = scipy.misc.imread("test2.png")
    print(time()-tic)

print("skimage.io.imread:")
for i in range(10):
    tic = time()
    im = skimage.io.imread("test2.png")
    print(time()-tic)

print("PIL.Image.open:")
for i in range(10):
    tic = time()
    pil_im = PIL.Image.open("test2.png")
    im = np.array(pil_im)
    print(time()-tic)
