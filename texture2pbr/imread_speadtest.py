import cv2
from time import time
import scipy.misc
import skimage.io
import PIL
import numpy as np

import asyncio
import multiprocessing
from multiprocessing import Process, Queue

print("cv2.misc.imread:")
tic = time()
for i in range(10):
    im = cv2.imread("test2.png")
    im = cv2.resize(im, (100,100))
print(time()-tic)
"""
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
"""

class Processor:
    def __call__(self,file_name):
        im = cv2.imread(file_name)
        im = cv2.resize(im, (100,100))
        return im

tic = time()
pool = multiprocessing.Pool()
files = ['test2.png'] * 10


proc = Processor()
#proc = cv2.imread
results = pool.map(proc,files)
print(time()-tic)
