import numpy as np
import matplotlib.pyplot as plt
import tomopy # library for tomography methods / testing benchmark
import seaborn as sns
import numba as nb 
from numba import njit, prange, float32, float64


import skimage.data as skdata 
import skimage.transform

from utilities import radon2d_sidon, siddon_line_projector, matrixradon2d


camera_phantom = skdata.camera().astype(float)/255. # camera phantom
pig_phantom = plt.imread('pig.jpg', format='jpg') # pig phantom
shepp_logan_phantom = skdata.shepp_logan_phantom() # Shepp-Logan phantom (for tomo)

fig, axs = plt.subplots(1, 3, figsize=(12, 8), sharey=True)
axs[0].imshow(camera_phantom, cmap='gray')
axs[1].imshow(pig_phantom)
axs[2].imshow(shepp_logan_phantom)
#plt.show()
print(camera_phantom.shape, shepp_logan_phantom.shape, shepp_logan_phantom.shape)


camera_phantom = skimage.transform.resize(camera_phantom, (128,128))
pig_phantom = skimage.transform.resize(pig_phantom, (128,128))
shepp_logan_phantom = skimage.transform.resize(shepp_logan_phantom, (64, 64))

fig, axs = plt.subplots(1, 3, figsize=(12, 8), sharey=False)
axs[0].imshow(camera_phantom, cmap='gray')
axs[1].imshow(pig_phantom)
axs[2].imshow(shepp_logan_phantom)
#plt.show()
print(camera_phantom.shape, pig_phantom.shape, shepp_logan_phantom.shape)

# projection matrix for Radon transform
projector = matrixradon2d(64, 32, 32, 1.0)
sinogram = (projector @ shepp_logan_phantom.reshape((-1,1))) #.reshape((32, 32))



