import cv2 as cv
import numpy as np
import skimage
from scipy import ndimage
from matplotlib import pyplot
import scipy.ndimage as ndi

# image=np.ones([100,100])
# diag=len(np.diag(image//2))
# image=np.pad(image,pad_width=diag+10)
# _=np.linspace(-1,1,image.shape[0])
# xv,yv=np.meshgrid(_,_)
# # # fig,ax=pyplot.subplots(3,1)
# # # ax[0].pcolor(xv,yv,image)
# image[(xv-0.1)**2+(yv-0.2)**2<0.01]=2
# image[(xv-0.3)**2+(yv-0.3)**2<0.01]=2
# # # image_rot=skimage.transform.rotate(image,45)
# # # ax[1].pcolor(xv,yv,image)
# # # ax[2].pcolor(xv,yv,image_rot)
# # # pyplot.show()
#
#
# # image = np.array([[0, 0, 0, 0, 0],
# #                   [0, 0, 3, 0, 0],
# #                   [0, 0, 2, 0, 0],
# #                   [0, 0, 1, 0, 0],
# #                   [0, 0, 0, 0, 0]])
#
# _=np.linspace(-1,1,image.shape[0])
# xv,yv=np.meshgrid(_,_)
# fig,ax=pyplot.subplots(1,1)
# ax.pcolor(xv,yv,image)
# pyplot.show()
# # kernel = np.ones((4, 4), np.uint8)
# # a = skimage.morphology.dilation(image, kernel)
# # print(a)
# # kernel = np.ones((3, 3), np.uint8)
# # b = skimage.morphology.dilation(image, kernel)
# # print(b)
#
#
#
# kernel=np.ones((1,1))
# d=skimage.morphology.dilation(image, kernel)
# print(d)
# labels, label_nb = ndimage.label(d)
# label_count = np.bincount(labels.ravel().astype(np.int))
# label_count[0] = 0
#
# mask = labels == label_count.argmax()
# mask = skimage.morphology.dilation(mask, np.ones((1, 1)))
# mask = ndimage.morphology.binary_fill_holes(mask)
# mask = skimage.morphology.dilation(mask, np.ones((3, 3)))
# masked_image = mask * image
# _=np.linspace(-1,1,image.shape[0])
# xv,yv=np.meshgrid(_,_)
#
# ## mask denoises
# fig,ax=pyplot.subplots(1,1)
# ax.pcolor(xv,yv,mask)
# pyplot.show()
# fig,ax=pyplot.subplots(1,1)
# ax.pcolor(xv,yv,masked_image)
# pyplot.show()
#
#
# # # dilation
# # image_1 = skimage.morphology.dilation(image, np.ones((5, 5)))
# # fig,ax=pyplot.subplots(1,1)
# # ax.pcolor(xv,yv,image_1)
# # pyplot.show()
# # # erosion
# # image_2 = skimage.morphology.erosion(image, np.ones((5, 5)))
# # fig,ax=pyplot.subplots(1,1)
# # ax.pcolor(xv,yv,image_2)
# # pyplot.show()
#
#
#
# # chop image
# mask = image == 0
# coords = np.array(np.nonzero(~mask))
# top_left = np.min(coords, axis=1)
# bottom_right = np.max(coords, axis=1)
#
# # Remove the background
# croped_image = image[top_left[0]:bottom_right[0],
#                 top_left[1]:bottom_right[1]]
#
# fig,ax=pyplot.subplots(1,1)
# ax.pcolor(croped_image)
# pyplot.show()
#
#
# height, width = image.shape
# final_image = np.zeros((512, 512))
# pad_left = int((512 - width) / 2)
# pad_top = int((512 - height) / 2)
# # Replace the pixels with the image's pixels
# final_image[pad_top:pad_top + height, pad_left:pad_left + width] = image
#
# fig,ax=pyplot.subplots(1,1)
# ax.pcolor(final_image)
# pyplot.show()


im1 = np.array([[93, 36,  87],
               [18, 49,  51],
               [45, 32,  63]])

# weights_smooth = [[0.11, 0.11, 0.11],
#            [0.11, 0.11, 0.11],
#            [0.11, 0.11, 0.11]]

# weights_shapen = [[0, 0., 0.],
#            [0., 2, 0.],
#            [0., 0., 0.]]
# im_filt = ndi.convolve(im1, weights)
# print(im_filt)

im_filt = ndi.median_filter(im1, size=3)
print(im_filt)
im_filt =ndi.maximum_filter(im1, size=3)
print(im_filt)
im_filt =ndi.uniform_filter(im1, size=3)
print(im_filt)
