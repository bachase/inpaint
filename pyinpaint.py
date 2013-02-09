import numpy as np
import matplotlib.pyplot as plt

rgb2ycbcr_mat = np.array([
    [0.299, 0.587, 0.114],
    [-0.168736,-0.331264,0.5],
    [0.5,-0.418688,-0.081312]
    ])

ycbcr_offsets = np.array([[16.],[128.],[128.]])
ycbcr_scales = np.array([[219.],[224.],[224.]])



def rgb2ycbcr(img):
    res = img.astype(np.float)/255.
    res = res.swapaxes(0,2)
    old_shape = res.shape
    res = res.reshape((3,-1))
    res = np.dot(rgb2ycbcr_mat,res)
    res = res * ycbcr_scales + ycbcr_offsets
    res.shape = old_shape
    res = np.swapaxes(res,2,0)
    return np.round(res).astype(np.uint8)

def ycbcr2rgb(img):
    res = img.astype(np.float)
    res = res.swapaxes(0,2)
    old_shape = res.shape
    res = res.reshape((3,-1))
    res = (res - ycbcr_offsets)/ycbcr_scales
    res = np.dot(np.linalg.inv(rgb2ycbcr_mat),res)
    res.shape = old_shape
    res = np.swapaxes(res,2,0)
    return np.round(res * 255.).astype(np.uint8)

a = plt.imread("img/dots_input.jpeg")
plt.figure()
plt.imshow(a)
plt.figure()
a1 = rgb2ycbcr(a)
plt.imshow(a1)
a2 = ycbcr2rgb(a1)
plt.figure()
plt.imshow(a2)
plt.show()


