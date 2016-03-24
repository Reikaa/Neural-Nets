import numpy as np
from skimage.measure import block_reduce
from scipy import misc, ndimage
import scipy

''' Check whether data manipulation works on actual data'''

# ------------------ PRACTICE ----------------------
# a = misc.imread("outfile.jpg")
# print a.shape
# # lx, ly = a.shape
# # rotate_image = rotate_image[lx / 4: - lx / 4, ly / 4: - ly / 4]
# rotate_image = ndimage.rotate(a, -20, reshape = False)
# print a.shape
# print rotate_image
# misc.imsave('outfile.jpg', rotate_image)

# ------------------ ACTUAL DATA ----------------------
import mnist_loader
# you can do this on either set
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# test on 5000 images
data = validation_data[:5000]
print len(data) # it's 5K images now`

# reshape for display
data1 = data + [(ndimage.rotate(x, -20, reshape =False),y) for x,y in data]
data2 = data1 + [(ndimage.rotate(x, 20, reshape =False),y) for x,y in data]
data = data2 + [(ndimage.rotate(x, 0),y) for x,y in data]

print len(data) # it's 20K images now
# convert them into 28x28 instead of col vector
data = [(im[0].reshape(28,28), im[1]) for im in data]
pic = data[6000][0]
scipy.misc.imsave('data.jpg', pic)
