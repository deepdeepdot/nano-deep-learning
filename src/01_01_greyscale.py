
import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("panda-corner.jpg")
nrows, ncols = img.shape[0], img.shape[1]
nchannels = img.shape[2]

greyed = np.zeros((nrows, ncols, nchannels), dtype=int)

for row in range(nrows):
    for col in range(ncols):
        avg = sum(img[row,col,:]) / 3
        greyed[row,col,:] = avg

plt.imshow(greyed), plt.show()
plt.imsave("out/panda-grey.png", greyed)
