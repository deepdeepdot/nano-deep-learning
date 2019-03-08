import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("panda-corner.jpg")
nrows, ncols, nchannels = img.shape[0], img.shape[1], img.shape[2]
half_rows, half_cols = int(nrows / 2), int(ncols / 2)

buffer = np.zeros((half_rows, half_cols, nchannels), dtype=int)

for row in range(2, half_rows-2):
    for col in range(2, half_cols-2):
        selection = img[row*2:(row+1)*2, col*2:(col+1)*2,:]
        for c in range(nchannels):
            buffer[row, col, c] = np.max(selection[:,:,c])

#plt.imshow(buffer)
#plt.show()

buffer = buffer.astype(np.uint8)
plt.imsave("out/panda-on-a-diet.jpg", buffer, quality=80)
