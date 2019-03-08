import matplotlib.pyplot as plt
import numpy as np

s = stride = 2 # Try 6?

img = plt.imread("panda-corner.jpg")
nrows, ncols = img.shape[0], img.shape[1]
nchannels = img.shape[2]

half_rows, half_cols = int(nrows / stride), int(ncols / stride)
print(nrows, ncols, half_rows, half_cols)

buffer = np.zeros((half_rows, half_cols, nchannels), dtype=int)

for row in range(half_rows-stride):
    for col in range(half_cols-stride):
        selection = img[row*s:(row+1)*s, col*s:(col+1)*s,:]
        r = selection[:,:,0]
        g = selection[:,:,1]
        b = selection[:,:,2]
        buffer[row, col, 0] = np.max(r)
        buffer[row, col, 1] = np.max(g)
        buffer[row, col, 2] = np.max(b)

#plt.imshow(buffer)
#plt.show()
buffer = buffer.astype(np.uint8)
plt.imsave("out/panda-on-a-diet.v2.png", buffer)
