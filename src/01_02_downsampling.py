import matplotlib.pyplot as plt
import numpy as np

s = stride = 2 # Try 6?

img = plt.imread("panda-corner.jpg")
nrows, ncols = img.shape[0], img.shape[1]
nchannels = img.shape[2]

half_rows, half_cols = int(nrows / stride), int(ncols / stride)
print(nrows, ncols, half_rows, half_cols)

buffer = np.zeros( \
    shape=(half_rows, half_cols, nchannels), dtype=int)

for row in range(half_rows-1):
    for col in range(half_cols-1):
        passed_top_left = row > stride and col > stride
        within_bottom_right = \
            row < half_rows - s and col < half_cols - s

        if passed_top_left and within_bottom_right:
            selection = img[row*s:(row+1)*s, col*s:(col+1)*s,:]
            r = selection[:,:,0]
            g = selection[:,:,1]
            b = selection[:,:,2]
            buffer[row, col, 0] = np.max(r) # if r.size != 0 else 0
            buffer[row, col, 1] = np.max(g) # if g.size != 0 else 0
            buffer[row, col, 2] = np.max(b) # if b.size != 0 else 0
        else:
            buffer[row, col, :] = 0

#plt.imshow(buffer)
#plt.show()
plt.imsave("out/panda-on-a-diet.png", buffer)
