import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("panda-corner.jpg")
nrows, ncols = img.shape[0], img.shape[1]
nchannels = img.shape[2]

sharpen = [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
]

buffer = np.zeros((nrows, ncols, 3))

for i in range(1, nrows-1):
    for j in range(1, ncols-1):
        for c in range(nchannels):
            source = img[i-1:i+2, j-1:j+2, c]
            buffer[i][j][c] = np.sum(np.multiply(source, sharpen))

buffer = np.clip(buffer, 0, 255).astype(int)
plt.imsave(f"out/panda-sharpen.png", buffer)

print("Done!")