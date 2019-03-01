import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("panda-corner.jpg")
nrows, ncols = img.shape[0], img.shape[1]
nchannels = img.shape[2]

blur = [
    [0.0625, 0.125, 0.0625],
    [0.125, 0.25, 0.125],
    [0.0625, 0.125, 0.0625]
]

buffer = np.zeros((nrows, ncols, 3))
# buffer = [[[0 for i in range(3)] for j in range(ncols)] for k in range(nrows)]

for i in range(1, nrows-1):
    for j in range(1, ncols-1):
        for c in range(nchannels):
            source = img[i-1:i+2, j-1:j+2, c]
            buffer[i][j][c] = np.sum(np.multiply(source, blur))

buffer = np.clip(buffer, 0, 255).astype(int)
plt.imsave(f"out/panda-blur.png", buffer)

print("Done!")