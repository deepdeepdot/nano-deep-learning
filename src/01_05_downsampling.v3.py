import matplotlib.pyplot as plt
import numpy as np

def downsample(img, stride=2):
    nrows, ncols = img.shape[0], img.shape[1]
    nfilters = img.shape[2]
    s = stride

    half_rows, half_cols = int(nrows / stride), int(ncols / stride)

    buffer = np.zeros((half_rows, half_cols, nfilters), dtype=int)

    for row in range(stride, half_rows-stride):
        for col in range(stride, half_cols-stride):
            selection = img[row*s:(row+1)*s, col*s:(col+1)*s,:]
            for c in range(nfilters):
                buffer[row, col, c] = np.max(selection[:,:,c]) # maybe np.average()?

    buffer = buffer.astype(np.uint8)
    return buffer


img = plt.imread("panda-corner.jpg")

for stride in [2, 3, 4]:
    down = downsample(img, stride)
    plt.imsave(f"out/panda-on-a-diet.{stride}.png", down)

