import matplotlib.pyplot as plt
import numpy as np

def upsample(input, output, stride=4):
    img = plt.imread(input)
    nrows, ncols = img.shape[0], img.shape[1]
    nfilters = img.shape[2]

    buffer = np.zeros((stride * nrows, stride * ncols, nfilters), dtype=int)

    s = stride
    for row in range(nrows):
        for col in range(ncols):
            buffer[row*s:(row+1)*s, col*s:(col+1)*s, :] = img[row, col, :]

    plt.imsave(output, buffer.astype(np.uint8))
    # plt.imshow(buffer)

upsample("panda-corner.jpg", "panda-corner.x4.jpg")
upsample("panda-corner.jpg", "panda-corner.x2.jpg", stride=2)
