import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("panda-corner.jpg")
nrows, ncols = img.shape[0], img.shape[1]
nchannels = img.shape[2]

# Kernels taken from: http://setosa.io/ev/image-kernels/
kernels = {
    "outline": [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ],
    "sharpen": [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ],
    "right_sobel": [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ],
    "blur": [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625]
    ],
    "emboss": [
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ]
}

def create_kernel_image(img, kernel):
    kernel = np.array(kernel).reshape((1, 3*3)) # so we can dot the source
    buffer = np.zeros((nrows, ncols, 3))

    for i in range(1, nrows-2):
        for j in range(1, ncols-2):
            for c in range(nchannels):
                source = img[i:i+3, j:j+3, c].reshape((3*3, 1))
                # buffer[i][j][c] = np.sum(np.multiply(source, kernel))
                # x10: Massive performance gains for matrix multiply with GPUs
                buffer[i][j][c] = np.matmul(kernel, source)

    buffer = np.clip(buffer, 0, 255).astype(int)
    return buffer

# For testing:
# kernels = {
#     "sharpen.test.v99": [
#         [0, -1, 0],
#         [-1, 5, -1],
#         [0, -1, 0]
#     ]
# }

for label, kernel in kernels.items():
    buffer = create_kernel_image(img, kernel)
    plt.imsave(f"out/panda-{label}.png", buffer)

print("Done!")