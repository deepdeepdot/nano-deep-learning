## Image filters

### Assignment preparation

1. Download and install Anaconda.
2. Create the *nanos* environment with python=3.6 and matplotlib and numpy

        # Download and install Anaconda
        $ conda create -n nanos python=3.6 ipython
        $ conda activate nanos
        $ conda install matplotlib
        $ conda install numpy

3. Clone the repository: github.com/deepdeepdot/nano-deep-learning-in-7-sessions

        $ git clone https://github.com/deepdeepdot/nano-deep-learning-in-7-sessions.git
        $ cd nano-deep-learning-in-7-sessions

4. Run the examples in the the *src* folder

        $ cd src
        $ python 01_01_greyscale.py
        $ python 01_02_downsampling.py
        $ python 01_06_image_kernel.py

        # Check in the /out for the resulting images
        # Pick your own images (and replace panda-corner.jpg)
        # To keep the sci-fi theme, pick few images in this space, or if you have some strong preference, pick your own.

5. Go over the assignment: downsampling, upsampling, greys and tints and image filters. The last exercise in *Matrix* is optional, but see if you can figure it out


### Image compression using strides

1. Image stride, we have covered stride = 2 and this reduces the size of the image by a quarter (1/2 width * 1/2 height = 1/4 image).
Extend this implementation to support strides of size: 2, 3, 4, etc

        downsampled_image = downsample(image, stride=4)

We have covered downsampling, the reverse is upsampling.

One way to upsample is the following

        1 2  ---->  1 1  2 2
        3 4         1 1  2 2
                    3 3  4 4
                    3 3  4 4

Another one way is to fill up with zeros

        1 2  ---->  1 0  2 0
        3 4         0 0  0 0
                    3 0  4 0
                    0 0  0 0


2. We used np.max(), but instead we could use np.average() when pooling. Compare the different image results.

3. Extra: When downsampling, we have reduced the size of the image; but when saving to PNG, we increased the size of the saved image. Why? Try different image formats, justify the numbers.


### Greys, Tints

1. We learned how to make grayscale images, let's create 'red', or 'green', or 'blue' versions.

        colored_image = transform_image(image, filter='red')

2. How about yellow? How about any rgb color?
3. Extra: How can transform to a Hue of 270 degrees (magenta/purple)
   See: https://en.wikipedia.org/wiki/HSL_and_HSV#Conversion_RGB_to_HSL/HSV_used_commonly_in_software_programming


### Filters

We have seen how to apply a filter and save an image.
Given a map of filters, produce different images

    filters = {
        "sobelius": [....],
        "blah_blah": [....]
    }

    should save the images:
        imagename_sobelius.png
        imagename_blah_blah.png
    
    Complete the list of filters from:
    http://setosa.io/ev/image-kernels/


### Matrix

Recall our optimized kernel transformation code

    emboss = [
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ]

    buffer = np.zeros((nrows, ncols, 3))

    sharpen = np.array(sharpen).reshape((1, 3*3)) # so we can dot the source

    for i in range(nrows):
        for j in range(ncols):
            if (i > 0 and j > 0) and (i < nrows-1 and j < ncols-1):
                for c in range(nchannels):
                    source = img[i-1:i+2, j-1:j+2, c].reshape((3*3, 1))
                    # x10: Massive performance gains for matrix multiply with GPUs
                    buffer[i][j][c] = np.matmul(emboss, source)

    buffer = np.clip(buffer, 0, 255).astype(int)


Our optimization involved using the dot product of the blur and source

Our original computation was

    buffer[i][j][c] = np.sum(np.multiply(source, blur))

But we wanted to convert to:

    buffer[i][j][c] = np.dot(blur, source)

In order to this, we needed to reshape our matrices to vectors

    blur: 3x3 -> 1 x (3*3)
    source: 3x3 -> (3*3) x 1

    blur = np.array(blur).reshape((1, 3*3))
    source = source.reshape((3*3, 1))


But first we needed to reshape both the blur filter and the source

    blur: 1 x (3*3)
    source: (3*3) x 1

    blur = np.array(blur).reshape((1, 3*3))
    source = source.reshape((3*3, 1))
    buffer[i][j][c] = np.dot(blur, source)


Let's rewrite the code, so that we compute:

    buffer[i][j][c] = np.dot(source, blur)

(Hint: we need to reshape diffrently)

Will the image result be the same? Why?

