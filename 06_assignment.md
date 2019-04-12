### Fashion MNIST VAE encoder (Tensorflow.js demo)

https://github.com/tensorflow/tfjs-examples/tree/master/fashion-mnist-vae

    // Need nodejs, yarn
    npm install -g yarn

    git clone https://github.com/tensorflow/tfjs-examples.git
    cd tfjs-examples/fashion-mnist-vae

    # Make sure you switched to Python 2.7 (for some depedencies)
    conda activate python_2.7
    yarn install

    yarn download-data
    yarn train
    yarn serve-model

    # separate terminal
    yarn serve-client

    # Open in a browser
    http://localhost:1234/


### Deep Image Prior (using Pytorch)

https://dmitryulyanov.github.io/deep_image_prior
* JPEG artifacts removal
* Inpainting (painting blank/corrupted sections)
* Super-resolution
* Denoising
* Inpainting (watermark removal)

        git clone https://github.com/DmitryUlyanov/deep-image-prior
        cd deep-image-prior
        conda activate nanos
        conda install jupyter
        conda env create -f environment.yml
        # conda install --yes --file requirements.txt

        # Share the kernel for the Jupyter notebook
        jupyter notebook .

        # Go to localhost:8000/<project>.ypnb
        # Change kernel to `nanos`
        # Execute each cell

On your local laptop or computer, if you don't have a CUDA GPU,
replace the following three lines

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark =True
        dtype = torch.cuda.FloatTensor

With the following:

        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark =True
        dtype = torch.FloatTensor

And re-run the experiments, the experience is about 10 times slower than with GPU.

If you  want to run on Google's Colab:

        !git clone hhttps://github.com/DmitryUlyanov/deep-image-prior.git
        !cd deep-image-prior

Use Deep Image Prior to fix your family photos.


### Style Transfer with webcam

