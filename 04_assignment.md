# Assignment 4: Image Classifier

For this assignment, you'll need to accomplish these three items:
* Run the CIFAR 10 project and save the results
* Watch MIT's CNN: Computer Vision
* Implement one of the projects using ML5.js
    - You can choose any of the ML5.js projects<br>
    These are in increasing order of difficulty:
        * Applying Style Transfer
        * Mustachio or Mexican Sombrero?
        * Image classifier
        * Training Style Transfer


## 1. CIFAR 10

The CIFAR 10 data set is located at:<br>
https://www.cs.toronto.edu/~kriz/cifar.html

In class, we have seen cifar 10 in action with convnet.js<br>
https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html


Follow the steps for training the CIFAR
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

Deliverables: "04_cifar.txt"
It should be placed in your github, folder /hw4


A) Copy/paste all the output from running this tutorial

It should contains lines like the following (but many more lines)

    [1,  2000] loss: 2.201
    [1,  4000] loss: 1.875
    ... (more lines)

    Accuracy of the network on the 10000 test images:

    Accuracy of plane : 57 %
    Accuracy of   car : 61 %
    Accuracy of  bird : 26 %
    ... (more lines)


B) Re-train the model for a longer time. This can be achieved by increasing the number of epochs, say from `range(2)` to `rnge(10)` and find out how much improvement we get for the model and the class predictions.

    for epoch in range(10):  # loop over the dataset multiple times

Copy/paste all the iteratons with the predicted loss

It will contains lines similar to:
    [3,  2000] loss: 1.039
    [3,  4000] loss: 1.066
    [3,  6000] loss: 1.049
    ... (more lines)

    Accuracy of the network on the 10000 test images: 60 %

    Accuracy of plane : 68 %
    Accuracy of   car : 65 %
    Accuracy of  bird : 44 %
    ... (more lines)


C) After you execute all the python code using ipython, save your ipython session

%save my-cifar-pytorch.py 1-20 # this will save the first 20 lines, maybe you need more lines?


## 2. Watch MIT's CNN: Computer Vision

https://www.youtube.com/watch?v=H-HVZJ7kGI0&index=1&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI

Questions
* What's a Convolution?
* What's ReLU?
* What's Upsampling?


## 3. ML5.js Projects

### A. Implementing Style Transfer

What's Style Transfer?<br>
See: https://github.com/NVIDIA/FastPhotoStyle

Based on a working example for Fast Style Transfer
https://ml5js.org/docs/style-transfer-image-example

Deliverables
* Create a web page in which the user can choose any image URL for the image as the source
* Support extra models designed for ML5
https://github.com/ml5js/ml5-data-and-models/tree/master/models/style-transfer


Feature Enhancements (optional)
* Support for image file upload (need a web server app)
* Support for drag n drop of images
* Support custom styles by using a custom model derived from the project below


### B. Mustachio or Mexican Sombrero?

Modify the Posenet example, so instead of 7 red dots, place a sombrero on top someone's head.
Or place a mustache or beard on the face (below the eyes and nose)
Or be creative and try a baseball cap or some Dracula teeth
https://ml5js.org/docs/posenet-webcam


### C. Image Classifier

Follow the steps to create your image classifier using ml5.js
https://ml5js.org/docs/training-introduction

#### Youtube: The Coding Train

* A beginner's guide to Machine Learning with ml5.js
https://www.youtube.com/watch?v=jmznx0Q1fP0&vl=en

* ml5.js: Image Classification with MobileNet
https://www.youtube.com/watch?v=yNkAuWz5lnY


### D. Training Style Transfer

https://ml5js.org/docs/training-styletransfer

It requires running on a GPU system (like paperspace.com)
And process 15 GB of image data from the COCO dataset.
http://cocodataset.org/#home

Using a Titan X, it took 4 to 6 hours for training.
On a Mac, it can take several months running with CPU.


