### VI. Neural Networks


### Topics

* Tensorflow.js, api and demos
* Big Picture of Deep Learning
* Deep Learning Problems
* Model Zoo
* Neural Networks
* DeepMind


### Tensorflow.js API

* Demos<br>
https://www.tensorflow.org/js/demos/

* Examples<br>
https://github.com/tensorflow/tfjs-examples

* APIs in 5 mins<br>
https://towardsdatascience.com/50-tensorflow-js-api-explained-in-5-minutes-tensorflow-js-cheetsheet-4f8c7f9cc8b2

* API<br>
https://js.tensorflow.org/api/latest/


### TF.js: Teachable Machine with Tensorflow

* Demo: https://teachablemachine.withgoogle.com/

* Tutorial:<br>
https://observablehq.com/@nsthorat/how-to-build-a-teachable-machine-with-tensorflow-js

* Youtube:<br>https://www.youtube.com/watch?v=3BhkeY974Rg


### TF.js: Fashion MNIST VAE

* https://blog.keras.io/building-autoencoders-in-keras.html

* https://github.com/tensorflow/tfjs-examples/tree/master/fashion-mnist-vae


### Big Picture

* AI -> Machine Learning -> Deep Learning (NN)
* Neural Networks are old, but booming now:
    - Big Data: Model Zoo
    - Hardware: GPUs (CUDA), TPUs, embedded/iot
    - Software: Keras, Tensorflow, Pytorch/Caffe, tensorflow.js, ml5.js, magenta.js, TFLite


### Deep Learning Problems

| Problem Type    | Subcategory                  | Error loss<br>cost function
|-----------------|------------------------------|----------------------
| Classification  | binary: 2 classes            | cross-entropy logistic regression
|                 | multiple-class single-label  | softmax regression
| Regression      | scalar, numeric              | L1-norm, L2-norm 


### Binary Classification

* HotDog or Not
* https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3

* https://www.engadget.com/2017/05/15/not-hotdog-app-hbo-silicon-valley/

* Sentiment analysis: positive vs negative
* IMDB movie reviews: https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/3.5-classifying-movie-reviews.ipynb


### Multiple class Classfication

* MNIST Digits: classifying images into digits
* CIFAR 10: classifying images into categories
* MNIST Fashion: classifying images of clothes 
* Udacity MNIST Fashion:
https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l03c01_classifying_images_of_clothing.ipynb


### Regression

* Celsius to Farenheit (Udacity)
https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l02c01_celsius_to_fahrenheit.ipynb

* House Prediction
    - https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/3.7-predicting-house-prices.ipynb
    - https://towardsdatascience.com/machine-learning-project-predicting-boston-house-prices-with-regression-b4e47493633d
    - https://github.com/rromanss23/Machine_Leaning_Engineer_Udacity_NanoDegree/blob/master/projects/boston_housing/boston_housing.ipynb


### Deep Learning workflow

* Steps
    * 1) Training: Create the model (weights)
    * 2) Inference: Make predictions with the model

* Example: StyleTransfer
    * 1) Python/Tensorflow
        * StyleTransfer + Training => Model (weights)
    * 2) Web Javascript/HTML
        * ml5.js + Model => Predictions


### Model Zoo

* Pre-trained Models? https://modelzoo.co/

* https://github.com/BVLC/caffe/wiki/Model-Zoo
* https://github.com/tensorflow/models
* https://github.com/pytorch/vision/tree/master/torchvision/models

* Neural Network Zoo<br>
http://www.asimovinstitute.org/neural-network-zoo/


### Neural Networks

* Mind: How to Build a Neural Network (Part One)
https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/

* Intro to Neural Networks<br>
https://victorzhou.com/blog/intro-to-neural-networks/

* Neural Network Tutorial<br>
http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial

* Andrej Karpathy<br>
http://karpathy.github.io/neuralnets/


#### Andrew Trask (nn in 13 lines of Python)

    import numpy as np
    X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
    y = np.array([[0,1,1,0]]).T
    alpha,hidden_dim = (0.5,4)
    synapse_0 = 2*np.random.random((3,hidden_dim)) - 1
    synapse_1 = 2*np.random.random((hidden_dim,1)) - 1
    for j in range(60000):
        layer_1 = 1/(1+np.exp(-(np.dot(X,synapse_0))))
        layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))
        layer_2_delta = (layer_2 - y)*(layer_2*(1-layer_2))
        layer_1_delta = layer_2_delta.dot(synapse_1.T) * (layer_1 * (1-layer_1))
        synapse_1 -= (alpha * layer_1.T.dot(layer_2_delta))
        synapse_0 -= (alpha * X.T.dot(layer_1_delta))    


### Andrew Trask
* Part 1: https://iamtrask.github.io/2015/07/12/basic-python-network/
* Part 2: https://iamtrask.github.io/2015/07/27/python-network-part2/


### More Neural Networks

* Build a Neural Network<br>
https://enlight.nyc/projects/neural-network/

* Simple NN in PyTorch<br>
https://medium.com/dair-ai/a-simple-neural-network-from-scratch-with-pytorch-and-google-colab-c7f3830618e0

* Udacity 187<br>
https://classroom.udacity.com/courses/ud187


### RNN Revisited

* Understanding LSTM Networks<br>
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

* PyTorch: RNN<br>
https://medium.com/dair-ai/building-rnns-is-fun-with-pytorch-and-google-colab-3903ea9a3a79


### Style GANS

* https://www.lyrn.ai/2018/12/26/a-style-based-generator-architecture-for-generative-adversarial-networks/
* https://arxiv.org/abs/1812.04948
* https://github.com/NVlabs/stylegan


### DeepMind - Atari Games

* https://deepmind.com/research/dqn/
* https://github.com/deepmind/dqn
* https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf


### Reinforcement Learning

* https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2
* https://gym.openai.com/docs/
* http://blog.dota2.com/
https://www.theverge.com/2018/6/25/17492918/openai-dota-2-bot-ai-five-5v5-matches
* Alphago<br>
https://www.netflix.com/ca/title/80190844


### Google Coral

* https://coral.withgoogle.com/
* https://aiyprojects.withgoogle.com/edge-tpu


### Tensorflow.js References

* Youtube: Coding Train
https://www.youtube.com/playlist?list=PLRqwX-V7Uu6YIeVA3dNxbR9PYj4wV31oQ

* Youtube: DeepLizard
https://www.youtube.com/playlist?list=PLZbbT5o_s2xr83l8w44N_g3pygvajLrJ-
