# handwritingAI
A neural network to classify handwritten numbers. The code
is based off of [this book by Michael Nielsen](http://neuralnetworksanddeeplearning.com/chap1.html).

# Setup environment
```
python -m venv .
source venv/bin/activate
pip install -r requirements.txt
```

# Data
The data are a part of the mnist module.

All data is taken from the MNIST dataset curated by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges from [this website](http://yann.lecun.com/exdb/mnist/).

# Run
Running `python handwritingAI.py --train` will train the AI with the default number of layers (1 hidden layer with 10 nodes), a learning rate of 3 (eta = 3), and a mini-batch size of 10. After the training is finished, the AI's information will be saved in `configs/config.conf`.

To use this default config file to predict against the test data, run `python handwritingAI.py --predict`.

To do all of this all at once, run `python handwritingAI.py --trian --predict`.

For a full listing of all options, run `python handwritingAI.py -h`.

# Viewing the Images
The program has support for displaying the images.

To display the first 100 images with matplotlib of the training dataset, run `python handwritingAI.py -n 100 --train -d`. 

To save that figure as an image, run `python handwritingAI.py -n 100 --train -d --save-image image.png`.

![predicted values](https://github.com/K20shores/handwritingAI/blob/master/images/train_100.png "Training Images")

Similarly, to display first 100 images with matplotlib of the test dataset, run `python handwritingAI.py -n 100 --train -d`. 

You can tell the program to color the images according to the how a particular AI configuration predicts the values, run `python handwritingAI.py -c <config> -n 100 --train -d -C`.

![predicted values](https://github.com/K20shores/handwritingAI/blob/master/images/predict_100.png"Prediction results")

# Resources used to make thie AI.
* Inspired by [this blog](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6).
* Program design followed from reading this online textbook [http://neuralnetworksanddeeplearning.com/chap1.html](http://neuralnetworksanddeeplearning.com/chap1.html).
