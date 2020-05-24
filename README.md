# handwritingAI
Self-made Neural Network to classify handwritten numbers

# Setup environment
`python3 -m venv .`
`pip install -r requirements.txt`

# Download the data
The training data can be found (here)[http://yann.lecun.com/exdb/mnist/].

## Training data
`wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz`
`wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz`
`gzip -d *.gz`

## Test data
`wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz`
`wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz`
`gzip -d *.gz`

# Run
Modify `make_graph.py` and then run `python3 handwritingAI.py`

# Resources used to make thie AI.
* Inspired by (this blog)[https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6].
* Program design followed from reading this online textbook (http://neuralnetworksanddeeplearning.com/chap1.html)[http://neuralnetworksanddeeplearning.com/chap1.html].
