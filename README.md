# Artificial Neural Networks

This repository was created to store examples of artifical neural networks, both from scratch, and using popular libraries like tensor flow and PyTorch.

## 1. Dense Layer
A custom Dense Layer is implemented in `include/dense.py`.

This class contains three methods: `__init__()`, `forward()`, and `backward()`. The initialization function defines random values for the weights and biases based on the input data, number of neurons, and selected type of initialization, which will be elaborated on in the next section.

### 1.1 Types of Initialization
Two types of initialization are provided for the dense layer: `"he"` and `"xavier"`. 

`"he"` initialization (a.k.a Kaiming intitialization) initializaes the weights as a guassian distribution with a mean of 0 and a variance of $\frac{2}{ninputs}$. This mitigates vanishing or exploding gradients.

`"xavier"` initialization is designed to keep variance for activations and gradients roughly constant across layers. It does this by setting the weights to a distribution where the mean is 0 and the variance is $\frac{6}{ninputs + noutputs}$.