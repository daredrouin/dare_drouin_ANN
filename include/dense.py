import numpy as np

class LayerDense():

    def __init__(self, ninputs, nneurons, initialization = "he"):
        if initialization == "he":
            self.weights = np.random.randn(ninputs, nneurons) * np.sqrt(2 / ninputs)
        elif initialization == "xavier":
            limit = np.sqrt(6 / (ninputs + nneurons))
            self.weights = np.random.randn(-limit, limit, (ninputs, nneurons))
        else:
            self.weights = np.random.randn(ninputs, nneurons) * 0.01
        
        self.biases = np.zeros((1, nneurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
 
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims = True)
        self.dinputs = np.dot(dvalues, self.weights.T)