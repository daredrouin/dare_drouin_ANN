import numpy as np

class Activation_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_LeakyReLU:
    
    def __init__(self, alpha = 0.01):
        self.alpha = alpha

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)
   
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] *= self.alpha

class Activation_Sigmoid:

    def forward(self, inputs):
        self.output = np.clip(1 / (1 + np.exp(-inputs)), 1e-7, 1 - 1e-7)

    def backward(self, dvalues):
        sigm = self.output
        deriv = sigm * (1 / sigm)
        self.dinputs = deriv * dvalues

class Activation_Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        jacobian_matrices = (
            np.einsum('ij,jk->ijk', self.output, np.eye(self.output.shape[1]))
            - np.einsum('ij,ik->ijk', self.output, self.output)
        )
        self.dinputs = np.einsum('ijk,ik->ij', jacobian_matrices, dvalues)
