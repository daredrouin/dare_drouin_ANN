import numpy as np

class Optimizer_SGD():

    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        weight_updates = -self.learning_rate * layer.dweights
        bias_updates = -self.learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

class Optimizer_Adam():

    def __init__(self, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.iteration = 0

    def update_params(self, layer):
        if not hasattr(layer, 'm'):
            layer.m = np.zeros_like(layer.weights)
            layer.v = np.zeros_like(layer.weights)
            layer.m_b = np.zeros_like(layer.biases)
            layer.v_b = np.zeros_like(layer.biases)
        
        self.iteration += 1

        layer.m = self.beta1 * layer.m + (1 - self.beta1) * layer.dweights
        layer.v = self.beta2 * layer.v + (1 - self.beta2) * (layer.dweights ** 2)
        m_hat = layer.m / (1 - self.beta1 ** self.iteration)
        v_hat = layer.v / (1 - self.beta2 ** self.iteration)
        layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) * self.eps)

        layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * layer.dbiases
        layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * (layer.dbiases ** 2)
        m_hat_b = layer.m_b / (1 - self.beta1 ** self.iteration)
        v_hat_b = layer.v_b / (1 - self.beta2 ** self.iteration)
        layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.eps)

