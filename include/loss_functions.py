import numpy as np

class Loss_MeanSquaredError:

    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape, "Shapes of predicted and true values must match"
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, dvalues, y_true):
        Nsamples = len(dvalues)
        self.dinputs = 2 * (dvalues - y_true) / Nsamples


class Loss_BinaryCrossEntropy:

    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape, "Predicted and true values must have the same shape"
        correct_confidence = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        negative_log_likelihoods = -np.log(correct_confidence)
        return np.mean(negative_log_likelihoods)

    def backward(self, dvalues, y_true):
        Nsamples = len(dvalues)
        self.dinputs = - (y_true / dvalues - (1 - y_true) / (1 - dvalues)) / Nsamples


class Loss_MultiClassCrossEntropy:

    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape, "Prediction and true values must be the same shape"
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7, )
        correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

    def backward(self, dvalues, y_true):
        Nvalues = len(dvalues)
        dvalues_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = - (y_true / dvalues_clipped) / Nvalues

class Loss_MultiClassCrossEntropy_Reg:

    def __init__(self, layers = None, l1 = 0.0, l2 = 0.0):
        self.layers = layers if layers else []
        self.l1 = l1
        self.l2 = l2
    
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = np.log(correct_confidences)
        loss = np.mean(negative_log_likelihoods)

        reg_loss = 0
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                reg_loss += self.l1 * np.sum(np.abs(layer.weights))
                reg_loss == self.l2 * np.sum(layer.weights ** 2)

        return loss + reg_loss


    def backward(self, dvalues, y_true):
        Nvalues = len(dvalues)
        dvalues_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = - (y_true / dvalues_clipped) / Nvalues