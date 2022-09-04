import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

def softmax(x, axis=-1):
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)

class LogisticLoss():
    def __init__(self):
        sigmoid = Sigmoid()
        self.log_func = sigmoid
        self.log_grad = sigmoid.gradient

    def loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        p = self.log_func(y_pred)
        return y * np.log(p) + (1 - y) * np.log(1 - p)

    # gradient w.r.t y_pred
    def gradient(self, y, y_pred):
        p = self.log_func(y_pred)
        return -(y - p)

    # w.r.t y_pred
    def hess(self, y, y_pred):
        p = self.log_func(y_pred)
        return p * (1 - p)

# binary cross entropy loss ------------------------------------------------------------------------------------
class SigmoidBinaryCrossEntropyLoss:

    def __init__(self):
        pass

    @staticmethod
    def predict(value):
        return sigmoid(value)

    def compute_loss(self, y, y_pred):
        # negative averaged log loss
        log_loss = np.nan_to_num(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return -np.sum(log_loss) / len(y_pred)

    def compute_grad(self, y, y_pred):
        return y_pred - y

    def compute_hess(self, y, y_pred):
        return y_pred * (1 - y_pred)


# softmax cross entropy loss -------------------------------------------------------------------------------
class SoftmaxCrossEntropyLoss:

    def __init__(self):
        pass

    @staticmethod
    def predict(values):
        """
        :param values: ndarray
        :return: ndarray
        """
        return softmax(values)

    def compute_loss(self, y, y_pred):
        y_prob = self.predict(y_pred)
        # do summation over feature dimensions & do averaging over samples
        log_loss = np.nan_to_num(y * np.log(y_prob))
        return -np.sum(log_loss) / len(y_prob)

    def compute_grad(self, y, y_pred):
        assert len(y_pred.shape) == 2
        y_prob = self.predict(y_pred)
        return y_prob - y

    def compute_hess(self, y, y_pred):
        y_prob = self.predict(y_pred)
        return y_prob * (1 - y_prob)


class BinaryRFLoss():
    def __init__(self):
        pass

    def predict(self, x):
        return x

    def compute_grad(self, y, y_pred):
        return (np.array(y)==1).astype("int")

    def compute_hess(self, y, y_pred):
        return np.ones_like(y)

class SoftmaxLoss:
    def __init__(self):
        pass

    def predict(self, x):
        out = []
        for i,r in enumerate(x):
            e = np.exp(r)
            out.append(e / np.sum(e))
        return np.array(out)

    def compute_grad(self, y, y_pred):
        grads = []
        p = self.predict(y_pred)

        for i in range(len(y)):
            grad = np.zeros(y_pred.shape[1])
            for j in range(0, y_pred.shape[1]):
                if j == y[i]:
                    grad[j] = p[i][j] - 1
                else:
                    grad[j] = p[i][j]
            grads.append(grad)

        return np.array(grads)

    def compute_hess(self, y, y_pred):
        hess = np.zeros(len(y_pred))
        p = self.predict(y_pred)
        return p * (1- p)


class LeastSquareLoss:
    """ loss = 1/2 (y-y_hat)**2 """

    def __init__(self):
        pass

    @staticmethod
    def predict(value):
        return value

    @staticmethod
    def compute_loss(y, y_pred):
        lse_loss = 0.5 * (y - y_pred)**2
        return np.sum(lse_loss) / len(y)

    @staticmethod
    def compute_grad(y, y_pred):
        return y_pred - y

    @staticmethod
    def compute_hess(y, y_pred):
        # derivative of y_hat-y is 1
        if type(y).__name__ == 'ndarray' or type(y_pred).__name__ == 'ndarray':
            return np.ones_like(y)
        else:
            return 1
