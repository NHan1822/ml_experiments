import numpy as np

class Module:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def parameters(self):
        return []

    def zero_grad(self):
        for p, g in self.parameters():
            g.fill(0.0)


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        self.dW += self.x.T @ grad_output
        self.db += grad_output.sum(axis=0)
        return grad_output @ self.W.T

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU(Module):
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask


class Sigmoid(Module):
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, grad_output):
        return grad_output * self.out * (1 - self.out)


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, grad_output):
        for l in reversed(self.layers):
            grad_output = l.backward(grad_output)
        return grad_output

    def parameters(self):
        params = []
        for l in self.layers:
            params += l.parameters()
        return params

    def zero_grad(self):
        for l in self.layers:
            l.zero_grad()

class CrossEntropyLoss(Module):
    def forward(self, logits, y):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probs = exp / exp.sum(axis=1, keepdims=True)
        self.y = y
        N = logits.shape[0]
        loss = -np.log(self.probs[np.arange(N), y] + 1e-9).mean()
        return loss

    def backward(self):
        N = self.y.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.y] -= 1
        grad /= N
        return grad

class SGD:
    def __init__(self, params, lr=1e-2):
        self.params = params
        self.lr = lr

    def step(self):
        for p, g in self.params:
            p -= self.lr * g

    def zero_grad(self):
        for _, g in self.params:
            g.fill(0.0)
