import numpy as np

from engine import Value, Tensor


class Module:
    """
    Base class for every layer.
    """
    def forward(self, *args, **kwargs):
        """Depends on functionality"""
        pass

    def __call__(self, *args, **kwargs):
        """For convenience we can use model(inp) to call forward pass"""
        return self.forward(*args, **kwargs)

    def parameters(self):
        """Return list of trainable parameters"""
        return []


class Linear(Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        """Initializing model"""
        # Create Linear Module
        stdv = 1 / np.sqrt(in_features)
        self.w = Tensor([Value(np.random.uniform(-stdv, stdv, size=(in_features,out_features)))])
        self.b = Value(0)
        self.bias = bias

    def forward(self, inp):
        """Y = W * x + b"""
        return self.w.dot(inp) + self.b

    def parameters(self):
        return self.w.parameters + [self.b]


class ReLU(Module):
    """The most simple and popular activation function"""
    def forward(self, inp):
        # Create ReLU Module
        return inp.relu()


class CrossEntropyLoss(Module):
    """Cross-entropy loss for multi-class classification"""
    def forward(self, inp, label):
        # Create CrossEntropy Loss Module
        return -label.dot(np.log(inp))
