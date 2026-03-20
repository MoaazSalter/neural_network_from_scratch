import numpy as np

class Dense:
    def __init__(self, input_dimension, output_dimension, activation):
        # Kaiming Initialization
        self.w = np.random.randn(output_dimension, input_dimension) * np.sqrt(2 / input_dimension)
        self.b = np.zeros((1, output_dimension))
        self.activation = activation

    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.w.T) + self.b
        self.a = self.activation.forward(self.z)
        return self.a
    
    def backward(self, da):
        dz = self.activation.backward(da)
        m = self.x.shape[0]
        self.dw = (1/m) * np.dot(dz.T, self.x)
        self.db = (1/m) * np.sum(dz, axis=0, keepdims=True)
        dx = np.dot(dz, self.w)
        return dx