import numpy as np

class ReLU:
    def forward(self, z):
        self.z = z
        return np.maximum(0, z)

    def backward(self, da):
        return da * (self.z > 0)