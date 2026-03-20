#import numpy for vectorized implementation
import numpy as np

class Sigmoid:
    def forward(self, z):
        self.a = 1 / (1+ np.exp(-z))
        return self.a

    def backward(self, da):
        dz = da * (self.a * (1 - self.a))
        return dz

# __local test__  note: i used llm for the input and expected numbers
if __name__ == "__main__":
    z = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    sig = Sigmoid()

    # __test forward
    a = sig.forward(z)
    print(f"forward results: {a}")
    print("the expected output : (0.1192 0.2689 0.5 0.7311 0.8808)")
    # __test backward

    dz = sig.backward(np.ones_like(z))
    print(f"backward results: {dz}")
    print("the expected output: (0.1049 0.1966 0.25 0.1966 0.1049)")