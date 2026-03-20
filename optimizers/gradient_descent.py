class Gradient_descent:
    def __init__(self, learning_rate = 0.01):
        self.lr = learning_rate

    def step(self, layers):
        for layer in layers:
            layer.w -= self.lr * layer.dw
            layer.b -= self.lr * layer.db

# __local test__  note: i used llm for the input and expected numbers
if __name__ =="__main__":
    import numpy as np
    import sys
    import os

    root = os.path.dirname(os.path.dirname(__file__)) # get the project root directory
    sys.path.append(root) # add the project root directory for python to import the modules

    from layers.dense import Dense
    from activations.sigmoid import Sigmoid

    layer = Dense(2, 1, Sigmoid())

    layer.w = np.array([[1.0, -1.0]])
    layer.b = np.array([[0.0]])

    layer.dw = np.array([[0.1, 0.1]])
    layer.db = np.array([[0.2]])

    opt = Gradient_descent(learning_rate=0.1)
    opt.step([layer])

    print(f"updated weight: {layer.w}")
    print("expected output: [[ 0.99 -1.01]]")
    print(f"updated bios: {layer.b}")
    print("expected output: [[-0.02]]")