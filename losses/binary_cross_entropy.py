import numpy as np
import sys

class Binary_cross_entropy:
    def forward(self, y, y_hat):
        self.y = y
        self.y_hat = np.clip(y_hat, sys.float_info.epsilon, (1 - sys.float_info.epsilon)) # to prevent log(y_hat) when y_hat=0
        loss = -np.mean((self.y * np.log(self.y_hat)) + ((1 - self.y) * np.log(1 - self.y_hat)))
        return loss

    def backward(self):
        return -(self.y / self.y_hat - (1 - self.y) / (1 - self.y_hat))
    

# __local test__  note: i used llm for the input and expected numbers  
if __name__ == "__main__":
    loss_fn = Binary_cross_entropy()

    y = np.array([[1], [0], [1]])
    y_hat = np.array([[0.9], [0.2], [0.7]])

    loss = loss_fn.forward(y, y_hat)
    grad = loss_fn.backward()

    print("Loss:", loss)
    print("expected: 0.228")
    print("Gradient:", grad)
    print("expected: [[-0.033], [0.066], [-0.100]]")