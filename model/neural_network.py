class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_func = None
        self.optimizer = None
    
    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_func = loss
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def fit(self, x, y, epochs = 1000, verbose = 100):
        if self.optimizer is None or self.loss_func is None:
            raise ValueError("please compwile with .compile() before you fit your model to your data")
        
        for epoch in range(epochs):
            y_hat = self.forward(x)
            loss = self.loss_func.forward(y, y_hat)
            d_loss = self.loss_func.backward()
            self.backward(d_loss)
            self.optimizer.step(self.layers)
            if epoch % verbose == 0:
                print(f"epoch {epoch}, loss = {loss:.4f}")
    
    def predict(self, x):
        return self.forward(x)