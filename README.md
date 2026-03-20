# Neural Network From Scratch

A fully functional neural network implemented using **only NumPy** — no PyTorch, no TensorFlow, no Keras. Every component including forward propagation, backpropagation, and gradient descent is written from scratch.

Tested on the **Wisconsin Breast Cancer dataset**, achieving **96.49% test accuracy** and **0.986 recall**.

---

## Results

| Metric | Value |
|---|---|
| Train Accuracy | 99.56% |
| Test Accuracy | 96.49% |
| Precision | 0.959 |
| Recall | 0.986 |
| F1-Score | 0.973 |

```
Confusion Matrix (Test Set):

                Predicted
                Neg    Pos
Actual  Neg      39      3
        Pos       1     71
```

---

## Project Structure

```
neural_network_from_scratch/
│
├── activations/
│   ├── sigmoid.py          # Sigmoid activation (output layer)
│   └── relu.py             # ReLU activation (hidden layers)
│
├── layers/
│   └── dense.py            # Fully connected layer + Kaiming initialization
│
├── losses/
│   └── binary_cross_entropy.py   # BCE loss + gradient
│
├── model/
│   └── neural_network.py   # Training loop orchestrator
│
├── optimizers/
│   └── gradient_descent.py # Vanilla gradient descent
│
└── breast_cancer.ipynb     # End-to-end demo on real dataset
```

---

## How It Works

### Kaiming (He) Initialization
Before training begins, weights need to start at sensible values. Weights are initialized from a normal distribution scaled by `√(2/n_in)`. That keep the activation variance stable as signals travel through layers — preventing vanishing or exploding gradients from the very first forward pass.

### Forward Pass
The input is passed through each layer sequentially. In every Dense layer, two operations happen: a linear transformation `z = x·wᵀ + b`, followed by a non-linear activation function applied to `z`. The result becomes the input to the next layer, and this repeats until the final layer produces a prediction.

### Loss Computation
The prediction is compared to the true label using a loss function, which returns a single number measuring how wrong the network is. The lower the loss, the better the predictions.

### Backward Pass (Backpropagation)
The loss gradient flows backward through every layer using the chain rule. Each layer receives `dL/da` from the layer ahead of it, computes the gradients for its own weights (`dw`) and biases (`db`), then passes `dL/dx` back to the layer behind it.

### Parameter Update
Once every layer has its gradients, the optimizer steps through each layer and updates its parameters:
```
w = w - lr * dw
b = b - lr * db
```
This repeats for every epoch until the loss converges.

---

## Model Interface

### `NeuralNetwork`
The main class that ties all components together.

```python
model = NeuralNetwork()
```

---

#### `model.add(layer)`
Adds a layer to the network. Layers are stacked in the order they are added.

| Parameter | Type | Description |
|---|---|---|
| `layer` | `Dense` | A layer instance to append to the network |

```python
model.add(Dense(input_dim, output_dim, activation))
```

---

#### `model.compile(optimizer, loss)`
Attaches an optimizer and a loss function to the network. Must be called before `fit()`.

| Parameter | Type | Description |
|---|---|---|
| `optimizer` | `Gradient_descent` | The optimization algorithm |
| `loss` | `Binary_cross_entropy` | The loss function |

```python
model.compile(
    optimizer=Gradient_descent(learning_rate=0.5),
    loss=Binary_cross_entropy()
)
```

---

#### `model.fit(x, y, epochs, verbose)`
Runs the training loop — forward pass, loss, backprop, and parameter update — for the given number of epochs.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x` | `np.ndarray` | required | Training features, shape `(m, n_features)` |
| `y` | `np.ndarray` | required | True labels, shape `(m, 1)` |
| `epochs` | `int` | `1000` | Number of full passes over the training data |
| `verbose` | `int` | `100` | Print loss every `verbose` epochs |

```python
model.fit(X_train, y_train, epochs=500, verbose=50)
```

---

#### `model.predict(x)`
Runs a forward pass and returns the raw output of the final layer.

| Parameter | Type | Description |
|---|---|---|
| `x` | `np.ndarray` | Input data, shape `(m, n_features)` |

**Returns:** `np.ndarray` of shape `(m, 1)` — probabilities for binary classification when using a Sigmoid output layer.

```python
y_pred = model.predict(X_test)

# convert probabilities to binary labels
y_binary = (y_pred >= 0.5).astype(int)
```

---

### `Dense(input_dim, output_dim, activation)`
A fully connected layer. The only layer type currently supported.

| Parameter | Type | Description |
|---|---|---|
| `input_dim` | `int` | Number of features coming into this layer |
| `output_dim` | `int` | Number of neurons in this layer |
| `activation` | `Sigmoid` or `ReLU` | Activation function applied after the linear step |

```python
Dense(30, 16, ReLU())    # hidden layer
Dense(16, 1, Sigmoid())  # output layer for binary classification
```

---

### Activations

| Class | Use Case | Import |
|---|---|---|
| `ReLU` | Hidden layers | `from activations.relu import ReLU` |
| `Sigmoid` | Binary classification output | `from activations.sigmoid import Sigmoid` |

---

### Losses

| Class | Use Case | Import |
|---|---|---|
| `Binary_cross_entropy` | Binary classification | `from losses.binary_cross_entropy import Binary_cross_entropy` |

---

### Optimizers

| Class | Parameter | Default | Import |
|---|---|---|---|
| `Gradient_descent` | `learning_rate` | `0.01` | `from optimizers.gradient_descent import Gradient_descent` |

---

## Requirements

```
numpy
scikit-learn   # only used for dataset loading, splitting, and scaling
```

---

## Future Work

- **Activations**: Tanh, Leaky ReLU, ELU, Softmax (for multiclass)
- **Losses**: Mean Squared Error, Categorical Cross-Entropy
- **Optimizers**: Momentum, RMSProp, Adam

---

## Author

**Moaaz Ahmed**
- GitHub: [@Moaaz Ahmed](https://github.com/MomoSalter)