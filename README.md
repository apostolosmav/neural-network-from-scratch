# Simple Neural Network from Scratch (Python + NumPy)

This project implements a simple feedforward neural network with one hidden layer **from scratch** using only NumPy and Pandas.  
It supports training via backpropagation and making predictions without relying on any deep learning libraries.

---

## Project Goal

The goal of this project is to provide a minimal yet complete implementation of a neural network to help understand the mathematics and logic behind **forward propagation**, **error calculation**, and **backpropagation**.

---

## Features

- **Customizable architecture**: Define number of input, hidden, and output nodes.
- **One hidden layer** network with sigmoid activation.
- **Weight initialization** with random values between -0.5 and 0.5.
- **Backpropagation** implemented manually.
- **One-hot encoded targets** for classification tasks.
- Works with Pandas DataFrames for training and testing.

---

## Requirements

Install Python 3.x and the following packages:

```bash
pip install numpy pandas
```

---

## Usage

### 1. Import and Initialize

```python
from neuronal_netz import NeuronalNetz  # or the filename you save it as

# Example: 64 input nodes, 100 hidden nodes, 10 output nodes, learning rate = 0.1
nn = NeuronalNetz(input_nodes=64, hidden_nodes=100, output_nodes=10, learning_rate=0.1)
```

### 2. Train the Model

```python
nn.fit(X_train, y_train, epochs=500)
```
- `X_train`: Pandas DataFrame of training features.
- `y_train`: Pandas DataFrame or Series of integer class labels.
- `epochs`: Number of training passes through the dataset.

### 3. Make Predictions

```python
predictions = nn.predict(X_test)
print(predictions)
```

---

## How It Works

1. **Initialization**  
   - Randomly initializes weights for input-to-hidden and hidden-to-output connections.

2. **Forward Propagation**  
   - Uses the sigmoid activation function to calculate outputs from the hidden and output layers.

3. **Error Calculation**  
   - Computes the difference between the predicted output and the target values.

4. **Backpropagation**  
   - Updates weights based on the calculated error and learning rate.

---

## Example Workflow

```python
import pandas as pd
import numpy as np

# Generate dummy dataset
X_train = pd.DataFrame(np.random.rand(100, 64))
y_train = pd.Series(np.random.randint(0, 10, size=100))

X_test = pd.DataFrame(np.random.rand(10, 64))

# Initialize and train network
nn = NeuronalNetz(64, 100, 10, 0.1)
nn.fit(X_train, y_train, epochs=500)

# Predict
print("Predictions:", nn.predict(X_test))
```

---

## License

This project is licensed under the MIT License.

---

## Author

Developed as a learning exercise to better understand neural networks without high-level libraries.
