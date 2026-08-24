# MNIST Digit Classifier — Built From Scratch with NumPy

A 2-layer neural network that classifies handwritten digits (0–9) from the classic MNIST dataset, implemented **from scratch using only NumPy and Pandas** — no TensorFlow, no PyTorch, no `sklearn` for the model itself. Every part of forward propagation, backpropagation, and gradient descent is hand-derived and hand-coded.

This project was built to genuinely understand what's happening *inside* a neural network — the matrix math, the gradients, the update rule — rather than treating it as a black box behind a `model.fit()` call.

## Why this project

Frameworks like PyTorch and TensorFlow abstract away the calculus. The goal here was the opposite: derive the forward and backward pass equations by hand (see the math notes below) and translate them directly into NumPy matrix operations, so the mechanics of how a neural network actually learns are fully transparent.

## Network Architecture

A simple fully-connected feedforward network:

```
Input Layer      Hidden Layer       Output Layer
  784 units   →     10 units    →     10 units
 (28x28 img)      (ReLU)            (Softmax)
```

- **Input**: Each 28×28 grayscale digit image is flattened into a 784-length vector and normalized to `[0, 1]`.
- **Hidden layer (Layer 1)**: 10 neurons, `ReLU` activation.
- **Output layer (Layer 2)**: 10 neurons (one per digit, 0–9), `Softmax` activation, producing a probability distribution over the 10 classes.

## Math

The full forward propagation, backpropagation, and parameter update derivations are worked out by hand and included as `math_notes.png` in this repo.

**Forward propagation:**
```
Z[1] = W[1]·X + b[1]
A[1] = ReLU(Z[1])
Z[2] = W[2]·A[1] + b[2]
A[2] = Softmax(Z[2])
```

**Backward propagation:**
```
dZ[2] = A[2] − Y                     (Y = one-hot encoded label)
dW[2] = (1/m) · dZ[2]·A[1]ᵀ
db[2] = (1/m) · Σ dZ[2]
dZ[1] = W[2]ᵀ·dZ[2] ⊙ ReLU'(Z[1])
dW[1] = (1/m) · dZ[1]·Xᵀ
db[1] = (1/m) · Σ dZ[1]
```

**Parameter update (gradient descent):**
```
W[1] := W[1] − α·dW[1]        b[1] := b[1] − α·db[1]
W[2] := W[2] − α·dW[2]        b[2] := b[2] − α·db[2]
```
where `α` is the learning rate.

## Dataset

- **Source**: [Digit Recognizer (MNIST)](https://www.kaggle.com/competitions/digit-recognizer/data) — `train.csv`, 42,000 labeled 28×28 grayscale images of handwritten digits.
- **Split**: 1,000 images held out as a dev/validation set; the remaining ~41,000 used for training.
- **Preprocessing**: Pixel values normalized from `[0, 255]` to `[0, 1]` by dividing by 255.

> **Note:** `train.csv` is not included in this repo (see [Setup](#setup) below to download it).

## Results

Trained for 500 iterations of batch gradient descent with a learning rate of `0.10`:

| Iteration | Training Accuracy |
|-----------|-------------------|
| 0         | ~10.2%            |
| 100       | ~64.7%            |
| 300       | ~81.6%            |
| 490       | ~84.6%            |

**Final dev set accuracy: ~84.2%**

Not state-of-the-art (a CNN would do far better), but a strong result for a plain 2-layer network with only 10 hidden units, trained with vanilla batch gradient descent and no regularization, momentum, or learning rate scheduling.

## Project Structure

```
.
├── MNIST_from_scratch.ipynb   # Main notebook: data loading, model, training, evaluation
├── math_notes.png             # Hand-derived forward/backprop math
├── README.md
└── requirements.txt
```

## Setup

This notebook was developed locally in **Jupyter Notebook** (not Kaggle/Colab), so the data path needs to point to wherever you place the dataset on your machine.

1. Clone the repo:
   ```bash
   git clone https://github.com/FarhanAlam2001/mnist-neural-network-from-scratch.git
   cd mnist-neural-network-from-scratch
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download `train.csv` from the [Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer/data) and place it in a `data/` folder (or update the path in the notebook to match where you save it).

4. Update the file path in the notebook's data-loading cell to point to your local copy of `train.csv`.

5. Launch Jupyter Notebook and run all cells:
   ```bash
   jupyter notebook
   ```

## Requirements

```
numpy
pandas
matplotlib
```

## Key Implementation Details

- **Weight initialization**: Weights and biases initialized as random values in `[-0.5, 0.5]`.
- **Activation functions**: `ReLU` for the hidden layer, `Softmax` for the output layer.
- **Loss**: Implicit cross-entropy via the `dZ[2] = A[2] - Y` gradient (the standard softmax + cross-entropy combined gradient).
- **Labels**: One-hot encoded on the fly inside `backward_prop`.
- **Optimization**: Vanilla batch gradient descent (no momentum, no Adam, no mini-batching — the full training set is used every iteration).
- **Evaluation**: Accuracy is computed by comparing `argmax` predictions against true labels, both on training data (printed every 10 iterations) and on the held-out dev set.

## Suggested Repo Name

**`mnist-neural-network-from-scratch`**

Other good alternatives:
- `numpy-mnist-classifier`
- `mnist-nn-from-scratch`
- `handwritten-digit-classifier-numpy`

## Possible Improvements

- Add a hidden layer or more hidden units to raise accuracy
- Implement mini-batch gradient descent
- Add L2 regularization to reduce overfitting
- Try momentum or Adam optimization
- Track and plot loss/accuracy curves over training
- Add a confusion matrix for per-digit error analysis

## License

MIT
