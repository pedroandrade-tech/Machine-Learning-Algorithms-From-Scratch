# Logistic Regression - Vectorized Implementation

A from-scratch implementation of Logistic Regression using NumPy for binary classification on the Framingham Heart Study dataset.

## Overview

This notebook implements the core logistic regression algorithm manually, focusing on understanding the mathematical foundations of the algorithm rather than using pre-built ML libraries.

## Dataset

**Framingham Heart Study** - Cardiovascular disease prediction dataset
- 4,238 samples with 15 features (demographics, risk factors, health metrics)
- Binary classification: 10-year CHD risk prediction
- Train/Test Split: 80/20 (3,390 / 848 samples)

## Implementation Details

### ✅ Implemented From Scratch

- **Sigmoid function**: Logistic activation function
- **Cost function**: Binary cross-entropy (log loss)
- **Gradient computation**: Fully vectorized gradient calculation
- **Gradient descent**: Iterative optimization algorithm
- **Prediction function**: Binary classification with configurable threshold

### ⚠️ External Dependencies (Non-Algorithm)

The following sklearn utilities are used for **data preprocessing only**:
- `train_test_split`: Dataset splitting
- `SimpleImputer`: Missing value imputation
- `StandardScaler`: Feature normalization
- `confusion_matrix`, `classification_report`: Evaluation metrics

**Note**: These are infrastructure utilities and do not affect the core algorithm implementation, which is 100% from scratch.

## Mathematical Foundation

### Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))
```

### Cost Function (Binary Cross-Entropy)
```
J(w,b) = -1/m * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
```

### Gradient Descent Update
```
w = w - α * ∂J/∂w
b = b - α * ∂J/∂b
```

Where:
- `∂J/∂w = 1/m * X^T * (h(X) - y)` (vectorized)
- `∂J/∂b = 1/m * Σ(h(x) - y)`

## Results

### Model Performance

- **Test Accuracy**: 63.68%
- **Precision (Class 1)**: 0.24
- **Recall (Class 1)**: 0.66
- **F1-Score (Class 1)**: 0.35

### Analysis

The model shows high recall but low precision for the positive class (CHD risk). This is expected given:
1. Highly imbalanced dataset (~15% positive cases)
2. Medical prediction task where missing true positives is costly

The relatively low accuracy reflects the challenge of predicting cardiovascular risk from limited features and class imbalance. 

## Key Features

- **Fully vectorized operations** using NumPy for computational efficiency
- **Epsilon smoothing** (1e-5) to prevent log(0) errors
- **Modular design** with separate functions for cost, gradient, and prediction
- **Training history tracking** for convergence analysis

## Usage

```python
# Initialize parameters
initial_w = np.zeros(n_features)
initial_b = 0.0

# Train model
w, b, J_history, w_history = gradient_descent(
    X_train_scaled,
    y_train,
    initial_w,
    initial_b,
    compute_cost,
    compute_gradient,
    alpha=0.001,        # learning rate
    num_iters=1000,    # iterations
    lambda_=0           # regularization
)

# Make predictions
predictions = predict(X_test_scaled, w, b, threshold=0.5)

# Evaluate
accuracy = np.mean(predictions == y_test) * 100
```

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Educational Purpose

This implementation prioritizes **clarity and understanding** over production-ready performance. It demonstrates:
- How logistic regression works mathematically
- The power of vectorization in machine learning
- Gradient descent optimization from first principles
- Binary classification fundamentals


## License

Educational project - Free to use and modify.
