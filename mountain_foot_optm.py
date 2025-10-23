import numpy as np

# Initialize weight matrix
w = np.random.randn(10, 3072) * 0.001
best_loss = float('inf')

# Define loss function and gradient function
def L(x, y, w):
    """Compute mean squared error loss"""
    predictions = x.dot(w.T)
    return np.mean((predictions - y) ** 2)

def gradient(x, y, w):
    """Compute gradient of loss function with respect to w"""
    predictions = x.dot(w.T)  # (500, 10)
    error = predictions - y   # (500, 10)
    grad = 2 * error.T.dot(x) / x.shape[0]  # (10, 3072)
    return grad

# Generate training data
xTr_cols = np.random.randn(500, 3072)
yTr = np.random.randn(500, 10)

# Optimization parameters
learning_rate = 0.01
momentum = 0.9
velocity = np.zeros_like(w)  # Momentum term
patience = 100  # Early stopping patience
no_improvement_count = 0
loss_history = []

print("Starting optimization...")

for i in range(10000):
    # Compute gradient
    grad = gradient(xTr_cols, yTr, w)
    
    # Update with momentum
    velocity = momentum * velocity - learning_rate * grad
    w += velocity
    
    # Compute loss
    loss = L(xTr_cols, yTr, w)
    loss_history.append(loss)
    
    # Adaptive learning rate decay
    if i > 0 and i % 500 == 0:
        learning_rate *= 0.95  # Decay learning rate by 5% every 500 iterations
        print(f"Iteration {i}: Learning rate decayed to {learning_rate:.6f}")
    
    # Update best loss and check for early stopping
    if loss < best_loss:
        best_loss = loss
        no_improvement_count = 0
        best_w = w.copy()  # Save best weights
        
        if i % 100 == 0:  # Print every 100 iterations
            print(f'Iteration {i}: Loss = {loss:.6f}')
    else:
        no_improvement_count += 1
        # Early stopping if no improvement for patience iterations
        if no_improvement_count >= patience:
            print(f'Early stopping at iteration {i}: No improvement for {patience} consecutive iterations')
            w = best_w  # Restore best weights
            break

print(f'Optimization completed, final loss: {best_loss:.6f}')

# Optional: Plot loss curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.yscale('log')  # Use log scale for better visualization of convergence
plt.xlabel('Iterations')
plt.ylabel('Loss (log scale)')
plt.title('Loss Convergence Curve')
plt.grid(True)
plt.show()