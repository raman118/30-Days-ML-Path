import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.costs = []

    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """Train the logistic regression model"""
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            z = X.dot(self.weights) + self.bias
            predictions = self.sigmoid(z)

            # Compute cost (cross-entropy loss)
            cost = self.compute_cost(y, predictions)
            self.costs.append(cost)

            # Compute gradients
            dw = (1 / n_samples) * X.T.dot(predictions - y)
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def compute_cost(self, y_true, y_pred):
        """Compute cross-entropy cost function"""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost

    def predict_proba(self, X):
        """Predict class probabilities"""
        z = X.dot(self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)


# Generate sample data
print("Generating sample dataset...")
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1, random_state=42)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train and test
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Train the model
print("\nTraining logistic regression model...")
model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate accuracy
train_accuracy = np.mean(y_train_pred == y_train)
test_accuracy = np.mean(y_test_pred == y_test)

print(f"\nResults:")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Final weights: {model.weights}")
print(f"Final bias: {model.bias:.4f}")

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Cost function
axes[0].plot(model.costs)
axes[0].set_title('Cost Function Over Iterations')
axes[0].set_xlabel('Iterations')
axes[0].set_ylabel('Cost')
axes[0].grid(True)


# Plot 2: Decision boundary
def plot_decision_boundary(X, y, model, ax):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(mesh_points)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap=plt.cm.RdYlBu)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    ax.set_title('Decision Boundary')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')


plot_decision_boundary(X_test, y_test, model, axes[1])
plt.tight_layout()
plt.show()

# Test with manual example
print("\nManual Test Example:")
print("Testing with sample point [1.0, -0.5]:")
test_point = np.array([[1.0, -0.5]])
prob = model.predict_proba(test_point)[0]
pred = model.predict(test_point)[0]
print(f"Probability: {prob:.4f}")
print(f"Prediction: {pred} ({'Class 1' if pred == 1 else 'Class 0'})")