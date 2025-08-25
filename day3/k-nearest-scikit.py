# k-Nearest Neighbors with Scikit-Learn: Clean & Professional Implementation
# Author: Your Name
# Date: 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("=" * 50)
print("k-NEAREST NEIGHBORS WITH SCIKIT-LEARN")
print("=" * 50)

# ================================
# 1. BASIC CLASSIFICATION EXAMPLE
# ================================
print("\n1. BASIC CLASSIFICATION EXAMPLE")
print("-" * 30)

# Load the famous Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

print(f"Dataset shape: {X.shape}")
print(f"Classes: {iris.target_names}")
print(f"Features: {iris.feature_names}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Create and train k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nResults with k=5:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Predictions: {y_pred}")
print(f"Actual:      {y_test}")

# ================================
# 2. FINDING OPTIMAL k
# ================================
print("\n\n2. FINDING OPTIMAL k")
print("-" * 30)

# Test different k values
k_range = range(1, 21)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

# Find optimal k
optimal_k = k_range[np.argmax(k_scores)]
best_score = max(k_scores)

print(f"Optimal k: {optimal_k}")
print(f"Best CV score: {best_score:.4f}")

# Plot k vs accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7,
            label=f'Optimal k = {optimal_k}')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Finding Optimal k for k-NN')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# ================================
# 3. FEATURE SCALING IMPACT
# ================================
print("\n3. FEATURE SCALING IMPACT")
print("-" * 30)

# Create dataset with different scales
np.random.seed(42)
n_samples = 300
X_small = np.random.normal(0, 1, (n_samples, 1))  # Feature 1: scale 0-5
X_large = np.random.normal(0, 100, (n_samples, 1))  # Feature 2: scale 0-500
X_mixed = np.hstack([X_small, X_large])
y_mixed = (X_small.ravel() + X_large.ravel() / 100 > 0).astype(int)

X_train_mix, X_test_mix, y_train_mix, y_test_mix = train_test_split(
    X_mixed, y_mixed, test_size=0.3, random_state=42
)

# Without scaling
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train_mix, y_train_mix)
accuracy_unscaled = knn_unscaled.score(X_test_mix, y_test_mix)

# With scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_mix)
X_test_scaled = scaler.transform(X_test_mix)

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train_mix)
accuracy_scaled = knn_scaled.score(X_test_scaled, y_test_mix)

print(f"Accuracy without scaling: {accuracy_unscaled:.4f}")
print(f"Accuracy with scaling:    {accuracy_scaled:.4f}")
print(f"Improvement:              {accuracy_scaled - accuracy_unscaled:.4f}")

# Visualize the difference
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train_mix[:, 0], X_train_mix[:, 1], c=y_train_mix, alpha=0.7)
plt.xlabel('Feature 1 (scale: ~1)')
plt.ylabel('Feature 2 (scale: ~100)')
plt.title('Original Data (Different Scales)')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train_mix, alpha=0.7)
plt.xlabel('Feature 1 (standardized)')
plt.ylabel('Feature 2 (standardized)')
plt.title('After Standard Scaling')
plt.colorbar()

plt.tight_layout()
plt.show()

# ================================
# 4. HYPERPARAMETER TUNING
# ================================
print("\n4. HYPERPARAMETER TUNING")
print("-" * 30)

# Use the scaled iris data
scaler = StandardScaler()
X_train_iris_scaled = scaler.fit_transform(X_train)
X_test_iris_scaled = scaler.transform(X_test)

# Define parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Grid search
knn = KNeighborsClassifier()
grid_search = GridSearchCV(
    knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)

print("Performing grid search...")
grid_search.fit(X_train_iris_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Test the best model
best_knn = grid_search.best_estimator_
test_accuracy = best_knn.score(X_test_iris_scaled, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# ================================
# 5. DETAILED EVALUATION
# ================================
print("\n5. DETAILED EVALUATION")
print("-" * 30)

# Get predictions
y_pred_best = best_knn.predict(X_test_iris_scaled)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=iris.target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

# ================================
# 6. REGRESSION EXAMPLE
# ================================
print("\n6. k-NN REGRESSION EXAMPLE")
print("-" * 30)

# Generate regression data
from sklearn.datasets import make_regression

X_reg, y_reg = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Train k-NN regressor
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_reg, y_train_reg)

# Predictions
y_pred_reg = knn_reg.predict(X_test_reg)
r2_score = knn_reg.score(X_test_reg, y_test_reg)

print(f"R² Score: {r2_score:.4f}")

# Visualize regression results
plt.figure(figsize=(10, 6))
plt.scatter(X_train_reg, y_train_reg, alpha=0.6, label='Training Data')
plt.scatter(X_test_reg, y_test_reg, alpha=0.6, label='Test Data')

# Create smooth prediction line
X_plot = np.linspace(X_reg.min(), X_reg.max(), 100).reshape(-1, 1)
y_plot = knn_reg.predict(X_plot)
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='k-NN Prediction')

plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('k-NN Regression Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ================================
# 7. DISTANCE METRICS COMPARISON
# ================================
print("\n7. DISTANCE METRICS COMPARISON")
print("-" * 30)

# Compare different distance metrics
metrics = ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
metric_scores = {}

for metric in metrics:
    if metric == 'minkowski':
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric, p=3)
    else:
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric)

    cv_scores = cross_val_score(knn, X_train_iris_scaled, y_train, cv=5)
    metric_scores[metric] = cv_scores.mean()

    print(f"{metric:12}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Visualize metric comparison
plt.figure(figsize=(10, 6))
plt.bar(metric_scores.keys(), metric_scores.values())
plt.title('k-NN Performance by Distance Metric')
plt.ylabel('Cross-Validation Accuracy')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.show()

# ================================
# 8. PRACTICAL TIPS
# ================================
print("\n8. PRACTICAL TIPS & BEST PRACTICES")
print("-" * 30)

tips = [
    "1. Always scale your features (StandardScaler or MinMaxScaler)",
    "2. Use cross-validation to find optimal k",
    "3. Try odd values of k to avoid ties in classification",
    "4. Consider 'distance' weighting for better results",
    "5. k-NN works best with < 50 features (curse of dimensionality)",
    "6. Remove irrelevant features to improve performance",
    "7. k-NN is slow on large datasets - consider approximation methods",
    "8. Good baseline model - simple and interpretable"
]

for tip in tips:
    print(tip)

# ================================
# 9. WHEN TO USE k-NN
# ================================
print("\n9. WHEN TO USE k-NN")
print("-" * 30)

print("✅ USE k-NN WHEN:")
use_cases = [
    "• Small to medium datasets (< 10K samples)",
    "• Non-linear decision boundaries",
    "• Mixed data types (with proper preprocessing)",
    "• Need a simple, interpretable model",
    "• Local patterns are important",
    "• Building a baseline model"
]

for case in use_cases:
    print(case)

print("\n❌ AVOID k-NN WHEN:")
avoid_cases = [
    "• Very large datasets (> 100K samples)",
    "• High-dimensional data (> 50 features)",
    "• Need real-time predictions",
    "• Limited memory/storage",
    "• Features have very different scales (without scaling)",
    "• Data is very noisy"
]

for case in avoid_cases:
    print(case)

print("\n" + "=" * 50)
print("k-NN IMPLEMENTATION COMPLETE!")
print("=" * 50)

# Summary of key parameters
print("\n📋 KEY PARAMETERS SUMMARY:")
print(f"• n_neighbors: {optimal_k} (found via cross-validation)")
print("• weights: 'distance' (usually better than 'uniform')")
print("• metric: 'euclidean' (most common, try others)")
print("• algorithm: 'auto' (let sklearn choose)")
print("• Always use StandardScaler() for preprocessing!")