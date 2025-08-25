# Random Forest Implementation using Scikit-Learn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.datasets import make_classification, make_regression, load_iris, load_boston
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: CLASSIFICATION EXAMPLE
# =============================================================================

print("=" * 60)
print("RANDOM FOREST CLASSIFICATION")
print("=" * 60)

# Load sample data (Iris dataset)
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
)

# Create Random Forest Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=5,  # Maximum depth of trees
    min_samples_split=2,  # Minimum samples to split
    min_samples_leaf=1,  # Minimum samples in leaf
    random_state=42,  # For reproducibility
    n_jobs=-1  # Use all processors
)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Feature importance
print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': rf_classifier.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance)

# Cross-validation
cv_scores = cross_val_score(rf_classifier, X_iris, y_iris, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# =============================================================================
# PART 2: REGRESSION EXAMPLE
# =============================================================================

print("\n" + "=" * 60)
print("RANDOM FOREST REGRESSION")
print("=" * 60)

# Create synthetic regression data
X_reg, y_reg = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    noise=0.1,
    random_state=42
)

# Split the data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Create Random Forest Regressor
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_regressor.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_reg = rf_regressor.predict(X_test_reg)

# Evaluate the model
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Feature importance for regression
feature_names = [f'Feature_{i}' for i in range(X_reg.shape[1])]
reg_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_regressor.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nFeature Importance (Regression):")
print(reg_importance)

# =============================================================================
# PART 3: HYPERPARAMETER TUNING EXAMPLE
# =============================================================================

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create Random Forest
rf_tuning = RandomForestClassifier(random_state=42, n_jobs=-1)

# Grid search with cross-validation
grid_search = GridSearchCV(
    rf_tuning,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit grid search (using smaller dataset for speed)
print("Running Grid Search...")
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Test the best model
best_rf = grid_search.best_estimator_
best_pred = best_rf.predict(X_test)
best_accuracy = accuracy_score(y_test, best_pred)
print(f"Test accuracy with best parameters: {best_accuracy:.4f}")

# =============================================================================
# PART 4: COMPARISON WITH SINGLE DECISION TREE
# =============================================================================

print("\n" + "=" * 60)
print("RANDOM FOREST vs SINGLE DECISION TREE")
print("=" * 60)

from sklearn.tree import DecisionTreeClassifier

# Single Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

# Random Forest (simple)
rf_simple = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_simple.fit(X_train, y_train)
rf_pred = rf_simple.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"Single Decision Tree Accuracy: {dt_accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"Improvement: {rf_accuracy - dt_accuracy:.4f}")

# =============================================================================
# PART 5: OUT-OF-BAG (OOB) SCORE
# =============================================================================

print("\n" + "=" * 60)
print("OUT-OF-BAG (OOB) SCORE")
print("=" * 60)

# Random Forest with OOB score
rf_oob = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,  # Enable OOB scoring
    random_state=42
)

rf_oob.fit(X_train, y_train)
oob_accuracy = rf_oob.oob_score_

print(f"Out-of-Bag Score: {oob_accuracy:.4f}")
print("OOB score provides an unbiased estimate of model performance")
print("without needing a separate validation set.")

# =============================================================================
# PART 6: PRACTICAL TIPS
# =============================================================================

print("\n" + "=" * 60)
print("PRACTICAL TIPS FOR RANDOM FOREST")
print("=" * 60)

tips = """
1. Default Parameters:
   - n_estimators: Start with 100, increase if needed
   - max_features: 'sqrt' for classification, 'auto' for regression
   - max_depth: None (let trees grow deep), but consider pruning for speed

2. Key Hyperparameters to Tune:
   - n_estimators: More trees = better performance but slower
   - max_depth: Control overfitting
   - min_samples_split: Prevent overfitting on small datasets
   - max_features: Control randomness and overfitting

3. Performance Tips:
   - Use n_jobs=-1 for parallel processing
   - Use oob_score=True to avoid separate validation set
   - Consider warm_start=True for incremental learning

4. When to Use Random Forest:
   ✓ Mixed data types (numerical + categorical)
   ✓ Non-linear relationships
   ✓ Need feature importance
   ✓ Robust to outliers required
   ✓ Good baseline model

5. When NOT to Use:
   ✗ Very high-dimensional sparse data (use linear models)
   ✗ Simple linear relationships (use linear models)
   ✗ Need probability calibration (use calibration techniques)
   ✗ Memory/speed critical applications
"""

print(tips)


# =============================================================================
# PART 7: SIMPLE PREDICTION FUNCTION
# =============================================================================

def simple_random_forest_prediction(X_train, y_train, X_test, task='classification'):
    """
    Simple function to train Random Forest and make predictions

    Parameters:
    X_train, y_train: Training data
    X_test: Test data
    task: 'classification' or 'regression'

    Returns:
    predictions, model, feature_importance
    """

    if task == 'classification':
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Feature importance
    feature_importance = model.feature_importances_

    return predictions, model, feature_importance


# Example usage of the simple function
print("\n" + "=" * 60)
print("USING SIMPLE PREDICTION FUNCTION")
print("=" * 60)

predictions, trained_model, importance = simple_random_forest_prediction(
    X_train, y_train, X_test, task='classification'
)

simple_accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy using simple function: {simple_accuracy:.4f}")
print(f"Top 2 most important features: {np.argsort(importance)[-2:]}")