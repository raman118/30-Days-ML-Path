# Logistic Regression with Scikit-Learn: Professional Implementation
# Complete workflow for binary classification using logistic regression

# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.datasets import load_breast_cancer
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("All libraries imported successfully")

# =============================================================================
# 2. LOAD AND EXPLORE DATA
# =============================================================================

# Load the breast cancer dataset (binary classification)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print("Dataset Information:")
print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"\nTarget distribution:")
print(y.value_counts())
print(f"Target names: {data.target_names}")

# Display basic statistics
print("\nDataset Statistics:")
print(X.describe())

# Check for missing values
print(f"\nMissing values: {X.isnull().sum().sum()}")

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# =============================================================================

# Create visualization subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Target distribution
y.value_counts().plot(kind='bar', ax=axes[0, 0], color=['lightcoral', 'skyblue'])
axes[0, 0].set_title('Target Class Distribution')
axes[0, 0].set_xlabel('Class (0=Malignant, 1=Benign)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].tick_params(axis='x', rotation=0)

# Feature correlation heatmap (first 10 features)
top_features = X.columns[:10]
corr_matrix = X[top_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', ax=axes[0, 1])
axes[0, 1].set_title('Feature Correlation Matrix (First 10 Features)')

# Distribution of key features by target
key_features = ['mean radius', 'mean texture']
for i, feature in enumerate(key_features):
    combined_data = pd.concat([X[feature], y], axis=1)
    sns.boxplot(data=combined_data, x='target', y=feature, ax=axes[1, i])
    axes[1, i].set_title(f'{feature.title()} by Target Class')
    axes[1, i].set_xlabel('Class (0=Malignant, 1=Benign)')

plt.tight_layout()
plt.show()

# =============================================================================
# 4. DATA PREPROCESSING
# =============================================================================

print("\nData Preprocessing:")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Training target distribution:")
print(y_train.value_counts())

# Feature scaling - crucial for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed")

# =============================================================================
# 5. MODEL TRAINING - BASIC LOGISTIC REGRESSION
# =============================================================================

print("\nTraining Logistic Regression Model:")

# Initialize and train the model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = lr_model.predict(X_train_scaled)
y_test_pred = lr_model.predict(X_test_scaled)
y_train_proba = lr_model.predict_proba(X_train_scaled)[:, 1]
y_test_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Calculate basic metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Number of coefficients: {len(lr_model.coef_[0])}")
print(f"Intercept: {lr_model.intercept_[0]:.4f}")

# =============================================================================
# 6. COMPREHENSIVE MODEL EVALUATION
# =============================================================================

print("\nComprehensive Model Evaluation:")

# Calculate all important metrics
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_proba)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

# Classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=data.target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"\nConfusion Matrix:")
print(cm)

# =============================================================================
# 7. VISUALIZATION - MODEL PERFORMANCE
# =============================================================================

# Create performance visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.target_names,
            yticklabels=data.target_names, ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_xlabel('Predicted Label')
axes[0, 0].set_ylabel('True Label')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC Curve (AUC = {roc_auc:.2f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve')
axes[0, 1].legend(loc="lower right")

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_test_proba)
axes[1, 0].plot(recall_vals, precision_vals, color='blue', lw=2)
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Precision-Recall Curve')

# Feature Importance (absolute coefficients)
feature_importance = np.abs(lr_model.coef_[0])
top_10_indices = np.argsort(feature_importance)[-10:]
top_10_features = X.columns[top_10_indices]
top_10_coefficients = feature_importance[top_10_indices]

axes[1, 1].barh(range(10), top_10_coefficients)
axes[1, 1].set_yticks(range(10))
axes[1, 1].set_yticklabels([feat.replace(' ', '\n') for feat in top_10_features])
axes[1, 1].set_xlabel('Absolute Coefficient Value')
axes[1, 1].set_title('Top 10 Most Important Features')

plt.tight_layout()
plt.show()

# =============================================================================
# 8. HYPERPARAMETER TUNING
# =============================================================================

print("\nHyperparameter Tuning:")

# Define parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Train best model
best_model = grid_search.best_estimator_
y_test_pred_best = best_model.predict(X_test_scaled)
y_test_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate best model
best_accuracy = accuracy_score(y_test, y_test_pred_best)
best_f1 = f1_score(y_test, y_test_pred_best)
best_roc_auc = roc_auc_score(y_test, y_test_proba_best)

print(f"Best model test accuracy: {best_accuracy:.4f}")
print(f"Best model F1-score: {best_f1:.4f}")
print(f"Best model ROC AUC: {best_roc_auc:.4f}")

# =============================================================================
# 9. CROSS-VALIDATION
# =============================================================================

print("\nCross-Validation Results:")

# Perform k-fold cross-validation
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")

# =============================================================================
# 10. MODEL INTERPRETATION
# =============================================================================

print("\nModel Interpretation:")

# Get feature coefficients
coefficients = best_model.coef_[0]
feature_names = X.columns

# Create coefficient DataFrame
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("Top 10 Most Important Features:")
print(coef_df.head(10)[['Feature', 'Coefficient']])

# Odds ratios
print("\nOdds Ratios for Top 10 Features:")
odds_ratios = np.exp(coef_df.head(10)['Coefficient'])
for i, (feature, odds_ratio) in enumerate(zip(coef_df.head(10)['Feature'], odds_ratios)):
    print(f"{feature}: {odds_ratio:.3f}")

# =============================================================================
# 11. PREDICTION ON NEW DATA
# =============================================================================

print("\nMaking Predictions on New Data:")

# Example: predict on first 5 test samples
sample_indices = [0, 1, 2, 3, 4]
sample_data = X_test_scaled[sample_indices]
sample_predictions = best_model.predict(sample_data)
sample_probabilities = best_model.predict_proba(sample_data)

print("Sample Predictions:")
for i, idx in enumerate(sample_indices):
    actual = y_test.iloc[idx]
    predicted = sample_predictions[i]
    prob_malignant = sample_probabilities[i][0]
    prob_benign = sample_probabilities[i][1]

    print(f"Sample {idx}: Actual={data.target_names[actual]}, "
          f"Predicted={data.target_names[predicted]}, "
          f"P(Malignant)={prob_malignant:.3f}, P(Benign)={prob_benign:.3f}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)
print(f"Final Model: {best_model}")
print(f"Test Accuracy: {best_accuracy:.4f}")
print(f"Test F1-Score: {best_f1:.4f}")
print(f"Test ROC AUC: {best_roc_auc:.4f}")
print(f"Cross-validation Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("=" * 60)