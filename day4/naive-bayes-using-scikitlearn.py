# Naive Bayes Implementation using Scikit-Learn
# Complete examples for all three variants: Gaussian, Multinomial, and Bernoulli

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.datasets import make_classification, fetch_20newsgroups
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("NAIVE BAYES IMPLEMENTATION USING SCIKIT-LEARN")
print("=" * 60)

# =============================================================================
# 1. GAUSSIAN NAIVE BAYES - For Continuous Features
# =============================================================================

print("\n1. GAUSSIAN NAIVE BAYES EXAMPLE")
print("-" * 40)

# Generate synthetic dataset with continuous features
X_continuous, y_continuous = make_classification(
    n_samples=1000,
    n_features=4,
    n_redundant=0,
    n_informative=4,
    n_clusters_per_class=1,
    random_state=42
)

# Create feature names for better understanding
feature_names = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4']
df_continuous = pd.DataFrame(X_continuous, columns=feature_names)
df_continuous['target'] = y_continuous

print("Dataset shape:", X_continuous.shape)
print("Class distribution:")
print(pd.Series(y_continuous).value_counts())

# Split the data
X_train_gauss, X_test_gauss, y_train_gauss, y_test_gauss = train_test_split(
    X_continuous, y_continuous, test_size=0.3, random_state=42, stratify=y_continuous
)

# Initialize and train Gaussian Naive Bayes
gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train_gauss, y_train_gauss)

# Make predictions
y_pred_gauss = gaussian_nb.predict(X_test_gauss)
y_pred_proba_gauss = gaussian_nb.predict_proba(X_test_gauss)

# Evaluate performance
accuracy_gauss = accuracy_score(y_test_gauss, y_pred_gauss)
print(f"\nGaussian NB Accuracy: {accuracy_gauss:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_gauss, y_pred_gauss))

# Display some predictions with probabilities
print("\nSample Predictions (first 10):")
print("Actual | Predicted | Prob_Class_0 | Prob_Class_1")
print("-" * 50)
for i in range(10):
    print(f"{y_test_gauss[i]:6d} | {y_pred_gauss[i]:9d} | {y_pred_proba_gauss[i][0]:11.4f} | {y_pred_proba_gauss[i][1]:11.4f}")

# =============================================================================
# 2. MULTINOMIAL NAIVE BAYES - For Count/Frequency Data (Text Classification)
# =============================================================================

print("\n\n2. MULTINOMIAL NAIVE BAYES EXAMPLE - TEXT CLASSIFICATION")
print("-" * 60)

# Sample text data for demonstration
documents = [
    "I love this movie, it's amazing and fantastic",
    "This film is terrible and boring, waste of time",
    "Great acting and wonderful story, highly recommend",
    "Awful movie, poor plot and bad acting",
    "Excellent cinematography and outstanding performance",
    "Horrible film, completely disappointed",
    "Beautiful story with great characters",
    "This movie is boring and predictable",
    "Fantastic film with amazing visuals",
    "Terrible acting and weak storyline",
    "Love the plot and character development",
    "Boring movie, fell asleep watching it",
    "Outstanding performance by all actors",
    "Poor quality film, not worth watching",
    "Excellent direction and cinematography"
]

# Labels: 1 = positive, 0 = negative
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

print(f"Number of documents: {len(documents)}")
print(f"Positive reviews: {sum(labels)}")
print(f"Negative reviews: {len(labels) - sum(labels)}")

# Convert text to numerical features using CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=100)
X_text = vectorizer.fit_transform(documents)

print(f"\nFeature matrix shape: {X_text.shape}")
print("Sample feature names:", vectorizer.get_feature_names_out()[:10])

# Split the data
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_text, labels, test_size=0.3, random_state=42, stratify=labels
)

# Initialize and train Multinomial Naive Bayes
multinomial_nb = MultinomialNB(alpha=1.0)  # alpha is the smoothing parameter
multinomial_nb.fit(X_train_multi, y_train_multi)

# Make predictions
y_pred_multi = multinomial_nb.predict(X_test_multi)
y_pred_proba_multi = multinomial_nb.predict_proba(X_test_multi)

# Evaluate performance
accuracy_multi = accuracy_score(y_test_multi, y_pred_multi)
print(f"\nMultinomial NB Accuracy: {accuracy_multi:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_multi, y_pred_multi, target_names=['Negative', 'Positive']))

# Test with new documents
test_docs = [
    "This movie is absolutely amazing and wonderful",
    "Terrible film with horrible acting and boring plot"
]

test_vectors = vectorizer.transform(test_docs)
test_predictions = multinomial_nb.predict(test_vectors)
test_probabilities = multinomial_nb.predict_proba(test_vectors)

print("\nPredictions for new documents:")
for i, doc in enumerate(test_docs):
    sentiment = "Positive" if test_predictions[i] == 1 else "Negative"
    prob_neg, prob_pos = test_probabilities[i]
    print(f"\nDocument: '{doc[:50]}...'")
    print(f"Predicted: {sentiment}")
    print(f"Probabilities - Negative: {prob_neg:.4f}, Positive: {prob_pos:.4f}")

# =============================================================================
# 3. BERNOULLI NAIVE BAYES - For Binary Features
# =============================================================================

print("\n\n3. BERNOULLI NAIVE BAYES EXAMPLE - BINARY FEATURES")
print("-" * 50)

# Create binary features (presence/absence of words)
binary_vectorizer = CountVectorizer(binary=True, stop_words='english', max_features=50)
X_binary = binary_vectorizer.fit_transform(documents)

print(f"Binary feature matrix shape: {X_binary.shape}")
print("Sample of binary matrix (first document):")
print(X_binary[0].toarray())

# Split the data
X_train_bern, X_test_bern, y_train_bern, y_test_bern = train_test_split(
    X_binary, labels, test_size=0.3, random_state=42, stratify=labels
)

# Initialize and train Bernoulli Naive Bayes
bernoulli_nb = BernoulliNB(alpha=1.0)
bernoulli_nb.fit(X_train_bern, y_train_bern)

# Make predictions
y_pred_bern = bernoulli_nb.predict(X_test_bern)
y_pred_proba_bern = bernoulli_nb.predict_proba(X_test_bern)

# Evaluate performance
accuracy_bern = accuracy_score(y_test_bern, y_pred_bern)
print(f"\nBernoulli NB Accuracy: {accuracy_bern:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_bern, y_pred_bern, target_names=['Negative', 'Positive']))

# =============================================================================
# 4. COMPARISON OF ALL THREE METHODS
# =============================================================================

print("\n\n4. COMPARISON OF NAIVE BAYES VARIANTS")
print("-" * 40)

# For fair comparison, let's use the same text data with different preprocessing
print("Accuracy Comparison on Text Classification:")
print(f"Multinomial NB (Count features): {accuracy_multi:.4f}")
print(f"Bernoulli NB (Binary features):  {accuracy_bern:.4f}")

# =============================================================================
# 5. HYPERPARAMETER TUNING EXAMPLE
# =============================================================================

print("\n\n5. HYPERPARAMETER TUNING - SMOOTHING PARAMETER")
print("-" * 50)

# Test different alpha values for smoothing
alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
accuracies = []

print("Alpha Value | Accuracy")
print("-" * 20)

for alpha in alpha_values:
    nb_model = MultinomialNB(alpha=alpha)
    nb_model.fit(X_train_multi, y_train_multi)
    pred = nb_model.predict(X_test_multi)
    acc = accuracy_score(y_test_multi, pred)
    accuracies.append(acc)
    print(f"{alpha:10.1f} | {acc:.4f}")

best_alpha = alpha_values[np.argmax(accuracies)]
print(f"\nBest alpha value: {best_alpha}")

# =============================================================================
# 6. FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print("\n\n6. FEATURE IMPORTANCE ANALYSIS")
print("-" * 35)

# Get feature log probabilities for each class
feature_names = vectorizer.get_feature_names_out()
log_probs_neg = multinomial_nb.feature_log_prob_[0]  # Class 0 (Negative)
log_probs_pos = multinomial_nb.feature_log_prob_[1]  # Class 1 (Positive)

# Calculate feature importance as difference in log probabilities
feature_importance = log_probs_pos - log_probs_neg

# Sort features by importance
sorted_indices = np.argsort(feature_importance)

print("Top 10 words indicating NEGATIVE sentiment:")
for i in sorted_indices[:10]:
    print(f"{feature_names[i]:15s}: {feature_importance[i]:.4f}")

print("\nTop 10 words indicating POSITIVE sentiment:")
for i in sorted_indices[-10:]:
    print(f"{feature_names[i]:15s}: {feature_importance[i]:.4f}")

# =============================================================================
# 7. PRACTICAL TIPS AND BEST PRACTICES
# =============================================================================

print("\n\n7. PRACTICAL IMPLEMENTATION TIPS")
print("-" * 35)

print("""
Best Practices for Naive Bayes:

1. Data Preprocessing:
   - Handle missing values appropriately
   - For text: remove stop words, apply stemming/lemmatization
   - For continuous data: check for normal distribution assumption

2. Feature Selection:
   - Remove highly correlated features (violates independence assumption)
   - Use feature selection techniques for high-dimensional data
   - Consider TF-IDF for text data instead of raw counts

3. Hyperparameter Tuning:
   - Tune smoothing parameter (alpha) using cross-validation
   - For Gaussian NB: consider feature scaling if needed

4. Model Selection:
   - Use Gaussian NB for continuous features
   - Use Multinomial NB for count/frequency data
   - Use Bernoulli NB for binary/boolean features

5. Evaluation:
   - Use cross-validation for robust performance estimation
   - Check confusion matrix for class-wise performance
   - Consider probability calibration for better probability estimates
""")

# =============================================================================
# 8. CROSS-VALIDATION EXAMPLE
# =============================================================================

print("\n8. CROSS-VALIDATION EVALUATION")
print("-" * 30)

from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(multinomial_nb, X_text, labels, cv=5, scoring='accuracy')

print("5-Fold Cross-Validation Results:")
print(f"Individual fold scores: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\n" + "=" * 60)
print("NAIVE BAYES IMPLEMENTATION COMPLETE")
print("=" * 60)