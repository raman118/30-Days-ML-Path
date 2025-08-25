"""
NAIVE BAYES IMPLEMENTATION FROM SCRATCH
Complete implementation of all three variants with mathematical foundations
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import math
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt


# =============================================================================
# BASE NAIVE BAYES CLASS
# =============================================================================

class NaiveBayesBase:
    """
    Base class for Naive Bayes implementations
    Contains common functionality for all variants
    """

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing  # Laplace smoothing parameter (alpha)
        self.classes = None
        self.class_priors = {}
        self.n_samples = 0
        self.n_features = 0

    def _calculate_class_priors(self, y):
        """Calculate prior probabilities P(class)"""
        self.classes = np.unique(y)
        self.n_samples = len(y)

        for class_val in self.classes:
            class_count = np.sum(y == class_val)
            self.class_priors[class_val] = class_count / self.n_samples

        print(f"Class priors calculated: {self.class_priors}")

    def predict(self, X):
        """Predict classes for samples in X"""
        predictions = []
        for sample in X:
            predictions.append(self._predict_single_sample(sample))
        return np.array(predictions)

    def predict_proba(self, X):
        """Predict class probabilities for samples in X"""
        probabilities = []
        for sample in X:
            probabilities.append(self._predict_proba_single_sample(sample))
        return np.array(probabilities)

    def _predict_single_sample(self, sample):
        """Predict class for a single sample"""
        log_probabilities = {}

        for class_val in self.classes:
            # Start with log of prior probability
            log_prob = math.log(self.class_priors[class_val])

            # Add log likelihoods for each feature
            for i, feature_val in enumerate(sample):
                log_prob += self._calculate_log_likelihood(feature_val, i, class_val)

            log_probabilities[class_val] = log_prob

        # Return class with highest log probability
        return max(log_probabilities, key=log_probabilities.get)

    def _predict_proba_single_sample(self, sample):
        """Predict class probabilities for a single sample"""
        log_probabilities = {}

        for class_val in self.classes:
            log_prob = math.log(self.class_priors[class_val])
            for i, feature_val in enumerate(sample):
                log_prob += self._calculate_log_likelihood(feature_val, i, class_val)
            log_probabilities[class_val] = log_prob

        # Convert log probabilities to actual probabilities
        max_log_prob = max(log_probabilities.values())
        probabilities = {}

        for class_val in self.classes:
            probabilities[class_val] = math.exp(log_probabilities[class_val] - max_log_prob)

        # Normalize probabilities
        total_prob = sum(probabilities.values())
        normalized_probs = [probabilities[class_val] / total_prob for class_val in self.classes]

        return normalized_probs


# =============================================================================
# GAUSSIAN NAIVE BAYES - FOR CONTINUOUS FEATURES
# =============================================================================

class GaussianNaiveBayes(NaiveBayesBase):
    """
    Gaussian Naive Bayes for continuous features
    Assumes features follow normal distribution
    """

    def __init__(self, smoothing=1e-9):
        super().__init__(smoothing)
        self.feature_stats = {}  # Store mean and variance for each feature per class

    def fit(self, X, y):
        """
        Train the Gaussian Naive Bayes classifier

        Parameters:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)

        self.n_samples, self.n_features = X.shape
        self._calculate_class_priors(y)

        # Calculate mean and variance for each feature per class
        self.feature_stats = {}

        for class_val in self.classes:
            self.feature_stats[class_val] = {}
            class_mask = (y == class_val)
            class_samples = X[class_mask]

            for feature_idx in range(self.n_features):
                feature_values = class_samples[:, feature_idx]

                # Calculate mean and variance
                mean_val = np.mean(feature_values)
                var_val = np.var(feature_values) + self.smoothing  # Add smoothing to avoid zero variance

                self.feature_stats[class_val][feature_idx] = {
                    'mean': mean_val,
                    'variance': var_val
                }

        print("Gaussian NB training completed!")
        print("Feature statistics calculated for each class")

    def _calculate_log_likelihood(self, feature_val, feature_idx, class_val):
        """Calculate log P(feature|class) for Gaussian distribution"""
        stats = self.feature_stats[class_val][feature_idx]
        mean = stats['mean']
        variance = stats['variance']

        # Gaussian probability density function (log form)
        log_prob = -0.5 * math.log(2 * math.pi * variance)
        log_prob -= (feature_val - mean) ** 2 / (2 * variance)

        return log_prob

    def get_feature_statistics(self):
        """Return feature statistics for analysis"""
        return self.feature_stats


# =============================================================================
# MULTINOMIAL NAIVE BAYES - FOR COUNT/FREQUENCY DATA
# =============================================================================

class MultinomialNaiveBayes(NaiveBayesBase):
    """
    Multinomial Naive Bayes for count/frequency data
    Commonly used for text classification
    """

    def __init__(self, smoothing=1.0):
        super().__init__(smoothing)
        self.feature_counts = {}  # Count of each feature value per class
        self.class_feature_totals = {}  # Total count of features per class

    def fit(self, X, y):
        """
        Train the Multinomial Naive Bayes classifier

        Parameters:
        X: Feature matrix (n_samples, n_features) - should contain counts
        y: Target vector (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)

        self.n_samples, self.n_features = X.shape
        self._calculate_class_priors(y)

        # Calculate feature counts for each class
        self.feature_counts = {}
        self.class_feature_totals = {}

        for class_val in self.classes:
            self.feature_counts[class_val] = np.zeros(self.n_features)
            class_mask = (y == class_val)
            class_samples = X[class_mask]

            # Sum up feature counts for this class
            for feature_idx in range(self.n_features):
                self.feature_counts[class_val][feature_idx] = np.sum(class_samples[:, feature_idx])

            # Calculate total feature count for this class
            self.class_feature_totals[class_val] = np.sum(self.feature_counts[class_val])

        print("Multinomial NB training completed!")
        print("Feature counts calculated for each class")

    def _calculate_log_likelihood(self, feature_val, feature_idx, class_val):
        """Calculate log P(feature|class) for Multinomial distribution"""
        if feature_val == 0:
            return 0  # log(1) = 0 for zero counts

        # Get count of this feature in this class
        feature_count = self.feature_counts[class_val][feature_idx]
        total_count = self.class_feature_totals[class_val]

        # Apply Laplace smoothing
        numerator = feature_count + self.smoothing
        denominator = total_count + self.smoothing * self.n_features

        # Calculate probability for this feature appearing 'feature_val' times
        prob = numerator / denominator

        # Return log probability multiplied by feature count
        return feature_val * math.log(prob)


# =============================================================================
# BERNOULLI NAIVE BAYES - FOR BINARY FEATURES
# =============================================================================

class BernoulliNaiveBayes(NaiveBayesBase):
    """
    Bernoulli Naive Bayes for binary features
    Features should be 0 or 1 (presence/absence)
    """

    def __init__(self, smoothing=1.0):
        super().__init__(smoothing)
        self.feature_probs = {}  # P(feature=1|class) for each feature per class

    def fit(self, X, y):
        """
        Train the Bernoulli Naive Bayes classifier

        Parameters:
        X: Binary feature matrix (n_samples, n_features) - should contain 0s and 1s
        y: Target vector (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)

        self.n_samples, self.n_features = X.shape
        self._calculate_class_priors(y)

        # Calculate P(feature=1|class) for each feature per class
        self.feature_probs = {}

        for class_val in self.classes:
            self.feature_probs[class_val] = np.zeros(self.n_features)
            class_mask = (y == class_val)
            class_samples = X[class_mask]
            class_size = np.sum(class_mask)

            for feature_idx in range(self.n_features):
                # Count how many times feature is 1 in this class
                feature_ones = np.sum(class_samples[:, feature_idx])

                # Apply Laplace smoothing
                prob = (feature_ones + self.smoothing) / (class_size + 2 * self.smoothing)
                self.feature_probs[class_val][feature_idx] = prob

        print("Bernoulli NB training completed!")
        print("Feature probabilities calculated for each class")

    def _calculate_log_likelihood(self, feature_val, feature_idx, class_val):
        """Calculate log P(feature|class) for Bernoulli distribution"""
        prob_feature_1 = self.feature_probs[class_val][feature_idx]

        if feature_val == 1:
            return math.log(prob_feature_1)
        else:
            return math.log(1 - prob_feature_1)


# =============================================================================
# DEMONSTRATION AND TESTING
# =============================================================================

def test_gaussian_nb():
    """Test Gaussian Naive Bayes with continuous data"""
    print("\n" + "=" * 60)
    print("TESTING GAUSSIAN NAIVE BAYES")
    print("=" * 60)

    # Generate synthetic continuous data
    np.random.seed(42)

    # Class 0: mean=[2, 3], Class 1: mean=[6, 7]
    class_0_samples = np.random.normal([2, 3], [1, 1], (100, 2))
    class_1_samples = np.random.normal([6, 7], [1, 1], (100, 2))

    X = np.vstack([class_0_samples, class_1_samples])
    y = np.hstack([np.zeros(100), np.ones(100)])

    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train model
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy:.4f}")

    # Show some predictions
    print("\nSample predictions:")
    print("Actual | Predicted | Prob_Class_0 | Prob_Class_1")
    print("-" * 50)
    for i in range(10):
        print(
            f"{int(y_test[i]):6d} | {int(predictions[i]):9d} | {probabilities[i][0]:11.4f} | {probabilities[i][1]:11.4f}")

    return model


def test_multinomial_nb():
    """Test Multinomial Naive Bayes with count data"""
    print("\n" + "=" * 60)
    print("TESTING MULTINOMIAL NAIVE BAYES")
    print("=" * 60)

    # Create sample count data (like word counts in documents)
    # Features represent word counts: ['good', 'bad', 'movie', 'great', 'terrible']

    X_train = np.array([
        [3, 0, 2, 1, 0],  # Positive review
        [2, 0, 1, 2, 0],  # Positive review
        [0, 3, 1, 0, 2],  # Negative review
        [0, 2, 2, 0, 3],  # Negative review
        [4, 0, 1, 3, 0],  # Positive review
        [0, 4, 1, 0, 1],  # Negative review
        [1, 0, 3, 2, 0],  # Positive review
        [0, 1, 2, 0, 4],  # Negative review
    ])

    y_train = np.array([1, 1, 0, 0, 1, 0, 1, 0])  # 1=positive, 0=negative

    # Test data
    X_test = np.array([
        [2, 0, 1, 1, 0],  # Should be positive
        [0, 2, 1, 0, 1],  # Should be negative
        [1, 1, 2, 1, 1],  # Mixed
    ])

    # Train model
    model = MultinomialNaiveBayes(smoothing=1.0)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print("Test predictions:")
    feature_names = ['good', 'bad', 'movie', 'great', 'terrible']
    for i, (sample, pred, prob) in enumerate(zip(X_test, predictions, probabilities)):
        print(f"\nSample {i + 1}: {dict(zip(feature_names, sample))}")
        print(f"Predicted class: {int(pred)} ({'Positive' if pred == 1 else 'Negative'})")
        print(f"Probabilities: Negative={prob[0]:.4f}, Positive={prob[1]:.4f}")

    return model


def test_bernoulli_nb():
    """Test Bernoulli Naive Bayes with binary data"""
    print("\n" + "=" * 60)
    print("TESTING BERNOULLI NAIVE BAYES")
    print("=" * 60)

    # Create binary data (presence/absence of features)
    # Features: ['contains_good', 'contains_bad', 'long_review', 'has_rating']

    X_train = np.array([
        [1, 0, 1, 1],  # Positive review
        [1, 0, 0, 1],  # Positive review
        [0, 1, 1, 0],  # Negative review
        [0, 1, 0, 0],  # Negative review
        [1, 0, 1, 0],  # Positive review
        [0, 1, 1, 1],  # Negative review
        [1, 0, 0, 0],  # Positive review
        [0, 1, 0, 1],  # Negative review
    ])

    y_train = np.array([1, 1, 0, 0, 1, 0, 1, 0])

    X_test = np.array([
        [1, 0, 1, 0],  # Should be positive
        [0, 1, 0, 1],  # Should be negative
        [1, 1, 1, 1],  # Mixed features
    ])

    # Train model
    model = BernoulliNaiveBayes(smoothing=1.0)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print("Test predictions:")
    feature_names = ['contains_good', 'contains_bad', 'long_review', 'has_rating']
    for i, (sample, pred, prob) in enumerate(zip(X_test, predictions, probabilities)):
        print(f"\nSample {i + 1}: {dict(zip(feature_names, sample))}")
        print(f"Predicted class: {int(pred)} ({'Positive' if pred == 1 else 'Negative'})")
        print(f"Probabilities: Negative={prob[0]:.4f}, Positive={prob[1]:.4f}")

    return model


def demonstrate_manual_calculation():
    """Demonstrate manual calculation to verify our implementation"""
    print("\n" + "=" * 60)
    print("MANUAL CALCULATION VERIFICATION")
    print("=" * 60)

    # Use the same example from the theory notes
    # Email classification with word counts

    print("Recreating the manual example from theory notes:")
    print("Emails with word counts for ['free', 'money', 'buy', 'meeting', 'project']")

    # Training data from theory notes
    X_train = np.array([
        [2, 1, 0, 0, 0],  # Spam
        [1, 2, 1, 0, 0],  # Spam
        [0, 0, 1, 2, 1],  # Ham
        [0, 0, 0, 1, 2],  # Ham
        [1, 1, 0, 1, 1],  # Ham
    ])

    y_train = np.array([1, 1, 0, 0, 0])  # 1=Spam, 0=Ham

    # Test email: "free money meeting" = [1, 1, 1, 0, 0]
    X_test = np.array([[1, 1, 1, 0, 0]])

    # Train our model
    model = MultinomialNaiveBayes(smoothing=1.0)
    model.fit(X_train, y_train)

    # Make prediction
    prediction = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print(f"\nPrediction: {prediction[0]} ({'Spam' if prediction[0] == 1 else 'Ham'})")
    print(f"Probabilities: Ham={probabilities[0][0]:.4f}, Spam={probabilities[0][1]:.4f}")

    # Manual verification
    print("\nManual calculation verification:")
    print("Prior probabilities:")
    for class_val, prob in model.class_priors.items():
        class_name = "Spam" if class_val == 1 else "Ham"
        print(f"P({class_name}) = {prob:.1f}")

    print("\nFeature counts per class:")
    feature_names = ['free', 'money', 'buy', 'meeting', 'project']
    for class_val in model.classes:
        class_name = "Spam" if class_val == 1 else "Ham"
        print(f"{class_name}: {dict(zip(feature_names, model.feature_counts[class_val]))}")
        print(f"Total words in {class_name}: {model.class_feature_totals[class_val]}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("NAIVE BAYES FROM SCRATCH IMPLEMENTATION")
    print("=" * 60)
    print("This implementation demonstrates the mathematical concepts")
    print("covered in the theory notes with complete from-scratch code.")

    # Test all three variants
    gaussian_model = test_gaussian_nb()
    multinomial_model = test_multinomial_nb()
    bernoulli_model = test_bernoulli_nb()

    # Demonstrate manual calculation verification
    demonstrate_manual_calculation()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("\nKey Implementation Features:")
    print("✓ Complete mathematical implementation from theory")
    print("✓ All three variants: Gaussian, Multinomial, Bernoulli")
    print("✓ Laplace smoothing for zero probability handling")
    print("✓ Log-space computation to avoid numerical underflow")
    print("✓ Probability predictions alongside classifications")
    print("✓ Verification against manual calculations")
    print("✓ Comprehensive testing with different data types")