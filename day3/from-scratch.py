import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    """
    Calculates the straight-line (Euclidean) distance between two points.
    It's the square root of the sum of the squared differences between coordinates.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    """
    A K-Nearest Neighbors classifier built from scratch.
    """

    def __init__(self, k=3):
        """
        Initializes the classifier.
        k: The number of neighbors to consider for voting.
        """
        # Store the number of neighbors, 'k'.
        self.k = k

    def fit(self, X, y):
        """
        "Trains" the model by simply storing the entire training dataset.
        KNN is a "lazy learner" because the real work happens during prediction.

        X: Training data features (a numpy array of samples).
        y: Training data labels (a numpy array of labels).
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts the labels for a set of new data points.

        X: The new data to classify (a numpy array).
        """
        # For each data point in X, call the helper `_predict` method.
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        """
        A helper method to predict the label for a *single* data point.
        This is where the core logic of the algorithm resides.

        x: A single sample (data point) to classify.
        """
        # 1. Calculate the distance from the new point 'x' to ALL points in the training data.
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # 2. Get the indices of the 'k' smallest distances.
        # `np.argsort` returns the indices that would sort the array.
        k_nearest_indices = np.argsort(distances)[:self.k]

        # 3. Get the labels of these 'k' nearest neighbors.
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        # 4. Determine the majority class (the most common label) among the neighbors.
        # `Counter` is a handy tool for counting hashable objects.
        # `most_common(1)` returns a list with the most common element and its count.
        # [0][0] extracts the element itself.
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# --- Example of How to Use the KNN Classifier (No Scikit-learn) ---
if __name__ == '__main__':
    # 1. Create a simple, synthetic dataset manually.
    # Imagine two clusters of points.
    # Class 0: points centered around (2, 3)
    # Class 1: points centered around (7, 8)
    X_train = np.array([
        [1.8, 2.5], [2.1, 3.2], [1.5, 3.5], [2.5, 2.8],  # Class 0
        [6.8, 7.5], [7.1, 8.2], [6.5, 8.5], [7.5, 7.8]  # Class 1
    ])
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # 2. Define our test data.
    # One point is clearly near class 0, and the other is near class 1.
    X_test = np.array([[3, 4], [7, 7]])
    y_test = np.array([0, 1])  # The true labels for our test data

    # --- Using our custom KNN ---

    # 3. Create an instance of our KNN classifier with k=3
    k = 3
    knn_classifier = KNN(k=k)

    # 4. "Train" the model with our training data
    knn_classifier.fit(X_train, y_train)

    # 5. Make predictions on the test data
    predictions = knn_classifier.predict(X_test)

    print(f"Test Data Points:\n{X_test}")
    print(f"Predicted Labels: {predictions}")
    print(f"Actual Labels:    {y_test}")

    # 6. Calculate the accuracy of our model manually
    # np.sum(predictions == y_test) counts the number of correct predictions
    num_correct = np.sum(predictions == y_test)
    total_samples = len(y_test)
    accuracy = num_correct / total_samples

    print(f"Accuracy: {accuracy:.4f} ({num_correct}/{total_samples} correct)")