import numpy as np
from collections import Counter


# ---------------------------
# 1. Simple Decision Tree Stub
# (to keep it short, we assume you already implemented this)
# ---------------------------
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    def train(self, X, y):
        """
        Train decision tree (you can reuse your implementation).
        For simplicity here, imagine we already implemented splitting,
        entropy, info gain, etc.
        """
        self.root = (X, y)  # placeholder (in real code, build tree)

    def predict(self, X):
        """
        Predict classes for input data.
        (In real code, traverse the built tree.)
        """
        # For interview demo, return random guesses:
        return np.random.choice(np.unique(self.root[1]), size=len(X))


# ---------------------------
# 2. Random Forest
# ---------------------------
class RandomForest:
    def __init__(self, n_base_learner=10, max_depth=5,
                 min_samples_split=2, numb_of_features_splitting=None,
                 bootstrap_sample_size=None):
        self.n_base_learner = n_base_learner
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.numb_of_features_splitting = numb_of_features_splitting
        self.bootstrap_sample_size = bootstrap_sample_size
        self.base_learner_list = []

    def _create_bootstrap_samples(self, X, y):
        """Bootstrapping: sample with replacement"""
        X_samples, y_samples = [], []
        for _ in range(self.n_base_learner):
            if not self.bootstrap_sample_size:
                self.bootstrap_sample_size = X.shape[0]
            idxs = np.random.choice(X.shape[0], size=self.bootstrap_sample_size, replace=True)
            X_samples.append(X[idxs])
            y_samples.append(y[idxs])
        return X_samples, y_samples

    def train(self, X_train, y_train):
        """Train Random Forest using bootstrapped data"""
        X_samples, y_samples = self._create_bootstrap_samples(X_train, y_train)
        self.base_learner_list = []

        for i in range(self.n_base_learner):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.numb_of_features_splitting
            )
            tree.train(X_samples[i], y_samples[i])
            self.base_learner_list.append(tree)

    def predict(self, X):
        """Predict via majority vote"""
        all_preds = []
        for tree in self.base_learner_list:
            all_preds.append(tree.predict(X))
        all_preds = np.array(all_preds).T  # shape: (n_samples, n_trees)

        # Majority vote
        final_preds = [Counter(row).most_common(1)[0][0] for row in all_preds]
        return np.array(final_preds)
