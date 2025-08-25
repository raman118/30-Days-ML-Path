import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.datasets import make_classification, make_regression
import pandas as pd

# Set style for better plots
plt.style.use('seaborn-v0_8')
np.random.seed(42)


class SVMImplementation:
    """
    A comprehensive SVM implementation class using scikit-learn
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def linear_svm_example(self):
        """
        Example 1: Linear SVM on a linearly separable dataset
        """
        print("=" * 60)
        print("EXAMPLE 1: LINEAR SVM - LINEARLY SEPARABLE DATA")
        print("=" * 60)

        # Generate linearly separable data
        X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1,
                                   random_state=42)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create and train Linear SVM
        svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
        svm_linear.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = svm_linear.predict(X_test_scaled)

        # Print results
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Number of support vectors: {svm_linear.n_support_}")
        print(f"Support vector indices: {svm_linear.support_}")

        # Plot decision boundary
        self.plot_decision_boundary(X_train_scaled, y_train, svm_linear, "Linear SVM")

        return svm_linear, X_train_scaled, X_test_scaled, y_train, y_test

    def non_linear_svm_example(self):
        """
        Example 2: Non-linear SVM with RBF kernel
        """
        print("\n" + "=" * 60)
        print("EXAMPLE 2: NON-LINEAR SVM - RBF KERNEL")
        print("=" * 60)

        # Generate non-linearly separable data (moons dataset)
        X, y = datasets.make_moons(n_samples=200, noise=0.3, random_state=42)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create and train RBF SVM
        svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm_rbf.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = svm_rbf.predict(X_test_scaled)

        # Print results
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Number of support vectors: {svm_rbf.n_support_}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Plot decision boundary
        self.plot_decision_boundary(X_train_scaled, y_train, svm_rbf, "RBF SVM")

        return svm_rbf, X_train_scaled, X_test_scaled, y_train, y_test

    def kernel_comparison(self):
        """
        Example 3: Comparing different kernels
        """
        print("\n" + "=" * 60)
        print("EXAMPLE 3: KERNEL COMPARISON")
        print("=" * 60)

        # Generate circular data
        X, y = datasets.make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Different kernels to compare
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        results = {}

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        for i, kernel in enumerate(kernels):
            # Train SVM with different kernels
            if kernel == 'poly':
                svm = SVC(kernel=kernel, C=1.0, degree=3, random_state=42)
            else:
                svm = SVC(kernel=kernel, C=1.0, random_state=42)

            svm.fit(X_train_scaled, y_train)
            y_pred = svm.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            results[kernel] = {
                'accuracy': accuracy,
                'n_support_vectors': svm.n_support_.sum()
            }

            print(f"{kernel.upper()} Kernel - Accuracy: {accuracy:.4f}, Support Vectors: {svm.n_support_.sum()}")

            # Plot decision boundary for each kernel
            self.plot_decision_boundary_subplot(X_train_scaled, y_train, svm,
                                                f"{kernel.upper()} Kernel", axes[i])

        plt.tight_layout()
        plt.show()

        return results

    def hyperparameter_tuning(self):
        """
        Example 4: Hyperparameter tuning using Grid Search
        """
        print("\n" + "=" * 60)
        print("EXAMPLE 4: HYPERPARAMETER TUNING")
        print("=" * 60)

        # Load iris dataset
        iris = datasets.load_iris()
        X, y = iris.data, iris.target

        # Use only first two classes for binary classification
        mask = y != 2
        X, y = X[mask], y[mask]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }

        # Perform grid search
        svm = SVC(random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        # Get best parameters
        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validation score:", f"{grid_search.best_score_:.4f}")

        # Test on test set
        best_svm = grid_search.best_estimator_
        y_pred = best_svm.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        print("Test accuracy:", f"{test_accuracy:.4f}")

        return grid_search.best_estimator_

    def multiclass_svm(self):
        """
        Example 5: Multiclass SVM classification
        """
        print("\n" + "=" * 60)
        print("EXAMPLE 5: MULTICLASS SVM")
        print("=" * 60)

        # Load full iris dataset (3 classes)
        iris = datasets.load_iris()
        X, y = iris.data, iris.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train multiclass SVM
        svm_multi = SVC(kernel='rbf', C=1.0, gamma='scale',
                        decision_function_shape='ovr', random_state=42)
        svm_multi.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = svm_multi.predict(X_test_scaled)

        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=iris.target_names, yticklabels=iris.target_names)
        plt.title('Confusion Matrix - Multiclass SVM')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        return svm_multi

    def svm_regression_example(self):
        """
        Example 6: SVM for Regression (SVR)
        """
        print("\n" + "=" * 60)
        print("EXAMPLE 6: SUPPORT VECTOR REGRESSION (SVR)")
        print("=" * 60)

        # Generate regression data
        X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train SVR with different kernels
        kernels = ['linear', 'rbf', 'poly']

        plt.figure(figsize=(15, 5))

        for i, kernel in enumerate(kernels):
            # Train SVR
            svr = SVR(kernel=kernel, C=100, gamma='auto', epsilon=0.1)
            svr.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = svr.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)

            print(f"{kernel.upper()} SVR - MSE: {mse:.2f}")

            # Plot results
            plt.subplot(1, 3, i + 1)
            plt.scatter(X_test, y_test, color='red', alpha=0.6, label='True values')

            # Sort for plotting
            sorted_indices = np.argsort(X_test.flatten())
            X_test_sorted = X_test[sorted_indices]
            X_test_scaled_sorted = X_test_scaled[sorted_indices]
            y_pred_sorted = svr.predict(X_test_scaled_sorted)

            plt.plot(X_test_sorted, y_pred_sorted, color='blue', label='SVR prediction')
            plt.title(f'{kernel.upper()} SVR (MSE: {mse:.2f})')
            plt.xlabel('X')
            plt.ylabel('y')
            plt.legend()

        plt.tight_layout()
        plt.show()

    def cross_validation_example(self):
        """
        Example 7: Cross-validation for model evaluation
        """
        print("\n" + "=" * 60)
        print("EXAMPLE 7: CROSS-VALIDATION")
        print("=" * 60)

        # Load breast cancer dataset
        cancer = datasets.load_breast_cancer()
        X, y = cancer.data, cancer.target

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Different SVM configurations
        svm_configs = {
            'Linear SVM': SVC(kernel='linear', C=1.0, random_state=42),
            'RBF SVM (C=1)': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
            'RBF SVM (C=10)': SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42),
            'Polynomial SVM': SVC(kernel='poly', C=1.0, degree=3, random_state=42)
        }

        results = {}

        for name, svm in svm_configs.items():
            # Perform 5-fold cross-validation
            cv_scores = cross_val_score(svm, X_scaled, y, cv=5, scoring='accuracy')
            results[name] = {
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'scores': cv_scores
            }

            print(f"{name}:")
            print(f"  Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  Individual scores: {cv_scores}")
            print()

        return results

    def plot_decision_boundary(self, X, y, model, title):
        """
        Plot decision boundary for 2D data
        """
        plt.figure(figsize=(10, 8))

        # Create a mesh
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Make predictions on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points)
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)

        # Plot data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')

        # Plot support vectors
        plt.scatter(X[model.support_, 0], X[model.support_, 1],
                    s=100, facecolors='none', edgecolors='black', linewidths=2,
                    label='Support Vectors')

        plt.title(f'{title}\nSupport Vectors: {len(model.support_)}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.colorbar(scatter)
        plt.show()

    def plot_decision_boundary_subplot(self, X, y, model, title, ax):
        """
        Plot decision boundary for subplot
        """
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points)
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
        ax.scatter(X[model.support_, 0], X[model.support_, 1],
                   s=100, facecolors='none', edgecolors='black', linewidths=2)
        ax.set_title(f'{title}\nSV: {len(model.support_)}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')


def main():
    """
    Main function to run all SVM examples
    """
    print("SUPPORT VECTOR MACHINE (SVM) IMPLEMENTATION USING SCIKIT-LEARN")
    print("=" * 70)

    # Create SVM implementation instance
    svm_impl = SVMImplementation()

    # Run all examples
    try:
        # Example 1: Linear SVM
        svm_impl.linear_svm_example()

        # Example 2: Non-linear SVM
        svm_impl.non_linear_svm_example()

        # Example 3: Kernel comparison
        svm_impl.kernel_comparison()

        # Example 4: Hyperparameter tuning
        svm_impl.hyperparameter_tuning()

        # Example 5: Multiclass SVM
        svm_impl.multiclass_svm()

        # Example 6: SVM Regression
        svm_impl.svm_regression_example()

        # Example 7: Cross-validation
        svm_impl.cross_validation_example()

        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()