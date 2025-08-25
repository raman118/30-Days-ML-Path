import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cvxopt
import cvxopt.solvers

# Suppress CVXOPT output for cleaner display
cvxopt.solvers.options['show_progress'] = False


class SVMFromScratch:
    """
    Support Vector Machine implementation from scratch

    This implementation uses the Sequential Minimal Optimization (SMO) algorithm
    for solving the quadratic programming problem and also includes a CVXOPT-based
    solver for educational purposes.
    """

    def __init__(self, C=1.0, kernel='linear', gamma=1.0, degree=3, coef0=0.0,
                 tolerance=1e-3, max_iterations=1000):
        """
        Initialize SVM parameters

        Parameters:
        - C: Regularization parameter
        - kernel: Kernel type ('linear', 'polynomial', 'rbf')
        - gamma: Kernel coefficient for RBF and polynomial kernels
        - degree: Degree for polynomial kernel
        - coef0: Independent term in polynomial kernel
        - tolerance: Tolerance for stopping criterion
        - max_iterations: Maximum number of iterations
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        # Initialize parameters
        self.alphas = None
        self.b = 0
        self.X = None
        self.y = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
        self.n_support_vectors = 0

    def _kernel_function(self, x1, x2):
        """
        Compute kernel function between two vectors
        """
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'polynomial':
            return (self.gamma * np.dot(x1, x2) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _compute_kernel_matrix(self, X1, X2=None):
        """
        Compute the kernel matrix between X1 and X2
        """
        if X2 is None:
            X2 = X1

        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel_function(X1[i], X2[j])

        return K

    def fit_cvxopt(self, X, y):
        """
        Fit SVM using CVXOPT quadratic programming solver
        This method solves the dual optimization problem directly
        """
        print("Training SVM using CVXOPT solver...")

        n_samples, n_features = X.shape
        self.X = X
        self.y = y

        # Compute kernel matrix
        K = self._compute_kernel_matrix(X)

        # Set up quadratic programming problem
        # Dual problem: maximize Σα_i - 1/2 Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
        # Subject to: 0 ≤ α_i ≤ C and Σα_i y_i = 0

        # Objective function: 1/2 x^T P x + q^T x
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n_samples))

        # Inequality constraints: -α_i ≤ 0 and α_i ≤ C
        G_std = np.diag(np.ones(n_samples) * -1)  # -α_i ≤ 0
        G_slack = np.diag(np.ones(n_samples))  # α_i ≤ C
        G = cvxopt.matrix(np.vstack((G_std, G_slack)))

        h_std = cvxopt.matrix(np.zeros(n_samples))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self.C)
        h = cvxopt.matrix(np.hstack((h_std, h_slack)))

        # Equality constraint: Σα_i y_i = 0
        A = cvxopt.matrix(y.reshape(1, -1).astype(float))
        b = cvxopt.matrix(0.0)

        # Solve quadratic programming problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Extract alphas
        alphas = np.ravel(solution['x'])

        # Find support vectors (α > tolerance)
        support_vector_indices = alphas > self.tolerance
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]
        self.support_vector_alphas = alphas[support_vector_indices]
        self.n_support_vectors = len(self.support_vector_alphas)

        print(f"Found {self.n_support_vectors} support vectors")

        # Compute bias term b
        # For any support vector: y_i(Σα_j y_j K(x_j, x_i) + b) = 1
        if self.n_support_vectors > 0:
            # Use first support vector to compute bias
            sv_idx = 0
            self.b = self.support_vector_labels[sv_idx]

            for i in range(self.n_support_vectors):
                self.b -= (self.support_vector_alphas[i] *
                           self.support_vector_labels[i] *
                           self._kernel_function(self.support_vectors[i],
                                                 self.support_vectors[sv_idx]))

        self.alphas = alphas

        return self

    def fit_smo(self, X, y):
        """
        Fit SVM using Sequential Minimal Optimization (SMO) algorithm
        This is a more practical implementation for larger datasets
        """
        print("Training SVM using SMO algorithm...")

        n_samples, n_features = X.shape
        self.X = X
        self.y = y

        # Initialize alphas and bias
        self.alphas = np.zeros(n_samples)
        self.b = 0

        # Compute kernel matrix (cache for efficiency)
        self.K = self._compute_kernel_matrix(X)

        # SMO main loop
        num_changed = 0
        examine_all = True
        iteration = 0

        while (num_changed > 0 or examine_all) and iteration < self.max_iterations:
            num_changed = 0

            if examine_all:
                # Examine all examples
                for i in range(n_samples):
                    num_changed += self._examine_example(i)
            else:
                # Examine examples where 0 < alpha < C
                for i in range(n_samples):
                    if 0 < self.alphas[i] < self.C:
                        num_changed += self._examine_example(i)

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            iteration += 1

        # Extract support vectors
        support_vector_indices = self.alphas > self.tolerance
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]
        self.support_vector_alphas = self.alphas[support_vector_indices]
        self.n_support_vectors = len(self.support_vector_alphas)

        print(f"SMO converged after {iteration} iterations")
        print(f"Found {self.n_support_vectors} support vectors")

        return self

    def _examine_example(self, i1):
        """
        Examine example i1 and try to find a second example to optimize
        """
        y1 = self.y[i1]
        alpha1 = self.alphas[i1]
        E1 = self._decision_function_single(i1) - y1

        r1 = E1 * y1

        # Check KKT conditions
        if ((r1 < -self.tolerance and alpha1 < self.C) or
                (r1 > self.tolerance and alpha1 > 0)):

            # Try to find second example using heuristics
            if self._take_step_heuristic(i1, E1):
                return 1

            # Try all non-zero and non-C alphas
            non_bound_indices = np.where((self.alphas > 0) & (self.alphas < self.C))[0]
            np.random.shuffle(non_bound_indices)

            for i2 in non_bound_indices:
                if self._take_step(i1, i2):
                    return 1

            # Try all examples
            all_indices = np.arange(len(self.alphas))
            np.random.shuffle(all_indices)

            for i2 in all_indices:
                if self._take_step(i1, i2):
                    return 1

        return 0

    def _take_step_heuristic(self, i1, E1):
        """
        Use heuristic to choose second example
        """
        # Find example with maximum |E1 - E2|
        errors = [self._decision_function_single(i) - self.y[i]
                  for i in range(len(self.y))]

        if E1 > 0:
            i2 = np.argmin(errors)
        else:
            i2 = np.argmax(errors)

        if abs(E1 - errors[i2]) > self.tolerance:
            return self._take_step(i1, i2)

        return False

    def _take_step(self, i1, i2):
        """
        Try to optimize alpha[i1] and alpha[i2]
        """
        if i1 == i2:
            return False

        alpha1, alpha2 = self.alphas[i1], self.alphas[i2]
        y1, y2 = self.y[i1], self.y[i2]

        E1 = self._decision_function_single(i1) - y1
        E2 = self._decision_function_single(i2) - y2

        s = y1 * y2

        # Compute bounds L and H
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)

        if L == H:
            return False

        # Compute eta
        k11 = self.K[i1, i1]
        k12 = self.K[i1, i2]
        k22 = self.K[i2, i2]
        eta = k11 + k22 - 2 * k12

        if eta > 0:
            # Compute new alpha2
            alpha2_new = alpha2 + y2 * (E1 - E2) / eta

            # Clip alpha2
            if alpha2_new >= H:
                alpha2_new = H
            elif alpha2_new <= L:
                alpha2_new = L
        else:
            # eta <= 0, compute objective function at bounds
            f1 = y1 * (E1 + self.b) - alpha1 * k11 - s * alpha2 * k12
            f2 = y2 * (E2 + self.b) - s * alpha1 * k12 - alpha2 * k22

            L1 = alpha1 + s * (alpha2 - L)
            H1 = alpha1 + s * (alpha2 - H)

            Lobj = L1 * f1 + L * f2 + 0.5 * L1 ** 2 * k11 + 0.5 * L ** 2 * k22 + s * L * L1 * k12
            Hobj = H1 * f1 + H * f2 + 0.5 * H1 ** 2 * k11 + 0.5 * H ** 2 * k22 + s * H * H1 * k12

            if Lobj < Hobj - self.tolerance:
                alpha2_new = L
            elif Lobj > Hobj + self.tolerance:
                alpha2_new = H
            else:
                alpha2_new = alpha2

        # Check if change is significant
        if abs(alpha2_new - alpha2) < self.tolerance * (alpha2_new + alpha2 + self.tolerance):
            return False

        # Compute new alpha1
        alpha1_new = alpha1 + s * (alpha2 - alpha2_new)

        # Update bias
        b_old = self.b

        if 0 < alpha1_new < self.C:
            self.b = E1 + y1 * (alpha1_new - alpha1) * k11 + y2 * (alpha2_new - alpha2) * k12 + b_old
        elif 0 < alpha2_new < self.C:
            self.b = E2 + y1 * (alpha1_new - alpha1) * k12 + y2 * (alpha2_new - alpha2) * k22 + b_old
        else:
            b1 = E1 + y1 * (alpha1_new - alpha1) * k11 + y2 * (alpha2_new - alpha2) * k12 + b_old
            b2 = E2 + y1 * (alpha1_new - alpha1) * k12 + y2 * (alpha2_new - alpha2) * k22 + b_old
            self.b = (b1 + b2) / 2

        # Update alphas
        self.alphas[i1] = alpha1_new
        self.alphas[i2] = alpha2_new

        return True

    def _decision_function_single(self, i):
        """
        Compute decision function for single example i
        """
        result = 0
        for j in range(len(self.alphas)):
            if self.alphas[j] > 0:
                result += self.alphas[j] * self.y[j] * self.K[j, i]
        result += self.b
        return result

    def decision_function(self, X):
        """
        Compute decision function for input X
        """
        if self.support_vectors is None:
            raise ValueError("Model has not been trained yet!")

        n_samples = X.shape[0]
        decision_scores = np.zeros(n_samples)

        for i in range(n_samples):
            score = 0
            for j in range(self.n_support_vectors):
                score += (self.support_vector_alphas[j] *
                          self.support_vector_labels[j] *
                          self._kernel_function(self.support_vectors[j], X[i]))
            decision_scores[i] = score + self.b

        return decision_scores

    def predict(self, X):
        """
        Make predictions on input X
        """
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        """
        Compute accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_support_vectors(self):
        """
        Get support vectors
        """
        return {
            'support_vectors': self.support_vectors,
            'support_vector_labels': self.support_vector_labels,
            'support_vector_alphas': self.support_vector_alphas,
            'n_support_vectors': self.n_support_vectors
        }


class SVMVisualizer:
    """
    Utility class for visualizing SVM results
    """

    @staticmethod
    def plot_decision_boundary(X, y, svm, title="SVM Decision Boundary"):
        """
        Plot decision boundary and support vectors
        """
        plt.figure(figsize=(12, 8))

        # Create a mesh
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Make predictions on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.decision_function(mesh_points)
        Z = Z.reshape(xx.shape)

        # Plot decision boundary and margins
        plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap=plt.cm.RdYlBu)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', linestyles='solid')
        plt.contour(xx, yy, Z, levels=[-1, 1], linewidths=2, colors='black', linestyles='dashed')

        # Plot data points
        colors = ['red' if label == -1 else 'blue' for label in y]
        plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, s=50)

        # Plot support vectors
        if svm.support_vectors is not None:
            plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1],
                        s=200, facecolors='none', edgecolors='black', linewidths=3,
                        label=f'Support Vectors ({svm.n_support_vectors})')

        plt.title(f'{title}\nC={svm.C}, Kernel={svm.kernel}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.colorbar(label='Decision Function')
        plt.grid(True, alpha=0.3)
        plt.show()

    @staticmethod
    def compare_solvers(X, y):
        """
        Compare CVXOPT and SMO solvers
        """
        print("=" * 60)
        print("COMPARING CVXOPT AND SMO SOLVERS")
        print("=" * 60)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train with CVXOPT
        svm_cvxopt = SVMFromScratch(C=1.0, kernel='rbf', gamma=0.1)
        svm_cvxopt.fit_cvxopt(X_train_scaled, y_train)

        # Train with SMO
        svm_smo = SVMFromScratch(C=1.0, kernel='rbf', gamma=0.1)
        svm_smo.fit_smo(X_train_scaled, y_train)

        # Compare results
        print(f"\nCVXOPT Solver:")
        print(f"  Training Accuracy: {svm_cvxopt.score(X_train_scaled, y_train):.4f}")
        print(f"  Test Accuracy: {svm_cvxopt.score(X_test_scaled, y_test):.4f}")
        print(f"  Support Vectors: {svm_cvxopt.n_support_vectors}")

        print(f"\nSMO Solver:")
        print(f"  Training Accuracy: {svm_smo.score(X_train_scaled, y_train):.4f}")
        print(f"  Test Accuracy: {svm_smo.score(X_test_scaled, y_test):.4f}")
        print(f"  Support Vectors: {svm_smo.n_support_vectors}")

        return svm_cvxopt, svm_smo, X_train_scaled, y_train


def demo_linear_svm():
    """
    Demonstrate linear SVM on linearly separable data
    """
    print("=" * 60)
    print("DEMO 1: LINEAR SVM")
    print("=" * 60)

    # Generate linearly separable data
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               n_classes=2, random_state=42)

    # Convert labels to -1, 1
    y = np.where(y == 0, -1, 1)

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train SVM
    svm = SVMFromScratch(C=1.0, kernel='linear')
    svm.fit_cvxopt(X_scaled, y)

    print(f"Training Accuracy: {svm.score(X_scaled, y):.4f}")

    # Visualize
    SVMVisualizer.plot_decision_boundary(X_scaled, y, svm, "Linear SVM")

    return svm, X_scaled, y


def demo_nonlinear_svm():
    """
    Demonstrate non-linear SVM with RBF kernel
    """
    print("\n" + "=" * 60)
    print("DEMO 2: NON-LINEAR SVM (RBF KERNEL)")
    print("=" * 60)

    # Generate non-linearly separable data
    X, y = make_blobs(n_samples=100, centers=2, n_features=2,
                      cluster_std=0.8, random_state=42)

    # Make data non-linearly separable by creating circular pattern
    for i in range(len(X)):
        if np.linalg.norm(X[i]) < 2:
            y[i] = 0
        else:
            y[i] = 1

    # Convert labels to -1, 1
    y = np.where(y == 0, -1, 1)

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train SVM with RBF kernel
    svm = SVMFromScratch(C=1.0, kernel='rbf', gamma=1.0)
    svm.fit_smo(X_scaled, y)  # Use SMO for RBF kernel

    print(f"Training Accuracy: {svm.score(X_scaled, y):.4f}")

    # Visualize
    SVMVisualizer.plot_decision_boundary(X_scaled, y, svm, "RBF SVM")

    return svm, X_scaled, y


def demo_kernel_comparison():
    """
    Compare different kernels
    """
    print("\n" + "=" * 60)
    print("DEMO 3: KERNEL COMPARISON")
    print("=" * 60)

    # Generate circular data
    np.random.seed(42)
    n_samples = 100

    # Inner circle (class -1)
    angles1 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    radii1 = np.random.uniform(0, 1, n_samples // 2)
    X1 = np.column_stack([radii1 * np.cos(angles1), radii1 * np.sin(angles1)])
    y1 = np.full(n_samples // 2, -1)

    # Outer circle (class +1)
    angles2 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    radii2 = np.random.uniform(1.5, 2.5, n_samples // 2)
    X2 = np.column_stack([radii2 * np.cos(angles2), radii2 * np.sin(angles2)])
    y2 = np.full(n_samples // 2, 1)

    # Combine data
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test different kernels
    kernels = ['linear', 'polynomial', 'rbf']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, kernel in enumerate(kernels):
        print(f"\nTesting {kernel.upper()} kernel...")

        if kernel == 'polynomial':
            svm = SVMFromScratch(C=1.0, kernel=kernel, degree=2, gamma=1.0)
        else:
            svm = SVMFromScratch(C=1.0, kernel=kernel, gamma=1.0)

        svm.fit_smo(X_scaled, y)
        accuracy = svm.score(X_scaled, y)
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Support Vectors: {svm.n_support_vectors}")

        # Plot decision boundary
        plt.subplot(1, 3, i + 1)
        h = 0.02
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.decision_function(mesh_points)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap=plt.cm.RdYlBu)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

        colors = ['red' if label == -1 else 'blue' for label in y]
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=colors, alpha=0.6)
        plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1],
                    s=100, facecolors='none', edgecolors='black', linewidths=2)

        plt.title(f'{kernel.upper()} Kernel\nAccuracy: {accuracy:.3f}, SV: {svm.n_support_vectors}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run all SVM demonstrations
    """
    print("SUPPORT VECTOR MACHINE - FROM SCRATCH IMPLEMENTATION")
    print("=" * 70)
    print("This implementation includes:")
    print("1. CVXOPT-based quadratic programming solver")
    print("2. Sequential Minimal Optimization (SMO) algorithm")
    print("3. Linear, Polynomial, and RBF kernels")
    print("4. Comprehensive visualization tools")
    print("=" * 70)

    try:
        # Demo 1: Linear SVM
        demo_linear_svm()

        # Demo 2: Non-linear SVM
        demo_nonlinear_svm()

        # Demo 3: Kernel comparison
        demo_kernel_comparison()

        # Demo 4: Solver comparison
        print("\n" + "=" * 60)
        print("DEMO 4: SOLVER COMPARISON")
        print("=" * 60)

        # Generate data for solver comparison
        X, y = make_classification(n_samples=50, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1,
                                   random_state=42)
        y = np.where(y == 0, -1, 1)

        svm_cvxopt, svm_smo, X_scaled, y_train = SVMVisualizer.compare_solvers(X, y)

        # Visualize both results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # CVXOPT result
        plt.subplot(1, 2, 1)
        SVMVisualizer.plot_decision_boundary(X_scaled, y_train, svm_cvxopt, "CVXOPT Solver")

        # SMO result
        plt.subplot(1, 2, 2)
        SVMVisualizer.plot_decision_boundary(X_scaled, y_train, svm_smo, "SMO Solver")

        plt.tight_layout()
        plt.show()

        print("\n" + "=" * 70)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()