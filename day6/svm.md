# Support Vector Machine (SVM) - Complete Notes

## 1. Introduction and Motivation

Support Vector Machine (SVM) is a powerful supervised learning algorithm used for both classification and regression tasks. The fundamental idea behind SVM is to find the optimal hyperplane that separates different classes in the feature space with maximum margin.

### Why do we use SVMs?

1. **Maximum Margin Principle**: SVMs find the hyperplane that maximizes the margin between classes, leading to better generalization
2. **Kernel Trick**: Ability to handle non-linearly separable data by mapping to higher dimensions
3. **Robust to Outliers**: The decision boundary depends only on support vectors, making it less sensitive to outliers
4. **Effective in High Dimensions**: Performs well even when the number of features exceeds the number of samples
5. **Memory Efficient**: Uses only support vectors for prediction, not the entire training dataset

## 2. Mathematical Foundation

### 2.1 Linear SVM for Binary Classification

Consider a binary classification problem with training data:
- {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
- where xᵢ ∈ ℝᵈ and yᵢ ∈ {-1, +1}

The goal is to find a hyperplane that separates the two classes:
**w^T x + b = 0**

where:
- w = weight vector (normal to the hyperplane)
- b = bias term
- x = input feature vector

### 2.2 Distance from Point to Hyperplane

The distance from a point xᵢ to the hyperplane is:
**distance = |w^T xᵢ + b| / ||w||**

For correct classification:
- If yᵢ = +1, then w^T xᵢ + b ≥ 0
- If yᵢ = -1, then w^T xᵢ + b ≤ 0

This can be combined as: **yᵢ(w^T xᵢ + b) ≥ 0**

### 2.3 Margin Definition

The margin is the minimum distance from the hyperplane to the closest data points. To maximize margin, we need to:

1. Normalize the hyperplane equation so that for the closest points: |w^T xᵢ + b| = 1
2. The margin becomes: **M = 2/||w||**

### 2.4 Hard Margin SVM Optimization Problem

To maximize margin (2/||w||), we minimize ||w||:

**Minimize: (1/2)||w||²**

**Subject to: yᵢ(w^T xᵢ + b) ≥ 1 for all i = 1, 2, ..., n**

This is a quadratic programming problem with linear constraints.

### 2.5 Lagrangian Formulation

The Lagrangian for the optimization problem:

**L(w, b, α) = (1/2)||w||² - Σᵢ₌₁ⁿ αᵢ[yᵢ(w^T xᵢ + b) - 1]**

where αᵢ ≥ 0 are Lagrange multipliers.

### 2.6 Karush-Kuhn-Tucker (KKT) Conditions

Taking partial derivatives and setting them to zero:

**∂L/∂w = 0 ⟹ w = Σᵢ₌₁ⁿ αᵢyᵢxᵢ**

**∂L/∂b = 0 ⟹ Σᵢ₌₁ⁿ αᵢyᵢ = 0**

The KKT conditions also require:
- αᵢ ≥ 0
- yᵢ(w^T xᵢ + b) - 1 ≥ 0
- αᵢ[yᵢ(w^T xᵢ + b) - 1] = 0 (complementary slackness)

### 2.7 Dual Problem

Substituting the KKT conditions back into the Lagrangian gives the dual problem:

**Maximize: Σᵢ₌₁ⁿ αᵢ - (1/2)Σᵢ₌₁ⁿΣⱼ₌₁ⁿ αᵢαⱼyᵢyⱼ(xᵢ^T xⱼ)**

**Subject to:**
- **Σᵢ₌₁ⁿ αᵢyᵢ = 0**
- **αᵢ ≥ 0 for all i**

## 3. Support Vectors

From the complementary slackness condition, αᵢ > 0 only when yᵢ(w^T xᵢ + b) = 1. These points are called **support vectors** - they lie exactly on the margin boundary and determine the hyperplane.

The decision function becomes:
**f(x) = sign(Σᵢ∈SV αᵢyᵢ(xᵢ^T x) + b)**

where SV is the set of support vector indices.

## 4. Soft Margin SVM

Real-world data is often not linearly separable. Soft margin SVM allows some misclassification by introducing slack variables ξᵢ ≥ 0:

**yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ**

The optimization problem becomes:

**Minimize: (1/2)||w||² + C Σᵢ₌₁ⁿ ξᵢ**

**Subject to:**
- **yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ**
- **ξᵢ ≥ 0**

where C is the regularization parameter controlling the trade-off between margin maximization and classification error.

### 4.1 Soft Margin Dual Problem

**Maximize: Σᵢ₌₁ⁿ αᵢ - (1/2)Σᵢ₌₁ⁿΣⱼ₌₁ⁿ αᵢαⱼyᵢyⱼ(xᵢ^T xⱼ)**

**Subject to:**
- **Σᵢ₌₁ⁿ αᵢyᵢ = 0**
- **0 ≤ αᵢ ≤ C for all i**

## 5. Kernel Methods

For non-linearly separable data, we map the input space to a higher-dimensional feature space using a mapping function φ(x).

The dual problem becomes:
**Maximize: Σᵢ₌₁ⁿ αᵢ - (1/2)Σᵢ₌₁ⁿΣⱼ₌₁ⁿ αᵢαⱼyᵢyⱼ K(xᵢ, xⱼ)**

where K(xᵢ, xⱼ) = φ(xᵢ)^T φ(xⱼ) is the kernel function.

### Common Kernel Functions:

1. **Linear Kernel**: K(xᵢ, xⱼ) = xᵢ^T xⱼ
2. **Polynomial Kernel**: K(xᵢ, xⱼ) = (γxᵢ^T xⱼ + r)ᵈ
3. **RBF (Gaussian) Kernel**: K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)
4. **Sigmoid Kernel**: K(xᵢ, xⱼ) = tanh(γxᵢ^T xⱼ + r)

## 6. Mathematical Example: Solving SVM by Hand

### Problem Setup:
Consider a 2D binary classification problem with 4 training points:
- x₁ = (1, 1), y₁ = +1
- x₂ = (2, 2), y₂ = +1  
- x₃ = (0, 0), y₃ = -1
- x₄ = (1, 0), y₄ = -1

### Step 1: Check Linear Separability
Plot the points to verify they are linearly separable. The positive points (1,1) and (2,2) should be separable from negative points (0,0) and (1,0).

### Step 2: Set up the Dual Problem
The dual optimization problem is:

**Maximize: α₁ + α₂ + α₃ + α₄ - (1/2)[α₁²(1) + α₂²(8) + α₃²(0) + α₄²(1) + 2α₁α₂(4) + 2α₁α₃(0) + 2α₁α₄(1) + 2α₂α₃(0) + 2α₂α₄(2) + 2α₃α₄(0)]**

**Subject to:**
- α₁ + α₂ - α₃ - α₄ = 0
- αᵢ ≥ 0 for all i

### Step 3: Compute Dot Products
- x₁^T x₁ = 1² + 1² = 2
- x₂^T x₂ = 2² + 2² = 8
- x₃^T x₃ = 0² + 0² = 0
- x₄^T x₄ = 1² + 0² = 1
- x₁^T x₂ = 1×2 + 1×2 = 4
- x₁^T x₃ = 1×0 + 1×0 = 0
- x₁^T x₄ = 1×1 + 1×0 = 1
- x₂^T x₃ = 2×0 + 2×0 = 0
- x₂^T x₄ = 2×1 + 2×0 = 2
- x₃^T x₄ = 0×1 + 0×0 = 0

### Step 4: Solve the Quadratic Programming Problem
Using the constraint α₁ + α₂ = α₃ + α₄ and solving the KKT conditions, we find:

Optimal solution: α₁ = 0, α₂ = 1/3, α₃ = 1/3, α₄ = 0

### Step 5: Find w and b
**w = α₂y₂x₂ + α₃y₃x₃ = (1/3)(+1)(2,2) + (1/3)(-1)(0,0) = (2/3, 2/3)**

To find b, use a support vector (x₂):
**b = y₂ - w^T x₂ = 1 - (2/3, 2/3)^T (2,2) = 1 - 8/3 = -5/3**

### Step 6: Final Hyperplane
**Decision function: f(x) = sign((2/3)x₁ + (2/3)x₂ - 5/3)**

**Hyperplane equation: (2/3)x₁ + (2/3)x₂ - 5/3 = 0**
**Simplified: 2x₁ + 2x₂ - 5 = 0**

## 7. Types of Data Suitable for SVM

### 7.1 Ideal Data Characteristics:
1. **High-dimensional data**: Text classification, gene expression data
2. **Clear margin of separation**: Data with distinct class boundaries
3. **Limited training samples**: SVMs work well with small datasets
4. **Binary classification**: Natural fit, though multiclass extensions exist

### 7.2 Data Types:
1. **Text Data**: Document classification, spam detection
2. **Image Data**: Face recognition, object detection
3. **Bioinformatics**: Gene classification, protein structure prediction
4. **Financial Data**: Credit scoring, fraud detection

### 7.3 Preprocessing Requirements:
1. **Feature Scaling**: Normalize features to similar scales
2. **Handling Categorical Data**: Convert to numerical format
3. **Dimensionality**: Works well in high dimensions but consider feature selection

## 8. Advantages and Disadvantages

### 8.1 Advantages:
1. **Effective in high dimensions**: Performs well when features > samples
2. **Memory efficient**: Uses only support vectors
3. **Versatile**: Different kernel functions for different data types
4. **Robust**: Less prone to overfitting, especially in high dimensions
5. **Global optimum**: Convex optimization guarantees global solution

### 8.2 Disadvantages:
1. **No probabilistic output**: Doesn't provide probability estimates directly
2. **Sensitive to feature scaling**: Requires preprocessing
3. **Kernel choice**: Selecting appropriate kernel and parameters can be challenging
4. **Large datasets**: Training time is O(n²) to O(n³), slow for large datasets
5. **Noisy data**: Sensitive to overlapping classes

### 8.3 Parameter Sensitivity:
1. **C parameter**: Controls trade-off between margin and misclassification
   - High C: Hard margin (low bias, high variance)
   - Low C: Soft margin (high bias, low variance)

2. **Kernel parameters**: 
   - RBF γ: Controls influence of single training example
   - Polynomial degree: Controls complexity of decision boundary

## 9. Multiclass SVM

SVMs are inherently binary classifiers. For multiclass problems:

### 9.1 One-vs-Rest (OvR):
Train k binary classifiers, one for each class against all others.

### 9.2 One-vs-One (OvO):
Train k(k-1)/2 binary classifiers for each pair of classes.

### 9.3 Decision:
- OvR: Choose class with highest decision function value
- OvO: Majority voting among all binary classifiers

## 10. SVM for Regression (SVR)

SVM can be adapted for regression by introducing an ε-insensitive loss function:

**Loss = 0 if |y - f(x)| ≤ ε, otherwise |y - f(x)| - ε**

The optimization becomes:
**Minimize: (1/2)||w||² + C Σᵢ(ξᵢ + ξᵢ*)**

**Subject to:**
- **yᵢ - w^T xᵢ - b ≤ ε + ξᵢ**
- **w^T xᵢ + b - yᵢ ≤ ε + ξᵢ***
- **ξᵢ, ξᵢ* ≥ 0**

## 11. Summary

Support Vector Machines provide a principled approach to classification and regression by maximizing the margin between classes. The key insights are:

1. The optimal hyperplane depends only on support vectors
2. The kernel trick allows handling non-linear relationships
3. The regularization parameter C controls the bias-variance trade-off
4. SVMs work particularly well for high-dimensional, small-sample problems

The mathematical foundation ensures that SVM finds globally optimal solutions for the given optimization problem, making it a robust and theoretically well-grounded machine learning algorithm.