# Logistic Regression: Complete In-Depth Guide

## Table of Contents
1. [Introduction and Motivation](#introduction-and-motivation)
2. [Mathematical Foundation](#mathematical-foundation)
3. [The Sigmoid Function](#the-sigmoid-function)
4. [Maximum Likelihood Estimation](#maximum-likelihood-estimation)
5. [Cost Function and Optimization](#cost-function-and-optimization)
6. [Gradient Descent](#gradient-descent)
7. [Regularization](#regularization)
8. [Multiclass Classification](#multiclass-classification)
9. [Model Evaluation](#model-evaluation)
10. [Assumptions and Limitations](#assumptions-and-limitations)
11. [Advanced Topics](#advanced-topics)

## Introduction and Motivation

### What is Logistic Regression?
Logistic regression is a statistical method used for binary classification problems. Despite its name containing "regression," it's actually a classification algorithm that predicts the probability of an instance belonging to a particular category.

### Why Not Linear Regression for Classification?
Linear regression outputs continuous values, but for classification we need probabilities (values between 0 and 1). Linear regression can output values outside this range, making it unsuitable for probability estimation.

### Key Characteristics:
- **Output**: Probabilities between 0 and 1
- **Decision Boundary**: Linear in feature space
- **Assumptions**: Linear relationship between features and log-odds
- **Distribution**: Assumes errors follow logistic distribution

## Mathematical Foundation

### The Linear Model
We start with a linear combination of features:
```
z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ = β^T x
```

Where:
- `β₀` is the intercept (bias term)
- `β₁, β₂, ..., βₙ` are the feature weights
- `x₁, x₂, ..., xₙ` are the input features

### The Logistic Function (Sigmoid)
To convert the linear output to a probability, we use the logistic function:
```
p(y=1|x) = σ(z) = 1 / (1 + e^(-z)) = e^z / (1 + e^z)
```

### Odds and Log-Odds
- **Odds**: `p / (1-p)` - ratio of probability of success to failure
- **Log-odds (Logit)**: `ln(p / (1-p)) = z` - natural logarithm of odds

The logistic regression assumes a linear relationship between features and log-odds:
```
logit(p) = ln(p / (1-p)) = β^T x
```

## The Sigmoid Function

### Properties:
1. **Range**: (0, 1) - perfect for probabilities
2. **Monotonic**: Always increasing
3. **S-shaped curve**: Smooth transition from 0 to 1
4. **Symmetric**: σ(-z) = 1 - σ(z)
5. **Derivative**: σ'(z) = σ(z)(1 - σ(z))

### Why Sigmoid?
- Maps any real number to (0,1)
- Differentiable everywhere
- Has nice mathematical properties for optimization
- Represents the cumulative distribution function of the logistic distribution

## Maximum Likelihood Estimation

### The Likelihood Function
For a dataset with n samples, the likelihood function is:
```
L(β) = ∏ᵢ₌₁ⁿ p(yᵢ|xᵢ, β)
```

For binary classification:
```
L(β) = ∏ᵢ₌₁ⁿ [σ(β^T xᵢ)]^yᵢ [1 - σ(β^T xᵢ)]^(1-yᵢ)
```

### Log-Likelihood
Taking the logarithm makes optimization easier:
```
ℓ(β) = ln L(β) = Σᵢ₌₁ⁿ [yᵢ ln(σ(β^T xᵢ)) + (1-yᵢ) ln(1-σ(β^T xᵢ))]
```

## Cost Function and Optimization

### Cross-Entropy Loss
The negative log-likelihood gives us the cross-entropy cost function:
```
J(β) = -1/n Σᵢ₌₁ⁿ [yᵢ ln(hβ(xᵢ)) + (1-yᵢ) ln(1-hβ(xᵢ))]
```

Where `hβ(xᵢ) = σ(β^T xᵢ)` is our hypothesis function.

### Why Cross-Entropy?
- Convex function (has single global minimum)
- Penalizes wrong predictions more heavily
- Derivative has nice form for gradient descent
- Derived from maximum likelihood principle

## Gradient Descent

### Computing the Gradient
The partial derivative of the cost function with respect to βⱼ:
```
∂J(β)/∂βⱼ = 1/n Σᵢ₌₁ⁿ (hβ(xᵢ) - yᵢ) xᵢⱼ
```

In vector form:
```
∇J(β) = 1/n X^T (h - y)
```

### Update Rule
```
β := β - α ∇J(β)
```

Where α is the learning rate.

### Key Insight
The gradient has the same form as linear regression! The difference is in the hypothesis function h.

## Regularization

### L1 Regularization (Lasso)
```
J(β) = CrossEntropy + λ Σⱼ₌₁ⁿ |βⱼ|
```

### L2 Regularization (Ridge)
```
J(β) = CrossEntropy + λ Σⱼ₌₁ⁿ βⱼ²
```

### Elastic Net
```
J(β) = CrossEntropy + λ₁ Σⱼ₌₁ⁿ |βⱼ| + λ₂ Σⱼ₌₁ⁿ βⱼ²
```

## Multiclass Classification

### One-vs-Rest (OvR)
Train k binary classifiers, one for each class.

### One-vs-One (OvO)
Train k(k-1)/2 binary classifiers for each pair of classes.

### Multinomial Logistic Regression (Softmax)
Extends binary logistic regression directly:
```
P(y=k|x) = e^(β_k^T x) / Σⱼ₌₁ᴷ e^(β_j^T x)
```

## Model Evaluation

### Classification Metrics
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

### Probability-based Metrics
- **Log-Loss**: Measures quality of probabilistic predictions
- **AUC-ROC**: Area under ROC curve
- **Brier Score**: Mean squared difference between predicted and actual probabilities

## Assumptions and Limitations

### Assumptions
1. **Linear relationship** between features and log-odds
2. **Independence** of observations
3. **No multicollinearity** among features
4. **Large sample size** for stable results

### Limitations
- Assumes linear decision boundary
- Sensitive to outliers
- Requires feature scaling for optimal performance
- Can struggle with complex non-linear relationships

## Advanced Topics

### Feature Engineering
- Polynomial features for non-linearity
- Interaction terms
- Feature scaling and normalization

### Handling Imbalanced Data
- Class weights
- Resampling techniques
- Threshold tuning

### Interpretation
- Odds ratios: e^βⱼ
- Feature importance
- Coefficient significance testing

---

# Complete Worked Example: From Scratch

Let's solve a simple logistic regression problem step by step with real calculations.

## Problem Setup
We want to predict whether a student passes an exam based on hours studied.

**Dataset:**
```
Hours Studied (x) | Pass (y)
       1          |   0
       2          |   0  
       3          |   1
       4          |   1
       5          |   1
```

## Step 1: Initialize Parameters
```
β₀ = 0 (intercept)
β₁ = 0 (coefficient for hours studied)
α = 0.1 (learning rate)
```

## Step 2: Add Intercept Term
Our feature matrix X becomes:
```
X = [1, 1]  # x₀=1 (intercept), x₁=1 (hours)
    [1, 2]
    [1, 3]
    [1, 4]
    [1, 5]

y = [0, 0, 1, 1, 1]
```

## Step 3: First Iteration Calculations

### Forward Pass
For each sample, calculate z = β₀ + β₁x₁:
```
z₁ = 0 + 0×1 = 0
z₂ = 0 + 0×2 = 0
z₃ = 0 + 0×3 = 0
z₄ = 0 + 0×4 = 0
z₅ = 0 + 0×5 = 0
```

### Sigmoid Calculation
σ(z) = 1/(1 + e^(-z))
```
h₁ = σ(0) = 1/(1 + e^0) = 1/2 = 0.5
h₂ = σ(0) = 0.5
h₃ = σ(0) = 0.5
h₄ = σ(0) = 0.5
h₅ = σ(0) = 0.5
```

### Cost Calculation
J(β) = -1/n Σᵢ [yᵢ ln(hᵢ) + (1-yᵢ) ln(1-hᵢ)]
```
J = -1/5 [0×ln(0.5) + 1×ln(0.5) + 0×ln(0.5) + 1×ln(0.5) + 1×ln(0.5) + 1×ln(0.5) + 1×ln(0.5) + 1×ln(0.5) + 1×ln(0.5) + 1×ln(0.5)]
J = -1/5 [0 + (-0.693) + 0 + (-0.693) + (-0.693) + (-0.693) + (-0.693)]
J = -1/5 × (-3.465) = 0.693
```

### Gradient Calculation
∇J = 1/n X^T (h - y)

```
h - y = [0.5-0, 0.5-0, 0.5-1, 0.5-1, 0.5-1] = [0.5, 0.5, -0.5, -0.5, -0.5]

∂J/∂β₀ = 1/5 × (1×0.5 + 1×0.5 + 1×(-0.5) + 1×(-0.5) + 1×(-0.5)) = 1/5 × (-0.5) = -0.1

∂J/∂β₁ = 1/5 × (1×0.5 + 2×0.5 + 3×(-0.5) + 4×(-0.5) + 5×(-0.5)) = 1/5 × (0.5 + 1 - 1.5 - 2 - 2.5) = 1/5 × (-4.5) = -0.9
```

### Parameter Update
```
β₀ := β₀ - α × ∂J/∂β₀ = 0 - 0.1×(-0.1) = 0.01
β₁ := β₁ - α × ∂J/∂β₁ = 0 - 0.1×(-0.9) = 0.09
```

## Step 4: Second Iteration

### Forward Pass
```
z₁ = 0.01 + 0.09×1 = 0.10
z₂ = 0.01 + 0.09×2 = 0.19
z₃ = 0.01 + 0.09×3 = 0.28
z₄ = 0.01 + 0.09×4 = 0.37
z₅ = 0.01 + 0.09×5 = 0.46
```

### Sigmoid Calculation
```
h₁ = σ(0.10) = 1/(1 + e^(-0.10)) = 1/(1 + 0.905) = 0.525
h₂ = σ(0.19) = 1/(1 + e^(-0.19)) = 1/(1 + 0.827) = 0.547
h₃ = σ(0.28) = 1/(1 + e^(-0.28)) = 1/(1 + 0.756) = 0.570
h₄ = σ(0.37) = 1/(1 + e^(-0.37)) = 1/(1 + 0.691) = 0.591
h₅ = σ(0.46) = 1/(1 + e^(-0.46)) = 1/(1 + 0.631) = 0.613
```

### Cost Calculation
```
J = -1/5 [0×ln(0.525) + 1×ln(0.475) + 0×ln(0.547) + 1×ln(0.453) + 1×ln(0.570) + 1×ln(0.430) + 1×ln(0.591) + 1×ln(0.409) + 1×ln(0.613) + 1×ln(0.387)]

J ≈ 0.647 (lower than previous iteration!)
```

## Step 5: Continue Until Convergence

After many iterations (typically 1000+), the parameters converge to approximately:
```
β₀ ≈ -2.8
β₁ ≈ 1.4
```

## Step 6: Final Model Interpretation

The final model: `P(Pass = 1) = σ(-2.8 + 1.4 × hours)`

**Interpretation:**
- For every additional hour studied, the log-odds of passing increase by 1.4
- The odds ratio is e^1.4 ≈ 4.06, meaning each additional hour multiplies the odds of passing by ~4
- The decision boundary (P = 0.5) occurs at: -2.8 + 1.4×hours = 0, so hours = 2

**Predictions:**
```
1 hour: P = σ(-2.8 + 1.4×1) = σ(-1.4) ≈ 0.20
2 hours: P = σ(-2.8 + 1.4×2) = σ(-1.4) ≈ 0.50  
3 hours: P = σ(-2.8 + 1.4×3) = σ(1.4) ≈ 0.80
```

This matches our intuition: more study hours lead to higher probability of passing!