# Decision Trees in Machine Learning - Comprehensive Notes

## 1. Introduction and Mathematical Foundation

A decision tree is a hierarchical model that makes predictions by recursively splitting the feature space into regions, where each region corresponds to a leaf node with an associated prediction. Mathematically, a decision tree can be represented as a function f: ℝᵈ → Y, where d is the dimensionality of the input space and Y is the output space.

For a dataset D = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}, where xᵢ ∈ ℝᵈ and yᵢ ∈ Y, the tree partitions the input space into disjoint regions R₁, R₂, ..., Rₘ such that:
- ⋃ᵢ₌₁ᵐ Rᵢ = ℝᵈ
- Rᵢ ∩ Rⱼ = ∅ for i ≠ j

Each region Rᵢ is associated with a prediction ĉᵢ, typically:
- **Classification**: ĉᵢ = mode{yⱼ : xⱼ ∈ Rᵢ}
- **Regression**: ĉᵢ = mean{yⱼ : xⱼ ∈ Rᵢ}

## 2. Tree Construction Algorithm

### 2.1 Greedy Recursive Splitting

The tree is built using a greedy, top-down approach that recursively splits nodes to minimize impurity. At each node t, we find the optimal split by solving:

**θ* = argmin_{θ} [p_left(θ) × I(t_left(θ)) + p_right(θ) × I(t_right(θ))]**

Where:
- θ = (j, s) represents a split on feature j at threshold s
- p_left(θ) = |D_left|/|D| and p_right(θ) = |D_right|/|D|
- I(t) is the impurity measure at node t

### 2.2 Impurity Measures

**For Classification:**

1. **Gini Impurity**: I_Gini(t) = 1 - Σᵢ₌₁ᵏ pᵢ²(t)
   - Where pᵢ(t) is the proportion of class i at node t
   - Range: [0, 1-1/k], where k is the number of classes
   - Minimum when all samples belong to one class

2. **Entropy**: I_Entropy(t) = -Σᵢ₌₁ᵏ pᵢ(t) log₂(pᵢ(t))
   - Range: [0, log₂(k)]
   - Maximum when classes are equally distributed

3. **Misclassification Error**: I_Error(t) = 1 - maxᵢ pᵢ(t)

**For Regression:**

1. **Mean Squared Error**: I_MSE(t) = (1/n_t) Σᵢ∈t (yᵢ - ȳ_t)²
   - Where ȳ_t is the mean target value at node t

2. **Mean Absolute Error**: I_MAE(t) = (1/n_t) Σᵢ∈t |yᵢ - median_t|

### 2.3 Information Gain

The quality of a split is measured by information gain:

**IG(D, θ) = I(D) - Σᵥ∈{left,right} (|Dᵥ|/|D|) × I(Dᵥ)**

Where I(D) is the impurity before the split and the second term is the weighted average impurity after the split.

## 3. Mathematical Example: Building a Classification Tree

### Dataset
Consider a binary classification problem with 8 samples:

| Sample | Feature 1 (x₁) | Feature 2 (x₂) | Class (y) |
|--------|---------------|---------------|-----------|
| 1      | 2.5           | 3.0           | A         |
| 2      | 3.0           | 4.0           | A         |
| 3      | 1.0           | 2.0           | A         |
| 4      | 4.0           | 1.0           | B         |
| 5      | 3.5           | 2.5           | B         |
| 6      | 2.0           | 1.5           | B         |
| 7      | 1.5           | 3.5           | A         |
| 8      | 4.5           | 2.0           | B         |

### Step 1: Calculate Root Node Impurity
- Total samples: n = 8
- Class A: 4 samples, Class B: 4 samples
- P(A) = P(B) = 0.5

**Gini Impurity**: I_Gini(root) = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5

**Entropy**: I_Entropy(root) = -(0.5 × log₂(0.5) + 0.5 × log₂(0.5)) = -(2 × 0.5 × (-1)) = 1.0

### Step 2: Find Best Split

**Candidate Split 1: x₁ ≤ 2.5**
- Left: {1, 3, 6, 7} → Classes: {A, A, B, A} → 3A, 1B
- Right: {2, 4, 5, 8} → Classes: {A, B, B, B} → 1A, 3B

Left node Gini: I_Gini(left) = 1 - (3/4)² - (1/4)² = 1 - 9/16 - 1/16 = 6/16 = 0.375
Right node Gini: I_Gini(right) = 1 - (1/4)² - (3/4)² = 1 - 1/16 - 9/16 = 6/16 = 0.375

Information Gain: IG = 0.5 - (4/8 × 0.375 + 4/8 × 0.375) = 0.5 - 0.375 = 0.125

**Candidate Split 2: x₂ ≤ 2.5**
- Left: {4, 5, 6, 8} → Classes: {B, B, B, B} → 0A, 4B
- Right: {1, 2, 3, 7} → Classes: {A, A, A, A} → 4A, 0B

Left node Gini: I_Gini(left) = 1 - (0/4)² - (4/4)² = 1 - 0 - 1 = 0
Right node Gini: I_Gini(right) = 1 - (4/4)² - (0/4)² = 1 - 1 - 0 = 0

Information Gain: IG = 0.5 - (4/8 × 0 + 4/8 × 0) = 0.5 - 0 = 0.5

**Best Split**: x₂ ≤ 2.5 with IG = 0.5 (perfect split!)

### Step 3: Tree Structure
```
Root: x₂ ≤ 2.5?
├── True: Class B (4 samples, all B)
└── False: Class A (4 samples, all A)
```

## 4. Types of Data and Application Conditions

### 4.1 Suitable Data Types

**Numerical Features:**
- Continuous variables (age, income, temperature)
- Discrete numerical variables (number of children, ratings)
- Trees handle these by finding optimal thresholds

**Categorical Features:**
- Nominal variables (color, brand, location)
- Ordinal variables (education level, satisfaction rating)
- Binary splits for nominal: feature ∈ {subset} vs feature ∉ {subset}
- Natural ordering splits for ordinal variables

**Mixed Data:**
- Decision trees naturally handle datasets with both numerical and categorical features
- No need for extensive preprocessing or encoding

### 4.2 Optimal Application Conditions

**Dataset Characteristics:**
1. **Non-linear relationships**: Trees excel when the relationship between features and target is non-linear
2. **Interaction effects**: Can naturally capture feature interactions without explicit feature engineering
3. **Interpretability requirements**: When model explainability is crucial
4. **Heterogeneous data**: Mixed data types with different scales

**Problem Types:**
- Binary and multi-class classification
- Regression problems
- Feature selection and importance ranking
- Missing value handling (surrogate splits)

**Scenarios:**
- Medical diagnosis systems
- Credit approval decisions
- Marketing campaign targeting
- Risk assessment models
- Rule extraction from complex systems

## 5. Advantages and Disadvantages

### 5.1 Advantages

**Interpretability:**
- Visual representation as flowcharts
- Easy to explain decisions to stakeholders
- Can extract explicit rules: IF-THEN statements

**No Assumptions:**
- No assumption about data distribution
- No assumption about linear relationships
- Robust to outliers in feature space

**Preprocessing Minimal:**
- Handles missing values through surrogate splits
- No need for feature scaling or normalization
- Natural handling of categorical variables

**Computational Efficiency:**
- Fast prediction: O(log n) for balanced trees
- Parallelizable training algorithms
- Memory efficient for sparse datasets

**Feature Selection:**
- Automatic feature selection during splitting
- Provides feature importance measures
- Handles irrelevant features naturally

### 5.2 Disadvantages

**Overfitting Tendency:**
- High variance, especially with deep trees
- Can memorize training data
- Poor generalization without regularization

**Bias Issues:**
- Greedy algorithm may not find globally optimal tree
- Bias toward features with more possible splits
- Instability: small data changes can drastically change tree structure

**Limited Expressiveness:**
- Axis-parallel splits only (standard trees)
- Difficulty with linear relationships
- Cannot capture smooth functions well

**Class Imbalance:**
- May create biased trees toward majority class
- Splitting criteria may not handle imbalanced data optimally

## 6. Addressing Overfitting

### 6.1 Pre-pruning (Early Stopping)

**Maximum Depth Control:**
- Limit tree depth: typically 3-10 for interpretability
- Mathematical constraint: depth ≤ d_max

**Minimum Samples per Node:**
- Stop splitting if n_samples < min_samples_split
- Typical values: 2-20 depending on dataset size

**Minimum Information Gain:**
- Stop if IG(split) < threshold
- Prevents splits that don't significantly improve purity

**Maximum Leaf Nodes:**
- Limit total number of leaves: |leaves| ≤ max_leaves
- Controls overall tree complexity

### 6.2 Post-pruning

**Cost Complexity Pruning (α-pruning):**

Minimize: Cost_α(T) = Error(T) + α × |leaves(T)|

Where:
- Error(T) = Σᵢ₌₁|ˡᵉᵃᵛᵉˢ| n_i × I(i) / n_total
- α is the complexity parameter
- |leaves(T)| is the number of leaf nodes

**Algorithm:**
1. Grow full tree T₀
2. For each internal node t, calculate: α_t = [Error(prune(t)) - Error(t)] / (|leaves(t)| - 1)
3. Prune node with smallest α_t
4. Repeat until only root remains
5. Use cross-validation to select optimal α

**Reduced Error Pruning:**
1. Split data into training and pruning sets
2. Build tree on training set
3. For each internal node, test if pruning improves validation error
4. Prune if validation error decreases or remains same

### 6.3 Ensemble Methods

**Random Forests:**
- Build multiple trees with bootstrap sampling
- Feature bagging: √d features per split
- Reduces variance through averaging

**Gradient Boosting:**
- Sequential tree building
- Each tree corrects previous ensemble errors
- Mathematical formulation:
  F_m(x) = F_{m-1}(x) + γ_m h_m(x)
  Where h_m(x) is trained on pseudo-residuals

## 7. Addressing Underfitting

### 7.1 Increase Model Complexity

**Deeper Trees:**
- Increase max_depth parameter
- Allow more granular partitioning
- Risk: May lead to overfitting

**Lower Stopping Criteria:**
- Reduce min_samples_split
- Reduce min_samples_leaf
- Allow smaller information gains

### 7.2 Feature Engineering

**Feature Interactions:**
- Create polynomial features: x₁ × x₂, x₁²
- Domain-specific feature combinations
- Binning continuous variables

**Feature Transformation:**
- Log transforms for skewed distributions
- Scaling for better split threshold selection
- Domain knowledge incorporation

### 7.3 Alternative Splitting Criteria

**Oblique Trees:**
- Linear combinations as split functions
- Split: w₁x₁ + w₂x₂ + ... + wₐxₐ ≤ threshold
- More expressive than axis-parallel splits

**Multivariate Trees:**
- Multiple features in splitting decisions
- Can capture linear relationships better
- Higher computational complexity

## 8. Advanced Mathematical Concepts

### 8.1 Bias-Variance Decomposition

For a decision tree predictor ĥ(x), the expected prediction error can be decomposed as:

**E[(y - ĥ(x))²] = σ² + Bias²(ĥ(x)) + Var(ĥ(x))**

Where:
- **Noise**: σ² = E[(y - f(x))²] (irreducible error)
- **Bias²**: [f(x) - E[ĥ(x)]]² (systematic error)
- **Variance**: E[(ĥ(x) - E[ĥ(x)])²] (prediction variability)

**Decision Tree Characteristics:**
- **High Variance**: Small changes in training data → large changes in tree structure
- **Low Bias**: Can fit complex patterns with sufficient depth
- **Trade-off**: Pruning increases bias but reduces variance

### 8.2 VC Dimension and Generalization

The VC dimension of decision trees with d features and maximum depth h is:

**VC_dim ≤ O(d × h × log(d × h))**

**Generalization bound** (with probability 1-δ):
R(h) ≤ R_emp(h) + √[(VC_dim × log(2n/VC_dim) + log(4/δ)) / (2n)]

Where:
- R(h) is true risk
- R_emp(h) is empirical risk
- n is training set size

### 8.3 Information Theory Foundation

**Mutual Information between feature X and target Y:**
I(X; Y) = Σₓ Σᵧ p(x,y) log(p(x,y)/(p(x)p(y)))

**Conditional Mutual Information:**
For split S on feature X:
I(Y; X|S) = Σₛ p(s) I(Y; X|X ∈ s)

**Optimal split maximizes:**
I(Y; X|S) = H(Y) - H(Y|S)

Where H(Y|S) = Σₛ p(s) H(Y|X ∈ s)

### 8.4 Statistical Significance Testing

**Chi-square test for split quality:**
χ² = Σᵢ Σⱼ (Oᵢⱼ - Eᵢⱼ)²/Eᵢⱼ

Where:
- Oᵢⱼ = observed frequency of class j in child i
- Eᵢⱼ = expected frequency under independence assumption
- df = (number of children - 1) × (number of classes - 1)

**P-value threshold**: Typically p < 0.05 to accept split

## 9. Hyperparameter Optimization

### 9.1 Key Hyperparameters

**Tree Structure:**
- max_depth: [3, 5, 7, 10, 15, None]
- min_samples_split: [2, 5, 10, 20]
- min_samples_leaf: [1, 2, 5, 10]
- max_features: ['sqrt', 'log2', None, 0.5, 0.8]

**Splitting Criteria:**
- criterion: ['gini', 'entropy', 'log_loss']
- splitter: ['best', 'random']
- max_leaf_nodes: [None, 10, 20, 50, 100]

### 9.2 Optimization Strategies

**Grid Search:**
- Exhaustive search over parameter grid
- Cross-validation for each combination
- Computationally expensive but thorough

**Random Search:**
- Random sampling from parameter distributions
- Often more efficient than grid search
- Good for high-dimensional parameter spaces

**Bayesian Optimization:**
- Uses probabilistic model of objective function
- Balances exploration vs exploitation
- Efficient for expensive evaluations

## 10. Practical Considerations

### 10.1 Handling Imbalanced Data

**Weighted Splits:**
- Modify impurity measures with class weights
- Weight inversely proportional to class frequency
- Gini_weighted = Σᵢ wᵢ pᵢ(1 - pᵢ)

**Balanced Sampling:**
- Stratified sampling for balanced training sets
- SMOTE for synthetic minority oversampling
- Cost-sensitive learning approaches

### 10.2 Missing Value Treatment

**Surrogate Splits:**
- Find alternative features that mimic primary split
- Correlation-based surrogate selection
- Maintains tree structure with missing data

**Imputation Strategies:**
- Mean/mode imputation for numerical/categorical
- Predictive imputation using other features
- Multiple imputation for uncertainty quantification

### 10.3 Feature Importance Calculation

**Impurity-based Importance:**
Importance(j) = Σₜ (pₜ × ΔI(t, j))

Where:
- pₜ = proportion of samples reaching node t
- ΔI(t, j) = information gain from splitting on feature j at node t

**Permutation Importance:**
1. Calculate baseline accuracy on validation set
2. Randomly permute feature j values
3. Recalculate accuracy
4. Importance = baseline_accuracy - permuted_accuracy

## 11. Model Evaluation Metrics

### 11.1 Classification Metrics

**Accuracy**: (TP + TN) / (TP + TN + FP + FN)
**Precision**: TP / (TP + FP)
**Recall**: TP / (TP + FN)
**F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

**AUC-ROC**: Area under ROC curve
- ROC plots TPR vs FPR at various thresholds
- Good for binary classification evaluation

### 11.2 Regression Metrics

**Mean Squared Error**: MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²
**Root Mean Squared Error**: RMSE = √MSE
**Mean Absolute Error**: MAE = (1/n) Σᵢ |yᵢ - ŷᵢ|
**R²**: 1 - (SS_res / SS_tot)

## 12. Conclusion

Decision trees provide an intuitive, interpretable approach to machine learning with strong theoretical foundations. Their ability to handle mixed data types and non-linear relationships makes them versatile tools. However, careful attention to overfitting through pruning and ensemble methods is essential for optimal performance. The mathematical framework underlying decision trees connects information theory, statistical learning theory, and optimization, providing rich theoretical insights into their behavior and limitations.

Understanding the bias-variance trade-off, appropriate regularization techniques, and proper evaluation methodologies ensures effective application of decision trees across diverse problem domains.