# Linear Regression Study Notes - Day [Date]

## 📚 Reading Overview

- **Primary**: ISLR Chapter 3.1-3.2 (Pages 59-82)
- **Algorithm Focus**: ESL Chapter 3.2 (Pages 43-55)
- **Math Deep Dive**: MML Chapter 9.1-9.2 (Pages 255-275)

---

## 🎯 Key Learning Objectives

- [ ] Understand simple and multiple linear regression
- [ ] Master least squares estimation
- [ ] Derive and interpret normal equations
- [ ] Grasp geometric interpretation of linear regression
- [ ] Explore probabilistic perspective of linear regression
- [ ] Connect optimization, geometry, and probability viewpoints

---

## 📖 ISLR Chapter 3.1-3.2: Linear Regression Fundamentals

### Simple Linear Regression - Deep Dive

#### Model Specification

- **Population Model**: $Y = \beta_0 + \beta_1X + \epsilon$
- **Sample Model**: $y_i = \beta_0 + \beta_1x_i + \epsilon_i$ for $i = 1, ..., n$

#### Key Assumptions

1. **Linearity**: Relationship between X and Y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: $\text{Var}(\epsilon_i) = \sigma^2$ (constant)
4. **Normality**: $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ (for inference)

#### Parameter Interpretation

- $\beta_0$ (intercept): Expected value of Y when X = 0
- $\beta_1$ (slope): Expected change in Y for one-unit increase in X
- $\epsilon$: Random error capturing unmeasured factors

#### Least Squares Estimation - Detailed Derivation

**Objective Function**: $$RSS(\beta_0, \beta_1) = \sum_{i=1}^{n}(y_i - \beta_0 - \beta_1x_i)^2$$

**Minimization**: Taking partial derivatives and setting to zero:

$$\frac{\partial RSS}{\partial \beta_0} = -2\sum_{i=1}^{n}(y_i - \beta_0 - \beta_1x_i) = 0$$

$$\frac{\partial RSS}{\partial \beta_1} = -2\sum_{i=1}^{n}x_i(y_i - \beta_0 - \beta_1x_i) = 0$$

**Normal Equations**: From the first equation: $n\beta_0 + \beta_1\sum_{i=1}^{n}x_i = \sum_{i=1}^{n}y_i$

This gives us: $\beta_0 = \bar{y} - \beta_1\bar{x}$

Substituting into the second equation: $$\sum_{i=1}^{n}x_i(y_i - (\bar{y} - \beta_1\bar{x}) - \beta_1x_i) = 0$$

**Final Estimates**: $$\hat{\beta_1} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}}$$

$$\hat{\beta_0} = \bar{y} - \hat{\beta_1}\bar{x}$$

Where:

- $S_{xy} = \sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$
- $S_{xx} = \sum_{i=1}^{n}(x_i - \bar{x})^2$

#### Statistical Properties

**Fitted Values**: $\hat{y_i} = \hat{\beta_0} + \hat{\beta_1}x_i$

**Residuals**: $e_i = y_i - \hat{y_i}$

**Residual Sum of Squares**: $RSS = \sum_{i=1}^{n}e_i^2$

**Total Sum of Squares**: $TSS = \sum_{i=1}^{n}(y_i - \bar{y})^2$

**Explained Sum of Squares**: $ESS = \sum_{i=1}^{n}(\hat{y_i} - \bar{y})^2$

**Key Identity**: $TSS = ESS + RSS$

**R-squared**: $R^2 = \frac{ESS}{TSS} = 1 - \frac{RSS}{TSS}$

#### Standard Errors and Inference

**Variance of Error**: $\hat{\sigma^2} = \frac{RSS}{n-2}$

**Standard Errors**: $$SE(\hat{\beta_1}) = \hat{\sigma}\sqrt{\frac{1}{S_{xx}}}$$

$$SE(\hat{\beta_0}) = \hat{\sigma}\sqrt{\frac{1}{n} + \frac{\bar{x}^2}{S_{xx}}}$$

**Confidence Intervals**:

- $\hat{\beta_1} \pm t_{n-2,\alpha/2} \cdot SE(\hat{\beta_1})$
- $\hat{\beta_0} \pm t_{n-2,\alpha/2} \cdot SE(\hat{\beta_0})$

**Hypothesis Testing**:

- $H_0: \beta_1 = 0$ vs $H_1: \beta_1 \neq 0$
- Test statistic: $t = \frac{\hat{\beta_1}}{SE(\hat{\beta_1})} \sim t_{n-2}$

### Multiple Linear Regression - Comprehensive Treatment

#### Model Specification

**Matrix Form**: $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$

Where:

- $\mathbf{y} = \begin{pmatrix} y_1 \ y_2 \ \vdots \ y_n \end{pmatrix}$ (n×1 response vector)
    
- $\mathbf{X} = \begin{pmatrix} 1 & x_{11} & x_{12} & \cdots & x_{1p} \ 1 & x_{21} & x_{22} & \cdots & x_{2p} \ \vdots & \vdots & \vdots & \ddots & \vdots \ 1 & x_{n1} & x_{n2} & \cdots & x_{np} \end{pmatrix}$ (n×(p+1) design matrix)
    
- $\boldsymbol{\beta} = \begin{pmatrix} \beta_0 \ \beta_1 \ \vdots \ \beta_p \end{pmatrix}$ ((p+1)×1 parameter vector)
    
- $\boldsymbol{\epsilon} = \begin{pmatrix} \epsilon_1 \ \epsilon_2 \ \vdots \ \epsilon_n \end{pmatrix}$ (n×1 error vector)
    

#### Interpretation of Coefficients

- $\beta_j$: Expected change in Y for one-unit increase in $X_j$, **holding all other variables constant**
- This is crucial - interpretation changes with multiple variables due to partial effects

#### Model Assessment

**F-Statistic**: Tests $H_0: \beta_1 = \beta_2 = ... = \beta_p = 0$ $$F = \frac{(TSS - RSS)/p}{RSS/(n-p-1)}$$

**Adjusted R-squared**: $R^2_{adj} = 1 - \frac{RSS/(n-p-1)}{TSS/(n-1)}$

- Penalizes for additional variables
- Can decrease when adding irrelevant variables

---

## 🔬 ESL Chapter 3.2: Mathematical Foundations - Extended Analysis

### Normal Equations - Complete Derivation

#### Matrix Calculus Approach

**Objective**: Minimize $J(\boldsymbol{\beta}) = ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2$

**Expanded Form**: $$J(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$ $$= \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$$

**Taking the Gradient**: $$\nabla_{\boldsymbol{\beta}} J(\boldsymbol{\beta}) = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$$

**Setting Equal to Zero**: $$\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$$

**Solution** (assuming $\mathbf{X}^T\mathbf{X}$ is invertible): $$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

#### Properties of the Solution

**Uniqueness**: Solution is unique if $\text{rank}(\mathbf{X}) = p+1$

**Invertibility Condition**: $(\mathbf{X}^T\mathbf{X})^{-1}$ exists iff columns of $\mathbf{X}$ are linearly independent

**When Rank is Deficient**:

- Multiple solutions exist
- Use pseudoinverse: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^+\mathbf{X}^T\mathbf{y}$

### Geometric Interpretation - Detailed Analysis

#### Vector Space Framework

- **Column Space**: $\mathcal{C}(\mathbf{X}) = {\mathbf{X}\boldsymbol{\beta} : \boldsymbol{\beta} \in \mathbb{R}^{p+1}}$
- **Null Space**: $\mathcal{N}(\mathbf{X}) = {\boldsymbol{\beta} : \mathbf{X}\boldsymbol{\beta} = \mathbf{0}}$

#### Orthogonal Projection

**Projection Matrix**: $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$

**Properties of H**:

1. **Symmetric**: $\mathbf{H}^T = \mathbf{H}$
2. **Idempotent**: $\mathbf{H}^2 = \mathbf{H}$
3. **Projects onto** $\mathcal{C}(\mathbf{X})$

**Fitted Values**: $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$

- $\hat{\mathbf{y}}$ is the orthogonal projection of $\mathbf{y}$ onto $\mathcal{C}(\mathbf{X})$

**Residuals**: $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = (\mathbf{I} - \mathbf{H})\mathbf{y}$

- $\mathbf{e} \perp \mathcal{C}(\mathbf{X})$
- $\mathbf{X}^T\mathbf{e} = \mathbf{0}$ (orthogonality condition)

#### Geometric Insights

- **Minimization**: Finding point in $\mathcal{C}(\mathbf{X})$ closest to $\mathbf{y}$
- **Pythagorean Theorem**: $||\mathbf{y}||^2 = ||\hat{\mathbf{y}}||^2 + ||\mathbf{e}||^2$
- **Best Approximation**: $\hat{\mathbf{y}}$ minimizes $||\mathbf{y} - \mathbf{z}||$ over all $\mathbf{z} \in \mathcal{C}(\mathbf{X})$

### Statistical Properties - Comprehensive Analysis

#### Unbiasedness

**Proof**: $$E[\hat{\boldsymbol{\beta}}] = E[(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}]$$ $$= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T E[\mathbf{y}]$$ $$= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \mathbf{X}\boldsymbol{\beta} = \boldsymbol{\beta}$$

#### Variance-Covariance Matrix

$$\text{Var}(\hat{\boldsymbol{\beta}}) = \text{Var}[(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}]$$ $$= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \text{Var}(\mathbf{y}) \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}$$ $$= \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$$

**Diagonal Elements**: $\text{Var}(\hat{\beta_j}) = \sigma^2[(\mathbf{X}^T\mathbf{X})^{-1}]_{jj}$

**Off-diagonal Elements**: $\text{Cov}(\hat{\beta_i}, \hat{\beta_j}) = \sigma^2[(\mathbf{X}^T\mathbf{X})^{-1}]_{ij}$

#### Gauss-Markov Theorem

**Statement**: Under assumptions of linearity, independence, homoscedasticity, OLS estimator is BLUE (Best Linear Unbiased Estimator)

**Proof Sketch**:

1. Consider any other linear unbiased estimator $\tilde{\boldsymbol{\beta}} = \mathbf{A}\mathbf{y}$
2. Unbiasedness requires $\mathbf{A}\mathbf{X} = \mathbf{I}$
3. Show $\text{Var}(\tilde{\boldsymbol{\beta}}) - \text{Var}(\hat{\boldsymbol{\beta}})$ is positive semidefinite

---

## 🎲 MML Chapter 9.1-9.2: Probabilistic Perspective - In-Depth Analysis

### Probabilistic Model Foundation

#### Likelihood Function

**Joint Likelihood**: $$L(\boldsymbol{\beta}, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \mathbf{x_i}^T\boldsymbol{\beta})^2}{2\sigma^2}\right)$$

**Matrix Form**: $$L(\boldsymbol{\beta}, \sigma^2) = (2\pi\sigma^2)^{-n/2} \exp\left(-\frac{1}{2\sigma^2}||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2\right)$$

**Log-Likelihood**: $$\ell(\boldsymbol{\beta}, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2$$

#### Maximum Likelihood Estimation

**MLE for β**: $$\frac{\partial \ell}{\partial \boldsymbol{\beta}} = \frac{1}{\sigma^2}\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = 0$$

This gives: $\hat{\boldsymbol{\beta}}_{MLE} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ (same as OLS!)

**MLE for σ²**: $$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{||\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}||^2}{2(\sigma^2)^2} = 0$$

This gives: $\hat{\sigma^2}_{MLE} = \frac{1}{n}||\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}||^2$

**Note**: MLE for σ² is biased; unbiased estimator uses $(n-p-1)$ instead of $n$

#### Distribution of Estimators

**Exact Distribution** (under normality): $$\hat{\boldsymbol{\beta}} \sim \mathcal{N}(\boldsymbol{\beta}, \sigma^2(\mathbf{X}^T\mathbf{X})^{-1})$$

$$\frac{(n-p-1)\hat{\sigma^2}}{\sigma^2} \sim \chi^2_{n-p-1}$$

**Independence**: $\hat{\boldsymbol{\beta}}$ and $\hat{\sigma^2}$ are independent

### Bayesian Linear Regression

#### Prior Specification

**Conjugate Prior**: $$\boldsymbol{\beta} | \sigma^2 \sim \mathcal{N}(\boldsymbol{\mu_0}, \sigma^2\mathbf{V_0})$$ $$\sigma^2 \sim \text{InvGamma}(a_0, b_0)$$

#### Posterior Derivation

**Posterior for β** (given σ²): $$\boldsymbol{\beta} | \mathbf{y}, \sigma^2 \sim \mathcal{N}(\boldsymbol{\mu_n}, \sigma^2\mathbf{V_n})$$

Where:

- $\mathbf{V_n} = (\mathbf{V_0}^{-1} + \mathbf{X}^T\mathbf{X})^{-1}$
- $\boldsymbol{\mu_n} = \mathbf{V_n}(\mathbf{V_0}^{-1}\boldsymbol{\mu_0} + \mathbf{X}^T\mathbf{y})$

**Limiting Cases**:

1. **Uninformative prior** ($\mathbf{V_0} \to \infty\mathbf{I}$): Posterior mean → MLE
2. **Strong prior**: Posterior shrinks toward prior mean

#### MAP Estimation

**Maximum A Posteriori**: $$\hat{\boldsymbol{\beta}}_{MAP} = \arg\max_{\boldsymbol{\beta}} p(\boldsymbol{\beta}|\mathbf{y}) = \boldsymbol{\mu_n}$$

**Connection to Regularization**:

- Ridge regression emerges from Gaussian prior
- Lasso regression from Laplace prior
- Shows regularization as Bayesian inference

### Predictive Distribution

#### Point Prediction

**New observation** $x__$: $$\hat{y__} = \mathbf{x_*}^T\hat{\boldsymbol{\beta}}$$

#### Predictive Uncertainty

**Prediction Variance**: $$\text{Var}(y_* | \mathbf{x__}) = \sigma^2(1 + \mathbf{x__}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x_*})$$

**Components**:

1. $\sigma^2$: Irreducible error
2. $\sigma^2\mathbf{x__}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x__}$: Parameter uncertainty

**Prediction Interval**: $$\hat{y__} \pm t_{n-p-1,\alpha/2} \sqrt{\hat{\sigma^2}(1 + \mathbf{x__}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x_*})}$$

---

## 🔗 Deep Connections & Unified Understanding

### The Trinity of Perspectives

#### Optimization View

- **Problem**: $\min_{\boldsymbol{\beta}} ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2$
- **Solution**: Normal equations
- **Interpretation**: Best fit in L2 sense

#### Geometric View

- **Problem**: Project $\mathbf{y}$ onto $\mathcal{C}(\mathbf{X})$
- **Solution**: Orthogonal projection
- **Interpretation**: Closest point in subspace

#### Probabilistic View

- **Problem**: $\max_{\boldsymbol{\beta}} p(\mathbf{y}|\mathbf{X}, \boldsymbol{\beta})$ under Gaussian errors
- **Solution**: Maximum likelihood
- **Interpretation**: Most likely parameters

### Why These Are Equivalent

1. **L2 Loss ↔ Gaussian Likelihood**: Quadratic loss function emerges naturally from Gaussian assumption
2. **Normal Equations ↔ Orthogonality**: Setting gradient to zero equivalent to orthogonality condition
3. **Projection ↔ Minimization**: Projection minimizes distance in Euclidean space

### Advanced Connections

#### Information Theory

- **Fisher Information**: $\mathcal{I}(\boldsymbol{\beta}) = \frac{1}{\sigma^2}\mathbf{X}^T\mathbf{X}$
- **Cramér-Rao Bound**: $\text{Var}(\hat{\boldsymbol{\beta}}) \geq \mathcal{I}(\boldsymbol{\beta})^{-1}$
- **Efficiency**: OLS achieves the bound (is efficient)

#### Regularization Connections

- **Ridge**: $\min ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda||\boldsymbol{\beta}||^2$ ↔ Gaussian prior
- **Lasso**: $\min ||\mathbf{y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda||\boldsymbol{\beta}||_1$ ↔ Laplace prior
- **Elastic Net**: Combination of Ridge and Lasso

---

## 📈 Practical Examples & Numerical Insights

### Simple Linear Regression Example

**Data**: Housing prices vs. size

- $n = 100$ observations
- $\bar{x} = 1500$ sq ft, $\bar{y} = 250k$
- $S_{xx} = 500000$, $S_{xy} = 50000000$

**Calculations**:

- $\hat{\beta_1} = 50000000/500000 = 100$ ($/sq ft)
- $\hat{\beta_0} = 250000 - 100 \times 1500 = 100000$ ($)
- **Interpretation**: Base price $100k + $100/sq ft

### Matrix Computation Example

For $n = 3$, $p = 1$: $$\mathbf{X} = \begin{pmatrix} 1 & 2 \ 1 & 4 \ 1 & 6 \end{pmatrix}, \mathbf{y} = \begin{pmatrix} 3 \ 7 \ 11 \end{pmatrix}$$

**Computation**: $$\mathbf{X}^T\mathbf{X} = \begin{pmatrix} 3 & 12 \ 12 & 56 \end{pmatrix}$$

$$(\mathbf{X}^T\mathbf{X})^{-1} = \frac{1}{168-144}\begin{pmatrix} 56 & -12 \ -12 & 3 \end{pmatrix} = \begin{pmatrix} 7/3 & -1/2 \ -1/2 & 1/8 \end{pmatrix}$$

$$\hat{\boldsymbol{\beta}} = \begin{pmatrix} 7/3 & -1/2 \ -1/2 & 1/8 \end{pmatrix}\begin{pmatrix} 1 & 1 & 1 \ 2 & 4 & 6 \end{pmatrix}\begin{pmatrix} 3 \ 7 \ 11 \end{pmatrix} = \begin{pmatrix} -1 \ 2 \end{pmatrix}$$

**Result**: $\hat{y} = -1 + 2x$

---

## 🧮 Important Formulas & Theorems Summary

### Core Estimators

- **OLS**: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$
- **Fitted values**: $\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{H}\mathbf{y}$
- **Residuals**: $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = (\mathbf{I} - \mathbf{H})\mathbf{y}$

### Variance Estimates

- **Parameter variance**: $\text{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$
- **Error variance**: $\hat{\sigma^2} = \frac{RSS}{n-p-1}$
- **Prediction variance**: $\text{Var}(\hat{y__}) = \sigma^2\mathbf{x__}^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x_*}$

### Key Identities

- **Orthogonality**: $\mathbf{X}^T\mathbf{e} = \mathbf{0}$
- **Decomposition**: $TSS = ESS + RSS$
- **R-squared**: $R^2 = 1 - \frac{RSS}{TSS} = \frac{ESS}{TSS}$

### Distributional Results

- **Parameters**: $\hat{\boldsymbol{\beta}} \sim \mathcal{N}(\boldsymbol{\beta}, \sigma^2(\mathbf{X}^T\mathbf{X})^{-1})$
- **Error variance**: $\frac{(n-p-1)\hat{\sigma^2}}{\sigma^2} \sim \chi^2_{n-p-1}$
- **Individual coefficients**: $\frac{\hat{\beta_j} - \beta_j}{SE(\hat{\beta_j})} \sim t_{n-p-1}$

---

## 🔍 Diagnostic Tools & Model Checking

### Residual Analysis

1. **Residuals vs. Fitted**: Check linearity and homoscedasticity
2. **Normal Q-Q Plot**: Check normality assumption
3. **Scale-Location**: Check homoscedasticity
4. **Residuals vs. Leverage**: Identify influential points

### Influence Measures

- **Leverage**: $h_{ii} = [\mathbf{H}]_{ii}$ (diagonal elements of hat matrix)
- **Cook's Distance**: $D_i = \frac{(\hat{\mathbf{y}} - \hat{\mathbf{y}}_{(i)})^T(\hat{\mathbf{y}} - \hat{\mathbf{y}}_{(i)})}{(p+1)\hat{\sigma^2}}$
- **Studentized Residuals**: $r_i = \frac{e_i}{\hat{\sigma}\sqrt{1-h_{ii}}}$

### Model Selection

- **AIC**: $AIC = n\log(RSS/n) + 2(p+1)$
- **BIC**: $BIC = n\log(RSS/n) + \log(n)(p+1)$
- **Cross-Validation**: Leave-one-out, k-fold
- **Adjusted R²**: Penalizes model complexity

---

## ❓ Advanced Questions & Research Directions

### Theoretical Questions

- [ ] How does the condition number of $\mathbf{X}^T\mathbf{X}$ affect numerical stability?
- [ ] What is the relationship between linear regression and principal components analysis?
- [ ] How do outliers affect the breakdown point of OLS?
- [ ] What are the finite-sample properties vs. asymptotic properties?

### Computational Considerations

- [ ] QR decomposition vs. normal equations for numerical solution
- [ ] Handling rank-deficient design matrices
- [ ] Iterative methods for large-scale problems (conjugate gradient)
- [ ] Distributed computing approaches

### Extensions & Generalizations

- [ ] Generalized Least Squares (GLS) for correlated/heteroscedastic errors
- [ ] Weighted Least Squares (WLS) for known heteroscedasticity
- [ ] Robust regression methods (Huber, bisquare)
- [ ] Nonparametric regression (kernel smoothing, splines)

### Modern Developments

- [ ] High-dimensional regression (p > n)
- [ ] Sparse regression and feature selection
- [ ] Online/streaming regression algorithms
- [ ] Causal inference and regression

---

## 🎯 Implementation Checklist

### From Scratch Implementation

- [ ] Write OLS solver using normal equations
- [ ] Implement using QR decomposition
- [ ] Add numerical stability checks
- [ ] Include statistical inference (t-tests, confidence intervals)

### Validation & Testing

- [ ] Compare with statistical software (R, Python)
- [ ] Test on simulated data with known parameters
- [ ] Verify distributional assumptions numerically
- [ ] Test edge cases (perfect collinearity, small samples)

### Practical Applications

- [ ] Real estate price prediction
- [ ] Stock return modeling
- [ ] Scientific data analysis
- [ ] A/B test analysis

---

## 📚 Related Topics for Future Study

### Immediate Extensions

- [[Polynomial Regression]] - Nonlinear relationships
- [[Interaction Terms]] - Variable interactions
- [[Categorical Variables]] - Dummy coding, contrasts
- [[Residual Analysis]] - Diagnostic plots and tests

### Statistical Theory

- [[Hypothesis Testing]] - F-tests, t-tests, multiple comparisons
- [[Confidence Intervals]] - Parameter and prediction intervals
- [[Bootstrap Methods]] - Nonparametric inference