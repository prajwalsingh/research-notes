In standard Ordinary Least Squares (OLS) regression, you predict a target $Y$ using a matrix of features $X$. The math relies on inverting the matrix product of the features: $(X^T X)^{-1}$.

However, if your features are highly correlated — for instance, 64 EEG channels that sit physically close to each other on the scalp — the $X^T X$ matrix becomes nearly singular (non-invertible). The regression coefficients blow up, the model becomes wildly unstable, and its predictions become useless.

**Principal Component Regression (PCR)** solves this by forcing the features to be mathematically independent (orthogonal) *before* the regression step ever runs.

## Step 1 — Standardization

Because PCA is highly sensitive to scale, you must mean-center and scale your variables first:

- Subtract the mean of each feature so every column sums to zero.
- Divide by the standard deviation so each feature has unit variance.
- Mean-center the target vector $Y$ as well.

## Step 2 — Principal Component Extraction

Instead of regressing $Y$ on $X$ directly, PCR extracts the principal components of $X$, typically via Singular Value Decomposition:

$$X = U D V^T$$

- $V$ (loadings) is a $p \times p$ matrix of eigenvectors of the covariance matrix $X^T X$ — the directions of maximum variance in feature space.
- $U D$ (scores) is the transformed data. Call this $Z = XV$.

$Z$ has the same shape as $X$, but every column is completely uncorrelated with every other column. $Z_1$ captures the most variance possible, $Z_2$ the second most, and so on.

## Step 3 — Dimensionality Reduction

Keeping all $p$ components of $Z$ and regressing reproduces standard OLS exactly. The power of PCR comes from dropping the noisy, low-variance components: you keep the first $k$ components, giving a truncated score matrix $Z_k$ of size $n \times k$, and discard the rest as presumed noise — like background biological noise in EEG.

## Step 4 — The Regression Step

Now run ordinary least squares, but predict $Y$ from $Z_k$ instead of $X$:

$$\hat{\gamma} = (Z_k^T Z_k)^{-1} Z_k^T Y$$

Because the columns of $Z_k$ are orthogonal by construction, $Z_k^T Z_k$ is diagonal — trivial to invert, and the multicollinearity problem that crashes standard OLS is gone.

## Step 5 — Transforming Back (optional)

$\hat{\gamma}$ tells you how much each principal component contributes to $Y$, but you usually want to know how much each *original* feature — a specific EEG channel — contributes. Project the coefficients back into the original feature space:

$$\hat{\beta}^{PCR} = V_k \hat{\gamma}$$

This gives an equation shaped just like standard regression, $\hat{Y} = X\hat{\beta}^{PCR}$, except the coefficients are now stable and implicitly regularized.

## The fatal flaw of PCR

PCR solves multicollinearity beautifully, but it has a serious blind spot: **the PCA step is completely unsupervised.**

When PCR computes $Z$ in Step 2, it looks only at $X$. It searches for the directions of highest variance while completely ignoring $Y$.

Imagine predicting a subject's visual response ($Y$) from 64 EEG channels ($X$). PCR assumes the direction of highest electrical variance — $Z_1$ — must be the most important predictor. But the highest-variance direction in raw EEG is often just eye blinks or jaw clenching. PCR will proudly keep the "jaw clenching" component and hand it to the regression, while discarding the tiny, low-variance micro-voltages that actually encode the visual stimulus.

## PCR vs. PLS

This is exactly why **Partial Least Squares (PLS)** is generally preferred over PCR in neuroscience and bioinformatics:

| | PCR | PLS |
|---|---|---|
| Optimizes | variance within $X$ | covariance between $X$ and $Y$ |
| Supervision | unsupervised | supervised |
| Small but predictive signal | discarded during truncation | promoted to the front |

If a signal has low variance but predicts $Y$ almost perfectly, PCR deletes it at the truncation step. PLS recognizes its predictive power, boosts it into an early latent component, and builds the representation around it instead.

> Rule of thumb: reach for PCR when you just need to kill multicollinearity and trust that "high variance" and "signal" roughly coincide. Reach for PLS the moment you suspect the two might not — which, for raw EEG next to eye and muscle artifacts, is most of the time.
