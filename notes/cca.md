Canonical Correlation Analysis (CCA) is a multivariate statistical technique used to uncover and quantify the underlying relationship between two multidimensional sets of variables. While standard correlation measures the relationship between two single variables, CCA finds the maximum possible correlation between two entire datasets describing the same subjects or objects.Think of it as having a dataset of physical metrics (height, weight, wingspan) and a dataset of athletic performance metrics (sprint speed, jump height, lifting weight) for the same group of athletes. CCA figures out how to optimally weight and blend the physical metrics into a single score, and blend the athletic metrics into another single score, so that those two newly blended scores correlate as strongly as possible. 

How It Works Mathematically

Given two sets of variables represented as random vectors $X \in \mathbb{R}^n$ and $Y \in \mathbb{R}^m$, CCA seeks coefficient vectors $a$ and $b$ to form linear combinations (called canonical variables):$$U = a^\top X$$$$V = b^\top Y$$The objective is to choose $a$ and $b$ such that the Pearson correlation between $U$ and $V$ is maximized:$$\rho(U, V) = \frac{\text{Cov}(U, V)}{\sqrt{\text{Var}(U)\text{Var}(V)}}$$Once the first pair of canonical variables $(U_1, V_1)$ is found, CCA can compute a second pair $(U_2, V_2)$ that captures the maximum remaining correlation, subject to the strict mathematical constraint that this new pair is completely uncorrelated (orthogonal) with the first pair. This process continues, extracting independent dimensions of shared variance, up to $\min(n, m)$ times.

Primary Use Cases

Multi-View Learning: Aligning different modalities of data that represent the same underlying concept, such as matching text descriptions to image features or audio signals to video frames.

Dimensionality Reduction: Compressing high-dimensional datasets into a few highly correlated components. It functions much like Principal Component Analysis (PCA), but instead of maximizing internal variance within one dataset, it maximizes the shared variance between two datasets.

Systems Neuroscience and Biology: Linking distinct measurement domains, such as finding the relationship between complex gene expression profiles and clinical disease symptoms, or mapping high-dimensional fMRI voxel activations to complex behavioral metrics.

## Finding a and b

Just as PCA reduces to finding the eigenvectors of a single covariance matrix, CCA reduces to solving a generalized eigenvalue problem across the cross-covariance matrices.To optimize the CCA formulation, we must express the objective function in terms of covariance matrices and apply a mathematical trick to handle scale invariance.

1. Matrix Notation for Covariance

Let $\Sigma_{XX}$ and $\Sigma_{YY}$ be the auto-covariance matrices of $X$ and $Y$, respectively, and let $\Sigma_{XY}$ (and its transpose $\Sigma_{YX}$) be the cross-covariance matrix between them.The objective function we want to maximize is:$$\rho = \frac{a^\top \Sigma_{XY} b}{\sqrt{(a^\top \Sigma_{XX} a)(b^\top \Sigma_{YY} b)}}$$

2. The Scale Invariance Trick

Notice that if you multiply the vector $a$ by any constant scalar, that scalar appears in both the numerator and the denominator and cancels out. The exact same is true for $b$. Because the correlation is invariant to the absolute scale of the vectors, we can mathematically force the variances of our new projections to be exactly 1 to simplify the math:$$a^\top \Sigma_{XX} a = 1$$$$b^\top \Sigma_{YY} b = 1$$Our optimization problem is now strictly constrained: maximize the covariance $a^\top \Sigma_{XY} b$ subject to those two unit-variance constraints.

3. Setting up the Lagrangian

To solve a constrained optimization problem, we use Lagrange multipliers ($\lambda$ and $\theta$). We construct the Lagrangian function $\mathcal{L}$:$$\mathcal{L}(a, b, \lambda, \theta) = a^\top \Sigma_{XY} b - \frac{\lambda}{2} (a^\top \Sigma_{XX} a - 1) - \frac{\theta}{2} (b^\top \Sigma_{YY} b - 1)$$

4. Taking the Derivatives

We find the maximum by taking the partial derivatives of $\mathcal{L}$ with respect to $a$ and $b$, and setting them to zero:Derivative with respect to $a$:$$\frac{\partial \mathcal{L}}{\partial a} = \Sigma_{XY} b - \lambda \Sigma_{XX} a = 0 \implies \Sigma_{XY} b = \lambda \Sigma_{XX} a$$Derivative with respect to $b$:$$\frac{\partial \mathcal{L}}{\partial b} = \Sigma_{YX} a - \theta \Sigma_{YY} b = 0 \implies \Sigma_{YX} a = \theta \Sigma_{YY} b$$If we multiply the first equation by $a^\top$ and the second by $b^\top$, we get:$$a^\top \Sigma_{XY} b = \lambda (a^\top \Sigma_{XX} a) = \lambda(1) = \lambda$$$$b^\top \Sigma_{YX} a = \theta (b^\top \Sigma_{YY} b) = \theta(1) = \theta$$Since $a^\top \Sigma_{XY} b$ is a scalar, it equals its own transpose ($b^\top \Sigma_{YX} a$). Therefore, $\lambda = \theta$, and this value is exactly the correlation $\rho$ we are trying to maximize!

5. Solving the Eigenvalue Problem

Now we substitute $b$ from the second derivative equation into the first to solve for $a$.First, isolate $b$:$$b = \frac{1}{\lambda} \Sigma_{YY}^{-1} \Sigma_{YX} a$$Substitute this $b$ back into the first derivative equation ($\Sigma_{XY} b = \lambda \Sigma_{XX} a$):$$\Sigma_{XY} \left( \frac{1}{\lambda} \Sigma_{YY}^{-1} \Sigma_{YX} a \right) = \lambda \Sigma_{XX} a$$Multiply both sides by $\lambda$ and isolate $a$ on the right side by multiplying by $\Sigma_{XX}^{-1}$:$$\Sigma_{XX}^{-1} \Sigma_{XY} \Sigma_{YY}^{-1} \Sigma_{YX} a = \lambda^2 a$$This final equation is the exact definition of a standard eigenvalue problem ($Mv = \lambda v$).The optimization is solved by calculating the eigenvectors and eigenvalues of the matrix $\Sigma_{XX}^{-1} \Sigma_{XY} \Sigma_{YY}^{-1} \Sigma_{YX}$. The largest eigenvalue ($\lambda^2$) gives the squared maximum correlation between the two datasets, and its corresponding eigenvector is the optimal weight vector $a$. You repeat the mirror-image process to find $b$. Just like PCA, finding the next orthogonal dimension of shared variance simply means looking at the eigenvector associated with the second-largest eigenvalue.

## From where did $\lambda$ appears in the equation

The $\lambda$ appears because we proved in the step immediately prior that $\lambda$ and $\theta$ are exactly the same number.

Here is the exact transition with the missing substitution step added back in.

From the derivative with respect to $b$, we originally had:$$\Sigma_{YX} a = \theta \Sigma_{YY} b$$

To isolate $b$, we multiply by the inverse covariance matrix and divide by $\theta$:$$b = \frac{1}{\theta} \Sigma_{YY}^{-1} \Sigma_{YX} a$$ Because we proved that $a^\top \Sigma_{XY} b = \lambda$ and $b^\top \Sigma_{YX} a = \theta$, and since a scalar equals its own transpose, we know that $\lambda = \theta$.

We simply substitute $\lambda$ in place of $\theta$ so that we can plug this $b$ equation back into the first derivative equation (which uses $\lambda$) to solve for a single variable:$$b = \frac{1}{\lambda} \Sigma_{YY}^{-1} \Sigma_{YX} a$$

By mathematically proving they are identical and consolidating both Lagrange multipliers into $\lambda$, we compress a two-variable system into a standard single-variable eigenvalue problem.