The coefficient 1.702 is used because it provides the closest mathematical approximation of the GELU (Gaussian Error Linear Unit) activation function when using a standard sigmoid function.

OpenAI adopted this specific approximation in models like GPT-2 and CLIP to make computation faster while maintaining high accuracy. Here is the exact mathematical breakdown of why that number is chosen:

1. The Exact GELU Definition

The exact GELU function is defined using the Cumulative Distribution Function (CDF) of a standard normal distribution, denoted as 

Φ(x): \(\text{GELU}(x)=x\cdot \Phi (x)=x\cdot P(X\le x),\text{\ where\ }X\sim \mathcal{N}(0,1)\)

Mathematically, this uses the error function (erf):

\(\text{GELU}(x)=x\cdot \frac{1}{2}\left[1+\text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]\)

2. The Sigmoid Approximation

Computing the error function (erf) directly is computationally expensive on hardware. To speed things up, researchers looked for a way to mimic the standard normal CDF (Φ(x)) using a simple logistic sigmoid function, which is defined as:

\(\sigma (x)=\frac{1}{1+e^{-x}}\)

To make the sigmoid function match the standard normal cumulative distribution as closely as possible, we scale the input by a constant k:

\(\Phi (x)\approx \sigma (k\cdot x)\)

3. Finding the Value of k

To find the absolute best value for k, mathematicians minimize the maximum difference between the two curves (or match their slopes at x=0).
• The derivative (slope) of the standard normal CDF Φ(x) at x = 0 is:

\(\Phi ^{\prime }(0)=\frac{1}{\sqrt{2\pi }}\approx 0.398942\)

• The derivative of the sigmoid function \(\sigma(k \cdot x)\) at x = 0 is:

\(\sigma ^{\prime }(0)=\frac{k}{4}=0.25\cdot k\)

To make the slopes match perfectly at the center (x=0):

\(\frac{k}{4}=\frac{1}{\sqrt{2\pi }}\)

\(k=\frac{4}{\sqrt{2\pi }}\approx 1.59576\)

However, matching just the center slope leaves slightly larger errors further out. If you instead optimize k to minimize the maximum absolute 
error across the entire curve, the optimal coefficient shifts slightly higher to exactly 1.702 (often rounded from 1.7017...).

By multiplying x by 1.702, the expression x * torch.sigmoid(1.702 * x) mimics the true, complex mathematical GELU curve with an error of less than 0.02 everywhere, allowing the model to run faster on modern GPUs without losing performance.
