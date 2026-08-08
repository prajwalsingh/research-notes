Converting a Large Language Model (LLM) into a Vision-Language-Action (VLA) policy relies on a clever bridge between **continuous physical controls** and **discrete text tokens**.

Because standard LLM tokenizers do not contain native "robot action tokens," the OpenVLA paradigm solves this by **discretizing continuous values into bins** and **overwriting unused tokens at the end of the LLM's existing vocabulary**.

Here is an in-depth breakdown of the complete round-trip pipeline: from continuous robot commands to LLM tokens during training, and back to continuous control during execution.

---

## Phase 1: Discretization (Continuous Action $\rightarrow$ Bin Index)

In LIBERO, a single timestep action $a$ is a 7-dimensional vector of continuous floating-point numbers normalized to $[-1, 1]$:


$$a = (a^{(1)}, a^{(2)}, a^{(3)}, a^{(4)}, a^{(5)}, a^{(6)}, a^{(7)}) \in [-1, 1]^7$$

* $a^{(1 \dots 3)}$: End-effector translation deltas $(\Delta x, \Delta y, \Delta z)$

* $a^{(4 \dots 6)}$: Rotation deltas $(\Delta \phi_1, \Delta \phi_2, \Delta \phi_3)$

* $a^{(7)}$: Gripper open/close command



To feed these continuous numbers into a classification-based language model, each dimension $d \in \{1 \dots 7\}$ is mapped into one of $K$ discrete bins (e.g., $K = 256$).

```
Continuous Scalar a^(d) ──> Quantile Binning ──> Bin Index c^(d) ∈ {1 ... K}

```

## 1. Percentile Boundary Calculation

Rather than dividing the strict min/max $[-1, 1]$ range uniformly—which wastes bin resolution on rare teleoperation jerks/outliers—we calculate the 1st percentile ($q_1^{(d)}$) and 99th percentile ($q_{99}^{(d)}$) across the training dataset for each dimension $d$.

The $K$ uniform bin edges are defined as:


$$\text{edge}_k^{(d)} = q_1^{(d)} + \frac{k}{K} \left( q_{99}^{(d)} - q_1^{(d)} \right), \quad k \in \{0, 1, \dots, K\}$$

## 2. Discretization Formula

A continuous scalar $a^{(d)}$ is mapped to a bin index $c^{(d)} \in \{1, \dots, K\}$ using quantization clipping:


$$c^{(d)} = \text{clip} \left( \text{digitize}\left( a^{(d)}, \{\text{edge}_k^{(d)}\} \right), 1, K \right)$$

---

## Phase 2: Vocabulary Overwriting (Bin Index $\rightarrow$ LLM Token ID)

Now we have 7 bin indices $c^{(1)}, \dots, c^{(7)}$, where each $c^{(d)} \in \{1, \dots, K\}$. But how does the LLM's tokenizer represent bin $c^{(d)}$?

Instead of extending the tokenizer vocabulary matrix (which requires changing embedding layer dimensions and initializing new weights), OpenVLA "hijacks" or **overwrites the $K$ least frequently used tokens at the very end of the existing vocabulary**.

```
Existing Vocabulary: [ "the", "cat", "robot", ..., <unused_31998>, <unused_31999> ]
                                                  ▲               ▲
                                                  │               │
Action Bins:                                    Bin 1    ...    Bin 256

```

## Mapping Formula

If the LLM's tokenizer has a **BASE** vocabulary size $V$, the last $K$ token IDs (from index $V - K$ to $V - 1$) are reserved exclusively to represent action bins:

$$\text{Token ID}(c^{(d)}) = V - K + (c^{(d)} - 1)$$

* Example: If $V = 49152$ and $K = 256$, the action bins map directly to token IDs $48896 \dots 49151$.
* Bin $1 \rightarrow$ Token ID $48896$
* Bin $256 \rightarrow$ Token ID $49151$

---

## Phase 3: The Sequence & Training Forward Pass

For any given timestep, a single robot action is formatted as a sequence of **7 consecutive action tokens**, generated in fixed dimension order:

$$\text{Action Sequence} = [\text{Token}(c^{(1)}), \text{Token}(c^{(2)}), \dots, \text{Token}(c^{(7)})]$$

## Sequence Layout

The full input sequence fed into the transformer decoder looks like:


$$\text{Sequence} = [\underbrace{H_v}_{\text{Visual Tokens}}, \quad \underbrace{X}_{\text{Instruction Prompt}}, \quad \underbrace{c^{(1)}, c^{(2)}, c^{(3)}, c^{(4)}, c^{(5)}, c^{(6)}, c^{(7)}}_{\text{7 Action Bin Tokens}}]$$

## How the Model Learns Positional Meaning

Notice that we use the **same** $K$ token slots for all 7 dimensions. The model knows that token #1 represents $\Delta x$ and token #7 represents the gripper command **purely through sequence position** via positional embeddings, not through separate sub-vocabularies.

---

## Phase 4: Inference Rollout (LLM Output $\rightarrow$ Continuous Control)

During evaluation, when the robot is operating closed-loop in the environment, the transformation runs in reverse:

```
1. Camera Image + Instruction ──> LLM Forward Pass
2. Autoregressively Predict 7 Tokens ──> [t_1, t_2, t_3, t_4, t_5, t_6, t_7]
3. Convert Token IDs ──> Bin Indices [c^(1) ... c^(7)]
4. De-binning Formula ──> Continuous Action [â^(1) ... â^(7)]
5. Pass to Simulator ──> env.step(â)

```

## De-Binning (De-Quantization) Formula

When the LLM predicts token ID $t_d$ for dimension $d$, we first convert it back to bin index $c^{(d)} = t_d - (V - K) + 1$.

To recover the continuous floating-point command $\hat{a}^{(d)}$, we take the **midpoint of bin $c^{(d)}$**:

$$\hat{a}^{(d)} = q_1^{(d)} + \left( c^{(d)} - \frac{1}{2} \right) \frac{q_{99}^{(d)} - q_1^{(d)}}{K}$$

## Step-by-Step Code Walkthrough

```python
import numpy as np


class ActionTokenizer:

    def __init__(self, dataset_actions, K=256, vocab_size=49152):
        self.K = K
        self.vocab_size = vocab_size

        # Compute 1st and 99th percentiles per dimension (shape: (7,))
        self.q1 = np.percentile(dataset_actions, 1, axis=0)
        self.q99 = np.percentile(dataset_actions, 99, axis=0)

    def encode_action(self, continuous_action):
        """Continuous action shape (7,) -> Token IDs shape (7,)"""
        bin_indices = []
        for d in range(7):
            # Compute bin edges
            edges = np.linspace(self.q1[d], self.q99[d], self.K + 1)
            # Digitize to bin 1...K
            bin_idx = np.digitize(continuous_action[d], edges)
            bin_idx = np.clip(bin_idx, 1, self.K)
            bin_indices.append(bin_idx)

        # Map bin indices 1..K to tail vocabulary token IDs
        token_ids = [
            self.vocab_size - self.K + (c - 1) for c in bin_indices
        ]
        return np.array(token_ids)

    def decode_action(self, token_ids):
        """Token IDs shape (7,) -> Continuous action shape (7,)"""
        # Convert Token IDs back to bin indices 1..K
        bin_indices = [
            t - (self.vocab_size - self.K) + 1 for t in token_ids
        ]

        continuous_action = []
        for d in range(7):
            c = bin_indices[d]
            # Bin midpoint formula
            bin_width = (self.q99[d] - self.q1[d]) / self.K
            val = self.q1[d] + (c - 0.5) * bin_width
            continuous_action.append(val)

        return np.array(continuous_action, dtype=np.float64)

```

## Key Trade-Offs to Keep in Mind

* **Quantization Error**: De-binning introduces an unavoidable discretization error bounded by half a bin width: $\frac{q_{99}^{(d)} - q_1^{(d)}}{2K}$ per step.


* **Bin Count Selection ($K$)**:
* **Higher $K$ (e.g., 256 or 512)**: Higher spatial precision, but makes the classification task harder for a lightweight LoRA adapter.


* **Lower $K$ (e.g., 64 or 128)**: Easier for the model to predict correct bins, but introduces coarse robot movements.


## 1. Why $K = 256$? (The Precision vs. Task Complexity Sweet Spot)

Choosing $K = 256$ is an intentional engineering compromise pioneered by robotics models like RT-1 and OpenVLA. It balances three competing demands:

## A. Spatial Precision (Resolution)

* If $K$ is too small (e.g., $K = 10$ or $K = 32$), the quantization gap between adjacent bins is large. This causes coarse, jerky, robot movements that miss delicate objects.
* For an end-effector translation delta range of $[-1, 1]$, $K = 256$ splits the range into steps of approximately $\frac{2}{256} \approx 0.0078$ units. This sub-centimeter granularity is fine enough for precise robotic manipulation tasks like grasping, opening drawers, or placing items on plates.

## B. Classification Complexity for the Language Model

* Language models treat action prediction as a **next-token classification problem** using Cross-Entropy loss.
* If $K$ is too large (e.g., $K = 2000$), the model has to choose from 2,000 possibilities per action dimension. This makes the classification task significantly harder, slows down training convergence, and requires larger LoRA ranks or full fine-tuning to learn.
* $K = 256$ keeps the classification problem lightweight enough for a small LoRA adapter (like rank $r = 16$) to learn efficiently from a few hundred demonstration episodes.

## C. Hardware and Byte Alignment

* $256 = 2^8$, which fits into a single unsigned 8-bit byte (`uint8`). This alignment simplifies data storage, tensor serialization, and memory allocation across data pipelines.

---

## 2. Why Quantile-Based (Percentile) Binning? (Handling Outliers)

Standard **uniform min-max binning** divides the strict range $[\text{min}, \text{max}]$ into equal-sized steps. However, quantile-based binning using the 1st ($q_1^{(d)}$) and 99th ($q_{99}^{(d)}$) percentiles is preferred due to real-world robotics data characteristics:

```
Uniform Min-Max (Flawed):
[Extreme Outlier Min] <----------------- Wasted Bins -----------------> [Extreme Outlier Max]
                     | Dense Action Region |

Quantile-Based (q_1 to q_99):
[Clipped] | <------------- Bins Focused Here (98% of Data) -------------> | [Clipped]

```

## A. Teleoperation Noise and Extreme Outliers

* Robot demonstration datasets (like LIBERO, ALOHA, or Bridge) are gathered by human operators using VR controllers or 3D mice.
* Human teleoperation naturally contains occasional sudden jerks, sensor glitches, or accidental fast movements.
* If a single extreme velocity spike occurs (e.g., a sudden delta of $+0.95$ when normal deltas are $\pm 0.05$), standard min-max binning stretches the bin range to fit that single outlier. As a result, 90% of the 256 bins are wasted on empty velocity ranges that the robot almost never visits.

## B. Concentrating Resolution Where It Matters

* In typical robot demonstrations, 98% of actions consist of small, smooth, continuous adjustments centered around zero.
* By calculating $q_1^{(d)}$ and $q_{99}^{(d)}$, the algorithm discards the top 1% and bottom 1% extreme outliers and distributes all $K = 256$ bins strictly across the dense operational region.
* Any rare action exceeding $q_{99}^{(d)}$ or falling below $q_1^{(d)}$ is safely clipped to bin $K$ or bin $1$. This preserves fine spatial resolution for 98% of normal trajectories without breaking the model when an extreme movement occurs.

Here is the breakdown of exactly how the percentiles are grouped and what the `digitize` function is doing mathematically.

## 1. How the Percentiles are Calculated (Independent vs. Stacked)

You do **not** stack all the action values together. The percentiles are calculated **separately for each of the 7 dimensions**.

According to the mathematical foundation notes, you must "compute the first and ninety-ninth percentile of that dimension’s values across the training data" for each specific dimension $d$.

**Why it must be done separately:**
The 7 dimensions represent fundamentally different physical quantities. Dimensions 1-3 are translations (meters), dimensions 4-6 are rotations (radians or axis-angles), and dimension 7 is the gripper state. The typical range of a rotation delta ($\Delta \phi_1$) might be completely different from a translation delta ($\Delta x$). If you stacked them all together into one massive pool before finding the percentiles, the smaller movements would be completely swallowed by the larger ones, ruining the bin resolution for that specific joint or axis.

In code, if your entire training dataset of actions has the shape `(Total_Timesteps, 7)`, you calculate the percentiles along the column axis (axis=0), resulting in 7 separate 1st percentiles and 7 separate 99th percentiles.

## 2. What the `digitize` Function Does

The `digitize` function (which maps directly to `numpy.digitize` in code) acts as a sorting hat. It takes a continuous floating-point number and figures out which discrete "bucket" (bin) it belongs to based on a list of boundaries.

In the formula:


$$c^{(d)} = \text{clip} \left( \text{digitize}\left( a^{(d)}, \{\text{edge}_k^{(d)}\} \right), 1, K \right)$$

Here is the exact step-by-step mechanical execution:

1. **The Inputs:** `digitize` looks at your specific continuous action value $a^{(d)}$ (e.g., `0.045`) and the array of bin edges $\{\text{edge}_k^{(d)}\}$ you created using the percentiles.
2. **The Search:** It scans through the bin edges to find where `0.045` fits. For example, if Bin 120 covers the range `[0.040, 0.050]`, the function recognizes that `0.045` falls squarely inside this interval.
3. **The Output:** It returns the integer index of that bin (e.g., `120`).

**Why the `clip(..., 1, K)` wrapper is there:**
Because the bin edges are built using the 1st and 99th percentiles, what happens if the network encounters a rare extreme outlier from the 99.9th percentile? The `digitize` function would normally return a bin index like `K + 1` or `K + 2` because the value exceeds the highest predefined edge.
The `clip` function catches these outliers and forces them into the nearest valid boundary. If `digitize` spits out a bin index greater than $K$, `clip` forcefully squishes it back down to exactly $K$. If it spits out an index less than 1, it snaps it to `1`.


Let's walk through a concrete numerical example to see exactly how a continuous 7-dimensional action vector is discretized into bin indices using the quantile-based formula.

## 1. The Setup: Defining the Data

Assume we have decided to use $K = 256$ bins.

First, we look at our entire training dataset and calculate the 1st percentile ($q_1^{(d)}$) and 99th percentile ($q_{99}^{(d)}$) for each of the 7 dimensions independently. Let's assume we got the following bounds:

* **$d=1 \ (\Delta x)$:** $q_1 = -0.100$, $q_{99} = 0.100$
* **$d=2 \ (\Delta y)$:** $q_1 = -0.100$, $q_{99} = 0.100$
* **$d=3 \ (\Delta z)$:** $q_1 = -0.050$, $q_{99} = 0.150$
* **$d=4 \ (\Delta \phi_1)$:** $q_1 = -0.500$, $q_{99} = 0.500$
* **$d=5 \ (\Delta \phi_2)$:** $q_1 = -0.500$, $q_{99} = 0.500$
* **$d=6 \ (\Delta \phi_3)$:** $q_1 = -1.000$, $q_{99} = 1.000$
* **$d=7 \ (\text{Gripper})$:** $q_1 = -1.000$, $q_{99} = 1.000$

Now, imagine during a specific timestep, the robot executes the following continuous action:


$$a = [0.025, -0.120, 0.080, 0.000, 0.450, 0.050, 1.000]$$

## 2. Step-by-Step Calculation for Dimension 1 ($\Delta x$)

Let's discretize $a^{(1)} = 0.025$.

**Step A: Find the total range and bin width**

* Range: $q_{99}^{(1)} - q_1^{(1)} = 0.100 - (-0.100) = 0.200$
* Bin Width: $\frac{0.200}{256} = 0.00078125$

**Step B: Define the edges**
Using the formula, the edges are constructed as:

* $\text{edge}_0 = -0.100 + 0 = -0.100$
* $\text{edge}_1 = -0.100 + 0.00078125 = -0.09921875$
* $\dots$
* $\text{edge}_{256} = 0.100$

**Step C: Digitize and Clip**
To find where $0.025$ falls, we measure its distance from the bottom edge ($q_1^{(1)}$) and divide by the bin width:


$$\text{Raw Position} = \frac{0.025 - (-0.100)}{0.00078125} = \frac{0.125}{0.00078125} = 160$$

Since it falls exactly on the boundary, `digitize` places it in bin index **160**. Because 160 is between 1 and 256, `clip` does nothing.
$c^{(1)} = 160$.

## 3. Step-by-Step Calculation for Dimension 2 ($\Delta y$) - An Outlier

Let's discretize $a^{(2)} = -0.120$.

Notice that $-0.120$ is smaller than our 1st percentile boundary $q_1^{(2)} = -0.100$. This is an extreme teleoperation outlier.

**Step A: Digitize**
Because $-0.120$ is less than $\text{edge}_0$ ($-0.100$), `digitize` assigns it an index of **0**.

**Step B: Clip**
The formula strictly bounds the indices:


$$c^{(2)} = \text{clip}(0, 1, 256) = 1$$


The outlier is safely caught and assigned to the lowest valid bin.

## 4. Summary Across All 7 Dimensions

Here is how the entire 7-dimensional vector is processed simultaneously:

| Dimension ($d$) | Action $a^{(d)}$ | Quantile Bounds $[q_1, q_{99}]$ | Raw `digitize` Position | Clipped Bin Index $c^{(d)}$ |
| --- | --- | --- | --- | --- |
| **1** ($\Delta x$) | $0.025$ | $[-0.10, 0.10]$ | $160$ | **160** |
| **2** ($\Delta y$) | $-0.120$ | $[-0.10, 0.10]$ | $0$ (Outlier) | **1** |
| **3** ($\Delta z$) | $0.080$ | $[-0.05, 0.15]$ | $166.4 \rightarrow 167$ | **167** |
| **4** ($\Delta \phi_1$) | $0.000$ | $[-0.50, 0.50]$ | $128$ | **128** |
| **5** ($\Delta \phi_2$) | $0.450$ | $[-0.50, 0.50]$ | $243.2 \rightarrow 244$ | **244** |
| **6** ($\Delta \phi_3$) | $0.050$ | $[-1.00, 1.00]$ | $134.4 \rightarrow 135$ | **135** |
| **7** (Gripper) | $1.000$ | $[-1.00, 1.00]$ | $256$ | **256** |

The final output of this discretization phase is a sequence of 7 integers: `[160, 1, 167, 128, 244, 135, 256]`. These integers are then mapped directly to the tail-end token IDs of the language model's vocabulary.


The $-\frac{1}{2}$ is included to specifically target the **midpoint (exact center)** of the predicted bin.

When the model predicts a discrete bin index $c^{(d)}$, it is essentially saying, *"The continuous value falls somewhere inside this specific bucket."* Because we don't know exactly where inside the bucket the original continuous value was, guessing the dead center of the bucket is the mathematically safest bet to minimize error.

Here is a step-by-step breakdown of how the math positions that point.

## 1. Defining the Bin Width

First, look at the right side of the term:


$$W = \frac{q_{99}^{(d)} - q_1^{(d)}}{K}$$


This represents the width of a single bin.

## 2. The 1-Based Indexing Problem

The bin index $c^{(d)}$ uses 1-based indexing (it ranges from $1$ to $K$). Because of this, if you multiply the index by the bin width without adjusting it, you end up pointing at the edges of the bin, not the center.

* **Pointing at the Upper Edge:**
If you calculate $q_1^{(d)} + c^{(d)} \times W$, you are calculating the starting percentile plus the full width of $c$ bins. This places your value exactly on the **upper (right) boundary** of the bin.
* **Pointing at the Lower Edge:**
If you subtract 1 and calculate $q_1^{(d)} + (c^{(d)} - 1) \times W$, you are placing the value exactly on the **lower (left) boundary** of the bin.

## 3. Finding the Center

To get to the middle of the bin, you want to jump over all the previous bins, but only go *halfway* through the current predicted bin.

By subtracting $0.5$ (or $\frac{1}{2}$), the formula calculates:


$$q_1^{(d)} + (c^{(d)} - 0.5) \times W$$


This perfectly positions the continuous estimate $\hat{a}^{(d)}$ at the bin's midpoint.

## 4. Minimizing Quantization Error

By always recovering the continuous value from the bin midpoint, the system guarantees that the maximum possible quantization error is bounded by exactly half a bin width.

If the true action was at the extreme left or extreme right edge of the bin, guessing the midpoint means your error is never larger than $\frac{q_{99}^{(d)} - q_1^{(d)}}{2K}$. If you didn't use the $-\frac{1}{2}$ and guessed the upper edge instead, your worst-case error could be a full bin width, making the robot's reconstructed movements significantly less accurate.