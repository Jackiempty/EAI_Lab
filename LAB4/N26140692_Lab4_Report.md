# Lab4 - Report
## 1. About Knowledge Distillation
### What Modes of Distillation is used in this Lab ?
This lab uses **two types of knowledge distillation**:

#### **(1) Response-Based Distillation**

* The teacher provides **logits** (pre-softmax outputs).
* The student is trained to match the teacher’s output distribution.
* Loss function is typically:

  $$
  \mathcal{L}_{KD} = (1-\alpha)\cdot CE + \alpha \cdot \tau^2\cdot KL(Q_S, Q_T)
  $$

#### **(2) Feature-Based Distillation**

* The teacher provides **intermediate feature maps**, not just final logits.
* The student tries to mimic these hidden representations.
* This requires dimension-alignment layers (connectors) because teacher & student channels/resolutions usually differ.
    
### What role do logits play in knowledge distillation? What effect does a higher temperature parameter have on logits conversion ?

#### **What logits do**

* **Logits** are the raw outputs of a model *before softmax*.
* They contain the **relative confidence relationships** between classes.
* Example:
  Teacher logits = `[10, 2, -3]` means teacher strongly prefers class 0.
* When doing distillation, we compare **teacher logits** and **student logits**, not their final predictions.

#### **Role in distillation**

* Teacher logits encode **dark knowledge**, meaning:

  * Relative similarity between classes
  * Secondary preferences (e.g., "class 3 is somewhat similar to class 2")

The student learns not only the correct label but also the teacher’s richer class structure.

#### **Effect of temperature (τ)**

Softened softmax:
$$
Q = \frac{\exp(z / \tau)}{\sum_i \exp(z_i / \tau)}
$$

* **Higher τ ⇒ logits are divided by a larger number**
* Makes the probability distribution **softer**, more uniform.
* Allows student to see **more of the teacher’s class relationships**, instead of a nearly one-hot vector.

##### Example

Teacher logits: `[10, 2, -3]`

* τ = 1
  → Softmax becomes extremely peaked (almost 1.0, 0, 0)

* τ = 5
  → Softmax distribution becomes softer
  (e.g. 0.75, 0.20, 0.05)

This softened distribution **contains much richer information**.
    
### In Feature-Based Knowledge Distillation, from which parts of the Teacher model do we extract features for knowledge transfer?

In feature-based KD, we extract feature maps from **intermediate layers** of the teacher, typically:

#### **From key representation blocks such as:**

* Convolutional blocks
* Residual blocks (e.g., ResNet’s BasicBlocks)
* Bottleneck layers
* Layer outputs before downsampling
* Any "semantic-representation-rich" layers

#### In this lab specifically (as typical for ResNet-like models):

Features are taken from:

* **Teacher’s intermediate CNN feature maps** (after certain convolution or residual layers)
* Matching layers on the student side produce corresponding feature maps
* Dimension connectors align channel/size mismatches

These feature pairs are then compared using L2 loss, SmoothL1 loss, or a custom feature distillation loss.

## 2. Response-Based KD

### How you choose the Temperature and alpha?
#### **1. How do we choose the Temperature (τ)?**

##### **Purpose of Temperature**

Temperature is used to soften the teacher and student logits:

$$
Q = \text{softmax}(z / \tau)
$$

##### **Effect**

* **Higher τ (e.g., 4–20)**
  → logits “soften” → more uniform distributions
  → student receives richer structural knowledge
  (“dark knowledge”: similarity between non-target classes)

* **Lower τ (close to 1)**
  → distribution becomes sharper (almost one-hot)
  → loses dark knowledge
  → becomes almost equivalent to normal CE training

##### **How to choose τ**

* Common values: **τ = 1, 2, 4, 5, 10**
* **Default best-practice: τ = 4** or **τ = 5**
  (balances stability & information richness)

##### **In practice**

* τ too small → KD effect weak
* τ too large → gradients become small and unstable
* Typical KD papers (Hinton et al., FitNets) use **τ = 4**

So in this lab, **τ = 4 or τ = 5 is a reasonable choice**.

#### **2. How do we choose α?**

α controls how much weight KD loss takes compared to CE loss:

$$
\mathcal{L}_{KD} = (1-\alpha)\cdot CE + \alpha \cdot \tau^2\cdot KL(Q_S,Q_T)
$$

##### Interpretation

* **(1 − α)** → weight on normal supervised loss
* **α** → weight on distillation loss

##### Choosing α

Typical values:

* **0.1 ~ 0.7** in literature
* Commonly used: **α = 0.5**

##### Practical impact:

* **α too small (<0.1)**
  → student almost ignores teacher
* **α too large (>0.7)**
  → student may overfit teacher mistakes
  → hard to optimize early in training

So **α = 0.5** is balanced and widely used.

##### Lab-appropriate recommendation

* Start with **α = 0.5**
* If student underfits teacher, increase α
* If student stops learning ground-truth labels, reduce α

### How you design the loss function?

#### The standard loss for Response-Based KD is:

$$
\mathcal{L}*{KD} = (1-\alpha),\mathcal{L}*{CE}(y, Q_S^{(1)}) + \alpha,\tau^2,KL(Q_S^{(\tau)}, Q_T^{(\tau)})
$$

Loss is composed of two parts:

#### **(A) Hard Loss (Supervised Loss)**

$$
\mathcal{L}_{CE}(y, Q_S^{(1)})
$$

* This is the normal cross-entropy between:

  * **ground-truth** label *y*
  * student prediction under **temperature = 1** (normal softmax)
* Makes sure the student still learns the real labels.

#### **(B) Soft Loss (Distillation Loss)**

$$
\tau^2 \cdot KL(Q_S^{(\tau)} ,|, Q_T^{(\tau)})
$$

* Uses **softened distributions** from teacher and student
* KL divergence measures how close the two distributions are
* **τ²** compensates gradient magnitude (Hinton’s paper)

#### Why KL divergence?

* Because KD compares **probability distributions**, not logits
* KL is the standard divergence for distributions
* Captures relative relationships among all classes

#### Why combine them?**

Using only CE → student learns labels but not teacher’s structure
Using only KD → student mimics teacher but may ignore correct labels

Combining both:

* CE ensures correctness
* KL ensures knowledge transfer

Thus giving best student generalization.

## 3. Feature-based KD

Please explain the following:
### How you extract features from the choosing intermediate layers?
To perform feature-based knowledge distillation, both the teacher and student networks must expose their intermediate feature maps. This is done by modifying the `forward()` function of the model so that it returns:

1. The final logits
2. A list of intermediate feature maps from selected layers

In our case, we extract features from the four major ResNet stages (`layer1`, `layer2`, `layer3`, and `layer4`):

```python
logits, [feature1, feature2, feature3, feature4]
```

These layers are selected because they provide progressively richer semantic representations. Early layers encode low-level details, while deeper layers capture high-level concepts. Returning these intermediate outputs allows the distillation module (Distiller) to access both the teacher’s and student’s internal representations during training.

Inside the `Distiller.forward()` method, we extract the features as follows:

```python
student_logits, student_features = self.student(x)
with torch.no_grad():
    teacher_logits, teacher_features = self.teacher(x)
```

The teacher is executed under `no_grad()` so its parameters remain frozen.

### How you design the loss function?
The goal of Feature-based KD is to make the student mimic the teacher’s internal representations. However, the feature maps from the teacher and student often have different channel dimensions. To resolve this mismatch, we introduce **connector layers**, typically implemented as 1×1 convolutions, to project the student’s feature maps into the teacher’s feature space.

Example connector initialization:

```python
self.connectors = nn.ModuleList([
    nn.Conv2d(student_dim[i], teacher_dim[i], kernel_size=1)
    for i in range(num_features)
])
```

After aligning dimensions, the feature distillation loss is computed using the **Mean Squared Error (MSE)** between the aligned student features and the teacher features:

```python
loss = 0
for i in range(len(student_features)):
    s = self.connectors[i](student_features[i])
    t = teacher_features[i].detach()
    loss += F.mse_loss(s, t)
```

This L2 loss encourages the student to learn feature representations that closely resemble the teacher’s, enabling better transfer of semantic knowledge beyond the outputs alone.


## 4. Comparison of student models w/ & w/o KD (5%)

Provide results according to the following structure:
|                            | loss     | accuracy |
| -------------------------- | -------- | -------- |
| Teacher from scratch       | 0.38     | 90.34     |
| Student from scratch       | 0.62     | 79.31     |
| Response-based student     | 1.20     | 86.77     |
| Featured-based student     | 0.61     | 87.27     |

## 5. Implementation Observations and Analysis (20%)
Based on the comparison results above:
### Did any KD method perform unexpectedly?
Nope, it perform quite well and truly improve the acc in same epoch number compare to the one without KD
### What do you think are the reasons?
First of all, I implemented it correctly XD; Ok let's be serious, I actually don't know the whole story behind the theory, so I just followed the instruction of the note and implemented it only, so I guess the reason will just be that the theory is right!
### If not, please share your observations during the implementation process, or what difficulties you encountered and how you solved them?
The major difficulty that I encountered is that by using Resnet50, I felt quite hard to push it even over 90% acc, I heard other classmates saying that it is quite easy so I guess that it is just me, 90% was the best I can do, so I feel a little shame that I couldn't train KD student with a higher teacher, if I were to do so, maybe it would be alot more easier to approach 85% in each KD training.