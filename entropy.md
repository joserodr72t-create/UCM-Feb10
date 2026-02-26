Entropy in decision trees is a numerical measure of how “mixed” the class labels are in a node: high entropy = very mixed (uncertain), low entropy = very pure (certain). [towardsdatascience](https://towardsdatascience.com/decision-trees-explained-entropy-information-gain-gini-index-ccp-pruning-4d78070db36c/)

***

## Intuition: disorder / impurity

- A node is **pure** if all samples have the same class → entropy should be 0 (no uncertainty). [bricaud.github](https://bricaud.github.io/personal-blog/entropy-in-decision-trees/)
- A node is **maximally impure** if classes are perfectly mixed (e.g., 50% positive, 50% negative in binary classification) → entropy should be maximal (1 for binary with log base 2). [towardsdatascience](https://towardsdatascience.com/decision-trees-explained-entropy-information-gain-gini-index-ccp-pruning-4d78070db36c/)
- Decision trees try to choose splits that turn a messy node into cleaner (lower-entropy) child nodes; this reduction in entropy is what we call **information gain**. [nebius](https://nebius.com/blog/posts/entropy-in-machine-learning)

***

## Formula

For a node with class distribution $p_1, p_2, \dots, p_n$ (each $p_i$ is the fraction of samples of class i in that node), entropy is:

$$
\text{Entropy} = -\sum_{i=1}^{n} p_i \log_2 p_i
$$

(terms with $p_i = 0$ are conventionally treated as $0 \cdot \log_2 0 = 0$). [geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/gini-impurity-and-entropy-in-decision-tree-ml/)

**Properties:**

- Entropy = 0 if one $p_i = 1$ and all others 0 (pure node)
- Entropy is maximal when all classes are equally likely  
  (for binary: $p_1 = p_2 = 0.5 \Rightarrow \text{Entropy} = 1$). [geeksforgeeks](https://www.geeksforgeeks.org/data-science/how-to-calculate-entropy-in-decision-tree/)

***

## Simple numeric examples

Assume binary classes: Positive (P) and Negative (N).

### Example 1: pure node

Node A: 10 samples, all Positive → $p(P)=1$, $p(N)=0$.

$$
\text{Entropy}(A) = -[1 \cdot \log_2 1 + 0 \cdot \log_2 0] = 0
$$

This node is perfectly pure; a decision tree would not need to split it further. [bricaud.github](https://bricaud.github.io/personal-blog/entropy-in-decision-trees/)

### Example 2: maximally mixed node

Node B: 10 samples, 5 Positive, 5 Negative → $p(P)=0.5$, $p(N)=0.5$.

$$
\text{Entropy}(B) = -[0.5 \log_2 0.5 + 0.5 \log_2 0.5] = -[0.5(-1) + 0.5(-1)] = 1
$$

This is the highest uncertainty for a binary problem. [bricaud.github](https://bricaud.github.io/personal-blog/entropy-in-decision-trees/)

### Example 3: moderately skewed node

Node C: 10 samples, 7 Positive, 3 Negative → $p(P)=0.7$, $p(N)=0.3$.

$$
\text{Entropy}(C) = -[0.7 \log_2 0.7 + 0.3 \log_2 0.3] \approx 0.881
$$

Uncertainty is lower than 1 but still not pure. [geeksforgeeks](https://www.geeksforgeeks.org/data-science/how-to-calculate-entropy-in-decision-tree/)

***

## How trees use entropy to split

At each node, the algorithm: [community.deeplearning](https://community.deeplearning.ai/t/confusion-about-the-concepts-of-entropy-and-information-gain-in-decision-tree/686640)

1. Computes the entropy of the current (parent) node.  
2. For each candidate feature and possible split threshold, it:
   - Partitions the data into child nodes (e.g. left/right).  
   - Computes the entropy of each child node.  
   - Computes the **weighted average** child entropy.  
3. Calculates **information gain** for that split:

$$
\text{Information Gain} = \text{Entropy(parent)} - \text{Weighted Entropy(children)}
$$

4. Chooses the split with the highest information gain (largest reduction in entropy).  

The weights are proportional to the number of samples going into each child node. [nebius](https://nebius.com/blog/posts/entropy-in-machine-learning)

***

## Concrete split example

Suppose the parent node has 10 samples: 6 Positive, 4 Negative.  
Parent entropy:

$$
p(P)=0.6,\quad p(N)=0.4,\quad
\text{Entropy(parent)} \approx 0.971
$$

Consider a split on some feature that yields:

- Left child: 6 samples, 5 Positive, 1 Negative  
  $p(P)=5/6 \approx 0.833,\ p(N)=1/6 \approx 0.167$  
  $\text{Entropy(left)} \approx 0.65$ (more pure). [geeksforgeeks](https://www.geeksforgeeks.org/data-science/how-to-calculate-entropy-in-decision-tree/)

- Right child: 4 samples, 1 Positive, 3 Negative  
  $p(P)=0.25,\ p(N)=0.75$  
  $\text{Entropy(right)} \approx 0.811$. [geeksforgeeks](https://www.geeksforgeeks.org/data-science/how-to-calculate-entropy-in-decision-tree/)

Weighted child entropy:

$$
\text{Entropy(children)} =
\frac{6}{10} \cdot 0.65 + \frac{4}{10} \cdot 0.811
\approx 0.39 + 0.324 = 0.714
$$

Information gain:

$$
\text{IG} = 0.971 - 0.714 \approx 0.257
$$

If another candidate split produced children with higher weighted entropy (e.g. 0.85), its information gain ($0.971 - 0.85 \approx 0.121$) would be smaller → the tree prefers the first split. [community.deeplearning](https://community.deeplearning.ai/t/confusion-about-the-concepts-of-entropy-and-information-gain-in-decision-tree/686640)

***

## Summary of key points

- Entropy quantifies class-label uncertainty in a node; 0 means pure, higher values mean more mixed classes. [geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/gini-impurity-and-entropy-in-decision-tree-ml/)
- Decision trees evaluate candidate splits by how much they reduce entropy (**information gain**). [towardsdatascience](https://towardsdatascience.com/decision-trees-explained-entropy-information-gain-gini-index-ccp-pruning-4d78070db36c/)
- The algorithm recursively selects splits with maximum information gain until nodes are pure enough or stopping criteria are met. [nebius](https://nebius.com/blog/posts/entropy-in-machine-learning)

If you want, next step I can show a small Python example computing entropy and information gain on a toy dataset and relate it to scikit-learn’s decision tree behavior.
