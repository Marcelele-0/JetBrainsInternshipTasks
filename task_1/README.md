# Word2Vec Implementation (NumPy)

This project implements a Word2Vec (Skip-gram with Negative Sampling) model using pure NumPy. It is designed to be clear, modular, and educational, demonstrating the core mathematical concepts behind word embeddings.

## Project Structure

The repository is organized to separate model logic, data processing, and utilities:

```text
task_1/
├── .venv/              # Virtual environment managed by uv
├── data/               # Stores generated embeddings and plots
├── src/
│   ├── __init__.py
│   ├── model.py        # Core Word2Vec class with manual gradient descent
│   ├── utils.py        # Mathematical helpers (sigmoid, cosine similarity)
│   └── preprocessing.py # Vocabulary building, subsampling, pair generation
├── tests/
│   └── test_model.py   # Unit tests ensuring model initialization and training steps work
├── main.py             # Orchestration script: loads data, trains model, visualizes results
├── pyproject.toml      # Dependency management (uv)
└── README.md           # Documentation
```

## Setup & Running with `uv`

This project uses **uv** for fast dependency management.

1. **Install uv** (if not installed):
   Follow instructions at [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).

2. **Run the project:**
   `uv` will automatically create a virtual environment and install dependencies (`numpy`, `matplotlib`, `scikit-learn`) on the first run.
   ```bash
   uv run main.py
   ```
   This will train the model on a small toy dataset, print loss progress, and save results to `data/`.

3. **Run tests:**
   ```bash
   uv run pytest
   ```

## Mathematical Assumptions

### Skip-Gram with Negative Sampling (SGNS)

We use the **Skip-gram** architecture, which predicts context words given a center word. To make training efficient, we approximate the full Softmax using **Negative Sampling**.

Instead of updating weights for the entire vocabulary (which is computationally expensive), we distinguish the true context word from $k$ noise words (negative samples) drawn from a unigram distribution.

### Loss Function

The objective is to maximize the probability of the true context word and minimize the probability of negative samples. The loss function for a single center word $w$ and context $c$ (with negative samples set $N$) is:

$$ J = - \log \sigma(u_c^T v_w) - \sum_{k \in N} \log \sigma(-u_k^T v_w) $$

Where:
* $v_w$ is the embedding vector of the center word (from $W_1$).
* $u_c$ is the embedding vector of the context word (from $W_2$).
* $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function.

### Gradient Derivation

To update the weights using Stochastic Gradient Descent (SGD), we compute the gradients of $J$ with respect to the input word vector $v_w$ and context vectors $u_x$.

Let $x$ be either the context word $c$ or a negative sample $k$. Let $t$ be the label ($t=1$ for context, $t=0$ for negative).
The probability predicted by the model is $y = \sigma(u_x^T v_w)$.

The derivative of the Binary Cross Entropy loss term for one pair with respect to the input score $z = u_x^T v_w$ is:

$$ \frac{\partial J}{\partial z} = \sigma(z) - t $$

Thus, the gradients for the vectors are:

1. **For the center word vector $v_w$:**
   It accumulates gradients from the positive context and all negative samples:
   $$ \frac{\partial J}{\partial v_w} = \sum_{x \in \{c\} \cup N} (\sigma(u_x^T v_w) - t_x) \cdot u_x $$

2. **For the context/negative vector $u_x$:**
   $$ \frac{\partial J}{\partial u_x} = (\sigma(u_x^T v_w) - t_x) \cdot v_w $$

In the code (`src/model.py`), this corresponds to:
```python
errors = probs - labels  # (sigma(z) - t)
dW2 = np.outer(errors, u_w)  # Update for context vectors
dW1 = np.dot(errors, V_context)  # Update for center vector
```

## Check Results

After running `uv run main.py`, check the `data/` folder:
* `data/embeddings.npy`: Saved vectors.
* `data/loss_plot.png`: Plot of training loss over epochs.
* `data/embeddings_pca.png`: 2D PCA projection of the word vectors.
