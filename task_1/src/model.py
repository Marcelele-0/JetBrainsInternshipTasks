import numpy as np
from .utils import sigmoid, cosine_similarity

class Word2Vec:
    def __init__(self, vocab_size: int, embedding_dim: int = 100, learning_rate: float = 0.01):
        self.v_size = vocab_size
        self.dim = embedding_dim
        self.lr = learning_rate
        
        # Weight Initialization (He initialization/standard normal)
        # Random initiation is crucial to break symmetry.
        # If we initialized with zeros, all neurons would learn the same features during backpropagation.
        # W1: Input matrix (Center words), W2: Output matrix (Context words)
        self.W1 = np.random.randn(self.v_size, self.dim) * 0.1
        self.W2 = np.random.randn(self.v_size, self.dim) * 0.1

    def train_step(self, center_word_idx: int, context_word_idx: int, negative_indices: list) -> float:
        """
        Single optimization step (forward + backward).
        Args: 
        - center_word_idx: index of the input word
        - context_word_idx: index of the context word (label=1)
        - negative_indices: list of indices for negative samples (label=0)
        """
        
        # 1. Forward Pass
        u_w = self.W1[center_word_idx] # Vector for center word (1, dim)
        
        # Gather all words for comparison (positive + negatives)
        indices = [context_word_idx] + list(negative_indices)
        labels = np.zeros(len(indices))
        labels[0] = 1.0 # The first one is the positive sample
        
        V_context = self.W2[indices] # Matrix (1+k, dim)
        
        # Calculate similarity (dot product)
        scores = np.dot(V_context, u_w) # (1+k,)
        probs = sigmoid(scores) # (1+k,)
        
        # 2. Loss (Binary Cross Entropy for SGNS)
        # Add small epsilon to avoid log(0)
        loss = - (labels * np.log(probs + 1e-10) + (1 - labels) * np.log(1 - probs + 1e-10)).sum()

        # 3. Gradients & Backward Pass
        # Error: (prediction - label)
        errors = probs - labels # (1+k,)
        
        # Gradient for W2 (Context): dJ/dV = error * u_w
        # np.outer creates the outer product matrix correctly tailored to update multiple vectors at once
        dW2 = np.outer(errors, u_w)
        
        # Gradient for W1 (Center): dJ/du = sum(error * v_context)
        # Summing the influence of all context vectors (positive and negative) on the center word
        dW1 = np.dot(errors, V_context)
        
        # 4. Parameter Update (SGD)
        self.W1[center_word_idx] -= self.lr * dW1
        self.W2[indices] -= self.lr * dW2
        
        return loss

    def get_embedding(self, word_idx: int) -> np.ndarray:
        """
        Returns the embedding vector for a given word index.
        Typically, we use W1 (input weights) as the final word embeddings.
        """
        return self.W1[word_idx]

    def save_embeddings(self, path: str):
        """Save embeddings to a .npy file."""
        np.save(path, self.W1)

    def load_embeddings(self, path: str):
        """Load embeddings from a .npy file."""
        self.W1 = np.load(path)
