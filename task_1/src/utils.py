import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid function.
    Uses np.clip to prevent overflow for large inputs.
    """
    return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

def cosine_similarity(v_a: np.ndarray, v_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Formula: (A . B) / (||A|| * ||B||)
    """
    dot_product = np.dot(v_a, v_b)
    norm_a = np.linalg.norm(v_a)
    norm_b = np.linalg.norm(v_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def get_negative_sampling_distribution(counts: np.ndarray, power: float = 0.75) -> np.ndarray:
    """
    Compute the probability distribution for negative sampling.
    Original paper suggests raising frequencies to the power of 3/4.
    """
    counts_pow = np.power(counts, power)
    return counts_pow / np.sum(counts_pow)
