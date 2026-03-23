import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from src.model import Word2Vec
from src.preprocessing import TextPreprocessor
from src.utils import cosine_similarity
import argparse
import os

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def main():
    # 0. Reproducibility
    set_seed(42)

    # 1. Setup Toy Dataset (simple sentences for demonstration)
    sentences = [
        ["cat", "purr"], 
        ["dog", "bark"], 
        ["cat", "meow"], 
        ["dog", "woof"],
        ["cat", "animal"],
        ["dog", "animal"],
        ["fish", "swim"],
        ["bird", "fly"],
        ["dog", "cat"],  # Shared context
        ["cat", "milk"],
    ] 
    print(f"Dataset: {len(sentences)} sentences")

    # 2. Preprocessing
    preprocessor = TextPreprocessor(min_count=1)
    preprocessor.build_vocab(sentences)

    # Subsampling
    # Apply subsampling to discard frequent words which reduces noise and speeds up training.
    sentences = preprocessor.subsample(sentences, threshold=1e-5)
    
    vocab_size = preprocessor.vocab_size
    embedding_dim = 10
    learning_rate = 0.05
    num_epochs = 200
    window_size = 2
    num_negatives = 3

    # 3. Model Initialization
    # Initialize Word2Vec model with Skip-gram architecture and Negative Sampling.
    model = Word2Vec(vocab_size=vocab_size, embedding_dim=embedding_dim, learning_rate=learning_rate)

    print(f"Training Word2Vec (Vocab: {vocab_size}, Dim: {embedding_dim})...")
    
    # 4. Training Loop
    losses = []
    
    # Use tqdm for progress bar
    progress_bar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    for epoch in progress_bar:
        epoch_loss = 0
        count = 0
        
        # Generator for each epoch to get fresh negatives
        train_data = preprocessor.generate_training_data(sentences, window_size, num_negatives)
        
        for center_id, context_id, negatives in train_data:
            loss = model.train_step(center_id, context_id, negatives)
            epoch_loss += loss
            count += 1
            
        if count > 0:
            epoch_loss /= count
        losses.append(epoch_loss)
            
        if epoch % 20 == 0:
            progress_bar.set_postfix({"Loss": f"{epoch_loss:.4f}"})

    # 5. Save Embeddings
    os.makedirs("data", exist_ok=True)
    model.save_embeddings("data/embeddings.npy")
    print("Embeddings saved to data/embeddings.npy")

    # 6. Evaluation (Similarity Check)
    word_pairs = [("cat", "dog"), ("cat", "purr"), ("dog", "bark"), ("cat", "fish")]
    print("\nSimilarities (Cosine):")
    for w1, w2 in word_pairs:
        if w1 in preprocessor.word2id and w2 in preprocessor.word2id:
            id1 = preprocessor.word2id[w1]
            id2 = preprocessor.word2id[w2]
            v1 = model.get_embedding(id1)
            v2 = model.get_embedding(id2)
            sim = cosine_similarity(v1, v2)
            print(f"Similarity ({w1}, {w2}): {sim:.4f}")
        else:
            print(f"Words {w1} or {w2} not in vocabulary.")

    # 7. Visualization (Loss)
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("data/loss_plot.png")
    print("Loss plot saved to data/loss_plot.png")

    # 8. Visualization (PCA)
    try:
        from sklearn.decomposition import PCA
        
        # Get all embeddings
        vectors = model.W1
        words = [preprocessor.id2word[i] for i in range(vocab_size)]
        
        pca = PCA(n_components=2)
        reduced_vectors = pca.fit_transform(vectors)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])
        
        for i, word in enumerate(words):
            plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]))
            
        plt.title("Word Embeddings Visualization (PCA)")
        plt.savefig("data/embeddings_pca.png")
        print("PCA plot saved to data/embeddings_pca.png")
        
    except ImportError:
        print("scikit-learn not installed. Skipping PCA visualization.")
    except Exception as e:
        print(f"PCA visualization failed: {e}")

if __name__ == "__main__":
    main()
