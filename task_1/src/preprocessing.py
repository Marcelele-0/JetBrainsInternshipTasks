import random
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Generator

class TextPreprocessor:
    def __init__(self, min_count: int = 5):
        self.min_count = min_count
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.word_counts: Dict[str, int] = {}
        self.total_words_count = 0
        self.vocab_size = 0

    def build_vocab(self, sentences: List[List[str]]) -> None:
        """
        Build vocabulary from a list of tokenized sentences.
        """
        all_words = [token for sentence in sentences for token in sentence]
        self.total_words_count = len(all_words)
        
        counts = Counter(all_words)
        self.word_counts = {w: c for w, c in counts.items() if c >= self.min_count}
        
        sorted_words = sorted(self.word_counts.keys(), key=lambda w: self.word_counts[w], reverse=True)
        
        self.word2id = {w: i for i, w in enumerate(sorted_words)}
        self.id2word = {i: w for i, w in enumerate(sorted_words)}
        self.vocab_size = len(self.word2id)
        
        print(f"Vocab size: {self.vocab_size}")

    def subsample(self, sentences: List[List[str]], threshold: float = 1e-5) -> List[List[str]]:
        """
        Apply subsampling to discard frequent words.
        Formula: P(w_i) = 1 - sqrt(t / f(w_i)).
        This technique accelerates training and improves vector quality for rare words.
        """
        subsampled_sentences = []
        if self.total_words_count == 0:
            return sentences

        for sentence in sentences:
            kept_words = []
            for word in sentence:
                if word not in self.word_counts:
                    # Rare words (below min_count) are usually discarded during vocab build or handled outside.
                    # If we encounter them here, let's keep them (or discard, depending on policy).
                    continue
                
                freq = self.word_counts[word] / self.total_words_count
                
                # Probability of discarding the word
                p_discard = 1.0 - np.sqrt(threshold / freq)
                
                # Keep word if random > discard probability
                # If freq < threshold, p_discard < 0, so random() > p_discard is always true (keep).
                if random.random() > p_discard:
                    kept_words.append(word)
            subsampled_sentences.append(kept_words)
        return subsampled_sentences

    def generate_training_data(self, sentences: List[List[str]], window_size: int, num_negatives: int) -> Generator[Tuple[int, int, List[int]], None, None]:
        """
        Generator yielding training samples: (center_word_idx, context_word_idx, list_of_negative_indices).
        Uses negative sampling.
        """
        if self.vocab_size == 0:
             return
             
        # Calculate negative sampling distribution based on word counts in vocab order
        vocab_counts = np.array([self.word_counts[self.id2word[i]] for i in range(self.vocab_size)])
        neg_dist = np.power(vocab_counts, 0.75)
        neg_dist /= np.sum(neg_dist)
        
        all_indices = np.arange(self.vocab_size)

        for sentence in sentences:
            # Convert words to IDs
            sentence_ids = [self.word2id[w] for w in sentence if w in self.word2id]
            
            for i, center_id in enumerate(sentence_ids):
                # Define context window with dynamic boundary checks
                start = max(0, i - window_size)
                end = min(len(sentence_ids), i + window_size + 1)
                
                context_indices = [sentence_ids[j] for j in range(start, end) if j != i]
                
                for context_id in context_indices:
                    # Sample negatives
                    # We want to exclude center and context words from negatives.
                    # For simplicity/performance, pure random choice often used.
                    negative_samples = np.random.choice(all_indices, size=num_negatives, p=neg_dist, replace=True)
                    
                    yield (center_id, context_id, negative_samples.tolist())

        for sentence in sentences:
            # Convert words to IDs
            sentence_ids = [self.word2id[w] for w in sentence if w in self.word2id]
            
            for i, center_id in enumerate(sentence_ids):
                # Define context window with dynamic boundary checks
                start = max(0, i - window_size)
                end = min(len(sentence_ids), i + window_size + 1)
                
                context_indices = [sentence_ids[j] for j in range(start, end) if j != i]
                
                for context_id in context_indices:
                    # Sample negatives
                    # We want to exclude center and context words from negatives, ideally
                    # For simplicity/performance, pure random choice often used, but we can try to filter
                    negative_samples = np.random.choice(all_indices, size=num_negatives, p=neg_dist, replace=True)
                    
                    yield (center_id, context_id, negative_samples.tolist())
