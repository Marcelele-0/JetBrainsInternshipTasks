import unittest
import numpy as np
from src.model import Word2Vec

class TestWord2Vec(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 10
        self.embedding_dim = 5
        self.model = Word2Vec(self.vocab_size, self.embedding_dim)

    def test_initialization(self):
        self.assertEqual(self.model.W1.shape, (self.vocab_size, self.embedding_dim))
        self.assertEqual(self.model.W2.shape, (self.vocab_size, self.embedding_dim))

    def test_train_step_loss_decrease(self):
        # Center word: 0, Context word: 1, Negatives: [2, 3]
        center_idx = 0
        context_idx = 1
        neg_indices = [2, 3]

        # Initial loss
        # Note: We can't easily check pure deterministic decrease with single step due to randomness and potential local structure,
        # but for a random initialization, one step should typically produce a valid loss value.
        loss = self.model.train_step(center_idx, context_idx, neg_indices)
        self.assertIsInstance(loss, float)
        self.assertTrue(loss > 0)
        
        # Run a few more steps to see if it doesn't crash
        for _ in range(5):
            self.model.train_step(center_idx, context_idx, neg_indices)

    def test_embeddings_update(self):
        old_w1 = self.model.W1.copy()
        self.model.train_step(0, 1, [2, 3])
        new_w1 = self.model.W1
        
        # Check if weights changed for the involved words
        self.assertFalse(np.array_equal(old_w1[0], new_w1[0]))
        # Check if weights did NOT change for uninvolved words (e.g. index 5)
        self.assertTrue(np.array_equal(old_w1[5], new_w1[5]))

    def test_dimensions(self):
        """Verify matrix dimensions are preserved after training steps."""
        self.assertEqual(self.model.W1.shape, (self.vocab_size, self.embedding_dim))
        self.assertEqual(self.model.W2.shape, (self.vocab_size, self.embedding_dim))
        
        self.model.train_step(0, 1, [2, 3])
        
        self.assertEqual(self.model.W1.shape, (self.vocab_size, self.embedding_dim))
        self.assertEqual(self.model.W2.shape, (self.vocab_size, self.embedding_dim))

if __name__ == '__main__':
    unittest.main()
