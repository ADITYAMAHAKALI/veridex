import unittest
import numpy as np
import math
from veridex.utils.confidence import (
    softmax_confidence,
    margin_confidence,
    entropy_confidence,
    distance_confidence,
    variance_confidence,
    default_confidence_for_heuristic
)

class TestConfidenceUtils(unittest.TestCase):
    
    def test_softmax_confidence(self):
        # Empty input
        self.assertEqual(softmax_confidence(np.array([])), 0.0)
        
        # Max confidence
        self.assertEqual(softmax_confidence(np.array([0.1, 0.2, 0.7])), 0.7)
        self.assertEqual(softmax_confidence(np.array([1.0, 0.0])), 1.0)
        
    def test_margin_confidence(self):
        # Too few classes
        self.assertEqual(margin_confidence(np.array([1.0]), top_k=2), 0.0)
        
        # Normal case
        probs = np.array([0.1, 0.2, 0.7])  # top2: 0.7, 0.2. margin: 0.5
        self.assertAlmostEqual(margin_confidence(probs), 0.5)
        
        # Less than top_k items avail (should return top item)
        # Actually logic says if < top_k return 0.0. Wait checking code.
        # Code: if probabilities.size < top_k: return 0.0
        self.assertEqual(margin_confidence(np.array([0.9]), top_k=2), 0.0)
        
        # Case where len(top_probs) >= 2 but original processing logic (redundant check in code?)
        # Code:
        # top_probs = np.sort(probabilities)[-top_k:]
        # if len(top_probs) >= 2: ...
        
        probs = np.array([0.6, 0.4])
        self.assertAlmostEqual(margin_confidence(probs), 0.2)

    def test_entropy_confidence(self):
        # Empty
        self.assertEqual(entropy_confidence(np.array([])), 0.0)
        
        # Low entropy (high confidence)
        probs = np.array([0.99, 0.01])
        conf = entropy_confidence(probs)
        self.assertGreater(conf, 0.9)
        
        # High entropy (low confidence) - uniform
        probs = np.array([0.5, 0.5]) # entropy 1 bit. max 1 bit. normalized 1.0. conf 0.0
        self.assertAlmostEqual(entropy_confidence(probs), 0.0)
        
        # Single element (max_entropy=0)
        probs = np.array([1.0])
        self.assertEqual(entropy_confidence(probs), 1.0)
        
    def test_distance_confidence(self):
        # Exact threshold match
        self.assertEqual(distance_confidence(0.5, threshold=0.5), 0.0)
        
        # Far from threshold (higher distance than threshold)
        # dist = |0.9 - 0.5| = 0.4
        # max_dist = 0.5 * 2 = 1.0. max_possible = max(|1.0-0.5|, 0.5) = 0.5
        # confidence = 0.4 / 0.5 = 0.8
        self.assertAlmostEqual(distance_confidence(0.9, threshold=0.5, max_distance=1.0), 0.8)
        
        # Higher is better flag
        # Base confidence 0.8. Add (0.9/1.0)*0.5 = 0.45. Sum 1.25. Clip 1.0.
        self.assertEqual(distance_confidence(0.9, threshold=0.5, max_distance=1.0, higher_is_better=True), 1.0)
        
        # Zero max_possible_distance protection (threshold=max_distance?)
        self.assertEqual(distance_confidence(0.0, threshold=0.0, max_distance=0.0), 0.0)

    def test_variance_confidence(self):
        # Single value
        self.assertEqual(variance_confidence([0.5]), 0.5)
        
        # Low variance (consistent)
        vals = [0.9, 0.91, 0.89]
        # Provide expected variance (e.g. variance of uniform distribution or random noise in [0,1])
        conf = variance_confidence(vals, expected_variance=0.1)
        self.assertGreater(conf, 0.9)
        
        # High variance (inconsistent)
        vals = [0.1, 0.9, 0.5]
        conf_low = variance_confidence(vals)
        self.assertLess(conf_low, 0.5)
        
        # Expected variance provided
        self.assertAlmostEqual(variance_confidence([0.5, 0.5], expected_variance=1.0), 1.0) # Variance 0. exp(-0)=1
        
        # Zero expected variance (and zero variance) -> 1.0
        self.assertEqual(variance_confidence([0.5, 0.5], expected_variance=0.0), 1.0)
        
        # Zero expected variance (and non-zero variance) -> 0.0
        self.assertEqual(variance_confidence([0.5, 0.6], expected_variance=0.0), 0.0)
        
        # Inverse=False (High variance = High confidence)
        # High var
        conf_high = variance_confidence(vals, inverse=False)
        self.assertGreater(conf_high, 0.0)

    def test_default_confidence_for_heuristic(self):
        # Known signal
        self.assertEqual(default_confidence_for_heuristic("frequency_artifacts"), 0.3)
        self.assertEqual(default_confidence_for_heuristic("aasist"), 0.9)
        
        # Unknown signal
        self.assertEqual(default_confidence_for_heuristic("unknown_signal"), 0.5)

if __name__ == '__main__':
    unittest.main()
