import unittest
from veridex.core.signal import BaseSignal, DetectionResult
from veridex.eval import evaluate_signal, EvaluationDataset
from veridex.eval.metrics import calculate_metrics

class MockSignal(BaseSignal):
    @property
    def name(self):
        return "mock_signal"

    @property
    def dtype(self):
        return "text"

    def run(self, input_data):
        # Determine score based on input content for predictability
        if "ai" in input_data:
            return DetectionResult(score=0.9, confidence=1.0)
        elif "human" in input_data:
            return DetectionResult(score=0.1, confidence=1.0)
        else:
            return DetectionResult(score=0.5, confidence=0.5)

class TestEvaluationFramework(unittest.TestCase):
    def test_metrics_calculation(self):
        y_true = [0, 1, 0, 1]
        y_scores = [0.1, 0.9, 0.2, 0.8]
        metrics = calculate_metrics(y_true, y_scores)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["auroc"], 1.0)

    def test_dataset_creation(self):
        data = [("human text", 0), ("ai text", 1)]
        dataset = EvaluationDataset.from_list(data)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.samples[0].label, 0)

    def test_evaluator_runner(self):
        data = [
            ("human text", 0),
            ("ai text", 1),
            ("uncertain", 0) # Should get 0.5 score
        ]
        signal = MockSignal()
        results = evaluate_signal(signal, data)

        self.assertEqual(results["signal_name"], "mock_signal")
        self.assertEqual(results["num_samples"], 3)
        self.assertEqual(results["num_errors"], 0)

        # Check metrics
        # Scores: 0.1, 0.9, 0.5
        # Labels: 0, 1, 0
        # Threshold 0.5:
        # Preds: 0, 1, 1
        # TP=1, TN=1, FP=1, FN=0
        # Accuracy: 2/3 = 0.666
        metrics = results["metrics"]
        self.assertAlmostEqual(metrics["accuracy"], 2/3, places=2)

if __name__ == '__main__':
    unittest.main()
