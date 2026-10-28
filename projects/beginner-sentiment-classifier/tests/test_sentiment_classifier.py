"""Basic tests for the sentiment classifier."""

import unittest

from src.sentiment_classifier import build_model, predict_sentiment


class SentimentClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model()

    def test_returns_a_supported_label(self):
        result = predict_sentiment(self.model, "I love this wonderful experience")
        self.assertIn(result, {"positive", "negative"})

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            predict_sentiment(self.model, "   ")

    def test_learns_simple_positive_example(self):
        self.assertEqual(predict_sentiment(self.model, "I love this"), "positive")

    def test_learns_simple_negative_example(self):
        self.assertEqual(predict_sentiment(self.model, "This is awful"), "negative")


if __name__ == "__main__":
    unittest.main()
