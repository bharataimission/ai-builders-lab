"""A tiny, beginner-friendly machine-learning sentiment classifier."""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Tiny teaching dataset only. It is not representative of real-world language.
TRAINING_EXAMPLES = [
    ("I love this", "positive"),
    ("This is wonderful", "positive"),
    ("I am happy with the result", "positive"),
    ("That was a great experience", "positive"),
    ("The product works really well", "positive"),
    ("I enjoyed learning this", "positive"),
    ("This is awful", "negative"),
    ("I hate this", "negative"),
    ("I am unhappy with the result", "negative"),
    ("That was a terrible experience", "negative"),
    ("The product works very badly", "negative"),
    ("I regret trying this", "negative"),
]


def build_model() -> Pipeline:
    """Create and train a text-classification pipeline."""
    texts = [text for text, _label in TRAINING_EXAMPLES]
    labels = [label for _text, label in TRAINING_EXAMPLES]

    model = Pipeline(
        [
            ("vectorizer", CountVectorizer()),
            ("classifier", MultinomialNB()),
        ]
    )
    model.fit(texts, labels)
    return model


def predict_sentiment(model: Pipeline, text: str) -> str:
    """Predict positive or negative sentiment for non-empty text."""
    if not text or not text.strip():
        raise ValueError("Please enter a sentence containing some text.")
    return str(model.predict([text.strip()])[0])


def main() -> None:
    """Run the interactive command-line demo."""
    model = build_model()
    print("Bharat AI Mission — Beginner Sentiment Classifier")
    print("Type a sentence to classify it, or type 'quit' to exit.")
    print("Note: this is a small learning demo, not a reliable sentiment service.")

    while True:
        try:
            sentence = input("\nEnter a sentence: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if sentence.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not sentence:
            print("Please enter some text.")
            continue

        try:
            sentiment = predict_sentiment(model, sentence)
            print(f"Prediction: {sentiment}")
        except ValueError as error:
            print(error)


if __name__ == "__main__":
    main()
