# Beginner Sentiment Classifier

A small, beginner-friendly machine learning project for the Bharat AI Mission community.

Build a simple text sentiment classifier that predicts whether a short sentence sounds **positive** or **negative**. It uses scikit-learn's `CountVectorizer` and `MultinomialNB`, with a tiny example dataset included in the code.

## Why this project?

- **Real machine learning:** trains a text classifier from examples.
- **No API key:** everything runs locally on your computer.
- **Beginner-friendly:** short code, clear comments, and a small test suite.
- **A starting point:** contributors can improve the dataset, evaluation, documentation, and eventually add Indian-language examples.

> Learning demo only. The training data is intentionally tiny and the predictions are not reliable enough for real-world decisions.

## Requirements

- Python 3.10 or newer
- `pip`

## Run it

From this folder, create and activate a virtual environment.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/sentiment_classifier.py
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/sentiment_classifier.py
```

Example interaction:

```text
Enter a sentence (or type 'quit' to exit): I love learning AI
Prediction: positive

Enter a sentence (or type 'quit' to exit): This was a terrible experience
Prediction: negative
```

Predictions may vary for unfamiliar words because the model is trained on a very small sample.

## Run tests

With the virtual environment activated:

```bash
python -m unittest discover -s tests -v
```

## How it works

1. The code defines a small set of example sentences with positive or negative labels.
2. `CountVectorizer` converts text into numeric word-count features.
3. `MultinomialNB` learns patterns associated with each label.
4. The trained model predicts a label for new text.

## Project structure

```text
beginner-sentiment-classifier/
├── README.md
├── requirements.txt
├── src/
│   └── sentiment_classifier.py
└── tests/
    └── test_sentiment_classifier.py
```

## Ideas for contributors

- Add more balanced, clearly licensed English training examples.
- Add tests for empty input and unknown words.
- Report accuracy on a held-out dataset instead of relying only on examples.
- Add a language selector and explore Indian-language sentiment data.
- Improve explanations of model limitations and bias.
- Add a simple web interface after the command-line version is stable.

Before adding a dataset, document its source, license, language, and any known limitations. Do not add private or sensitive text.

## Contributing

1. Check the repository's `CONTRIBUTING.md` and Code of Conduct.
2. Look for an open issue or create one describing your proposed change.
3. Make a focused change and run the tests.
4. Open a pull request using the repository's pull request template.

## License

This project is intended to be shared under the MIT License, consistent with the parent AI Builders Lab repository. Check the repository's license before redistributing this project separately.
