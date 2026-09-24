"""
train_model.py
---------------
Trains the Marathi News Categorizer model:
    raw headline text -> TF-IDF features -> Multinomial Naive Bayes

Run:
    python train_model.py

Reads:  data/marathi_news_dataset.csv   (headline, category)
Writes: model/vectorizer.pkl
        model/classifier.pkl
        model/categories.pkl
"""

import pickle
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/marathi_news_dataset.csv"
VECTORIZER_PATH = "model/vectorizer.pkl"
MODEL_PATH = "model/classifier.pkl"
CATEGORIES_PATH = "model/categories.pkl"


from preprocess import preprocess


def load_data(path):
    df = pd.read_csv(path, encoding="utf-8")
    headlines = df["headline"].apply(preprocess).tolist()
    labels = df["category"].tolist()
    return headlines, labels


def main():
    headlines, labels = load_data(DATA_PATH)
    print(f"Loaded {len(headlines)} labelled headlines.")

    X_train, X_test, y_train, y_test = train_test_split(
        headlines, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Two things matter for Marathi/Devanagari text with sklearn's TfidfVectorizer:
    #
    # 1. The default token_pattern (\b\w\w+\b) breaks Devanagari words apart,
    #    because combining vowel signs (matras, Unicode category Mc/Mn, e.g.
    #    the ी in नवीन) are NOT matched by \w in Python's re module. That
    #    would split "नवीन" into "नव" + "ीन" and destroy the vocabulary if we
    #    used word-level tokens with the default pattern.
    #
    # 2. Marathi is morphologically rich (case/gender/number suffixes attach
    #    to the stem: मुख्यमंत्री -> मुख्यमंत्र्यांनी, चित्रपट -> चित्रपटाचा).
    #    A pure whole-word vocabulary would treat every inflected form as an
    #    unseen, unrelated word, so a headline phrased slightly differently
    #    from the training templates would get no signal at all.
    #
    # Character n-grams (analyzer="char_wb", within word boundaries) solve
    # both problems at once: they naturally keep matras attached to their
    # base letter, and they let inflected forms share most of their n-grams
    # with the base stem (e.g. "मुख्यमंत्र" overlaps heavily whether or not
    # a case suffix follows), which is the standard trick for classifying
    # morphologically rich / low-resource languages without a stemmer.
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.3)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    categories = sorted(model.classes_.tolist())

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(CATEGORIES_PATH, "wb") as f:
        pickle.dump(categories, f)

    print(f"Saved vectorizer -> {VECTORIZER_PATH}")
    print(f"Saved model      -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
