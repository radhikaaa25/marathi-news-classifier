"""
app.py
------
Flask backend for the Marathi News Categorizer.

Endpoints:
    GET  /api/health        -> quick check that the server + model are up
    POST /api/categorize    -> { "headline": "..." }
                                returns top_predictions + reasoning_english,
                                matching the shape the frontend (script.js)
                                already expects.

Run:
    python app.py
Then open frontend/index.html (or index.html at the project root) in a
browser. The frontend calls this server at http://127.0.0.1:5000.
"""

import pickle
import re

from flask import Flask, jsonify, request
from flask_cors import CORS

VECTORIZER_PATH = "model/vectorizer.pkl"
MODEL_PATH = "model/classifier.pkl"
CATEGORIES_PATH = "model/categories.pkl"

app = Flask(__name__)
CORS(app)  # allow the static frontend (opened from file:// or another port) to call this API

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(CATEGORIES_PATH, "rb") as f:
    CATEGORIES = pickle.load(f)


from preprocess import preprocess


def explain(cleaned_headline, predicted_class):
    """
    Build a short, human-readable explanation.

    The model itself is trained on character n-grams (see train_model.py for
    why), which generalizes well to inflected Marathi words but doesn't
    produce readable "features" on its own (a raw n-gram like 'ित्र' means
    nothing to a reader). So for the explanation only, we re-score each
    *whole word* of the input headline by re-vectorizing it alone and taking
    its dot product against the predicted class's learned log-probabilities.
    This surfaces which actual Marathi words in the headline the model
    leaned on most, in a form a person can read.
    """
    class_index = list(model.classes_).index(predicted_class)
    log_probs = model.feature_log_prob_[class_index]

    words = cleaned_headline.split()
    if not words:
        return (
            f"No strong keywords were detected, so the model defaulted to its "
            f"most likely overall category, '{predicted_class}'."
        )

    word_scores = []
    for word in words:
        word_vec = vectorizer.transform([word])
        score = word_vec.multiply(log_probs).sum()
        word_scores.append((word, score))

    word_scores.sort(key=lambda x: x[1], reverse=True)
    top_words = [w for w, _ in word_scores[:2]]

    terms_str = "', '".join(top_words)
    return (
        f"The word(s) '{terms_str}' in the headline carry the strongest "
        f"statistical association with the '{predicted_class}' category, "
        f"based on character/word patterns learned from the training data."
    )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "categories": CATEGORIES})


@app.route("/api/categorize", methods=["POST"])
def categorize():
    data = request.get_json(silent=True) or {}
    headline = (data.get("headline") or "").strip()

    if not headline:
        return jsonify({"error": "Please provide a non-empty 'headline'."}), 400

    cleaned = preprocess(headline)
    if not cleaned:
        return jsonify({"error": "Headline must contain Marathi (Devanagari) text."}), 400

    vec = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(vec)[0]

    # rank all categories by probability, take top 3
    ranked = sorted(
        zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True
    )
    top3 = ranked[:3]
    predicted_class = top3[0][0]

    reasoning = explain(cleaned, predicted_class)

    response = {
        "top_predictions": [
            {"category": category, "score": float(score)} for category, score in top3
        ],
        "reasoning_english": reasoning,
        "preprocessed_text": cleaned,
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
