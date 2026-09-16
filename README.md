# 📰 Marathi News Categorizer

**An NLP mini-project that classifies Marathi news headlines into 12 categories using TF-IDF + Multinomial Naive Bayes, served by a Flask API and a Tailwind-styled web UI.**

*Mumbai University — BE Semester 7 (NLP syllabus)*

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20Naive%20Bayes-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/status-active-brightgreen)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
- [The Dataset](#the-dataset)
- [Design Notes & Known Limitations](#design-notes--known-limitations)
- [Retraining the Model](#retraining-the-model)
- [Future Improvements](#future-improvements)

---

## Overview

This project takes a Marathi news headline and predicts which of **12 categories** it belongs to — Politics, Sports, Entertainment, Health, Crime, Tech, Travel, Education, Fashion, Bhakti, Auto, or International.

It demonstrates the core NLP pipeline taught in the syllabus end-to-end:

```
Raw headline → Preprocessing → TF-IDF feature extraction → Multinomial Naive Bayes → Predicted category
```

...wired up to a real, working web app rather than a notebook demo.

## Features

- 🔤 **Real text classification** — TF-IDF + Multinomial Naive Bayes trained on labelled Marathi headlines, not keyword matching.
- ⚡ **Flask REST API** — a single `/api/categorize` endpoint returns the top 3 predicted categories with confidence scores.
- 🎨 **Responsive web UI** — Tailwind-styled interface with live results, confidence breakdown, and classification history.
- 🧠 **Explainability** — every prediction comes with a short, human-readable explanation of which words drove it.
- 🛡️ **Graceful offline fallback** — if the backend isn't running, the frontend falls back to a keyword-based simulation, clearly labelled as such.
- 🇮🇳 **Devanagari-aware preprocessing** — correctly handles quirks of Marathi text that trip up default NLP tooling (see [Design Notes](#design-notes--known-limitations)).

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, Tailwind CSS, vanilla JavaScript |
| Backend | Python, Flask, Flask-CORS |
| ML / NLP | scikit-learn (TF-IDF, Multinomial Naive Bayes), pandas |
| Data | Synthetically generated, template-based Marathi headline corpus |

## Project Structure

```
MarathiNLP/
├── index.html                      # Frontend UI
├── style.css                       # Custom styling on top of Tailwind
├── script.js                       # Frontend logic — calls the Flask API
├── README.md                       # You are here
└── backend/
    ├── app.py                      # Flask API — loads the trained model, serves predictions
    ├── train_model.py              # Trains TF-IDF + MultinomialNB, saves the model
    ├── generate_dataset.py         # Builds the synthetic training dataset
    ├── requirements.txt            # Python dependencies
    ├── data/
    │   └── marathi_news_dataset.csv    # 840 labelled headlines (70 per category)
    └── model/
        ├── vectorizer.pkl          # Fitted TF-IDF vectorizer
        ├── classifier.pkl          # Trained MultinomialNB model
        └── categories.pkl          # List of the 12 category labels
```

A trained model already ships in `model/`, so you can run the app immediately without retraining.

## Getting Started

### Prerequisites

- Python 3.9+
- A modern web browser

### 1. Clone and set up the backend

```bash
git clone <your-repo-url>
cd MarathiNLP/backend

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

The API starts at `http://127.0.0.1:5000`. Verify it's up:

```bash
curl http://127.0.0.1:5000/api/health
```

### 2. Open the frontend

Open `index.html` directly in a browser, or serve it with something like VS Code's Live Server extension. `script.js` calls `http://127.0.0.1:5000`, so **make sure the backend is running first**.

## How It Works

1. **Preprocessing** — strip everything except Devanagari characters and spaces, collapse whitespace.
2. **Tokenization** — split the cleaned text into word tokens.
3. **Feature extraction** — `TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))` converts text into weighted character n-gram vectors.
4. **Classification** — `MultinomialNB` scores the vector against all 12 learned category distributions.
5. **Explanation** — the model's learned word associations are used to surface which words in *your* headline most influenced the prediction.
6. **Response** — the top 3 categories (with confidence scores) and a plain-English explanation are returned to the frontend.

## API Reference

### `POST /api/categorize`

**Request**
```json
{ "headline": "नवीन तंत्रज्ञान बाजारात आलं" }
```

**Response**
```json
{
  "top_predictions": [
    { "category": "Tech", "score": 0.996 },
    { "category": "Auto", "score": 0.002 },
    { "category": "Politics", "score": 0.001 }
  ],
  "reasoning_english": "The word(s) 'आलं', 'नवीन' in the headline carry the strongest statistical association with the 'Tech' category, based on character/word patterns learned from the training data."
}
```

### `GET /api/health`

Returns `{ "status": "ok", "categories": [...] }` — useful for confirming the server and model loaded correctly.

## The Dataset

`backend/data/marathi_news_dataset.csv` contains 840 headlines (70 per category), **synthetically generated** by `generate_dataset.py` from category-specific Marathi vocabulary combined with sentence templates and city names.

Why synthetic? A real, license-clear, labelled Marathi news corpus large enough to train on wasn't available for this project, and scraping copyrighted news text isn't appropriate to redistribute in a student repo either. The generated data is enough for the model to genuinely learn per-category word/character patterns — verified against held-out headlines the model never saw during training — which is the point of the exercise.

> For a stronger, more realistic model, swap this CSV for a real labelled corpus (e.g. curated from Lokmat, Loksatta, or Saamana) using the same two columns: `headline,category`.

## Design Notes & Known Limitations

Two Devanagari-specific issues came up while building this — worth knowing for a report or viva:

1. **scikit-learn's default tokenizer breaks Devanagari words apart.** Its default `token_pattern` relies on Python's `\w`, which does *not* match combining vowel signs (matras — e.g. the `ी` in `नवीन`). Left unfixed, this silently splits `नवीन` into `नव` + `ीन` and corrupts the vocabulary.
2. **Marathi is morphologically rich.** Suffixes attach directly to word stems (`मुख्यमंत्री` → `मुख्यमंत्र्यांनी`), so whole-word features don't generalize to phrasing the model hasn't seen verbatim.

Both are solved here by using **character n-grams** (`analyzer="char_wb", ngram_range=(2, 4)`) instead of whole-word features — inflected forms share most of their n-grams with the base stem, so the model generalizes without needing a Marathi stemmer.

**Known limitation:** because training headlines come from templates, some genuinely ambiguous real-world phrasing can still be misclassified when a word is shared across categories (e.g. "हिवाळ्यात" / "in winter" appears in both Health and Travel templates). This is an honest limitation of a small n-gram model — see [Future Improvements](#future-improvements).

## Retraining the Model

```bash
cd backend
python generate_dataset.py   # regenerate the dataset (optional)
python train_model.py        # retrain and overwrite model/*.pkl, prints an accuracy report
```

## Future Improvements

- Swap the synthetic dataset for a real, labelled Marathi news corpus.
- Try a more powerful model (SVM, or a fine-tuned transformer such as MahaBERT/IndicBERT).
- Add proper stemming/lemmatization for Marathi to complement the n-gram approach.
- Deploy the Flask API (e.g. Render, Railway) and the frontend (e.g. Netlify, Vercel) for a live demo link.

---

<p align="center">Built as an NLP mini-project — Mumbai University, BE Semester 7</p>