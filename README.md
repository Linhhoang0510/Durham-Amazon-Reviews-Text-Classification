# Amazon Food Reviews — Multi-Class Text Classification

> Predicting review ratings (1–5) from raw text using classical ML and deep learning.

![Amazon](https://github.com/user-attachments/assets/a30f18b6-6e40-4c57-ae75-39c75899d674)

---

## Objective

Build and compare **four** machine-learning models that predict the star rating of an Amazon food review based solely on its written content, framing the problem as a **5-class classification** task.

## Dataset

| Property | Value |
|---|---|
| Source | [Amazon Food Reviews (Balanced)](https://www.kaggle.com/datasets) |
| Records | 540,031 reviews |
| Classes | 1 ★ – 5 ★
| Features used | `Summary` + `Text` → combined into `total_reviews` |

---

## Pipeline

```
Raw CSV
  │
  ▼
Data Cleaning ── handle nulls, strip whitespace, drop duplicates
  │
  ▼
EDA ── score distribution, class balance check
  │
  ▼
Text Preprocessing
  ├─ Tokenisation (NLTK word_tokenize)
  ├─ Stopword & punctuation removal
  ├─ Regex filtering (alphabetic tokens only)
  └─ Lemmatisation (WordNetLemmatizer + POS-tag mapping)
  │
  ▼
Feature Engineering
  ├─ TF-IDF vectors ──────────────► Complement Naïve Bayes
  ├─ Sentence embeddings ─────────► KNN (all-MiniLM-L6-v2)
  └─ Keras TextVectorization ─────► CNN / Bi-LSTM
  │
  ▼
Evaluation ── Accuracy · F1 (macro) · Precision · Recall · Confusion Matrix
```

---

## Models & Results

| # | Model | Representation | Accuracy | F1 (macro) |
|---|---|---|---|---|
| 1 | **Complement Naïve Bayes** (baseline) | TF-IDF | **0.543** | 0.420 |
| 2 | **KNN** (best *k* via 10-fold CV) | Sentence-BERT (`all-MiniLM-L6-v2`) | 0.449 | 0.380 |
| 3 | **CNN** | Keras `TextVectorization` → Embedding → Conv1D | — | — |
| 4 | **Bi-LSTM** | Keras `TextVectorization` → Embedding → Bidirectional LSTM | — | — |

> **Note:** CNN and LSTM results are available upon re-execution — the notebook cells were run in a Kaggle GPU session whose outputs were not persisted.

### Key hyperparameters

| Parameter | CNN | Bi-LSTM |
|---|---|---|
| Vocab size | 20,000 | 20,000 |
| Max sequence length | 200 | 200 |
| Embedding dim | 20 | 20 |
| Optimiser / LR | Adam / 0.01 | Adam / 0.01 |
| Early stopping patience | 4 epochs | 4 epochs |
| Batch size | 1,000 | 1,000 |

---

## Project Structure

```
Amz-Reviews-Classifier/
├── main.ipynb             # End-to-end pipeline: EDA → preprocessing → modelling → evaluation
├── trained_model.ipynb    # Model serialisation & inference helper (predict_food_review)
├── text_clean.py          # Reusable text-cleaning function (tokenise, clean, lemmatise)
├── document_embedding.py  # GloVe-based document embedding utility
├── Amazon-Food-Reviews.csv  # Dataset (git-ignored, ~115 MB)
├── .gitignore
└── README.md
```

---

## Quickstart

```bash
# 1 — Clone
git clone https://github.com/<your-username>/Amz-Reviews-Classifier.git
cd Amz-Reviews-Classifier

# 2 — Create environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3 — Install dependencies
pip install numpy pandas scikit-learn tensorflow nltk sentence-transformers seaborn matplotlib

# 4 — Download NLTK data (one-time)
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# 5 — Run
#   Open main.ipynb in Jupyter / VS Code and execute all cells.
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.x-154f5b)
![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-latest-blueviolet)

---

## License

This project is released for **educational and research purposes**. The dataset is sourced from Kaggle — please refer to the original dataset license for usage terms.
