## Project: Extractive & Abstractive Summarization

This project implements:

- **Extractive summarization** using TF‑IDF features and multiple classifiers:
  - RandomForest, Naive Bayes, SVM, Logistic Regression, Linear Regression.
- **Abstractive summarization** using:
  - A Groq‑hosted LLM baseline.
  - A fine‑tuned seq2seq model (e.g., BART) on the CNN/DailyMail dataset.
- Evaluation with **sentence‑level classification metrics** and **document‑level ROUGE‑1 / ROUGE‑L**.

The main entrypoint for extractive models is `main.py`.  
The LLM fine‑tuning and evaluation script is `llm_cnn.py`.

---

## 1. Environment Setup

### 1.1. Create and activate a virtual environment (recommended)

From the project root:

```bash
cd /Users/shloknanani/Desktop/ai_project

# Using venv (Python 3.x)
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate   # Windows PowerShell
```

### 1.2. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` includes:

- **scikit-learn**: classical ML models and metrics
- **nltk**: sentence tokenization
- **datasets**: Hugging Face datasets (CNN/DailyMail, etc.)
- **rouge-score**: ROUGE‑1 / ROUGE‑L metrics
- **joblib**: saving/loading trained models
- **requests**: Groq API calls
- **transformers**, **torch**: LLM fine‑tuning and inference

### 1.3. Download NLTK tokenizer data (once)

The code already tries to download `punkt` if missing, but you can also run:

```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## 2. Data

This project uses:

- **CNN/DailyMail** summarization dataset via Hugging Face  
  [`abisee/cnn_dailymail`](https://huggingface.co/datasets/abisee/cnn_dailymail).
- **Local lecture notes** stored under `data/` (e.g., `lecture_notes.txt`, `lecture_notes_ml_intro.txt`, etc.).

The Hugging Face datasets are downloaded automatically on first use and cached under your HF cache directory.

---

## 3. Running the Extractive Pipeline (`main.py`)

`main.py` has three modes:

- **`train`** – train and save an extractive summarization model.
- **`test`** – load a saved model, evaluate it, and show example summaries.
- **`llm`** – evaluate a Groq LLM baseline on CNN/DailyMail.

You can also choose the classifier with `--classifier`:

- `rf` (default) – RandomForest
- `nb` – Naive Bayes
- `svm` – Linear SVM
- `logreg` – Logistic Regression
- `linreg` – Linear Regression (regression turned into binary via threshold)

### 3.1. Train an extractive model

From the project root (with your virtualenv activated):

```bash
# Example: RandomForest (default)
python main.py --mode train --classifier rf

# Example: Logistic Regression
python main.py --mode train --classifier logreg
```

This will:

- Load labeled sentence‑level data via `get_training_data` in `src/data_loader.py`
  (CNN/DailyMail + optional HF dataset / local notes).
- Split into train/test (internally).
- Fit a `TextPreprocessor` (TF‑IDF with English stop words).
- Train the chosen classifier.
- Save:
  - Model to `artifacts/summary_model_<classifier>.joblib`  
    (e.g., `summary_model_rf.joblib`).
  - TF‑IDF vectorizer to `artifacts/tfidf_preprocessor.joblib`.

### 3.2. Evaluate a trained extractive model

```bash
# Evaluate RandomForest model
python main.py --mode test --classifier rf

# Evaluate SVM model
python main.py --mode test --classifier svm
```

This will:

1. Load the appropriate model + vectorizer from `artifacts/`.
2. Rebuild a train/test split for evaluation and print a **classification report**
   (precision, recall, F1 for labels 0/1).
3. Compute **document‑level ROUGE‑1 and ROUGE‑L** on a subset of CNN/DailyMail
   test articles.
4. Generate and print an extractive summary for a local lecture note file
   (see `lecture_file_to_summarize` in `main.py`).

Make sure you’ve run `--mode train` at least once before `--mode test`
so the artifacts exist.

---

## 4. Groq LLM Baseline (Abstractive)

`main.py --mode llm` evaluates an off‑the‑shelf Groq LLM on CNN/DailyMail.

### 4.1. Set up your Groq API key

Export your Groq API key as environment variables (shell example):

```bash
export GROQ_API_KEY="your_api_key_here"
# Optional: override default model
export GROQ_MODEL_NAME="llama-3.1-8b-instant"
```

### 4.2. Run the evaluation

```bash
python main.py --mode llm
```

This will:

- Fetch a small subset of CNN/DailyMail test articles.
- Call the Groq API to generate abstractive summaries.
- Compute and print ROUGE‑1 / ROUGE‑L F1 scores for the LLM baseline.

If `GROQ_API_KEY` is not set, the script will print a warning and skip scoring.

---

## 5. Fine‑Tuning a Seq2Seq LLM on CNN/DailyMail (`llm_cnn.py`)

`llm_cnn.py` handles fine‑tuning and evaluating a Hugging Face seq2seq model
(e.g., BART) on CNN/DailyMail.

### 5.1. Train a seq2seq model

Example (smaller run for local machines):

```bash
python llm_cnn.py \
  --mode train \
  --model_name facebook/bart-base \
  --output_dir artifacts/llm_cnn \
  --max_train_samples 20000 \
  --max_val_samples 2000 \
  --num_train_epochs 1
```

This will:

- Load the CNN/DailyMail dataset (`3.0.0`).
- Select up to `max_train_samples` training examples and
  `max_val_samples` validation examples.
- Tokenize `article` (inputs) and `highlights` (targets).
- Fine‑tune `model_name` using `Seq2SeqTrainer`.
- Save the fine‑tuned model and tokenizer to `output_dir`.

### 5.2. Evaluate a fine‑tuned model

After training, evaluate on a small test subset:

```bash
python llm_cnn.py \
  --mode eval \
  --output_dir artifacts/llm_cnn \
  --split test[:20]
```

This will:

- Load the tokenizer and model from `output_dir`.
- Run generation on the specified `split` of CNN/DailyMail.
- Compute and print average **ROUGE‑1** and **ROUGE‑L** F1 scores.

---

## 6. Files Overview

- **`main.py`**  
  Orchestrates data loading, preprocessing, training/testing of extractive models,
  Groq LLM baseline, and ROUGE evaluation.

- **`src/data_loader.py`**  
  - Loads CNN/DailyMail and optional HF summarization datasets.  
  - Uses **ROUGE‑L** overlap with gold summaries to label top‑k “important”
    sentences (label 1 vs 0).  
  - Loads local lecture notes and manual labels.

- **`src/preprocessor.py`**  
  - `TextPreprocessor` wrapping `TfidfVectorizer(stop_words="english")`.

- **`src/model.py`**  
  - `SummaryModel` wrapping a `RandomForestClassifier` with
    `class_weight="balanced"`, OOB scoring, etc.

- **`src/nb_model.py`, `src/svm_model.py`, `src/logreg_model.py`, `src/linreg_model.py`**  
  - Wrapper classes for Naive Bayes, SVM, Logistic Regression,
    and Linear Regression summary models.

- **`src/summarizer.py`**  
  - `Summarizer` that:
    - Splits a document into sentences.  
    - Computes probabilities (if available) or predictions.  
    - Ranks sentences by probability and selects the top `num_sentences`
      as the summary.

- **`llm_cnn.py`**  
  - Fine‑tuning and evaluation script for a seq2seq LLM (e.g., BART)
    on CNN/DailyMail.

- **`data/lecture_notes*.txt`**  
  - Local lecture notes used as additional training and demo data.

- **`requirements.txt`**  
  - Python dependencies.

- **`.gitignore`**  
  - Ignores `artifacts/`, large Parquet shards, etc. so model checkpoints
    and dataset caches are not committed.

---

## 7. Typical Workflow

1. **Set up environment**
   - Create venv, install `requirements.txt`, ensure `nltk` and `datasets` work.

2. **Train an extractive model**
   - `python main.py --mode train --classifier rf`

3. **Evaluate extractive model + see example summaries**
   - `python main.py --mode test --classifier rf`

4. **(Optional) Compare alternative classifiers**
   - Train/test with `--classifier nb`, `svm`, `logreg`, `linreg`.

5. **(Optional) Run Groq LLM baseline**
   - Set `GROQ_API_KEY`, run `python main.py --mode llm`.

6. **(Optional) Fine‑tune BART on CNN/DailyMail**
   - Train with `llm_cnn.py --mode train ...`, then evaluate with `--mode eval`.

---

## 8. Cited Dataset for YouTube Lecture Transcripts
https://www.kaggle.com/datasets/jfcaro/5000-transcripts-of-youtube-ai-related-videos

