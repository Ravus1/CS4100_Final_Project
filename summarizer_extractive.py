"""
Pipeline:
1. Load dataset (1 sentence per row)
2. Train/test split
3. Embed sentences using SentenceTransformers
4. Train LogisticRegression classifier
5. Evaluate with ROUGE
6. Save/load trained model
"""

import pandas as pd
import nltk
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer

# Download tokenizer just in case
nltk.download("punkt")

# CONFIGURATION

DATASET_PATH = "lecture_dataset.csv"
MODEL_SAVE_PATH = "trained_summarizer.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# 1. LOAD DATA 

def load_dataset(path):
    """
    Loads a dataset structured as:
      text: sentence
      label: 1 (important) or 0 (not important)
    """
    df = pd.read_csv(path)
    print("Dataset loaded:", df.shape)
    return df


# 2. EMBEDDINGS

def embed_sentences(sentences, model):
    """
    Converts sentences → numerical vectors using SBERT.
    """
    return model.encode(sentences, convert_to_numpy=True)


# 3. TRAIN MODEL

def train_classifier(X_train, y_train):
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    return clf


# 4. EVALUATION

def evaluate(model, embeddings_test, sentences_test, y_test):
    predictions = model.predict(embeddings_test)

    gold_summary = " ".join(
        [s for s, label in zip(sentences_test, y_test) if label == 1]
    )
    predicted_summary = " ".join(
        [s for s, p in zip(sentences_test, predictions) if p == 1]
    )

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(gold_summary, predicted_summary)

    print("\n===== ROUGE SCORES =====")
    print("ROUGE-1:", scores["rouge1"])
    print("ROUGE-L:", scores["rougeL"])
    print("========================")

    return scores


# 5. SAVE MODEL

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


# 6. LOAD MODEL 

def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded model from {path}")
    return model


# MAIN RUN FUNCTION

def main():

    print("\n=== STEP 1: LOADING DATA ===")
    df = load_dataset(DATASET_PATH)

    sentences = df["text"].tolist()
    labels = df["label"].tolist()
    print("Total sentences:", len(sentences))

    print("\n=== STEP 2: LOADING EMBEDDING MODEL ===")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Embedding sentences...")
    sentence_embeddings = embed_sentences(sentences, embedder)

    print("\n=== STEP 3: TRAIN/TEST SPLIT ===")
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        sentence_embeddings,
        labels,
        sentences,
        test_size=0.2,
        random_state=42
    )

    print("Train size:", len(X_train), "| Test size:", len(X_test))

    print("\n=== STEP 4: TRAINING CLASSIFIER ===")
    classifier = train_classifier(X_train, y_train)

    print("\n=== STEP 5: EVALUATION ===")
    evaluate(classifier, X_test, s_test, y_test)

    print("\n=== STEP 6: SAVING MODEL ===")
    save_model(classifier, MODEL_SAVE_PATH)

    print("\n=== Done! Model trained and saved. ===")


if __name__ == "__main__":
    main()
