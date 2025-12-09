from src.data_loader import get_training_data, load_and_process_hf_summarization_dataset
from src.preprocessor import TextPreprocessor
from src.model import SummaryModel
from src.nb_model import NaiveBayesSummaryModel
from src.svm_model import SVMSummaryModel
from src.logreg_model import LogisticRegressionSummaryModel
from src.linreg_model import LinearRegressionSummaryModel
from src.summarizer import Summarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import load_dataset
from rouge_score import rouge_scorer
from joblib import dump, load
import argparse
import os
import requests

# used AI for debugging, generating synthetic data, syntax.


def load_single_lecture_note(file_path):
    """Loads a single lecture note for summarization."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return ""


ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "summary_model_rf.joblib")
NB_MODEL_PATH = os.path.join(ARTIFACT_DIR, "summary_model_nb.joblib")
SVM_MODEL_PATH = os.path.join(ARTIFACT_DIR, "summary_model_svm.joblib")
LOGREG_MODEL_PATH = os.path.join(ARTIFACT_DIR, "summary_model_logreg.joblib")
LINREG_MODEL_PATH = os.path.join(ARTIFACT_DIR, "summary_model_linreg.joblib")
VECTORIZER_PATH = os.path.join(ARTIFACT_DIR, "tfidf_preprocessor.joblib")


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def generate_groq_summary(text: str, max_tokens: int = 256) -> str:
    """
    Call a Groq‑hosted LLM via API to generate an abstractive summary.

    The API key is read from the GROQ_API_KEY environment variable.
    Optionally, the model name can be overridden with GROQ_MODEL_NAME.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set; skipping Groq LLM baseline for this run.")
        return ""


    model_name = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that writes concise, factual"
                    "summaries of Text. Limit the summary to around 4 sentences."
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        if response.status_code != 200:
            print("Groq error details:", response.status_code, response.text)
            response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        print(f"Error calling Groq API for abstractive summary: {exc}")
        return ""


def train_main(classifier: str = "rf") -> None:
    """
    Train the summarization model and save the trained artifacts to disk.
    This step loads the data, performs a train split only, and persists the
    TF-IDF vectorizer and RandomForest model.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("Loading training data...")

    sentences, labels = get_training_data(
        use_cnn_dailymail=True,
        use_hf_lecture_dataset=False,
        sample_size=200000,
    )

    if not sentences:
        print("No training data found. Exiting.")
        return

    num_pos = sum(labels)
    total = len(labels)
    print(
        f"Final label distribution: {num_pos} positives, {total} total "
        f"({num_pos / max(1, total):.3f} positives)",
    )

    print("Splitting data into train and test sets (for later evaluation)...")
    X_train_text, _, y_train, _ = train_test_split(
        sentences,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    print("Preprocessing data (fitting TF-IDF on training set)...")
    preprocessor = TextPreprocessor()
    X_train = preprocessor.fit_transform(X_train_text)

    print("Training model...")
    if classifier == "rf":
        summary_model = SummaryModel()
        model_path = MODEL_PATH
    elif classifier == "nb":
        summary_model = NaiveBayesSummaryModel()
        model_path = NB_MODEL_PATH
    elif classifier == "svm":
        summary_model = SVMSummaryModel()
        model_path = SVM_MODEL_PATH
    elif classifier == "logreg":
        summary_model = LogisticRegressionSummaryModel()
        model_path = LOGREG_MODEL_PATH
    elif classifier == "linreg":
        summary_model = LinearRegressionSummaryModel()
        model_path = LINREG_MODEL_PATH
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    summary_model.train(X_train, y_train)
    print("Model training complete.")

    # Only RandomForest exposes an OOB score
    oob_method = getattr(summary_model, "oob_score", None)
    if callable(oob_method):
        oob = oob_method()
        if oob is not None:
            print(f"OOB score (RandomForest internal estimate): {oob:.4f}")

    dump(preprocessor, VECTORIZER_PATH)
    dump(summary_model, model_path)
    print(f"Saved trained model to {model_path} and vectorizer to {VECTORIZER_PATH}.")


def test_main(classifier) -> None:
    """
    Load a previously trained model and vectorizer and evaluate on the held‑out
    test split, plus document‑level ROUGE and a sample lecture summary.
    """
    if classifier == "rf":
        model_path = MODEL_PATH
    elif classifier == "nb":
        model_path = NB_MODEL_PATH
    elif classifier == "svm":
        model_path = SVM_MODEL_PATH
    elif classifier == "logreg":
        model_path = LOGREG_MODEL_PATH
    elif classifier == "linreg":
        model_path = LINREG_MODEL_PATH
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    if not (os.path.exists(model_path) and os.path.exists(VECTORIZER_PATH)):
        print("Trained artifacts not found. Run `python main.py --mode train` first.")
        return

    print("Loading trained artifacts...")
    preprocessor: TextPreprocessor = load(VECTORIZER_PATH)
    summary_model = load(model_path)

    print("Loading data and rebuilding train/test split for evaluation...")

    sentences, labels = get_training_data(
        use_cnn_dailymail=True,
        use_hf_lecture_dataset=True,
        sample_size=50000,
    )

    if not sentences:
        print("No data available for evaluation.")
        return

    _, X_test_text, _, y_test = train_test_split(
        sentences,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    X_test = preprocessor.transform(X_test_text)

    print("Evaluating sentence-level classification performance on test set...")
    y_pred = summary_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    summarizer = Summarizer(model=summary_model, preprocessor=preprocessor)

    # Document-level evaluation using CNN/DailyMail test split
    print("\nLoading CNN/DailyMail test split for document-level evaluation...")
    cnn_test = load_dataset(
        "cnn_dailymail",
        "3.0.0",
        split="test[:100]",
    )
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rougeL_scores = []


    for ex in cnn_test:
        doc_text = ex["article"]
        gold_summary = ex["highlights"]

        pred_summary = summarizer.summarize(doc_text, num_sentences=5)
        scores = rouge.score(gold_summary, pred_summary)

        rouge1_scores.append(scores["rouge1"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    if rouge1_scores:
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
        print(f"ROUGE-1 F1: {avg_rouge1:.4f}")
        print(f"ROUGE-L F1: {avg_rougeL:.4f}")
    
    # Manual summarization of a local lecture note
    lecture_file_to_summarize = "data/lecture_notes_ai_genetic.txt"
    print(f"\nLoading '{lecture_file_to_summarize}' for a manual summarization check...")
    lecture_to_summarize = load_single_lecture_note(lecture_file_to_summarize)

    if not lecture_to_summarize:
        print("No lecture notes to summarize. Exiting.")
        return

    print("Generating summary for local lecture note...")
    summary = summarizer.summarize(lecture_to_summarize, num_sentences=5)

    print("\n--- Original Text (truncated) ---")
    print(lecture_to_summarize[:1000], "...\n")
    print("--- Generated Summary ---")
    print(summary)


def test_llm_baseline() -> None:
    """
    Evaluate an off‑the‑shelf Groq LLM on CNN/DailyMail using ROUGE‑1 and ROUGE‑L.
    This does NOT use the trained extractive model, so it does not affect the
    existing evaluation pipeline.
    """
    print("\nLoading CNN/DailyMail test split for Groq LLM evaluation...")
    cnn_test = load_dataset(
        "cnn_dailymail",
        "3.0.0",
        split="test[:100]",
    )
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rougeL_scores = []

    for ex in cnn_test:
        doc_text = ex["article"]
        gold_summary = ex["highlights"]

        pred_summary = generate_groq_summary(doc_text)
        if not pred_summary:
            # If the API key is missing or the call failed, skip scoring
            continue

        scores = rouge.score(gold_summary, pred_summary)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    if rouge1_scores:
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
        print("\nROUGE scores for Groq LLM baseline:")
        print(f"ROUGE-1 F1: {avg_rouge1:.4f}")
        print(f"ROUGE-L F1: {avg_rougeL:.4f}")
    else:
        print("No ROUGE scores computed. Check GROQ_API_KEY and network connectivity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "test", "llm"],
        default="train",
        help=(
            "Run training (`train`), evaluation/testing of extractive models "
            "(`test`), or LLM baseline evaluation (`llm`)."
        ),
    )
    parser.add_argument(
        "--classifier",
        choices=["rf", "nb", "svm", "logreg", "linreg"],
        default="rf",
        help=(
            "Which classifier to use: 'rf' (RandomForest), 'nb' (Naive Bayes), "
            "'svm' (Support Vector Machine), 'logreg' (Logistic Regression), "
            "or 'linreg' (Linear Regression)."
        ),
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_main(classifier=args.classifier)
    elif args.mode == "test":
        test_main(classifier=args.classifier)
    else:
        # LLM baseline ignores the classifier argument
        test_llm_baseline()
