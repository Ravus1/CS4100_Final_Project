from src.data_loader import get_training_data, load_and_process_hf_summarization_dataset
from src.preprocessor import TextPreprocessor
from src.model import SummaryModel
from src.summarizer import Summarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import load_dataset
from rouge_score import rouge_scorer
from joblib import dump, load
import argparse
import os
# used AI for debugging, gereating synthetic data,syntax, etc.

def load_single_lecture_note(file_path):
    """Loads a single lecture note for summarization."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return ""

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "summary_model.joblib")
VECTORIZER_PATH = os.path.join(ARTIFACT_DIR, "tfidf_preprocessor.joblib")


def train_main() -> None:
    """
    Train the summarization model and save the trained artifacts to disk.
    This step loads the data, performs a train split only, and persists the
    TF-IDF vectorizer and RandomForest model.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("Loading training data...")
    sentences, labels = get_training_data(
        use_cnn_dailymail=True,
        use_hf_lecture_dataset=True,
        hf_dataset_name="TanveerAman/AMI-Corpus-Text-Summarization",
        hf_text_field="Dialogue",
        hf_summary_field="Summaries",
        hf_split="train[:500]",
        sample_size=400,
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

    # Train/test split, but here we only use the training portion for fitting.
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
    summary_model = SummaryModel()
    summary_model.train(X_train, y_train)
    print("Model training complete.")

    oob = summary_model.oob_score()
    if oob is not None:
        print(f"OOB score (RandomForest internal estimate): {oob:.4f}")

    dump(preprocessor, VECTORIZER_PATH)
    dump(summary_model, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH} and vectorizer to {VECTORIZER_PATH}.")


def test_main() -> None:
    """
    Load a previously trained model and vectorizer and evaluate on the held‑out
    test split, plus document‑level ROUGE and a sample lecture summary.
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        print("Trained artifacts not found. Run `python main.py --mode train` first.")
        return

    print("Loading trained artifacts...")
    preprocessor: TextPreprocessor = load(VECTORIZER_PATH)
    summary_model: SummaryModel = load(MODEL_PATH)

    print("Loading data and rebuilding train/test split for evaluation...")
    sentences, labels = get_training_data(
        use_cnn_dailymail=True,
        use_hf_lecture_dataset=True,
        hf_dataset_name="TanveerAman/AMI-Corpus-Text-Summarization",
        hf_text_field="Dialogue",
        hf_summary_field="Summaries",
        hf_split="train[:500]",
        sample_size=400,
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

    # Document-level evaluation using AMI validation split
    print("\nLoading AMI validation split for document-level evaluation...")
    ami_val = load_dataset(
        "TanveerAman/AMI-Corpus-Text-Summarization",
        split="validation[:10]",
    )
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rougeL_scores = []

    print("Evaluating document-level ROUGE scores on AMI validation examples...")
    for ex in ami_val:
        doc_text = ex["Dialogue"]
        gold_summary = ex["Summaries"]

        pred_summary = summarizer.summarize(doc_text, num_sentences=5)
        scores = rouge.score(gold_summary, pred_summary)

        rouge1_scores.append(scores["rouge1"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    if rouge1_scores:
        avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
        avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
        print("\nAverage document-level ROUGE scores on AMI validation[:10]:")
        print(f"ROUGE-1 F1: {avg_rouge1:.4f}")
        print(f"ROUGE-L F1: {avg_rougeL:.4f}")
    else:
        print("No AMI validation examples were evaluated.")

    # Manual summarization of a local lecture note
    lecture_file_to_summarize = "data/lecture_notes_model_evaluation.txt"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="Run training (`train`) or evaluation/testing (`test`).",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_main()
    else:
        test_main()