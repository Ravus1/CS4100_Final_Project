from src.data_loader import get_training_data
from src.preprocessor import TextPreprocessor
from src.model import SummaryModel
from src.summarizer import Summarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import load_dataset
from rouge_score import rouge_scorer
import os

def load_single_lecture_note(file_path):
    """Loads a single lecture note for summarization."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return ""

def main():
    """
    Main function to run the summarization pipeline.
    """
    # 1. Load and prepare training data
    # Use CNN/Daily Mail plus an additional lecture-style dataset (AMI Corpus)
    print("Loading training data...")
    sentences, labels = get_training_data(
        use_cnn_dailymail=True,
        use_hf_lecture_dataset=True,
        hf_dataset_name="TanveerAman/AMI-Corpus-Text-Summarization",
        hf_text_field="dialogue",
        hf_summary_field="summary",
        hf_split="train[:100]",
        sample_size=200,
    )

    if not sentences:
        print("No training data found. Exiting.")
        return

    # 2. Train/test split for sentence-level classification
    print("Splitting data into train and test sets...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        sentences,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    # 3. Preprocess the training data (fit on train only)
    print("Preprocessing data (fitting TF-IDF on training set)...")
    preprocessor = TextPreprocessor()
    X_train = preprocessor.fit_transform(X_train_text)
    X_test = preprocessor.transform(X_test_text)

    # 4. Train the model
    print("Training model...")
    summary_model = SummaryModel()
    summary_model.train(X_train, y_train)
    print("Model training complete.")

    # 5. Evaluate sentence-level performance
    print("Evaluating sentence-level classification performance on test set...")
    y_pred = summary_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 6. Create a summarizer for document-level evaluation
    summarizer = Summarizer(model=summary_model, preprocessor=preprocessor)

    # 7. Document-level evaluation using AMI validation split
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
        doc_text = ex["dialogue"]
        gold_summary = ex["summary"]

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

    # 8. (Optional) Summarize one of our local lecture notes for a manual check
    lecture_file_to_summarize = "data/lecture_notes_rl.txt"
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
    main()
