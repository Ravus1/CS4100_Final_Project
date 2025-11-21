from src.data_loader import get_training_data, load_lecture_notes
from src.preprocessor import TextPreprocessor
from src.model import SummaryModel
from src.summarizer import Summarizer

def main():
    """
    Main function to run the summarization pipeline.
    """
    # 1. Load and prepare training data
    print("Loading training data...")
    sentences, labels = get_training_data()

    if not sentences:
        print("No training data found. Exiting.")
        return

    # 2. Preprocess the training data
    print("Preprocessing data...")
    preprocessor = TextPreprocessor()
    X_train = preprocessor.fit_transform(sentences)

    # 3. Train the model
    print("Training model...")
    summary_model = SummaryModel()
    summary_model.train(X_train, labels)
    print("Model training complete.")

    # 4. Create a summarizer
    summarizer = Summarizer(model=summary_model, preprocessor=preprocessor)

    # 5. Load the lecture notes to be summarized
    print("\nLoading lecture notes for summarization...")
    lecture_to_summarize = load_lecture_notes() # Loads the sample by default

    if not lecture_to_summarize:
        print("No lecture notes to summarize. Exiting.")
        return

    # 6. Generate and print the summary
    print("Generating summary...")
    summary = summarizer.summarize(lecture_to_summarize, num_sentences=5)

    print("\n--- Original Text ---")
    print(lecture_to_summarize)
    print("\n--- Summary ---")
    print(summary)


if __name__ == "__main__":
    main()
