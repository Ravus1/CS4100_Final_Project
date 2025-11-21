def load_lecture_notes(file_path="data/lecture_notes.txt"):
    """
    Loads lecture notes from a file.
    This is where you would load your own training data.
    For now, it loads a sample file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return ""

def get_training_data():
   
    notes = load_lecture_notes()

    # every 3rd sentence is important in dummy data.
    sentences = notes.split('.')
    labels = [1 if i % 3 == 0 else 0 for i in range(len(sentences)) if sentences[i].strip()]
    # Filter out empty sentences that might result from splitting.
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Make sure we have the same number of sentences and labels.
    # The split and strip logic might create discrepancies.
    min_len = min(len(sentences), len(labels))
    sentences = sentences[:min_len]
    labels = labels[:min_len]

    return sentences, labels
