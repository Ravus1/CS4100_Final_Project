import nltk

# It's good practice to ensure the tokenizer data is available.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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
    """
    Loads lecture notes and provides manually labeled training data.
    The labels indicate whether a sentence is important for a summary (1) or not (0).
    """
    notes = load_lecture_notes()
    sentences = nltk.sent_tokenize(notes)

    important_sentences = [
        "The preceding discussions on search algorithms delved into methods for traversing state-space graphs from an initial state to a goal, emphasizing the construction and the preservation of a path throughout the process.",
        "However, many real-life problems do not focus on the specific path taken to reach the goal.",
        "With this in mind, we now turn our attention to local search methods.",
        "Local search algorithms offer two key advantages in scenarios where the path to the goal is inconsequential:",
        "Hill-climbing is a type of local search algorithm.",
        "Hill climbing iteratively moves in either an increasing/ascent or decreasing/descent direction based on the problem objective.",
        'The algorithm terminates when it reaches a "peak" in case of a maximization problem, i.e., no neighbor has a higher objective function value, or a "valley," when the neighbor has no lower value in case of a minimization problem.',
        "This particular approach is one of several related variants of hill-climbing, and is called Steepest-Ascent Hill Climbing.",
        "In contrast, another commonly seen implementation is to generate a neighboring state, and simply accept it if the score is better than the current state, rather than iterating over all neighbors to find the best one.",
        "This latter approach is known as First-Choice Hill Climbing.",
        "This final variant incorporates randomness in a slightly different way; it chooses randomly among all uphill/downhill moves, with the probability of selecting a particular move varying with the steepness of the ascent/descent.",
        "Hill-climbing approaches, while often effective, may fail to produce an optimal solution in certain instances.",
        "This may happen as a result of the search getting stuck in a local optimum.",
        "One of the most popular methods to circumvent getting stuck in local optima is to randomly restart the hill-climbing process with a different initial state.",
        "A second way to wiggle our way out of local optima is to allow steps that lead to a worse objective function value with a small probability."
    ]

    labels = [1 if s in important_sentences else 0 for s in sentences]

    # Ensure we have the same number of sentences and labels.
    if len(sentences) != len(labels):
        # This case should ideally not be hit if tokenization is consistent.
        # Fallback to ensure lists are of same length to prevent errors.
        min_len = min(len(sentences), len(labels))
        sentences = sentences[:min_len]
        labels = labels[:min_len]

    return sentences, labels
