import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

def check_nltk_data():
    """Checks if the 'punkt' tokenizer is available and provides instructions if not."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' tokenizer not found.")
       
check_nltk_data()

def split_into_sentences(text):
    """Splits text into sentences using NLTK."""
    return nltk.sent_tokenize(text)

class TextPreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, sentences):
        """Fits the vectorizer to the sentences and transforms them into a matrix of TF-IDF features."""
        return self.vectorizer.fit_transform(sentences)

    def transform(self, sentences):
        """Transforms sentences into a matrix of TF-IDF features using the fitted vectorizer."""
        return self.vectorizer.transform(sentences)
