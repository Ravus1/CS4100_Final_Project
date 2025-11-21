from src.preprocessor import split_into_sentences

class Summarizer:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def summarize(self, text, num_sentences=3):
        """
        Summarizes the text by selecting the most important sentences.
        """
        sentences = split_into_sentences(text)
        if not sentences:
            return ""

        X_test = self.preprocessor.transform(sentences)
        predictions = self.model.predict(X_test)

        # Get probabilities for the 'important' class (class 1)
        # Check if the model has predict_proba attribute
        if hasattr(self.model.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_test)[:, 1]
        else:
            # Fallback for models without predict_proba - use predictions as scores
            probabilities = predictions

        # Combine sentences with their importance scores
        scored_sentences = list(zip(sentences, probabilities))
        
        # Sort sentences by score in descending order
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # Select the top sentences based on predictions, but fall back to scores if not enough
        summary_sentences = []
        for i, (sentence, score) in enumerate(scored_sentences):
            if predictions[sentences.index(sentence)] == 1 and len(summary_sentences) < num_sentences:
                summary_sentences.append(sentence)

        # If not enough sentences were predicted as 1, take the top scored ones
        if len(summary_sentences) < num_sentences:
            top_sentences = [s for s, score in scored_sentences[:num_sentences]]
            # Add sentences that are not already in the summary
            for s in top_sentences:
                if s not in summary_sentences and len(summary_sentences) < num_sentences:
                    summary_sentences.append(s)

        # To maintain the original order of sentences, we sort them by their appearance in the original text
        summary_sentences.sort(key=lambda s: sentences.index(s))

        return " ".join(summary_sentences)
