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

        # Get probabilities for the 'important' class (class 1)
        # Prefer predict_proba if available; fall back to predicted labels as scores.
        if hasattr(self.model, "model") and hasattr(self.model.model, "predict_proba"):
            probabilities = self.model.model.predict_proba(X_test)[:, 1]
        else:
            probabilities = self.model.predict(X_test)

        # Rank sentences purely by probability and take the top-k
        scored_sentences = list(enumerate(probabilities))  # (index, score)
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        top_indices = [idx for idx, score in scored_sentences[:num_sentences]]
        # Sort chosen indices by original order to preserve flow
        top_indices.sort()
        summary_sentences = [sentences[idx] for idx in top_indices]

        return " ".join(summary_sentences)
