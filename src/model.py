from sklearn.ensemble import RandomForestClassifier

class SummaryModel:
    def __init__(self, n_estimators=300, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            oob_score=True,
            n_jobs=-1, 
            verbose=1,
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Predicts which sentences are important for the summary."""
        return self.model.predict(X_test)

    def oob_score(self):
        """
        Returns the out-of-bag (OOB) score of the underlying RandomForest,
        if available. This is an internal estimate of generalization error
        computed from samples not used in each tree's bootstrap.
        """
        # Attribute is only available after fit and when oob_score=True
        return getattr(self.model, "oob_score_", None)
