from sklearn.ensemble import RandomForestClassifier

class SummaryModel:
    def __init__(self, n_estimators=200, random_state=42):
        # Use class_weight='balanced' to address label imbalance (few positives)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )

    def train(self, X_train, y_train):
        """Trains the RandomForestClassifier model."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Predicts which sentences are important for the summary."""
        return self.model.predict(X_test)
