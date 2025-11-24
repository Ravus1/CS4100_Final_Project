import os
import pickle
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

# -----------------------------
# 0. Setup
# -----------------------------
nltk.download("punkt")

# -----------------------------
# 1. Load trained model
# -----------------------------
MODEL_FILE = "trained_summarizer.pkl"

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# -----------------------------
# 2. Load SBERT embedding model
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 3. Load lecture text from a file
# -----------------------------
# Change this to your file name
LECTURE_FILE = "my_lecture.txt"

if not os.path.exists(LECTURE_FILE):
    raise FileNotFoundError(f"{LECTURE_FILE} not found in folder!")

with open(LECTURE_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# -----------------------------
# 4. Split into sentences
# -----------------------------
sentences = sent_tokenize(text)

# -----------------------------
# 5. Embed sentences
# -----------------------------
embeddings = embedder.encode(sentences, convert_to_numpy=True)

# -----------------------------
# 6. Predict important sentences
# -----------------------------
preds = model.predict(embeddings)

# -----------------------------
# 7. Extract summary
# -----------------------------
summary = " ".join([s for s, p in zip(sentences, preds) if p == 1])

# -----------------------------
# 8. Print summary
# -----------------------------
print("\n===== ORIGINAL TEXT =====\n")
print(text)
print("\n===== SUMMARY =====\n")
print(summary)
