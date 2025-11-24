import os
import pickle
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDF reading
import re

# -----------------------------
# 1. Load trained model
# -----------------------------
MODEL_FILE = "trained_summarizer.pkl"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"{MODEL_FILE} not found in folder!")

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# -----------------------------
# 2. Load SBERT embedding model
# -----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 3. Load lecture text
# -----------------------------
LECTURE_FILE = "lecture-notes.pdf"  # Change to your file name (txt or pdf)

if not os.path.exists(LECTURE_FILE):
    raise FileNotFoundError(f"{LECTURE_FILE} not found in folder!")

def extract_text_from_pdf(pdf_path):
    """Extract text from each page of a PDF."""
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text() + " "
    return text

# Read file based on extension
file_ext = os.path.splitext(LECTURE_FILE)[1].lower()
if file_ext == ".txt":
    with open(LECTURE_FILE, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
elif file_ext == ".pdf":
    text = extract_text_from_pdf(LECTURE_FILE)
else:
    raise ValueError("Unsupported file type! Only .txt and .pdf are supported.")

# -----------------------------
# 4. Clean text
# -----------------------------
text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
text = re.sub(r"\s+", " ", text)  # collapse multiple spaces

# -----------------------------
# 5. Split into sentences using regex
# -----------------------------
sentences = re.split(r'(?<=[.!?]) +', text)

# -----------------------------
# 6. Embed sentences
# -----------------------------
embeddings = embedder.encode(sentences, convert_to_numpy=True)

# -----------------------------
# 7. Predict important sentences
# -----------------------------
preds = model.predict(embeddings)

# -----------------------------
# 8. Extract summary
# -----------------------------
summary = " ".join([s for s, p in zip(sentences, preds) if p == 1])

# -----------------------------
# 9. Print results
# -----------------------------
print("\n===== ORIGINAL TEXT (first 2000 chars) =====\n")
print(text[:2000] + " ...")  # first 2000 characters for readability
print("\n===== SUMMARY =====\n")
print(summary)
