"""
summarize_abstractive_chunked.py

Usage:
    python summarize_abstractive_chunked.py my_lecture.pdf
"""

import argparse
import os
import re
import fitz  # PyMuPDF
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text() + " "
    return text

def clean_text(text):
    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text, max_len=1000, overlap=200):
    """
    Split text into overlapping chunks of roughly max_len tokens/words.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_len, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_len - overlap  # move with overlap
    return chunks

def summarize_chunk(model, tokenizer, chunk, device, max_input_len=512, max_output_len=200):
    input_text = "summarize: " + chunk
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
        max_length=max_output_len,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
        use_cache=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("--model_dir", type=str, default="./t5_cnn_model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument("--max_output_len", type=int, default=200)
    parser.add_argument("--chunk_size", type=int, default=1000)  # words per chunk
    parser.add_argument("--chunk_overlap", type=int, default=200)
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model dir not found: {args.model_dir}")

    tokenizer = T5TokenizerFast.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(args.device)

    ext = os.path.splitext(args.input_file)[1].lower()
    if ext == ".pdf":
        raw_text = extract_text_from_pdf(args.input_file)
    elif ext == ".txt":
        with open(args.input_file, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
    else:
        raise ValueError("Unsupported file type. Use .pdf or .txt")

    text = clean_text(raw_text)

    # Chunk the text
    chunks = chunk_text(text, max_len=args.chunk_size, overlap=args.chunk_overlap)

    # Summarize each chunk
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        chunk_summary = summarize_chunk(model, tokenizer, chunk, args.device, args.max_input_len, args.max_output_len)
        summaries.append(chunk_summary)

    # Combine all chunk summaries into a final summary
    final_text = " ".join(summaries)
    print("\n===== FINAL SUMMARY =====\n")
    print(final_text)

if __name__ == "__main__":
    main()
