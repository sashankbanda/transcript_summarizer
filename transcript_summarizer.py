"""
Transcript Notes Generator (Improved Version)

What this version fixes:
- No hardcoded folder dependency
- Accepts folder path as input
- Handles missing folders safely
- Better text cleaning
- Safer duplicate removal
- More stable summary length control

Usage:
    python transcript_summarizer.py <path_to_transcript_folder>

Example:
    python transcript_summarizer.py transcripts
"""

import os
import re
import sys
from collections import Counter


def read_transcripts(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    combined_text = ""
    files_found = False

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".txt"):
            files_found = True
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                combined_text += f.read() + " "

    if not files_found:
        raise ValueError("No .txt files found in the folder.")

    return combined_text


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[.*?\]", "", text)  # remove timestamps like [00:01]
    return text.strip()


def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.split()) > 6]


def compute_word_frequencies(sentences):
    stopwords = set([
        "the","is","in","and","to","of","a","that","it","on","for","with","as","this","was","are","by","an","be","or","from","at",
        "so","if","but","we","they","you","i","he","she","them","his","her","their"
    ])

    words = []
    for sentence in sentences:
        for word in re.findall(r"\w+", sentence.lower()):
            if word not in stopwords:
                words.append(word)

    return Counter(words)


def score_sentences(sentences, word_freq):
    sentence_scores = {}
    for sentence in sentences:
        score = 0
        words = re.findall(r"\w+", sentence.lower())
        for word in words:
            score += word_freq.get(word, 0)
        sentence_scores[sentence] = score / (len(words) + 1)
    return sentence_scores


def remove_duplicates(sentences, threshold=0.75):
    unique = []

    for sentence in sentences:
        words_set = set(sentence.lower().split())
        is_duplicate = False

        for existing in unique:
            existing_set = set(existing.lower().split())
            overlap = words_set.intersection(existing_set)
            similarity = len(overlap) / max(len(words_set), 1)

            if similarity > threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(sentence)

    return unique


def summarize(sentences, sentence_scores, ratio=0.2):
    if not sentences:
        return []

    select_length = max(5, int(len(sentences) * ratio))
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    top_sentences = sorted_sentences[:select_length]

    # Keep original order
    return sorted(top_sentences, key=lambda s: sentences.index(s))


def format_notes(sentences):
    return "\n".join(f"- {sentence}" for sentence in sentences)


def main():
    if len(sys.argv) < 2:
        print("Please provide transcript folder path.")
        print("Example: python transcript_summarizer.py transcripts")
        sys.exit(1)

    folder_path = sys.argv[1]

    try:
        print("Reading transcripts...")
        text = read_transcripts(folder_path)

        print("Cleaning text...")
        text = clean_text(text)

        print("Splitting sentences...")
        sentences = split_sentences(text)

        print("Removing duplicates...")
        sentences = remove_duplicates(sentences)

        print("Computing frequencies...")
        word_freq = compute_word_frequencies(sentences)

        print("Scoring sentences...")
        sentence_scores = score_sentences(sentences, word_freq)

        print("Generating summary...")
        summary_sentences = summarize(sentences, sentence_scores, ratio=0.25)

        output_file = "summary_notes.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(format_notes(summary_sentences))

        print(f"Notes saved to {output_file}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
