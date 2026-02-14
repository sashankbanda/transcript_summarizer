"""
Transcript Notes Generator (Production Version)

Hybrid pipeline:
  1. Local extractive summarization (TF-IDF) reduces transcript to ~20%
  2. Optional Groq LLM polishes into structured notes + generates quizzes

Usage:
    python transcript_summarizer.py <path_to_transcript_folder>
    python transcript_summarizer.py transcripts --no-llm
    python transcript_summarizer.py transcripts --quiz -v
"""

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

# fmt: off
STOPWORDS: set[str] = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","aren't","as","at","be","because","been","before","being","below",
    "between","both","but","by","can","can't","cannot","could","couldn't","did",
    "didn't","do","does","doesn't","doing","don't","down","during","each","few",
    "for","from","further","get","got","had","hadn't","has","hasn't","have",
    "haven't","having","he","he'd","he'll","he's","her","here","here's","hers",
    "herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've",
    "if","in","into","is","isn't","it","it's","its","itself","just","know",
    "let","let's","like","make","me","might","more","most","my","myself","no",
    "nor","not","now","of","off","oh","ok","on","once","only","or","other",
    "ought","our","ours","ourselves","out","over","own","really","right","said",
    "same","say","she","she'd","she'll","she's","should","shouldn't","so",
    "some","such","take","than","that","that's","the","their","theirs","them",
    "themselves","then","there","there's","these","they","they'd","they'll",
    "they're","they've","thing","think","this","those","through","to","too",
    "under","until","up","us","very","want","was","wasn't","we","we'd","we'll",
    "we're","we've","well","were","weren't","what","what's","when","when's",
    "where","where's","which","while","who","who's","whom","why","why's","will",
    "with","won't","would","wouldn't","yeah","yes","yet","you","you'd","you'll",
    "you're","you've","your","yours","yourself","yourselves","going","gonna",
    "gotta","actually","basically","literally","stuff","things","kind","kinda",
}
# fmt: on

# Abbreviations that should NOT trigger sentence splits
ABBREVIATIONS: set[str] = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "ave", "blvd",
    "gen", "gov", "sgt", "cpl", "pvt", "capt", "lt", "col", "maj",
    "etc", "vs", "fig", "vol", "dept", "univ", "inc", "corp", "ltd",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct",
    "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "u.s", "u.k", "e.g", "i.e", "a.m", "p.m",
}

MIN_SENTENCE_WORDS: int = 6


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def read_transcripts(folder_path: str) -> str:
    """Read and concatenate all .txt files in `folder_path`."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    combined: list[str] = []
    txt_files = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith(".txt")
    )

    if not txt_files:
        raise ValueError(f"No .txt files found in {folder_path}")

    for filename in txt_files:
        filepath = os.path.join(folder_path, filename)
        logger.debug("Reading: %s", filepath)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
            combined.append(fh.read())

    logger.info("Read %d transcript file(s) from %s", len(txt_files), folder_path)
    return " ".join(combined)


def clean_text(text: str) -> str:
    """Normalize whitespace, remove timestamps and bracketed annotations."""
    text = re.sub(r"\[.*?\]", "", text)          # [00:01:23] or [Music]
    text = re.sub(r"\(.*?\)", "", text)           # (applause)
    text = re.sub(r"\d{1,2}:\d{2}(:\d{2})?", "", text)  # bare timestamps
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences with abbreviation awareness.

    Filters out sentences shorter than MIN_SENTENCE_WORDS.
    """
    # Insert split markers after sentence-ending punctuation,
    # but not after known abbreviations.
    tokens = text.split()
    breaks: list[int] = []

    for i, token in enumerate(tokens):
        if token.endswith((".", "!", "?")):
            # Check if this is an abbreviation
            word = token.rstrip(".!?").lower()
            if word not in ABBREVIATIONS and len(word) > 1:
                breaks.append(i)

    if not breaks:
        return [text] if len(tokens) >= MIN_SENTENCE_WORDS else []

    sentences: list[str] = []
    start = 0
    for brk in breaks:
        chunk = " ".join(tokens[start : brk + 1]).strip()
        if len(chunk.split()) >= MIN_SENTENCE_WORDS:
            sentences.append(chunk)
        start = brk + 1

    # Remainder
    if start < len(tokens):
        chunk = " ".join(tokens[start:]).strip()
        if len(chunk.split()) >= MIN_SENTENCE_WORDS:
            sentences.append(chunk)

    return sentences


# ---------------------------------------------------------------------------
# TF-IDF scoring
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Extract lowercase words, excluding stopwords."""
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]


def compute_tfidf(sentences: list[str]) -> dict[str, float]:
    """
    Compute TF-IDF scores for each word across all sentences.

    TF  = count of word in corpus / total words
    IDF = log(total sentences / sentences containing word)
    """
    if not sentences:
        return {}

    n_docs = len(sentences)
    corpus_words = _tokenize(" ".join(sentences))
    total_words = len(corpus_words) or 1

    # Term frequency across entire corpus
    tf: dict[str, float] = {}
    for word, count in Counter(corpus_words).items():
        tf[word] = count / total_words

    # Document frequency (how many sentences contain each word)
    df: dict[str, int] = {}
    for sentence in sentences:
        unique_words = set(_tokenize(sentence))
        for word in unique_words:
            df[word] = df.get(word, 0) + 1

    # TF-IDF
    tfidf: dict[str, float] = {}
    for word in tf:
        idf = math.log((n_docs + 1) / (df.get(word, 0) + 1)) + 1
        tfidf[word] = tf[word] * idf

    return tfidf


def score_sentences(
    sentences: list[str], tfidf: dict[str, float]
) -> dict[str, float]:
    """Score each sentence by average TF-IDF of its non-stopword tokens."""
    scores: dict[str, float] = {}
    for sentence in sentences:
        words = _tokenize(sentence)
        if not words:
            scores[sentence] = 0.0
            continue
        scores[sentence] = sum(tfidf.get(w, 0) for w in words) / len(words)
    return scores


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def remove_duplicates(
    sentences: list[str], threshold: float = 0.75
) -> list[str]:
    """
    Remove near-duplicate sentences using Jaccard similarity.

    Jaccard = |intersection| / |union|
    """
    unique: list[str] = []
    for sentence in sentences:
        words_a = set(sentence.lower().split())
        is_dup = False
        for existing in unique:
            words_b = set(existing.lower().split())
            intersection = len(words_a & words_b)
            union = len(words_a | words_b)
            if union > 0 and (intersection / union) > threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(sentence)
    return unique


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

def summarize(
    sentences: list[str],
    sentence_scores: dict[str, float],
    ratio: float = 0.25,
) -> list[str]:
    """
    Select top-scoring sentences while preserving original order.

    Args:
        sentences: All sentences.
        sentence_scores: Score per sentence.
        ratio: Fraction of sentences to keep (0.0 – 1.0).

    Returns:
        Selected sentences in their original order.
    """
    if not sentences:
        return []

    select_count = max(3, int(len(sentences) * ratio))
    ranked = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    top = set(ranked[:select_count])

    # Preserve original order
    return [s for s in sentences if s in top]


def format_bullet_notes(sentences: list[str]) -> str:
    """Format sentences as markdown bullet points."""
    return "\n".join(f"- {s}" for s in sentences)


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="transcript_summarizer",
        description="Generate study notes from transcript files using "
                    "extractive summarization + optional LLM polishing.",
    )
    parser.add_argument(
        "input",
        help="Path to folder containing .txt transcript files",
    )
    parser.add_argument(
        "-o", "--output",
        default="summary_notes.md",
        help="Output file path (default: summary_notes.md)",
    )
    parser.add_argument(
        "-r", "--ratio",
        type=float,
        default=0.25,
        help="Fraction of sentences to extract (default: 0.25)",
    )
    parser.add_argument(
        "--quiz",
        action="store_true",
        help="Generate quiz questions (saved to quiz.md)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM processing; output raw extractive summary only",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """
    Entry point.

    Returns:
        0 on success, 1 on failure.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # --- Logging setup ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        # --- Step 1: Read ---
        logger.info("Reading transcripts from: %s", args.input)
        raw_text = read_transcripts(args.input)
        logger.info("Raw text length: %d characters", len(raw_text))

        # --- Step 2: Clean ---
        text = clean_text(raw_text)

        # --- Step 3: Split ---
        sentences = split_sentences(text)
        logger.info("Sentences extracted: %d", len(sentences))

        if not sentences:
            logger.error("No usable sentences found in transcripts.")
            return 1

        # --- Step 4: Deduplicate ---
        sentences = remove_duplicates(sentences)
        logger.info("After deduplication: %d sentences", len(sentences))

        # --- Step 5: Score with TF-IDF ---
        tfidf = compute_tfidf(sentences)
        scores = score_sentences(sentences, tfidf)

        # --- Step 6: Extract top sentences ---
        extracted = summarize(sentences, scores, ratio=args.ratio)
        logger.info(
            "Extracted %d key sentences (%.0f%% of %d)",
            len(extracted),
            (len(extracted) / len(sentences)) * 100,
            len(sentences),
        )

        # --- Step 7: LLM polishing (optional) ---
        notes_output: str
        if args.no_llm:
            logger.info("LLM skipped (--no-llm flag)")
            notes_output = format_bullet_notes(extracted)
        else:
            from llm_handler import polish_notes, generate_quiz

            llm_notes = polish_notes(extracted)
            if llm_notes:
                notes_output = llm_notes
            else:
                logger.warning("LLM unavailable — falling back to extractive output")
                notes_output = format_bullet_notes(extracted)

            # --- Step 8: Quiz generation (optional) ---
            if args.quiz:
                quiz_output = generate_quiz(extracted)
                if quiz_output:
                    quiz_path = "quiz.md"
                    with open(quiz_path, "w", encoding="utf-8") as fh:
                        fh.write(quiz_output)
                    logger.info("Quiz saved to %s", quiz_path)
                else:
                    logger.warning("Quiz generation failed — skipping")

        # --- Step 9: Write output ---
        if os.path.exists(args.output):
            logger.warning("Overwriting existing file: %s", args.output)

        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(notes_output)

        logger.info("Notes saved to %s", args.output)
        return 0

    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    except ValueError as e:
        logger.error("%s", e)
        return 1
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
