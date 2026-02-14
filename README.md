# Transcript Summarizer

Generates study notes (and optional quizzes) from `.txt` transcript files using extractive summarization + optional Groq LLM polishing.

## Setup

```bash
pip install -r requirements.txt
```

For LLM features, create a `.env` file (or set env var):

```bash
cp .env.example .env
# Edit .env and add your Groq API key
# Get one free at https://console.groq.com/keys
```

## Usage

```bash
# Basic (with LLM polishing)
python transcript_summarizer.py <transcript_folder>

# Without LLM (fully offline, extractive only)
python transcript_summarizer.py <transcript_folder> --no-llm

# Custom output file and ratio
python transcript_summarizer.py transcripts -o my_notes.md -r 0.3

# Generate quiz
python transcript_summarizer.py transcripts --quiz

# Verbose logging
python transcript_summarizer.py transcripts -v
```

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `input` (positional) | — | Folder with `.txt` transcripts |
| `-o, --output` | `summary_notes.md` | Output file path |
| `-r, --ratio` | `0.25` | Fraction of sentences to extract |
| `--quiz` | off | Generate quiz → `quiz.md` |
| `--no-llm` | off | Skip Groq, pure extractive output |
| `-v, --verbose` | off | Debug-level logging |

## How It Works

1. Reads all `.txt` files from input folder
2. Cleans text (timestamps, annotations, whitespace)
3. Splits into sentences (abbreviation-aware)
4. Removes near-duplicates (Jaccard similarity)
5. Scores sentences with TF-IDF
6. Extracts top sentences (default 25%)
7. *(Optional)* Sends condensed text to Groq LLM for structured notes
8. *(Optional)* Generates quiz questions via LLM

## Tests

```bash
python -m unittest test_transcript_summarizer.py -v
```
