# Transcript Summarizer

Extracts key sentences from `.txt` transcript files and generates concise bullet-point notes using extractive summarization (no external APIs or ML libraries).

## Requirements

- Python 3.7+
- No external dependencies (stdlib only)

## Usage

```bash
python transcript_summarizer.py <path_to_transcript_folder>
```

### Example

```bash
python transcript_summarizer.py transcripts
```

## Output

Saves summarized notes to `summary_notes.txt` in the current directory.

## How It Works

1. Reads all `.txt` files from the given folder
2. Cleans text (removes timestamps, extra whitespace)
3. Splits into sentences and removes near-duplicates
4. Scores sentences by word frequency
5. Selects top 25% as summary notes
