"""
LLM Handler — Groq integration for transcript summarizer.

Sends pre-extracted sentences to Groq LLM for:
  1. Polishing into structured markdown notes
  2. Generating quiz questions (MCQ + short answer)

The LLM only receives condensed text (~20% of original),
keeping token usage and costs minimal.
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"

NOTES_SYSTEM_PROMPT = """You are a study notes generator. You receive key sentences extracted from a lecture/video transcript.

Your job:
1. Organize them into clean, structured markdown notes
2. Add section headings based on topic shifts
3. Use bullet points for key facts
4. Keep the language concise and direct — no fluff
5. Do NOT add information that isn't in the extracted sentences
6. Use markdown formatting (##, -, **bold** for key terms)

Output ONLY the notes in markdown, nothing else."""

QUIZ_SYSTEM_PROMPT = """You are a quiz generator. You receive key sentences extracted from a lecture/video transcript.

Your job:
1. Generate multiple choice questions (MCQs) and short answer questions
2. Base questions ONLY on the content provided
3. For MCQs: provide 4 options (A-D) with one correct answer
4. For short answers: keep expected answers to 1-2 sentences
5. Output in this exact markdown format:

## Quiz

### Multiple Choice

**Q1. [question]**
- A) [option]
- B) [option]
- C) [option]
- D) [option]

**Answer:** [letter]

### Short Answer

**Q1. [question]**

**Answer:** [answer]

Output ONLY the quiz in markdown, nothing else."""


def _get_groq_client() -> Optional[object]:
    """Initialize Groq client from env var. Returns None if unavailable."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set. Skipping LLM processing.")
        return None

    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        logger.warning("groq package not installed. Run: pip install groq")
        return None
    except Exception as e:
        logger.error("Failed to initialize Groq client: %s", e)
        return None


def _call_groq(client: object, system_prompt: str, user_content: str) -> Optional[str]:
    """Make a Groq API call. Returns response text or None on failure."""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Groq API call failed: %s", e)
        return None


def polish_notes(extracted_sentences: list[str]) -> Optional[str]:
    """
    Send extracted sentences to Groq for polishing into structured notes.

    Args:
        extracted_sentences: Key sentences from extractive summarization.

    Returns:
        Structured markdown notes, or None if LLM unavailable/failed.
    """
    client = _get_groq_client()
    if not client:
        return None

    user_content = "Here are the key sentences extracted from a transcript:\n\n"
    user_content += "\n".join(f"- {s}" for s in extracted_sentences)

    token_estimate = len(user_content.split())
    logger.info(
        "Sending %d sentences (~%d tokens) to Groq for note polishing",
        len(extracted_sentences),
        token_estimate,
    )

    result = _call_groq(client, NOTES_SYSTEM_PROMPT, user_content)
    if result:
        logger.info("LLM notes generated successfully")
    return result


def generate_quiz(
    extracted_sentences: list[str], num_questions: int = 5
) -> Optional[str]:
    """
    Generate quiz questions from extracted sentences using Groq.

    Args:
        extracted_sentences: Key sentences from extractive summarization.
        num_questions: Number of questions to generate.

    Returns:
        Quiz in markdown format, or None if LLM unavailable/failed.
    """
    client = _get_groq_client()
    if not client:
        return None

    user_content = (
        f"Generate {num_questions} questions from these key transcript sentences:\n\n"
    )
    user_content += "\n".join(f"- {s}" for s in extracted_sentences)

    token_estimate = len(user_content.split())
    logger.info(
        "Sending %d sentences (~%d tokens) to Groq for quiz generation",
        len(extracted_sentences),
        token_estimate,
    )

    result = _call_groq(client, QUIZ_SYSTEM_PROMPT, user_content)
    if result:
        logger.info("Quiz generated successfully")
    return result
