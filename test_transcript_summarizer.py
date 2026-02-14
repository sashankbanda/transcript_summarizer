"""
Unit tests for transcript_summarizer.

Run:
    python -m unittest test_transcript_summarizer.py -v
"""

import os
import tempfile
import unittest

from transcript_summarizer import (
    clean_text,
    compute_tfidf,
    format_bullet_notes,
    read_transcripts,
    remove_duplicates,
    score_sentences,
    split_sentences,
    summarize,
)


class TestCleanText(unittest.TestCase):
    def test_removes_timestamps(self):
        self.assertNotIn("[", clean_text("Hello [00:01:23] world"))

    def test_removes_annotations(self):
        text = clean_text("Hello (applause) world [Music]")
        self.assertNotIn("applause", text)
        self.assertNotIn("Music", text)

    def test_normalizes_whitespace(self):
        result = clean_text("Hello    world\n\tnow")
        self.assertEqual(result, "Hello world now")

    def test_empty_string(self):
        self.assertEqual(clean_text(""), "")


class TestSplitSentences(unittest.TestCase):
    def test_basic_split(self):
        text = "This is the first sentence here. This is the second sentence here."
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 2)

    def test_filters_short_sentences(self):
        text = "Hi there. This one is definitely long enough to pass the filter."
        sentences = split_sentences(text)
        # "Hi there." should be filtered out (< 6 words)
        for s in sentences:
            self.assertGreaterEqual(len(s.split()), 6)

    def test_empty_input(self):
        self.assertEqual(split_sentences(""), [])


class TestComputeTfidf(unittest.TestCase):
    def test_returns_dict(self):
        sentences = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning is a subset of machine learning.",
        ]
        result = compute_tfidf(sentences)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_empty_input(self):
        self.assertEqual(compute_tfidf([]), {})

    def test_unique_words_have_nonzero_scores(self):
        sentences = [
            "Python programming language is very popular today.",
            "Java programming language is also quite popular today.",
            "Python has great data science libraries and tools.",
        ]
        tfidf = compute_tfidf(sentences)
        # Words appearing in the corpus should have positive TF-IDF
        for word in ("python", "programming", "data", "science"):
            if word in tfidf:
                self.assertGreater(tfidf[word], 0)


class TestScoreSentences(unittest.TestCase):
    def test_scores_all_sentences(self):
        sentences = [
            "Machine learning is transforming the tech industry.",
            "The weather today is very nice and pleasant.",
        ]
        tfidf = compute_tfidf(sentences)
        scores = score_sentences(sentences, tfidf)
        self.assertEqual(len(scores), 2)

    def test_empty_input(self):
        self.assertEqual(score_sentences([], {}), {})


class TestRemoveDuplicates(unittest.TestCase):
    def test_keeps_unique(self):
        sentences = [
            "Machine learning is important for technology.",
            "The weather today is very nice and warm.",
        ]
        result = remove_duplicates(sentences)
        self.assertEqual(len(result), 2)

    def test_removes_near_duplicate(self):
        sentences = [
            "Machine learning is important for AI technology.",
            "Machine learning is important for AI advancement.",
        ]
        result = remove_duplicates(sentences, threshold=0.5)
        self.assertEqual(len(result), 1)

    def test_empty_input(self):
        self.assertEqual(remove_duplicates([]), [])


class TestSummarize(unittest.TestCase):
    def test_selects_top_sentences(self):
        sentences = [f"Sentence number {i} is here for testing." for i in range(20)]
        scores = {s: i for i, s in enumerate(sentences)}  # higher index = higher score
        result = summarize(sentences, scores, ratio=0.25)
        # Should select ~5 sentences (25% of 20)
        self.assertGreaterEqual(len(result), 3)
        self.assertLessEqual(len(result), 10)

    def test_preserves_order(self):
        sentences = ["First sentence is here now.", "Second sentence is here now.", "Third sentence is here now."]
        scores = {"First sentence is here now.": 1, "Second sentence is here now.": 3, "Third sentence is here now.": 2}
        result = summarize(sentences, scores, ratio=1.0)
        self.assertEqual(result, sentences)

    def test_empty_input(self):
        self.assertEqual(summarize([], {}, ratio=0.25), [])


class TestFormatBulletNotes(unittest.TestCase):
    def test_formatting(self):
        result = format_bullet_notes(["Hello world", "Foo bar"])
        self.assertIn("- Hello world", result)
        self.assertIn("- Foo bar", result)


class TestReadTranscripts(unittest.TestCase):
    def test_missing_folder(self):
        with self.assertRaises(FileNotFoundError):
            read_transcripts("/nonexistent/path/here")

    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                read_transcripts(tmpdir)

    def test_reads_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("Hello world from test transcript.")
            result = read_transcripts(tmpdir)
            self.assertIn("Hello world", result)


if __name__ == "__main__":
    unittest.main()
