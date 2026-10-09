"""Focused, offline checks for the lab's deterministic pieces."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import AIMessage

from check_citations import check
from research import REPORT_PATH, SOURCES_PATH, save_outputs, slugify, summarize
from tools import RetryableError, _hf_record, with_retry


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {"n": 1, "url": "https://example.org/one"},
            {"n": 2, "url": "https://example.org/two"},
        ]
        self.report = ("# Survey\nEvidence [1][2]. `Ignored [99]` and [77](https://example.org).\n"
                       "## References\n[1] One. https://example.org/one\n"
                       "[2] Two. https://example.org/two\n")

    def test_valid_report_and_grouped_citations(self):
        self.assertEqual(check(self.report, self.sources), [])
        self.assertEqual(check(self.report.replace("[1][2]", "[1-2]"), self.sources), [])

    def test_missing_duplicate_and_mismatched_references(self):
        broken = self.report.replace("[2] Two. https://example.org/two", "[1] Two. https://example.org/two")
        problems = check(broken, self.sources)
        self.assertTrue(any("exactly one reference" in issue for issue in problems))
        self.assertTrue(any("URL differs" in issue for issue in problems))
        self.assertTrue(any("duplicate URL" in issue for issue in
                            check(self.report, self.sources + [{"n": 3, "url": self.sources[0]["url"]}])))


class ToolTests(unittest.TestCase):
    def test_retry_waits_with_cap_and_not_after_last_failure(self):
        attempts = []

        def failing():
            attempts.append(1)
            raise RetryableError("busy", retry_after=100)

        with patch("tools.time.sleep") as sleep:
            with self.assertRaises(RetryableError):
                with_retry(failing, attempts=3, cap=7)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(sleep.call_args_list[0].args, (7,))

    def test_non_retryable_failure_is_not_repeated(self):
        with patch("tools.time.sleep") as sleep:
            with self.assertRaises(ValueError):
                with_retry(lambda: (_ for _ in ()).throw(ValueError("bad")))
        sleep.assert_not_called()

    def test_hf_flat_search_shape(self):
        record = _hf_record({"id": "2501.12345", "title": "Test", "ai_summary": "AI summary"}, True)
        self.assertEqual(record["url"], "https://huggingface.co/papers/2501.12345")
        self.assertEqual(record["summary"], "AI summary")


class RunnerTests(unittest.TestCase):
    def test_safe_slug(self):
        self.assertEqual(slugify("../../x"), "x")
        self.assertEqual(slugify("!!!"), "topic")
        self.assertLessEqual(len(slugify("A" * 100)), 60)

    def test_meta_counts_tool_calls_and_tokens(self):
        message = AIMessage(content="", tool_calls=[{"name": "task", "args": {}, "id": "1"}],
                            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        result = summarize([message], 2.12, "test-model")
        self.assertEqual(result["subagent_calls"], 1)
        self.assertEqual(result["tokens"], {"input": 10, "output": 5})
        self.assertEqual(result["elapsed_s"], 2.1)

    def test_failed_report_leaves_no_partial_output(self):
        files = {REPORT_PATH: b"# Draft\nNo citations", SOURCES_PATH: b"[]"}
        with TemporaryDirectory() as directory, patch("research.download", return_value=files):
            with self.assertRaises(RuntimeError):
                save_outputs(None, "test topic", [], 1, "model", Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
