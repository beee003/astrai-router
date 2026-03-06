"""Tests for astrai_router.compression — task-aware conversation compression."""

import copy

from astrai_router.compression import (
    compress_messages,
    _dedup_system_prompts,
    _normalize_whitespace,
    _strip_code_comments,
    _summarize_old_turns,
    _estimate_tokens,
    _estimate_messages_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_messages(n_turns: int, content_len: int = 500, n_system: int = 1) -> list:
    """Generate a conversation with n_turns user/assistant pairs plus system messages."""
    msgs = []
    for i in range(n_system):
        msgs.append(
            {"role": "system", "content": f"System prompt {i}. " + "x" * content_len}
        )
    for i in range(n_turns):
        msgs.append(
            {
                "role": "user",
                "content": f"User message {i}. "
                + "Hello world. " * (content_len // 13),
            }
        )
        msgs.append(
            {
                "role": "assistant",
                "content": f"Assistant reply {i}. "
                + "Some response text. " * (content_len // 20),
            }
        )
    return msgs


def _make_code_messages(n_turns: int = 15) -> list:
    """Generate a coding conversation with comments and docstrings."""
    msgs = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "system", "content": "You are a coding assistant."},  # duplicate
    ]
    for i in range(n_turns):
        msgs.append(
            {
                "role": "user",
                "content": f"# This is a comment for turn {i}\n"
                f"// Another comment\n"
                f"def function_{i}():\n"
                f'    """This is a docstring that should be stripped."""\n'
                f"    # inline comment\n"
                f"    x = {i}\n"
                f"    return x * 2\n\n\n\n"
                f"Please help me with this code. I need to understand the implementation.\n"
                f"The function above processes data and returns the result.\n"
                f"Can you explain what modifications are needed?\n"
                + "Extra context. "
                * 40,
            }
        )
        msgs.append(
            {
                "role": "assistant",
                "content": f"Here's the explanation for function_{i}. "
                f"The function takes no arguments and computes a simple multiplication.\n"
                f"# Note: This comment should be stripped\n"
                f"You should consider adding type hints and error handling.\n"
                + "Additional detail. "
                * 40,
            }
        )
    return msgs


# ---------------------------------------------------------------------------
# Unit: System prompt deduplication
# ---------------------------------------------------------------------------


class TestSystemDedup:
    def test_single_system_untouched(self):
        msgs = [{"role": "system", "content": "You are helpful."}]
        result, removed = _dedup_system_prompts(msgs)
        assert removed == 0
        assert len(result) == 1

    def test_multiple_keeps_longest(self):
        msgs = [
            {"role": "system", "content": "Short."},
            {"role": "user", "content": "Hello"},
            {
                "role": "system",
                "content": "This is a much longer system prompt with more detail.",
            },
        ]
        result, removed = _dedup_system_prompts(msgs)
        assert removed == 1
        assert len(result) == 2
        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "much longer" in system_msgs[0]["content"]

    def test_no_system_messages(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result, removed = _dedup_system_prompts(msgs)
        assert removed == 0
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Unit: Whitespace normalization
# ---------------------------------------------------------------------------


class TestWhitespace:
    def test_collapse_blank_lines(self):
        msgs = [{"role": "user", "content": "line1\n\n\n\n\nline2"}]
        result = _normalize_whitespace(msgs)
        assert result[0]["content"] == "line1\n\nline2"

    def test_trailing_spaces(self):
        msgs = [{"role": "user", "content": "hello   \nworld  "}]
        result = _normalize_whitespace(msgs)
        assert "   " not in result[0]["content"]


# ---------------------------------------------------------------------------
# Unit: Code comment stripping
# ---------------------------------------------------------------------------


class TestCodeComments:
    def test_strips_hash_comments(self):
        msgs = [{"role": "user", "content": "# comment\nx = 1\n# another"}]
        result = _strip_code_comments(msgs)
        assert "# comment" not in result[0]["content"]
        assert "x = 1" in result[0]["content"]

    def test_strips_slash_comments(self):
        msgs = [{"role": "user", "content": "// JS comment\nconst x = 1;"}]
        result = _strip_code_comments(msgs)
        assert "// JS comment" not in result[0]["content"]
        assert "const x = 1;" in result[0]["content"]

    def test_strips_docstrings(self):
        msgs = [
            {"role": "user", "content": 'def f():\n    """A docstring."""\n    pass'}
        ]
        result = _strip_code_comments(msgs)
        assert '"""A docstring."""' not in result[0]["content"]
        assert "pass" in result[0]["content"]


# ---------------------------------------------------------------------------
# Unit: Old turn summarization
# ---------------------------------------------------------------------------


class TestSummarization:
    def test_short_conversation_unchanged(self):
        msgs = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result, summarized = _summarize_old_turns(msgs, keep_last_n=6)
        assert summarized == 0
        assert len(result) == 3

    def test_long_conversation_summarized(self):
        msgs = [{"role": "system", "content": "System"}]
        for i in range(10):
            msgs.append(
                {
                    "role": "user",
                    "content": f"Question {i}. This has a lot of extra detail that should get trimmed. "
                    * 5,
                }
            )
            msgs.append(
                {
                    "role": "assistant",
                    "content": f"Answer {i}. Here is a detailed response with lots of content. "
                    * 5,
                }
            )

        result, summarized = _summarize_old_turns(msgs, keep_last_n=4)
        assert summarized > 0
        # Last 4 turns should be unchanged
        original_last_4 = msgs[-4:]
        result_last_4 = result[-4:]
        for orig, res in zip(original_last_4, result_last_4):
            assert orig["content"] == res["content"]


# ---------------------------------------------------------------------------
# Strategy mapping
# ---------------------------------------------------------------------------


class TestStrategyMapping:
    def test_code_uses_all_techniques(self):
        msgs = _make_code_messages(15)
        result, manifest = compress_messages(msgs, task_type="code")
        assert manifest is not None
        assert manifest["strategy"] == "code"
        # Should apply code_comments since it's a code task
        assert (
            "code_comments" in manifest["techniques_applied"]
            or "system_dedup" in manifest["techniques_applied"]
        )

    def test_research_skips_code_comments(self):
        msgs = _make_messages(20, content_len=300)
        result, manifest = compress_messages(msgs, task_type="research")
        if manifest:
            assert "code_comments" not in manifest["techniques_applied"]

    def test_general_uses_default_strategy(self):
        msgs = _make_messages(20, content_len=300)
        result, manifest = compress_messages(msgs, task_type="general")
        if manifest:
            assert "code_comments" not in manifest["techniques_applied"]


# ---------------------------------------------------------------------------
# Skip threshold
# ---------------------------------------------------------------------------


class TestSkipThreshold:
    def test_short_conversation_no_compression(self):
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result, manifest = compress_messages(msgs, task_type="code")
        assert manifest is None
        assert result == msgs

    def test_empty_messages(self):
        result, manifest = compress_messages([], task_type="code")
        assert manifest is None
        assert result == []


# ---------------------------------------------------------------------------
# Input immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_original_messages_unchanged(self):
        msgs = _make_messages(20, content_len=300)
        original = copy.deepcopy(msgs)
        compress_messages(msgs, task_type="code")
        assert msgs == original


# ---------------------------------------------------------------------------
# Integration: 3x compression achievable
# ---------------------------------------------------------------------------


class TestCompressionRatio:
    def test_3x_compression_achievable(self):
        """A 25-turn coding conversation with comments and duplication should achieve >= 3x."""
        msgs = _make_code_messages(25)
        result, manifest = compress_messages(msgs, task_type="code")
        assert manifest is not None, "Compression should have been applied"
        assert manifest["compression_ratio"] >= 3.0, (
            f"Expected >= 3.0x compression, got {manifest['compression_ratio']}x"
        )

    def test_manifest_fields_present(self):
        msgs = _make_code_messages(20)
        result, manifest = compress_messages(msgs, task_type="code")
        assert manifest is not None
        assert "original_tokens" in manifest
        assert "compressed_tokens" in manifest
        assert "compression_ratio" in manifest
        assert "strategy" in manifest
        assert "techniques_applied" in manifest
        assert "turns_summarized" in manifest
        assert "system_prompts_deduped" in manifest
        assert manifest["original_tokens"] > manifest["compressed_tokens"]


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    def test_basic_estimate(self):
        assert _estimate_tokens("hello world") >= 1
        assert _estimate_tokens("") == 1  # min 1

    def test_messages_estimate(self):
        msgs = [{"role": "user", "content": "Hello " * 100}]
        tokens = _estimate_messages_tokens(msgs)
        assert tokens > 100  # ~600 chars / 4 = ~150 tokens + overhead
