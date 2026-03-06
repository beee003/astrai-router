"""Context Compression — Task-aware conversation compression for token savings.

Compresses conversation context before sending to LLM providers.
Customer pays retail on original tokens, Astrai pays wholesale on compressed.
The spread is pure margin.

Follows pii_shield.py:strip_pii_from_messages() pattern.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token estimation (cheap heuristic — 1 token ≈ 4 chars)
# ---------------------------------------------------------------------------

def _extract_text(content) -> str:
    """Extract plain text from str or multimodal content array."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str)
            else ""
            for item in content
        )
    return ""


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _estimate_messages_tokens(messages: List[dict]) -> int:
    total = 0
    for m in messages:
        content = _extract_text(m.get("content") or "")
        total += _estimate_tokens(content) + 4  # role overhead
    return total


# ---------------------------------------------------------------------------
# Technique 1: System prompt deduplication
# ---------------------------------------------------------------------------

def _dedup_system_prompts(messages: List[dict]) -> Tuple[List[dict], int]:
    """Keep the longest system message, drop duplicates.

    Returns (deduped_messages, count_of_removed_system_msgs).
    """
    system_indices: List[int] = []
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            system_indices.append(i)

    if len(system_indices) <= 1:
        return messages, 0

    # Keep the longest system message
    longest_idx = max(system_indices, key=lambda i: len(_extract_text(messages[i].get("content") or "")))
    removed = 0
    result: List[dict] = []
    for i, m in enumerate(messages):
        if m.get("role") == "system" and i != longest_idx:
            removed += 1
            continue
        result.append(m)
    return result, removed


# ---------------------------------------------------------------------------
# Technique 2: Whitespace normalization
# ---------------------------------------------------------------------------

_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+$", re.MULTILINE)


def _normalize_whitespace(messages: List[dict]) -> List[dict]:
    """Collapse excessive blank lines and trailing whitespace."""
    out: List[dict] = []
    for m in messages:
        content = m.get("content") or ""
        if not isinstance(content, str):
            out.append(m)
            continue
        cleaned = _MULTI_BLANK.sub("\n\n", content)
        cleaned = _TRAILING_SPACES.sub("", cleaned)
        cleaned = cleaned.strip()
        out.append({**m, "content": cleaned})
    return out


# ---------------------------------------------------------------------------
# Technique 3: Code comment stripping
# ---------------------------------------------------------------------------

_LINE_COMMENT = re.compile(r"^[ \t]*(?:#|//)[^\n]*$", re.MULTILINE)
_DOCSTRING = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')


def _strip_code_comments(messages: List[dict]) -> List[dict]:
    """Remove line comments (#, //) and Python docstrings from message content."""
    out: List[dict] = []
    for m in messages:
        content = m.get("content") or ""
        if not isinstance(content, str):
            out.append(m)
            continue
        # Strip docstrings first (multi-line)
        cleaned = _DOCSTRING.sub("", content)
        # Strip line comments
        cleaned = _LINE_COMMENT.sub("", cleaned)
        # Collapse resulting blank lines
        cleaned = _MULTI_BLANK.sub("\n\n", cleaned)
        cleaned = cleaned.strip()
        out.append({**m, "content": cleaned})
    return out


# ---------------------------------------------------------------------------
# Technique 4: Old turn extractive summarization
# ---------------------------------------------------------------------------

def _first_sentence(text: str) -> str:
    """Extract the first sentence (up to 200 chars) as extractive summary."""
    text = text.strip()
    if not text:
        return ""
    # Find first sentence boundary
    for delim in (".\n", ". ", ".\t", "!\n", "! ", "?\n", "? "):
        idx = text.find(delim)
        if 0 < idx < 200:
            return text[: idx + 1]
    # No sentence boundary found — truncate
    if len(text) > 200:
        return text[:200] + "..."
    return text


def _summarize_old_turns(messages: List[dict], keep_last_n: int) -> Tuple[List[dict], int]:
    """Extractive-summarize turns older than the last N.

    Keeps system messages untouched (they live at the top).
    Returns (compressed_messages, turns_summarized).
    """
    # Separate system messages from conversation turns
    system_msgs: List[dict] = []
    conv_turns: List[dict] = []
    for m in messages:
        if m.get("role") == "system":
            system_msgs.append(m)
        else:
            conv_turns.append(m)

    if len(conv_turns) <= keep_last_n:
        return messages, 0

    old_turns = conv_turns[: -keep_last_n]
    recent_turns = conv_turns[-keep_last_n:]
    turns_summarized = 0

    summarized: List[dict] = []
    for turn in old_turns:
        content = turn.get("content") or ""
        if not isinstance(content, str):
            summarized.append(turn)
            continue
        summary = _first_sentence(content)
        if len(summary) < len(content):
            turns_summarized += 1
            summarized.append({**turn, "content": summary})
        else:
            summarized.append(turn)

    return system_msgs + summarized + recent_turns, turns_summarized


# ---------------------------------------------------------------------------
# Strategy mapping
# ---------------------------------------------------------------------------

_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "code": {
        "techniques": ["system_dedup", "whitespace", "code_comments", "summarize"],
        "keep_last_n": 4,
    },
    "research": {
        "techniques": ["system_dedup", "whitespace", "summarize"],
        "keep_last_n": 6,
    },
    "analysis": {
        "techniques": ["system_dedup", "whitespace", "summarize"],
        "keep_last_n": 6,
    },
}

_DEFAULT_STRATEGY = {
    "techniques": ["system_dedup", "whitespace", "summarize"],
    "keep_last_n": 6,
}

# Skip compression below this threshold (not worth the overhead)
_MIN_TOKENS_THRESHOLD = 2000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compress_messages(
    messages: List[dict],
    task_type: str = "general",
) -> Tuple[List[dict], Optional[Dict[str, Any]]]:
    """Compress conversation messages using task-aware strategies.

    Returns:
        (compressed_messages, manifest_dict_or_None)

    If compression is skipped (short conversation), manifest is None.
    Input list is never mutated.
    """
    if not messages:
        return messages, None

    original_tokens = _estimate_messages_tokens(messages)
    if original_tokens < _MIN_TOKENS_THRESHOLD:
        return list(messages), None

    strategy = _STRATEGIES.get(task_type, _DEFAULT_STRATEGY)
    techniques = strategy["techniques"]
    keep_last_n = strategy["keep_last_n"]

    # Work on a copy — never mutate input
    working = [dict(m) for m in messages]

    techniques_applied: List[str] = []
    system_prompts_deduped = 0
    turns_summarized = 0

    # Apply techniques in order (cheapest first)
    if "system_dedup" in techniques:
        working, deduped = _dedup_system_prompts(working)
        if deduped > 0:
            techniques_applied.append("system_dedup")
            system_prompts_deduped = deduped

    if "whitespace" in techniques:
        working = _normalize_whitespace(working)
        techniques_applied.append("whitespace")

    if "code_comments" in techniques:
        working = _strip_code_comments(working)
        techniques_applied.append("code_comments")

    if "summarize" in techniques:
        working, summarized = _summarize_old_turns(working, keep_last_n)
        if summarized > 0:
            techniques_applied.append("summarize")
            turns_summarized = summarized

    compressed_tokens = _estimate_messages_tokens(working)

    # If we didn't actually compress anything meaningful, skip
    if compressed_tokens >= original_tokens:
        return list(messages), None

    ratio = round(original_tokens / max(compressed_tokens, 1), 2)

    manifest = {
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "compression_ratio": ratio,
        "strategy": task_type,
        "techniques_applied": techniques_applied,
        "turns_summarized": turns_summarized,
        "system_prompts_deduped": system_prompts_deduped,
    }

    return working, manifest
