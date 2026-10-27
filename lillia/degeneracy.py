"""Detect generations that have stopped reasoning: repetition loops and raw enumeration.

Two backend-independent signals are measured over a trailing window of streamed reasoning text.
Both were calibrated on every saved reasoning trace in the local journal (2,522 traces, 156 of
them over 12,000 characters, scanned in 12,000-character windows) and on synthetic failure modes:

* ``compression_ratio`` (zlib level 6, compressed bytes / raw bytes). Legitimate reasoning never
  went below 0.247; a phrase loop measures ~0.007 and cyclic copying of a hex dump ~0.026.
* ``prose_density`` (language per 1,000 characters, measured on the text outside long hex
  identifiers). Legitimate 12,000-character windows never went below 53; counters, hex pairs,
  ``0x..`` lists, offset dumps and bare rows of hashes measure 0, while an annotated list of
  sha256 artifact names ("- <sha>.txt (12,345 bytes)") is judged on its annotations and scores
  about 90. Every non-ASCII letter counts as half a word, so reasoning in Cyrillic, Arabic,
  Devanagari, Thai or CJK is language; the rule is skipped only when at least a tenth of the
  window is non-ASCII letters, never for symbols, and a window with an explanatory line (eight or
  more words on one line) is given one more streaming check before it counts.

Either signal alone marks a window as degenerate. The compression threshold sits 2x under the
legitimate minimum and the density threshold about 1.7x under it, so genuinely long thinking is
never cut short by this module: it only reacts to text that carries no analysis. A
whole-generation check additionally uses the exact characters-per-token ratio when the server
reported completion tokens: reasoning-dominated saved responses measure 1.93 characters per
token or more (the four journaled reasoning-only overruns: 2.47 and 1.94 legitimate, 1.06 and
1.07 degenerate), so mostly-ASCII output below 1.6 is single-character tokens (hex digits,
numbers, symbols, escapes). Text that is mostly long hex identifiers with language around them
is exempt from that rule, since identifiers tokenize a character or two at a time.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field

WORD = re.compile(r"[A-Za-z]{3,}")
HEX_LETTERS = frozenset("abcdefABCDEF")
# Content-addressed identifiers (sha256 artifact names, git hashes) are references the model
# copies from context; they are neither language nor, on their own, evidence of enumeration.
HEX_RUN = re.compile(r"[0-9a-fA-F]{32,}")
# A line carrying this many words is narration or a caption: the surrounding table or grid is
# something the model is explaining, not a stream that has stopped reasoning.
EXPLANATORY_WORDS = 8
# Below this share of non-ASCII letters a window is judged by the density rule; above it the
# text is in a script the Latin word count cannot see, and only repetition is judged.
LETTER_SCRIPT_FRACTION = 0.1


def language_words(text: str) -> int:
    """Latin words of three or more letters with at least two letters outside a-f."""
    return sum(1 for word in WORD.findall(text) if sum(c not in HEX_LETTERS for c in word) >= 2)


def non_ascii_letters(text: str) -> int:
    return sum(1 for c in text if ord(c) > 127 and c.isalpha())


def non_ascii_letter_fraction(text: str) -> float:
    if not text:
        return 0.0
    return non_ascii_letters(text) / len(text)


def prose_density(text: str) -> float:
    """Language per 1,000 characters of text outside long hex identifiers.

    Counts Latin words (three or more letters, at least two outside a-f so hex strings and
    ``0xab`` tokens do not count) and every non-ASCII letter as half a word, so scripts without
    Latin letters or without spaces (Chinese, Thai, Devanagari with combining vowels) score as
    language. Long hex identifiers are removed from the measure rather than credited: an
    annotated list of them is judged on its annotations, a bare list of them scores zero.
    """
    if not text:
        return 0.0
    words = language_words(text)
    letters = non_ascii_letters(text)
    if not words and not letters:
        return 0.0
    run_chars = sum(len(match) for match in HEX_RUN.findall(text))
    return (words + letters / 2) * 1000 / max(1, len(text) - run_chars)


def compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    if not data:
        return 1.0
    return len(zlib.compress(data, 6)) / len(data)


def ascii_fraction(text: str) -> float:
    if not text:
        return 1.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


def has_explanatory_line(text: str) -> bool:
    return any(language_words(line) >= EXPLANATORY_WORDS for line in text.splitlines())


def hex_run_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(len(match) for match in HEX_RUN.findall(text)) / len(text)


@dataclass
class Verdict:
    degenerate: bool
    reason: str = ""
    metrics: dict = field(default_factory=dict)


REPEATED = "repeated text"
ENUMERATION = "raw numeric or hex sequences without language"


def assess_window(window: str, config) -> Verdict:
    """Judge one trailing window of reasoning text against the configured thresholds.

    ``metrics["low_density"]`` records that the window has too little language even when an
    explanatory line vetoes the verdict here; the stream monitor uses it to defer rather than
    exempt.
    """
    ratio = compression_ratio(window)
    density = prose_density(window)
    low_density = (
        density < config.degenerate_prose_density
        and non_ascii_letter_fraction(window) < LETTER_SCRIPT_FRACTION
    )
    explanatory = low_density and has_explanatory_line(window)
    metrics = {
        "window_chars": len(window),
        "compression_ratio": round(ratio, 4),
        "prose_density": round(density, 1),
        "low_density": low_density,
        "explanatory_line": explanatory,
    }
    if ratio < config.degenerate_compression_ratio:
        return Verdict(True, REPEATED, metrics)
    if low_density and not explanatory:
        return Verdict(True, ENUMERATION, metrics)
    return Verdict(False, "", metrics)


def assess_generation(
    reasoning: str,
    config,
    *,
    completion_tokens=None,
    content_chars: int = 0,
    tool_call_chars: int = 0,
) -> Verdict:
    """Classify a finished generation that produced no action.

    Only reasoning-only output is judged: once answer text or a tool call has started, the model
    was acting, and the token ratio would mix reasoning with structured output.
    """
    if not reasoning or content_chars or tool_call_chars:
        return Verdict(False, "", {})
    window = reasoning[-config.degenerate_window_chars :]
    verdict = assess_window(window, config)
    metrics = {"reasoning_chars": len(reasoning), **verdict.metrics}
    if type(completion_tokens) is int and completion_tokens > 0:
        metrics["chars_per_token"] = round(len(reasoning) / completion_tokens, 3)
    if verdict.degenerate:
        # The window's diagnosis (loop or enumeration) is the more specific one.
        return Verdict(True, verdict.reason, metrics)
    if type(completion_tokens) is int and completion_tokens > 0:
        chars_per_token = len(reasoning) / completion_tokens
        # Ideographic and other non-Latin scripts legitimately run near one character per
        # token, and a list of long hex identifiers surrounded by language tokenizes a
        # character or two at a time; both are judged by the window only.
        identifier_list = (
            hex_run_fraction(reasoning) >= 0.2 and prose_density(reasoning) >= config.degenerate_prose_density
        )
        if (
            chars_per_token < config.degenerate_chars_per_token
            and non_ascii_letter_fraction(reasoning) < LETTER_SCRIPT_FRACTION
            and not identifier_list
        ):
            return Verdict(True, "single-character tokens (hex digits, numbers or symbols)", metrics)
    return Verdict(False, "", metrics)


class StreamMonitor:
    """Incrementally watch streamed reasoning and report once its trailing window is degenerate.

    The check runs every ``step_chars`` after the first full window, so the cost is one zlib pass
    over the window every couple of thousand characters, and a runaway stream is caught within
    minutes rather than at the output limit. A repetition verdict is reported at once; a
    low-density window must be seen on two consecutive checks, so a table or grid the model is in
    the middle of writing out gets one more step to turn back into analysis. An explanatory line
    inside the window defers the verdict to the next check rather than exempting the stream, so
    sparse narration between blocks of raw output does not keep it alive indefinitely.
    """

    def __init__(self, config, step_chars: int = 2000):
        self.config = config
        self.window = config.degenerate_window_chars
        self.step = step_chars
        self.buffer = ""
        self.chars = 0
        self.next_check = self.window
        self.checks = 0
        self.enumeration_checks = 0

    def feed(self, delta: str) -> Verdict | None:
        if not delta:
            return None
        self.chars += len(delta)
        self.buffer = (self.buffer + delta)[-self.window :]
        if self.chars < self.next_check:
            return None
        self.next_check = self.chars + self.step
        self.checks += 1
        verdict = assess_window(self.buffer, self.config)
        if verdict.reason == REPEATED:
            self.enumeration_checks = 0
            verdict.metrics["reasoning_chars"] = self.chars
            return verdict
        if not verdict.metrics["low_density"]:
            self.enumeration_checks = 0
            return None
        self.enumeration_checks += 1
        if self.enumeration_checks < 2:
            return None
        metrics = {**verdict.metrics, "reasoning_chars": self.chars}
        return Verdict(True, ENUMERATION, metrics)
