from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from . import __version__

# The shortcut also selects [model_profiles."Qwen3.6-35B"] when the loaded config defines it: --model
# matches profile keys exactly, before and independently of this alias (load_config).
MODEL_ALIASES = {"Qwen3.6-35B": "Qwen3.6-35B-A3B"}
# Backends that answer without an API key. Ordered so the general web index runs first and the
# encyclopedic one catches what it misses. Mojeek and Ecosia are excluded (captcha/block page
# under HTTP 200) and so is Bing's RSS view, which returns well-formed results for the query's
# first token only -- see lillia/web.py for the evidence behind each exclusion.
SEARCH_BACKENDS = ("brave", "wikipedia", "ddg")
CONTEXT_TOKEN_MARGIN = 2048
# Give slow disk scans, network tools and indexing requests an hour to finish.
DEFAULT_TOOL_TIMEOUT_SECONDS = 3600
# Approximation for prompt budgeting, not a tokenizer. JSON escaping and chat
# templates differ between backends; the margin and early compaction add room.
CONTEXT_CHARS_PER_TOKEN = 3
# Per-request sampling fields a recovery retry may override on an OpenAI-compatible server.
SAMPLING_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repeat_penalty",
        "seed",
    }
)
# Tables a profile such as [cloud] or [model_profiles.NAME] may extend key by key; its other keys
# replace top-level settings.
PROFILE_SECTIONS = (
    "model",
    "subagent_model",
    "delegation",
    "web",
    "recovery",
    "embedding_model",
    "retrieval",
    "playbooks",
    "outcome_review",
)


@dataclass
class ModelConfig:
    base_url: str = "http://127.0.0.1:8080/v1"
    name: str = "Qwen3.8-27B"
    api_key_env: str = "LILLIA_API_KEY"
    max_tokens: int = 24576
    context_window_tokens: int = 0
    timeout_seconds: int = 0
    request_timeout_seconds: int = 0
    token_parameter: str = "max_tokens"
    include_usage: bool = True
    replay_reasoning: bool = False
    extra_body: dict = field(
        default_factory=lambda: {
            "reasoning_budget_tokens": 16384,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    )
    # Abort a streamed generation whose trailing reasoning window has stopped carrying language:
    # a repetition loop or raw hex/number enumeration. This never bounds genuine thinking; the
    # thresholds sit about 2x beyond the extremes of every saved reasoning trace (lillia/degeneracy.py).
    degenerate_detection: bool = True
    degenerate_window_chars: int = 12000
    degenerate_compression_ratio: float = 0.12
    degenerate_prose_density: float = 30.0
    # Whole-generation check when the server reports completion tokens: reasoning-dominated saved
    # responses measure 1.93 characters per token or more, spaced hex pairs about 1.21, base64 1.35.
    # Skipped for text in non-Latin scripts and for identifier lists surrounded by language.
    degenerate_chars_per_token: float = 1.6
    # Sampling overrides sent only on the retry that follows a degenerate generation.
    degenerate_retry_sampling: dict = field(default_factory=lambda: {"presence_penalty": 1.0})
    # With a reasoning_effort in extra_body (Ollama), the level sent when a request asks for no
    # reasoning at all. Some models answer "none" by thinking in the answer text instead.
    zero_budget_effort: str = "none"

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


@dataclass
class EmbeddingConfig:
    """Vector transport for the evidence index. Separate server, separate queue.

    ``dimensions`` truncates each vector to a leading slice (Matryoshka) and renormalizes.
    Qwen3-Embedding-8B is natively 4096 wide; 1024 measured 4/5 top-1 agreement with the
    full width on this codebase at a quarter of the storage and scan cost. Zero keeps the
    server's native width. Changing it invalidates stored vectors: rebuild the index.
    """

    enabled: bool = True
    base_url: str = "http://127.0.0.1:8080/v1"
    name: str = "Qwen3-Embedding-8B"
    api_key_env: str = "LILLIA_API_KEY"
    # Identity recorded with stored vectors; empty means ``name``. Set it only when ``name`` is a
    # different gateway route to the SAME weights, pooling and width (the --cloud profile's GPU2
    # route), so the existing index stays usable instead of being refused as another model's.
    index_model: str = ""
    dimensions: int = 1024
    batch_size: int = 16
    # Zero here means "use the module's bounded default", NOT "unlimited" as it does for a
    # chat backend: a search holds the index write lock across an embedding request, so an
    # unanswered one would hang the turn. Set a positive value to choose your own bound.
    timeout_seconds: int = 0
    request_timeout_seconds: int = 0
    # Qwen3-Embedding is asymmetric: a query carries an instruction, a document does not.
    query_instruction: str = (
        "Given a question about work done on this machine, "
        "retrieve the saved evidence passage that answers it."
    )

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    @property
    def vector_model(self) -> str:
        return self.index_model or self.name


@dataclass
class RetrievalConfig:
    """Lexical-plus-vector recall over saved artifacts.

    The lexical half is sqlite FTS5 and needs no model, so it stays available when the
    embedding server is down. Measured on the live store: 907 artifacts / 24,134 chunks
    index lexically in under a second, while embedding the same corpus takes hours at the
    measured 408 tok/s. Vectors are therefore backfilled incrementally, the bulk build is
    an explicit 'lillia index' run, and the tool reports its own coverage rather than
    implying the vector arm has seen everything.
    """

    enabled: bool = True
    chunk_chars: int = 1500
    chunk_overlap: int = 200
    max_results: int = 10
    # Vectors computed per search call while the backfill catches up. One call must not
    # turn into a multi-hour embed; 'lillia index' does the bulk work deliberately.
    backfill_chunks_per_call: int = 128
    # Ceiling on the brute-force cosine scan. Measured against the live index rather than
    # a synthetic one: 58.2 us per vector at 1024 dims, i.e. ~1.4s for the current 24,189-chunk
    # corpus and ~2.3s at this ceiling. Beyond it the scan stops, and because it runs
    # newest-first the chunks it drops are the oldest; the result says so.
    max_scan_chunks: int = 40000
    # Compaction source archives re-serialize rows already retired from the journal, and
    # measured 76% of artifact volume. They are indexed because read_artifact can reach
    # them and nothing else can, but identical chunks are collapsed in the results.
    index_archives: bool = True
    # Opt in for a particular deployment; the checked-in project enables this.
    auto_recall: bool = False
    # Index this worker's new artifacts in small batches while its turn runs.
    continuous_indexing: bool = False
    index_interval_seconds: float = 2.0
    index_batch_chunks: int = 4
    recall_max_results: int = 3
    recall_max_chars: int = 3500
    recall_timeout_seconds: float = 5.0
    recall_min_similarity: float = 0.65
    recall_cooldown_steps: int = 4


@dataclass
class WebConfig:
    """Outbound HTTP behaviour for fetch_url and web_search.

    ``contact`` is the single field that matters for access. Wikimedia (and several other
    large sites) reject requests whose User-Agent carries no way to reach the operator, and
    they reject a spoofed browser string just as hard as a bare tool name: only a parenthesised
    contact URL earns a 200. Keep it a public URL, never a personal email address.
    """

    contact: str = "https://github.com/lillia-full/lillia-full"
    user_agent: str = ""
    timeout_seconds: int = DEFAULT_TOOL_TIMEOUT_SECONDS
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.0
    retry_max_sleep_seconds: float = 10.0
    search_backends: list[str] = field(default_factory=lambda: ["brave", "wikipedia", "ddg"])
    search_max_results: int = 10

    def header(self) -> str:
        if self.user_agent:
            return self.user_agent
        return f"Lillia-Full/{__version__} (+{self.contact}) httpx/{httpx.__version__}"


@dataclass
class DelegationConfig:
    enabled: bool = True
    strategy: str = "aggressive"
    assessment_enabled: bool = True
    assessment_interval: int = 8
    assessment_max_tokens: int = 4096
    assessment_reasoning_tokens: int = 1024
    preflight_read_chars: int = 8000
    max_active: int = 2  # 0 disables the active-child cap (cloud profile).
    max_spawn_per_turn: int = 0
    max_steps: int = 0
    max_seconds: int = 0
    max_report_chars: int = 6000
    final_wait_seconds: int = 0
    max_model_calls: int = 0
    model_concurrency: int = 2  # Generation concurrency: 0 means unlimited.
    subagent_model_concurrency: int = 1


@dataclass
class RecoveryConfig:
    """Optional advisory reviews and an independent sustained repeated-read safeguard."""

    enabled: bool = True
    max_tokens: int = 2048
    reasoning_tokens: int = 256
    cooldown_tools: int = 6
    # Executed unchanged reads accumulate debt; new observations, completed experiments,
    # and newly passed criteria reduce it. Warn before ending partial with a checkpoint.
    # Independent of advisory reviews and delegation.
    # Zero disables this safeguard, e.g. when measuring model sampling behavior.
    read_loop_limit: int = 12


@dataclass
class PlaybookConfig:
    """Project-scoped historical procedures, never current verification evidence."""

    enabled: bool = False
    max_results: int = 2
    max_context_chars: int = 3500
    max_entries_per_project: int = 100
    max_age_days: int = 90


@dataclass
class OutcomeReviewConfig:
    """Bounded support review before delivering evidence-backed parent work."""

    enabled: bool = False
    max_tokens: int = 2048
    reasoning_tokens: int = 512
    max_context_chars: int = 24000
    timeout_seconds: int = 180


@dataclass
class Config:
    mode: str = "full"
    state_dir: str = "~/.local/share/lillia-full"
    max_steps: int = 0
    max_consecutive_tool_failures: int = 0
    # Completed generations without an answer or tools get bounded automatic recovery.
    # Zero disables this retry; ordinary text answers still finish normally.
    empty_response_retries: int = 2
    # Consecutive retries of a work generation that ended without any action: it reached the output
    # limit while still reasoning or answering, or was aborted as degenerate. Reasoning-only overruns
    # are condensed into working notes before the retry, so the thinking is carried forward rather
    # than lost; a usable response restores the allowance. Zero disables these retries.
    length_retries: int = 1
    max_turn_seconds: int = 0
    max_context_chars: int = 250000
    # Optional smaller working context for the parent. Zero preserves the shared ceiling.
    # Children and summary requests retain max_context_chars, subject to their model window.
    parent_context_chars: int = 0
    # Fractions of the input budget after reserving tool schemas and runtime state.
    compaction_trigger_ratio: float = 0.8
    compaction_keep_ratio: float = 1 / 3
    summary_max_tokens: int = 4096
    summary_reasoning_tokens: int = 512
    max_output_chars: int = 16000
    max_read_tools_per_round: int = 4
    # Default shell lifetime and minimum positive lifetime for all command tools.
    # Zero leaves the default unlimited and permits explicit per-call limits.
    command_timeout_seconds: int = DEFAULT_TOOL_TIMEOUT_SECONDS
    # Opt in for local stacks that should have every configured backend ready at launch.
    # Generic/remote installations retain on-demand requests and lexical fallback.
    startup_models: bool = False
    startup_timeout_seconds: int = 1800
    model: ModelConfig = field(default_factory=ModelConfig)
    subagent_model: ModelConfig | None = None
    delegation: DelegationConfig = field(default_factory=DelegationConfig)
    web: WebConfig = field(default_factory=WebConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    embedding_model: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    playbooks: PlaybookConfig = field(default_factory=PlaybookConfig)
    outcome_review: OutcomeReviewConfig = field(default_factory=OutcomeReviewConfig)

    def working_context_chars(self, *, is_subagent: bool = False) -> int:
        if not is_subagent and self.parent_context_chars:
            return min(self.max_context_chars, self.parent_context_chars)
        return self.max_context_chars

    def validate(self) -> None:
        if self.mode not in {"full", "assisted", "observe"}:
            raise ValueError("mode must be full, assisted, or observe")
        if type(self.empty_response_retries) is not int or self.empty_response_retries < 0:
            raise ValueError("empty_response_retries must be a nonnegative integer (0 disables retries)")
        if type(self.length_retries) is not int or self.length_retries < 0:
            raise ValueError("length_retries must be a nonnegative integer (0 disables retries)")
        for obj, names in (
            (
                self,
                (
                    "max_context_chars",
                    "summary_max_tokens",
                    "max_output_chars",
                    "max_read_tools_per_round",
                    "startup_timeout_seconds",
                ),
            ),
            (self.recovery, ("max_tokens", "cooldown_tools")),
            (
                self.delegation,
                (
                    "max_report_chars",
                    "assessment_interval",
                    "assessment_max_tokens",
                    "preflight_read_chars",
                ),
            ),
        ):
            for name in names:
                value = getattr(obj, name)
                if type(value) is not int or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
        for obj, name in (
            (self, "max_turn_seconds"),
            (self.delegation, "max_seconds"),
            (self.delegation, "final_wait_seconds"),
        ):
            value = getattr(obj, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer (0 disables the deadline)")
        for obj, name in (
            (self, "max_steps"),
            (self, "parent_context_chars"),
            (self, "max_consecutive_tool_failures"),
            (self, "command_timeout_seconds"),
            (self.delegation, "max_steps"),
            (self.delegation, "max_spawn_per_turn"),
            (self.delegation, "max_model_calls"),
            (self.delegation, "max_active"),
            (self.delegation, "model_concurrency"),
            (self.delegation, "subagent_model_concurrency"),
            (self.delegation, "assessment_reasoning_tokens"),
            (self, "summary_reasoning_tokens"),
            (self.recovery, "reasoning_tokens"),
        ):
            value = getattr(obj, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer (0 disables the cap)")
        if self.max_context_chars < 16000:
            raise ValueError("max_context_chars must be at least 16000")
        if 0 < self.parent_context_chars < 16000:
            raise ValueError("parent_context_chars must be 0 or at least 16000")
        for name in ("compaction_trigger_ratio", "compaction_keep_ratio"):
            value = getattr(self, name)
            if type(value) not in {int, float} or not 0 < value < 1:
                raise ValueError(f"{name} must be a number greater than 0 and less than 1")
        if self.compaction_keep_ratio >= self.compaction_trigger_ratio:
            raise ValueError("compaction_keep_ratio must be smaller than compaction_trigger_ratio")
        if self.summary_reasoning_tokens >= self.summary_max_tokens:
            raise ValueError("summary_reasoning_tokens must be smaller than summary_max_tokens")
        if type(self.recovery.enabled) is not bool:
            raise ValueError("recovery.enabled must be a boolean")
        read_loop_limit = self.recovery.read_loop_limit
        if type(read_loop_limit) is not int or read_loop_limit < 0:
            raise ValueError(
                "recovery.read_loop_limit must be a nonnegative integer (0 disables the safeguard)"
            )
        if 0 < read_loop_limit < 12:
            raise ValueError(
                "recovery.read_loop_limit must be 0 or at least 12 to allow recovery after warning"
            )
        if type(self.startup_models) is not bool:
            raise ValueError("startup_models must be a boolean")
        if self.recovery.reasoning_tokens >= self.recovery.max_tokens:
            raise ValueError("recovery.reasoning_tokens must be smaller than recovery.max_tokens")
        if type(self.delegation.enabled) is not bool:
            raise ValueError("delegation.enabled must be a boolean")
        if type(self.delegation.assessment_enabled) is not bool:
            raise ValueError("delegation.assessment_enabled must be a boolean")
        if self.delegation.strategy not in ("balanced", "aggressive"):
            raise ValueError("delegation.strategy must be balanced or aggressive")
        if not 500 <= self.delegation.max_report_chars <= 12000:
            raise ValueError("delegation.max_report_chars must be between 500 and 12000")
        if not isinstance(self.state_dir, str) or not self.state_dir:
            raise ValueError("state_dir must be a nonempty path")
        self.validate_model(self.model, "model")
        if self.subagent_model is not None:
            self.validate_model(self.subagent_model, "subagent_model")
        self.validate_web()
        self.validate_retrieval()
        self.validate_run_improvements()

    def validate_run_improvements(self) -> None:
        for name, cls in (("playbooks", PlaybookConfig), ("outcome_review", OutcomeReviewConfig)):
            section = getattr(self, name)
            if not isinstance(section, cls):
                raise ValueError(f"{name} must be a {cls.__name__}")
            if type(section.enabled) is not bool:
                raise ValueError(f"{name}.enabled must be a boolean")
        for section, field_name, low, high in (
            ("playbooks", "max_results", 1, 5),
            ("playbooks", "max_context_chars", 500, 12000),
            ("playbooks", "max_entries_per_project", 1, 1000),
            ("playbooks", "max_age_days", 1, 3650),
            ("outcome_review", "max_tokens", 256, 8192),
            ("outcome_review", "reasoning_tokens", 0, 8191),
            ("outcome_review", "max_context_chars", 4000, 100000),
            ("outcome_review", "timeout_seconds", 1, 3600),
        ):
            value = getattr(getattr(self, section), field_name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{section}.{field_name} must be an integer between {low} and {high}")
        if self.outcome_review.reasoning_tokens >= self.outcome_review.max_tokens:
            raise ValueError("outcome_review.reasoning_tokens must be smaller than outcome_review.max_tokens")

    def validate_retrieval(self) -> None:
        embedding, retrieval = self.embedding_model, self.retrieval
        if not isinstance(embedding, EmbeddingConfig):
            raise ValueError("embedding_model must be an EmbeddingConfig")
        if not isinstance(retrieval, RetrievalConfig):
            raise ValueError("retrieval must be a RetrievalConfig")
        for obj, section, name in (
            (embedding, "embedding_model", "enabled"),
            (retrieval, "retrieval", "enabled"),
            (retrieval, "retrieval", "index_archives"),
            (retrieval, "retrieval", "auto_recall"),
            (retrieval, "retrieval", "continuous_indexing"),
        ):
            if type(getattr(obj, name)) is not bool:
                raise ValueError(f"{section}.{name} must be a boolean")
        self.validate_endpoint(embedding, "embedding_model")
        if not isinstance(embedding.index_model, str):
            raise ValueError("embedding_model.index_model must be a string (empty means the model name)")
        if not isinstance(embedding.query_instruction, str):
            raise ValueError("embedding_model.query_instruction must be a string")
        for name in ("batch_size",):
            value = getattr(embedding, name)
            if type(value) is not int or not 1 <= value <= 256:
                raise ValueError(f"embedding_model.{name} must be between 1 and 256")
        for name in ("timeout_seconds", "request_timeout_seconds"):
            value = getattr(embedding, name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"embedding_model.{name} must be a nonnegative integer (0 disables the deadline)"
                )
        # Zero keeps the server's native width; anything else must be wide enough to rank.
        if type(embedding.dimensions) is not int or embedding.dimensions < 0:
            raise ValueError("embedding_model.dimensions must be a nonnegative integer (0 means native)")
        if embedding.dimensions and embedding.dimensions < 64:
            raise ValueError("embedding_model.dimensions must be 0 or at least 64")
        for name in ("chunk_chars", "max_results", "max_scan_chunks"):
            value = getattr(retrieval, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"retrieval.{name} must be a positive integer")
        if type(retrieval.backfill_chunks_per_call) is not int or retrieval.backfill_chunks_per_call < 0:
            raise ValueError(
                "retrieval.backfill_chunks_per_call must be a nonnegative integer (0 disables backfill)"
            )
        if type(retrieval.chunk_overlap) is not int or retrieval.chunk_overlap < 0:
            raise ValueError("retrieval.chunk_overlap must be a nonnegative integer")
        # Equal values would advance the cursor by zero characters and never terminate.
        if retrieval.chunk_overlap >= retrieval.chunk_chars:
            raise ValueError("retrieval.chunk_overlap must be smaller than retrieval.chunk_chars")
        if not 200 <= retrieval.chunk_chars <= 20000:
            raise ValueError("retrieval.chunk_chars must be between 200 and 20000")
        if retrieval.max_results > 50:
            raise ValueError("retrieval.max_results must be at most 50")
        for name, low, high in (
            ("index_batch_chunks", 1, 32),
            ("recall_max_results", 1, 5),
            ("recall_max_chars", 1000, 8000),
            ("recall_cooldown_steps", 1, 100),
        ):
            value = getattr(retrieval, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"retrieval.{name} must be between {low} and {high}")
        for name, low, high in (
            ("index_interval_seconds", 0.1, 300),
            ("recall_timeout_seconds", 0.1, 10),
            ("recall_min_similarity", 0.0, 1.0),
        ):
            value = getattr(retrieval, name)
            if type(value) not in {int, float} or not low <= value <= high:
                raise ValueError(f"retrieval.{name} must be between {low} and {high}")

    @staticmethod
    def validate_endpoint(obj, section: str) -> None:
        """Every configured backend answers to the same URL rules; none is exempt."""
        for name in ("base_url", "name", "api_key_env"):
            if not isinstance(getattr(obj, name), str) or not getattr(obj, name):
                raise ValueError(f"{section}.{name} must be a nonempty string")
        url = urlsplit(obj.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError(f"{section}.base_url must be an HTTP(S) URL without embedded credentials")
        if url.query or url.fragment:
            raise ValueError(f"{section}.base_url must not contain a query or fragment")

    @staticmethod
    def validate_model(model: ModelConfig, section: str) -> None:
        if not isinstance(model, ModelConfig):
            raise ValueError(f"{section} must be a ModelConfig")
        if type(model.max_tokens) is not int or model.max_tokens < 1:
            raise ValueError(f"{section}.max_tokens must be a positive integer")
        for name in ("timeout_seconds", "request_timeout_seconds"):
            value = getattr(model, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{section}.{name} must be a nonnegative integer (0 disables the deadline)")
        window = model.context_window_tokens
        if type(window) is not int or window < 0:
            raise ValueError(
                f"{section}.context_window_tokens must be a nonnegative integer (0 means unknown)"
            )
        if window and (window - model.max_tokens - CONTEXT_TOKEN_MARGIN) * CONTEXT_CHARS_PER_TOKEN < 16000:
            raise ValueError(
                f"{section}.context_window_tokens must leave at least 16000 estimated input characters "
                f"after reserving {section}.max_tokens and {CONTEXT_TOKEN_MARGIN} safety tokens"
            )
        for name in ("replay_reasoning", "include_usage"):
            if type(getattr(model, name)) is not bool:
                raise ValueError(f"{section}.{name} must be a boolean")
        Config.validate_endpoint(model, section)
        if model.token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(f"{section}.token_parameter must be max_tokens or max_completion_tokens")
        if not isinstance(model.extra_body, dict):
            raise ValueError(f"{section}.extra_body must be a table")
        reserved = {
            "model",
            "messages",
            "tools",
            "stream",
            "tool_choice",
            "n",
            "max_tokens",
            "max_completion_tokens",
        }
        if reserved & model.extra_body.keys():
            raise ValueError(f"{section}.extra_body cannot override request structure or token limits")
        if type(model.degenerate_detection) is not bool:
            raise ValueError(f"{section}.degenerate_detection must be a boolean")
        if (
            type(model.degenerate_window_chars) is not int
            or not 2000 <= model.degenerate_window_chars <= 200000
        ):
            raise ValueError(f"{section}.degenerate_window_chars must be between 2000 and 200000")
        for name, low, high in (
            ("degenerate_compression_ratio", 0.0, 1.0),
            ("degenerate_prose_density", 0.0, 1000.0),
            ("degenerate_chars_per_token", 0.0, 10.0),
        ):
            value = getattr(model, name)
            if type(value) not in {int, float} or not low <= value <= high:
                raise ValueError(f"{section}.{name} must be a number between {low} and {high}")
        effort = model.extra_body.get("reasoning_effort")
        if effort is not None and (not isinstance(effort, str) or not effort):
            raise ValueError(f"{section}.extra_body.reasoning_effort must be a nonempty string")
        if not isinstance(model.zero_budget_effort, str) or not model.zero_budget_effort:
            raise ValueError(f"{section}.zero_budget_effort must be a nonempty string")
        sampling = model.degenerate_retry_sampling
        if not isinstance(sampling, dict):
            raise ValueError(f"{section}.degenerate_retry_sampling must be a table")
        unknown = sampling.keys() - SAMPLING_KEYS
        if unknown:
            raise ValueError(
                f"{section}.degenerate_retry_sampling has unsupported keys: {', '.join(sorted(map(str, unknown)))}. "
                f"Supported: {', '.join(sorted(SAMPLING_KEYS))}"
            )
        for key, value in sampling.items():
            if type(value) not in {int, float}:
                raise ValueError(f"{section}.degenerate_retry_sampling.{key} must be a number")

    def validate_web(self) -> None:
        web = self.web
        for name in ("timeout_seconds", "retry_attempts", "search_max_results"):
            value = getattr(web, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"web.{name} must be a positive integer")
        if web.retry_attempts > 6:
            raise ValueError("web.retry_attempts must be at most 6")
        if web.search_max_results > 25:
            raise ValueError("web.search_max_results must be at most 25")
        for name in ("retry_backoff_seconds", "retry_max_sleep_seconds"):
            value = getattr(web, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value <= 60:
                raise ValueError(f"web.{name} must be a number greater than 0 and at most 60")
        if not isinstance(web.user_agent, str):
            raise ValueError("web.user_agent must be a string")
        if not isinstance(web.contact, str) or not web.contact.strip():
            raise ValueError("web.contact must be a nonempty string")
        if "@" in web.contact and not web.contact.startswith("mailto:"):
            raise ValueError(
                "web.contact looks like a bare email address; use a public URL, "
                "or write it as mailto:you@example.org if you intend to publish it"
            )
        if not web.user_agent and urlsplit(web.contact).scheme not in {"http", "https", "mailto"}:
            raise ValueError("web.contact must be an http(s) or mailto URL")
        if not isinstance(web.search_backends, list) or not web.search_backends:
            raise ValueError("web.search_backends must be a nonempty array")
        unknown = [b for b in web.search_backends if b not in SEARCH_BACKENDS]
        if unknown:
            raise ValueError(
                f"Unknown web.search_backends: {', '.join(map(str, unknown))}. "
                f"Supported: {', '.join(SEARCH_BACKENDS)}"
            )
        if len(set(web.search_backends)) != len(web.search_backends):
            raise ValueError("web.search_backends must not repeat a backend")


def model_profiles(data: dict) -> dict:
    """Pop [model_profiles] and check its shape, whether or not --model selects one of them."""
    profiles = data.pop("model_profiles", None)
    if profiles is None:
        return {}
    if not isinstance(profiles, dict):
        raise ValueError("model_profiles must be a TOML table")
    for key, profile in profiles.items():
        name = profile_name(key)
        if not isinstance(profile, dict):
            raise ValueError(f"{name} must be a TOML table")
        for section in PROFILE_SECTIONS:
            if section in profile and not isinstance(profile[section], dict):
                raise ValueError(f"{name}.{section} must be a TOML table")
    return profiles


def profile_name(key: str) -> str:
    # Quoted as in the TOML source: model_profiles."Qwen3.6-35B" (a bare dotted key would split).
    return f"model_profiles.{json.dumps(key)}"


def apply_profile(data: dict, profile: dict, name: str) -> None:
    """Overlay a profile table such as [cloud] on the base settings, one level deep.

    A key in [cloud.model] replaces that key in [model], and a top-level key replaces the top-level
    setting. A nested table such as extra_body replaces the whole value, so request fields meant for
    one kind of server (llama.cpp's reasoning budget) never reach another. Unmentioned settings stay.
    """
    for key, value in profile.items():
        if key in PROFILE_SECTIONS:
            if not isinstance(value, dict):
                raise ValueError(f"{name}.{key} must be a TOML table")
            base = data.get(key, {})
            if not isinstance(base, dict):
                raise ValueError(f"{key} must be a TOML table")
            data[key] = {**base, **value}
        else:
            data[key] = value


def load_config(path: Path | None = None, *, cloud: bool = False, model: str | None = None) -> Config:
    if path is None:
        candidates = [
            Path("config.local.toml"),
            Path("config.toml"),
            Path.home() / ".config/lillia-full/config.toml",
        ]
        path = next((p for p in candidates if p.is_file()), None)
    source = path.read_bytes() if path else b""
    data = tomllib.loads(source.decode()) if path else {}
    # [cloud] is a profile rather than settings: it changes nothing unless --cloud selects it.
    profile = data.pop("cloud", None)
    if profile is not None and not isinstance(profile, dict):
        raise ValueError("cloud must be a TOML table")
    # [model_profiles."NAME"] is a placement profile that --model NAME selects by exact key (no
    # alias resolution): the full gateway ID, LILLIA_MODEL and every other name select nothing.
    profiles = model_profiles(data)
    selected = model if model in profiles else None
    if cloud:
        if selected is not None:
            raise ValueError(
                f"--model {selected} selects the local placement profile [{profile_name(selected)}] "
                "and cannot be combined with --cloud; with --cloud, --model names an Ollama cloud model"
            )
        if profile is None:
            raise ValueError(
                f"--cloud needs a [cloud] table in {path or 'the configuration file'}; "
                "see the [cloud] section of config.toml"
            )
        apply_profile(data, profile, "cloud")
    if selected is not None:
        apply_profile(data, profiles[selected], profile_name(selected))
    known = {f.name for f in fields(Config)}
    if data.keys() - known:
        raise ValueError(f"Unknown config keys: {', '.join(sorted(data.keys() - known))}")
    model_data = data.pop("model", {})
    subagent_model_data = data.pop("subagent_model", None)
    delegation_data = data.pop("delegation", {})
    web_data = data.pop("web", {})
    recovery_data = data.pop("recovery", {})
    embedding_data = data.pop("embedding_model", {})
    retrieval_data = data.pop("retrieval", {})
    playbook_data = data.pop("playbooks", {})
    outcome_review_data = data.pop("outcome_review", {})
    for section, values in (("playbooks", playbook_data), ("outcome_review", outcome_review_data)):
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a TOML table")
    if not isinstance(embedding_data, dict):
        raise ValueError("embedding_model must be a TOML table")
    if not isinstance(retrieval_data, dict):
        raise ValueError("retrieval must be a TOML table")
    if not isinstance(delegation_data, dict):
        raise ValueError("delegation must be a TOML table")
    if not isinstance(model_data, dict):
        raise ValueError("model must be a TOML table")
    if subagent_model_data is not None and not isinstance(subagent_model_data, dict):
        raise ValueError("subagent_model must be a TOML table")
    if not isinstance(web_data, dict):
        raise ValueError("web must be a TOML table")
    if not isinstance(recovery_data, dict):
        raise ValueError("recovery must be a TOML table")
    parent = ModelConfig(**model_data)
    parent.base_url = os.environ.get("LILLIA_BASE_URL", parent.base_url).rstrip("/")
    # --model outranks LILLIA_MODEL, so the parent a selected profile names is kept. The other
    # environment overrides still win over any profile, as they do over [cloud].
    if selected is None:
        parent.name = os.environ.get("LILLIA_MODEL", parent.name)
    subagent_model = None
    if subagent_model_data is not None:
        subagent_model = ModelConfig(**subagent_model_data)
        subagent_model.base_url = os.environ.get("LILLIA_SUBAGENT_BASE_URL", subagent_model.base_url).rstrip(
            "/"
        )
        subagent_model.name = os.environ.get("LILLIA_SUBAGENT_MODEL", subagent_model.name)
        subagent_model.name = MODEL_ALIASES.get(subagent_model.name, subagent_model.name)
    web = WebConfig(**web_data)
    web.contact = os.environ.get("LILLIA_WEB_CONTACT", web.contact)
    embedding = EmbeddingConfig(**embedding_data)
    embedding.base_url = os.environ.get("LILLIA_EMBEDDING_BASE_URL", embedding.base_url).rstrip("/")
    embedding.name = os.environ.get("LILLIA_EMBEDDING_MODEL", embedding.name)
    config = Config(
        **data,
        model=parent,
        subagent_model=subagent_model,
        delegation=DelegationConfig(**delegation_data),
        web=web,
        recovery=RecoveryConfig(**recovery_data),
        embedding_model=embedding,
        retrieval=RetrievalConfig(**retrieval_data),
        playbooks=PlaybookConfig(**playbook_data),
        outcome_review=OutcomeReviewConfig(**outcome_review_data),
    )
    # Provenance only, outside dataclass fields: inherited children keep effective settings.
    config._source_path = str(path.resolve()) if path else None
    config._source_sha256 = hashlib.sha256(source).hexdigest() if path else None
    config._profile = "cloud" if cloud else "local"
    config._model_profile = selected
    # Each profile's (parent, subagent, embedder) names, None where it keeps the base value, so the CLI
    # can warn when a run uses a profile's parent without the rest of that placement.
    roles = ("model", "subagent_model", "embedding_model")
    config._profile_placements = {
        key: tuple(profile.get(role, {}).get("name") for role in roles) for key, profile in profiles.items()
    }
    config.state_dir = os.environ.get("LILLIA_STATE_DIR", config.state_dir)
    config.validate()
    return config
