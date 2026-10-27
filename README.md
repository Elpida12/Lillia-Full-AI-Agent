# Lillia-Full

A general-purpose AI assistant that runs on your machine. Give it a task; it can inspect files,
execute commands, work with processes, edit documents and configuration, fetch web pages, and verify
results. Coding is one use case alongside system maintenance, data work, research, and everyday automation.

**This checkout defaults to full access. Tools run on the host as your current OS user, without
per-command approval.** Root operations still require OS authorization. The model does not gain root
simply because full mode is enabled.

## Start here

The development environment is installed in `.venv/` on this machine:

```bash
./run.sh
```

Or use the installed command:

```bash
source .venv/bin/activate
lillia
lillia "Inspect disk usage and explain what is taking up the most space"
lillia run "Organize the CSV files in this directory by year" --cwd ~/Documents
```

`run.sh` works from any directory and uses this checkout's configuration, selecting
`config.local.toml` when present and otherwise `config.toml`. The agent's working directory
is the directory you launch it from, unless you pass `--cwd`.

For a fresh installation (Python 3.11+, Linux with Bash):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
./run.sh doctor
./run.sh
```

All three models use the existing llama-swap gateway at `http://127.0.0.1:8080/v1`.
This checkout sets `startup_models = true`: starting chat, running a task, or resuming a session
loads and checks every enabled model role before accepting work. The checks perform a small generation
for each chat model and an embedding request, with `startup_timeout_seconds = 1800` per role for cold
loads. A failed check reports the affected model and stops startup.

Lillia sends `system` messages throughout a conversation, so the model server's chat template must
accept them after the first message. Some templates embedded in Qwen GGUF files reject them; with
llama.cpp, supply a compatible template with `--chat-template-file`.

This machine uses the 32 GB MI50 and three P100s. `config.toml` describes this layout; the gateway
owns hardware placement.
The default layout, used by `./run.sh` without `--model`, is below; `--model Qwen3.6-35B`
selects a second local layout, described in its own section after this one.

| Role | Model | Placement | Window |
| --- | --- | --- | --- |
| Parent | `Qwen3.8-27B` | AMD MI50 (NUMA node 1) | 250,000 tokens, one slot |
| Subagent | `Qwen3.6-35B-A3B-GPU01` | P100 GPUs 0/1 (NUMA node 0) | 250,000 tokens, one slot |
| Embeddings | `Qwen3-Embedding-8B-GPU2` | P100 GPU2 (NUMA node 1) | 4096 dims, stored at 1024 |

The chat servers use Q8 KV caches, MTP speculative decoding, and `-c 250000 --parallel 1 -kvu`:
one **250,000-token** window per model shared by input and output. Each backend has its own
one-request queue, so a parent and child can generate concurrently. Two children can remain active
and overlap tool work while sharing the worker slot.

The gateway's routing matrix allows the P100 worker and both embedding entries alongside either
MI50 chat model. Loading `Qwen3.6-35B-A3B-MI50` swaps out `Qwen3.8-27B`, and vice versa; embeddings
and the P100 worker can stay loaded. Other gateway models run alone and unload the Lillia models.

The gateway owns GPU placement and server lifetimes. Each Lillia model has a 900-second idle TTL;
it can unload while Lillia is idle and reload on its next use. Startup checks install no keepalive
or boot service, and exiting Lillia leaves unloading to the gateway. Set `startup_models = false`
to load models only as needed; this is the library default when no project configuration enables it.

Earlier revisions ran the subagent and embedding models as always-on systemd units
(`lillia-workers.service` on `:8091`, `lillia-embeddings.service` on `:8092`). Both are now **stopped
and disabled**. Inspect the gateway with `systemctl status llama-swap` and
`sudo journalctl -u llama-swap -n 50`.

Thinking is explicitly enabled. The checked-in `reasoning_budget_tokens = -1` sets no client-side reasoning
cap (16,384 is only the library default), so reasoning and the answer share the whole **64,000-token**
generation allowance of the checked-in configuration; compaction, assessment, and recovery reviews send their own
smaller per-request budgets. The built-in default without this configuration is 24,576 tokens.
Ordinary chat requests, parent turns, child assignments, and final report collection have no time cutoff by default.
Prompt wording such as "an eight-hour investigation" does not create a runtime deadline. Time limits are
opt-in configuration settings; Ctrl-C cancels work.
Without a deadline, model prompts say to take the time needed and omit the elapsed-task counter.
Runtime updates and advisory reviews are not requests from the user to hurry.
The reasoning budget is a maximum, not a requirement to use that many tokens.

```bash
./run.sh models
./run.sh doctor
./run.sh                         # Default: Qwen3.8-27B
./run.sh --model Qwen3.6-35B      # 35B on the MI50, 250000 tokens; embedder on GPU2
./run.sh --model Qwen3.6-35B-A3B  # Standalone GPU2 + RAM route; does not select the MI50 profile
python main.py --model Qwen3.6-35B
./run.sh --model SOME_MODEL --base-url http://localhost:8080/v1
```

`Qwen3.6-35B` selects the 35B layout described in the next section: the parent runs the 35B on the
MI50, the subagents run the same 35B on the P100 worker, and embeddings run on GPU2. Only that exact
`--model` value selects it.
With a config file that has no such profile table, `Qwen3.6-35B` remains a plain shortcut for
`Qwen3.6-35B-A3B`.
The full ID `Qwen3.6-35B-A3B` selects the older standalone GPU2 + CPU route for the parent.
It does not apply the profile or adjust the context budget, and it conflicts with the GPU2 embedder.
Use `--model Qwen3.6-35B` for this deployment.
Exact server model IDs also work. Apart from the 35B layout, selection applies to the parent and its
context summaries; the configured `[subagent_model]` keeps children on its specified backend. Without that optional
section, children follow the parent model selection. Selection does not rewrite the default configuration. Precedence is
`--model`, then `LILLIA_MODEL`, then the config file, then the built-in `Qwen3.8-27B` default.
`LILLIA_MODEL=Qwen3.6-35B` behaves like the full ID and does not select the 35B layout.

`doctor` checks configuration, state storage, model discovery, and embedding/index status.
With `startup_models = true`, it also runs the same real readiness checks as session startup.
These small probes check inference availability; they do not prove tool-calling or retrieval quality.

## 35B layout (`--model Qwen3.6-35B`)

`--model Qwen3.6-35B` switches to a second local layout. The 35B model runs on the 32 GB MI50 for the
parent, the same 35B runs on the P100 worker for the subagents, and the embedding model runs on GPU2.
Runs without this option keep the default layout above.

```bash
./run.sh --model Qwen3.6-35B                  # interactive chat
./run.sh --model Qwen3.6-35B run "Summarize the TODOs in this repository"
./run.sh --model Qwen3.6-35B doctor           # names the model profile, checks all three roles
./run.sh --model Qwen3.6-35B index            # backfill vectors on the GPU2 embedder
./run.sh --model Qwen3.6-35B --background run --prompt-file request.txt
```

| Role | Model | Placement | Window |
| --- | --- | --- | --- |
| Parent | `Qwen3.6-35B-A3B-MI50` | AMD MI50, ROCm (NUMA node 1) | 250,000 tokens, one slot |
| Subagents | `Qwen3.6-35B-A3B-GPU01` | P100 GPUs 0/1 (NUMA node 0) | 250,000 tokens, own slot |
| Embeddings | `Qwen3-Embedding-8B-GPU2` | P100 GPU2 (NUMA node 1) | 4096 dims, stored at 1024 |

Since 2026-10-08 the parent generates on the MI50 slot (`model_concurrency = 1`) while children generate
on the P100 worker's own slot (`subagent_model_concurrency = 1`), so neither role waits for the other.
Before that change both roles shared the MI50 slot, and an 11.5-hour review run serialized every parent,
child, planner and summary generation through it: the parent spent six hours queued. Two children can
still be active and overlap tool work, sharing the worker slot. Both roles keep the 64,000-token output
allowance and uncapped reasoning of the default configuration.

`embedding_model.index_model = "Qwen3-Embedding-8B"` keeps the stored vectors' identity
(`Qwen3-Embedding-8B@1024`), so the existing index is used without a rebuild. Both local layouts
use batch size 32, `backfill_chunks_per_call = 128`, and a 5-second recall timeout.

**How the profile works.** The layout is the `[model_profiles."Qwen3.6-35B"]` table in `config.toml`.
It overlays the rest of the file the same way `[cloud]` does, and without that `--model` value it has no effect.

- Only the exact `--model` value `Qwen3.6-35B` selects it. The full ID `Qwen3.6-35B-A3B` and
  `LILLIA_MODEL=Qwen3.6-35B` do not; they select the standalone GPU2 + RAM entry for the parent.
- It cannot be combined with `--cloud`, and Lillia reports an error if both are given.
- While it is selected, `LILLIA_MODEL` is ignored, since `--model` outranks it.
- The endpoint and model environment overrides (`LILLIA_BASE_URL`, `LILLIA_SUBAGENT_BASE_URL`,
  `LILLIA_SUBAGENT_MODEL`, `LILLIA_EMBEDDING_BASE_URL`, `LILLIA_EMBEDDING_MODEL`) still apply on top
  of it, and startup warns when one is set. `--base-url` still changes only the parent.
- `--background` passes `--model Qwen3.6-35B` on to the detached run, which selects the whole layout again.
- The startup banner, `doctor` and `/status` (`model_profile`) name the active profile.

**Gateway.** `Qwen3.6-35B-A3B-MI50` uses the existing ROCm launcher with the UD-Q4_K_XL GGUF,
full GPU offload, Q8 K/V caches, MTP, and `-c 250000 --parallel 1 -kvu`. The server rounds its
allocation to 250,112 tokens; Lillia budgets exactly 250,000, shared by input, reasoning and output.

- The MI50's default 27B and this 35B are alternatives in the routing matrix. Switching layouts
  swaps the MI50 model, so avoid running default and 35B sessions simultaneously.
- Both layouts and `--cloud` share the GPU2 embedder. Open WebUI's CPU RAG embedder can also stay
  loaded. A cloud session does not load either local chat model.
- The P100 worker serves this profile's subagents and stays loaded for default-layout sessions; the
  routing matrix keeps it resident alongside the MI50 35B.
- The 35B route appears in `/v1/models` for model discovery; the GPU2 embedder stays hidden.
- Use the shortcut `--model Qwen3.6-35B` to select the complete profile. Passing the route's own
  ID changes only the parent.

## Cloud mode (`--cloud`)

`--cloud` runs the parent and subagent models on **Ollama Cloud** and keeps the embedding model
local, on GPU2. Everything else (tools, sessions, memories, the evidence index) is unchanged.

```bash
./run.sh --cloud                              # interactive chat
./run.sh --cloud run "Summarize the TODOs in this repository"
./run.sh --cloud doctor                       # account, models and embedder readiness
./run.sh --cloud --model kimi-k3:cloud        # another cloud parent for this run
./run.sh --cloud index                        # backfill vectors on the GPU2 embedder
./run.sh --cloud --resume                     # a session keeps working when you switch modes
```

| Role | Model | Where it runs |
| --- | --- | --- |
| Parent | `deepseek-v4.1-flash:cloud` (763B MoE, FP8, 1M context) | Ollama Cloud, through the local Ollama daemon |
| Subagents | `minimax-m3:cloud` (512K context) | Ollama Cloud; unlimited parallel children in Lillia |
| Embeddings | `Qwen3-Embedding-8B-GPU2` | llama-swap gateway, P100 GPU2, same weights as the CPU entry |

**Billing.** Chat requests go to the local daemon (`ollama.service`, `http://127.0.0.1:11434/v1`),
which forwards `:cloud` models to ollama.com and signs them with its own device key. That key is
signed in to your Ollama account, so the requests use your subscription's cloud usage. Lillia
itself holds no cloud credential. Startup and `doctor` confirm this from the daemon's `/api/me`:

```text
Ollama account <account> · plan <plan>: cloud generations through http://127.0.0.1:11434 draw on this subscription's usage.
```

A signed-out daemon or a free plan is reported in yellow. Plan limits apply account-wide, including
to Open WebUI's cloud use. Lillia places no limit on parallel cloud subagents or their generation
requests; Ollama can still queue or reject requests at its account/service limits. Usage is metered per token
at each model's rate on the plans introduced on
2026-08-31, or in 5-hour and weekly windows on older subscriptions; ollama.com/settings shows which.
When a usage limit is reached, Lillia reports the server's message (for example "you (...) have reached
your weekly usage limit") without retrying. Progress is saved, so resume once the limit resets.

**How the profile works.** `--cloud` overlays the `[cloud]` table of the config file on the rest
of it. That table is present in `config.toml`. A key in `[cloud.model]` replaces the same key in
`[model]`. A nested table such as
`[cloud.model.extra_body]` replaces the whole table, so llama.cpp request fields never reach Ollama.
Settings the profile does not mention keep their values. Without `--cloud` the table has no effect.
Environment overrides such as `LILLIA_MODEL` still apply on top of it, and startup warns when one is set.
`--background` passes `--cloud` on to the detached run.

The cloud profile sets `delegation.max_active`, `model_concurrency`, and
`subagent_model_concurrency` to `0` (unlimited). The parent can start 10 or more independent children
and all can generate concurrently, including when they share its cloud model. Set positive values in
`[cloud.delegation]` to impose your own limits. Without `--cloud`, the two-child limit and local
generation queues remain unchanged. Restart or resume Lillia to load these settings.

**Pricing.** Per million tokens (input / cached input / output): DeepSeek-V4.1-Flash $0.15 / $0.003 /
$0.60, doubled on weekdays 12:00-18:00 UTC (08:00-14:00 EDT); MiniMax-M3 $0.60 / $0.12 / $2.40.

**Reasoning.** Ollama has no reasoning token budget, only `reasoning_effort`. The main generation
uses `"max"`, the highest level Ollama documents. Reasoning and the answer share a 131,072-token
allowance for the parent and 65,536 for subagents, and neither may exceed what Ollama Cloud actually
serves. Measured on 2026-09-22:
- `deepseek-v4.1-flash` generated all 131,072 requested tokens.
- `minimax-m3` generated 72,887 tokens without being cut off.
- `glm-5.3-flash`, an earlier choice, stopped at exactly 65,536 even when more were requested.

The no-action recovery only retries a length finish that reaches `max_tokens`, so measure a model's
served cap before raising the value or switching models. Compaction summaries, delegation
assessments and recovery reviews ask for `"low"`. A request for no reasoning at all, such as the
startup probe or a summary retry, uses the model's `zero_budget_effort`. That is `"none"` for both
current models, which answer cleanly. GLM-5.3 needed `"low"`, because it answers `"none"` by writing
its thinking into the answer text. The `model_request` journal event records the effort actually sent.
Parallel tool calls are separated by call id as well as index, since MiniMax-M3 numbers every call 0.

**Failures.**
- Tool calls arrive whole, so a large file write is silent while it is being generated. 175 seconds for
  an 18,935-token argument was measured, and `timeout_seconds = 900` is the tolerated gap.
- A stream that breaks off before any answer text or tool call is repeated up to twice. Examples are a
  stall before the first byte, a reset, or a daemon that closes the stream without `[DONE]`.
  Each repeat is journaled as `model_stream_retry` and counts as a model call.
- A new cloud model must be registered with the daemon once: `ollama pull NAME:cloud`.

**Embedder on GPU2.**
- The gateway entry `Qwen3-Embedding-8B-GPU2` serves the same GGUF, pooling and width as the CPU
  entry, with FP32 cuBLAS compute.
- `embedding_model.index_model = "Qwen3-Embedding-8B"` keeps the stored vectors' identity, so both
  routes use one index without a rebuild. Measured against saved vectors on 2026-09-22: cosine
  0.99885 to 1.0, and 8/8 passages matched their own vector first.
- The routing matrix allows this embedder alongside either MI50 chat model, the P100 worker,
  and Open WebUI's CPU embedder. `--cloud` requests only the embedder, so it can run alongside
  either local layout without loading a local chat model.
- The `-GPU2` entry is hidden from `/v1/models`.

**Privacy and exposure.** In cloud mode the conversation, tool output and any file contents the agent
reads are sent to Ollama Cloud. Ollama states that prompt and response data are not logged or used for
training (https://ollama.com/pricing, FAQ). The daemon listens on `0.0.0.0:11434` and needs no
credentials. Anything that can reach that port, including the agent's own shell commands, can spend
the subscription.

To bypass the daemon, set `base_url = "https://ollama.com/v1"` and use names without `:cloud` in the
`[cloud.*model]` tables, then export an API key from ollama.com/settings/keys as `OLLAMA_API_KEY`.
Tool subprocesses inherit Lillia's environment, so the model's shell could read that key.

## Using it

### Conversations and resume

```bash
lillia chat
lillia sessions                    # list conversations, newest first
lillia sessions --pick             # list them, then choose one by number and resume it
lillia sessions --all              # also show sessions that have no messages
lillia sessions --limit 10         # list fewer rows (default 50)
lillia --resume                    # open the most recently updated conversation
lillia --resume SESSION_ID
lillia run --resume SESSION_ID     # continue outstanding work immediately
lillia run "Now check the result" --resume SESSION_ID
lillia show SESSION_ID
lillia export SESSION_ID --output conversation.json
```

`sessions` shows a message count and a subagent count per row. Those are usually what tells rows apart:
a title is only the first 100 characters of the opening prompt, so asking the same thing twice produces
two rows reading identically, and the counts are what distinguish a 4-message false start from the
127-message run you actually want. A `Directory` column appears only when the listed sessions span more
than one working directory.

Sessions with no messages are hidden unless you pass `--all`. Starting the CLI and leaving without a
prompt creates one, and `/clear` leaves one behind every time, so they accumulate; none of them has
anything to resume.

`--pick` is a flag on the `sessions` command, **not** a command inside chat — `lillia chat` has no
`/resume`. It needs an interactive terminal and exits with a message rather than hanging when output is
piped. A blank answer cancels. Once you choose a number it resumes exactly as `--resume SESSION_ID`
would, so a resumed session still keeps its original working directory.

Inside chat:

| Command | Action |
| --- | --- |
| `/continue` | Continue the outstanding request |
| `/clear` | Start a fresh conversation; keep saved memories and archive the previous session |
| `/status` | Show session, progress notes/evidence/reviews, token usage, and managed processes |
| `/agents` | Show child status, reported subgoal/next action, and advisory questions |
| `/mode full` | Execute tools without approval |
| `/mode assisted` | Ask before commands, file changes, memory changes, or network calls |
| `/mode observe` | Expose built-in read tools only; stop existing managed jobs |
| `/memory` | Show saved facts and preferences |
| `/forget KEY` | Remove one saved memory |
| `/exit` or Ctrl-D | Save and exit; stop managed jobs |
| Ctrl-C during work | Interrupt the turn and stop its managed jobs |

### Live terminal panel

While Lillia works, **You (queue) ›** remains editable. Type a follow-up and press **Enter** to queue it.
The terminal confirms submission and shows the number waiting. Messages are delivered in order after
the current model response and its tool-call batch finish, before the next work request. They update
the current task while preserving completed work, evidence, active jobs, and existing budgets.
A background command or subagent may still be running at that boundary; queuing a message does not
cancel it. A final answer prepared before an update is reconsidered with the new information.

Submitted messages are saved immediately. If you stop the turn or the connection fails, `/continue`
or `lillia run --resume SESSION_ID` delivers any messages still waiting. `/status` lists them. If you type a new
message while saved messages are pending, it joins them and continues that request. Unsubmitted drafts
stay in the input field across approval prompts and when work finishes; drafts are not saved on exit.
Up-arrow history and ordinary text editing work during a run. **Ctrl-C** still stops work. Slash commands
such as `/clear` and `/exit` run at the idle prompt; while working, they remain editable drafts.

Interactive terminals show a live panel at the bottom during work, while entering a prompt, and during
approval prompts. It shows the parent and each subagent's phase (queued, waiting for the model, thinking,
responding, using a tool, or finished), the number of active children, and **tokens overall**. Active
children stay at the top; small terminals show a compact layout and point to `/agents` for additional rows.
Normal conversation and tool output scroll above the panel.

During the parent's work requests, a separate **Reasoning (live)** box appears above the status panel
as soon as the server sends reasoning text. Each received chunk updates the box immediately, with screen
redraws capped at 30 frames per second. Text wraps to the terminal width and scrolls to the latest six
lines as it grows. When the generation ends or is interrupted, the full received reasoning is printed
once in a **Reasoning** box in the normal terminal scrollback, so you can scroll back and read it all.
This retains text beyond the live preview's six lines and 16,000-character buffer. Each retry gets its
own box, preserving the previous attempt's reasoning. The live preview starts fresh for each generation.
This uses the server's `reasoning_content` or `reasoning` stream; availability and chunk timing depend on
the model and server. Subagent and internal review requests retain their existing phase indicators.
The box is enabled automatically in interactive terminals, including with `--plain`. Redirected output
keeps the usual answer/tool log. Terminals shorter than ten rows show the compact status layout.

Assistant messages appear once each complete message arrives, rendered as Markdown with headings,
tables, lists, and code blocks. The panel continues showing progress during generation. Interrupted
responses are labeled incomplete and displayed as plain text. Use `--plain` to display raw Markdown;
redirected output also uses raw Markdown. Saved responses and JSON exports retain the original text,
and `lillia show SESSION_ID` displays the saved raw Markdown for inspection or copying.
Answers also render once when independent review marks the result partial. The review notice appears
separately, and the run keeps its partial status and exit code 2.

The token total covers the current conversation and all its subagents, including input, generated output,
reasoning, and context-summary requests. Repeated input context counts on each request, as reported by the
model server. Reasoning is already part of output usage and is not added a second time. This is a token
usage display, not a billing estimate.

While streaming, counts based on text length are marked `~` and **estimated**. Completed requests replace
their provisional count with server-reported usage when available, so the displayed number can adjust
downward. Backends without usage reporting and interrupted streams retain explicitly marked estimates.
An older or failed request with no measurable usage is shown as **unmetered**, rather than silently
counting it as zero. Estimates use roughly four characters per token and can differ substantially from
the model's tokenizer.

Usage is saved with the conversation and restored by `--resume`. `/clear` starts a new conversation with
a fresh counter; the old conversation retains its totals. `/status` also shows the token snapshot.
Redirected output and noninteractive runs use ordinary logs plus an end-of-turn token summary.

`/clear` resets conversation context and its summary, stops managed jobs from the previous conversation,
and keeps the current working directory, model, and autonomy mode. The previous conversation remains
available through `lillia sessions` and `--resume`. Saved cross-session memories are retained.
Subagents are also cancelled on interruption, turn end, `/clear`, or CLI exit.

`/memory` shows only durable facts saved with the `remember` tool, not conversation history or a log of
completed tasks. Running tasks does not automatically create memory entries. The agent is instructed to
save useful preferences and stable project/environment facts proactively, but saving still depends on
the model calling the tool. You can request a specific entry, for example: “Remember that I prefer concise
answers.” Memory writes require approval in assisted mode and are unavailable in observe mode. Saved
facts remain available to conversations using the same `state_dir` through `recall` and `/memory`.

Prompt memory is selected for the latest request. `remember` accepts `scope="relevant"` (the default),
`scope="project"`, or `scope="global"`. Use global only for preferences that apply across topics;
project scope uses the nearest Git root, or the working directory when there is no Git root.
Project entries are included only in that project. Other entries need distinctive topical matches;
a shared name in an otherwise unrelated note is insufficient. Selection uses words, without another
model or embedding call, and includes at most eight whole entries within 6,000 characters. It can miss
synonyms or ambiguous follow-ups; explicit `recall` still lists all entries and their scope metadata.

Old entries migrate to relevant scope without changing their text or dates. Updating a key without
`scope` preserves its existing scope. Entries carry the saved date, owning project when applicable,
and source session for new writes. Memories are historical leads; their current applicability still
needs checking. Selection uses the current request rather than prior tool output or old summaries.

To send a long prompt without terminal paste limits:

```bash
lillia run --prompt-file request.txt
cat request.txt | lillia run --prompt-file -
```

For large documents, tell Lillia the path so it can read selected portions. A single prompt is limited
to one third of the configured context character budget.

For a run that should survive closing its SSH terminal, use the optional detached launcher:

```bash
./run.sh --background run --prompt-file request.txt
```

This returns immediately with a PID and private log path. The normal session journal remains available
through `lillia sessions`, `show`, and `--resume`; the log also prints its session ID. Follow the log with
`tail -f /path/from/output`, or stop that PID gracefully with `kill -INT PID` to save progress and stop its
managed commands. `--background` requires explicit `run` and supports full or observe mode; it never
changes permissions automatically. Assisted mode, interactive chat, and `--prompt-file -` are rejected.
Prompt text is copied into the private run directory so the child does not depend on an open terminal
or the original prompt file. Background execution does not restart after a machine reboot or process crash.
Add `--max-seconds 3600` if you want a one-hour turn limit. Background execution itself adds no time limit.

### Autonomy modes

| Mode | Behavior |
| --- | --- |
| `full` (default here) | All registered tools execute without a confirmation prompt. No command denylist. |
| `assisted` | Read tools run directly. Each shell, write, or HTTP tool invocation requires approval. |
| `observe` | Only built-in local read tools are advertised and accepted. Shell and HTTP tools are unavailable. |

```bash
lillia --mode observe "Find large text files in this directory"
lillia --mode assisted "Update this service configuration"
lillia --mode full "Investigate and clean up expendable build caches"
```

In assisted mode with no interactive terminal, actions requiring approval are denied. Approval applies
to one invocation, including the whole command/script that invocation runs. There is no claim that
arbitrary shell commands can be reliably classified as harmless by inspecting their text.

These are application policies, **not an OS sandbox**. Full mode can read or change everything your
user can access. Observe mode can still read sensitive local files. Use a separate OS account or a
disposable VM if you need enforceable isolation. A root-capable agent can alter its own local controls.

## Tools in this version

- **Shell:** Bash with `pipefail`, explicit working directory, exit codes, merged stdout/stderr.
- **Programs:** `execute_program` runs a literal argv array with a declared command deadline and preserves
  that program's exit status; it supports Python scripts through `["python", "-c", "..."]`.
- **Processes:** start, poll, send text to stdin, close stdin, and terminate a process group.
- **Files:** read text with line numbers and SHA-256, create/replace files, exact text edits, directory listing,
  and bounded literal-text search reporting character offsets.
- **Recall:** `search_evidence` finds saved context artifacts by meaning and by literal terms, returning
  ranked candidate locations to read with `read_artifact`.
- **Web:** fetch HTTP(S) pages and text, extract readable HTML text and link destinations.
- **Memory:** save, recall, and forget durable facts across sessions.
- **Delegation:** spawn, wait for, list, and cancel subagents with inherited permissions.
- **Progress:** record durable work notes and inspect their context and linked original tool evidence.
- **Findings:** retain stable hypotheses, valid negative results, assumptions, and evidence provenance.
- **Verification:** task profiles, acceptance criteria, journal-backed checks, and completion outcomes.

Built-in file edits require the hash of the file that was read and save the prior bytes under
`backups/SESSION_ID/`. A changed file is rejected before replacement. Backups cover built-in file tools;
shell commands do not have automatic undo. Replacement preserves permission bits but does not promise
to preserve ownership, ACLs, extended attributes, or hard-link identity. Prefer native system tools for
files with special metadata.

File tools handle regular UTF-8 files up to 2 MiB. Search scans at most 2,000 files for up to 60 minutes,
skips symlinks and common dependency directories, and reports incomplete results. Use shell tools for
larger files, binary formats, recursive inventories, or specialized operations.

Processes return promptly with a handle when still running. Polling retrieves new output. Each shell
call starts a fresh shell: `cd` and exported variables do not carry to the next call. Use explicit `cwd`
or keep related commands in one script. Text stdin is supported; a PTY for full-screen terminal apps
is not implemented yet. Use `sudo -n` for preauthorized elevation so a password prompt cannot hang a run.

Prefer `execute_program` for generated scripts and checks. Supply `timeout_seconds` explicitly; positive
values are raised to at least `command_timeout_seconds` (3600 seconds / 60 minutes by default), and `0`
deliberately leaves the process unlimited. This minimum applies to `shell`, `execute_program`, and
`run_required_check`, including child agents; larger requested limits are preserved. Results report the
effective `timeout_seconds`. The short `wait_ms` polling window only returns control; it never kills a job.
Start with a small fixture or control before an expensive run. Results distinguish `running`, `succeeded`,
`failed`, `timed_out`, and `stopped`.
They expose a command fingerprint and full streamed-output hash; an optional `input_fingerprint` remains
a caller-declared label, not a verified input hash. Use stable `experiment_id` / `hypothesis_id` values
across retries. A successful shell command only establishes the shell's final exit status: `python bad.py;
cat log.txt` can exit zero despite the failed program. Every execution reports `outcome_verified = false`;
the requested result still needs appropriate checks.

Managed process handles last for the current CLI process only. Normal exit and Ctrl-C terminate ordinary
descendants in their process groups. On Linux, a command supervisor keeps the handle running until those
descendants exit, even when Bash finishes first and a background child redirects all output. Verification
stays pending while that command group is running. Stop, cancellation, and command deadlines apply to the
whole group. Deliberately detached processes, OS services, and jobs surviving an uncatchable termination
require separate inspection. This CLI is not a persistent service supervisor.

## Recalling saved evidence

Tool output larger than `max_output_chars` is replaced in context by a short excerpt plus a
`saved_output` path, and the full text is written to `artifacts/SESSION_ID/` under a name that is the
SHA-256 of its contents. Nothing is lost. What is lost is the *address*: compaction preserves one
archive path mechanically, and every other path survives only if the model copied it into its notes.
Measured on a real store of 42 summarized sessions, the median number of artifact paths still named in
a surviving summary was **1**; the worst session had 201 artifacts on disk and 10 still addressable.

`search_evidence` closes that gap. It maintains an index in `index.sqlite3` beside the journal and
searches it two ways:

- **Lexical** — sqlite FTS5 over every indexed chunk. No model, no network, no new dependency. The
  whole live corpus (907 artifacts, 24,134 chunks) indexes in under a second. A natural-language
  question is translated into an OR-join of its content words first, because a bare question is an
  implicit AND in FTS5 and reliably matches nothing.
- **Semantic** — cosine similarity over embeddings, for the case lexical search cannot reach: the
  recorded text and the question share no words.

Both arms are combined with reciprocal rank fusion, and each result is a **pointer** — `path` plus
`char_offset` — to read with `read_artifact`. Chunks whose text is byte-identical collapse into one
result rather than occupying two slots, which matters because compaction archives re-serialize text
that often also exists as its own artifact.

```bash
lillia index              # build or continue the index (lexical first, then vectors)
lillia index --rebuild    # discard and rebuild, e.g. after changing dimensions
lillia doctor             # probe the embedding endpoint and report index coverage
```

Indexing commits each completed batch, so an interrupted run resumes from saved vectors.
If the embedding server returns invalid values, indexing stops without storing that batch
and reports its chunk range. Repair the embedding endpoint, then run `./run.sh index`
again without `--rebuild`. The P100 embedding server requires FP32 CUDA matrix computation
for some inputs.

**A ranking is not a reading, and it is not evidence.** Cosine similarity measures topical proximity,
not whether a passage answers the question. On this corpus, off-topic queries scored 0.497-0.518 while
a question whose answer appears nowhere in the corpus still scored 0.668 — a confident-looking number
attached to text that merely shares the vocabulary. So `search_evidence` returns raw scores and never a
confidence label, states in its own result that absence from the results is not evidence that something
was never recorded, and points at bytes to read rather than summarizing them. A completed `search_files`
with `truncated: false` is real evidence of absence; a low similarity score is not.

The tool reports what it actually searched, not just what it found:

| Field | Meaning |
| --- | --- |
| `query_terms` | The exact OR-join the lexical arm matched on; a miss here is a vocabulary miss, not an absence |
| `index_current` | False when another process held the write lock, so very recent artifacts may be missing |
| `semantic_available` | True only if the vector arm actually compared vectors — a reachable server that scanned nothing is not coverage |
| `semantic_chunks_scanned` | How many vectors were compared |
| `vector_coverage` | Fraction of in-scope chunks that have vectors, for the scope actually searched |
| `pending_vectors` | Chunks still awaiting a vector |
| `scope` | `this session` or `all sessions`; counts describe that population only |
| `coverage_notes` | Plain-language notes for every way this search was partial |
| `degraded` | Why the vector arm did not run |

Each result carries `path`, `char_offset`, `chunk_chars`, `lexical_rank`, `semantic_rank`,
`similarity` and `duplicate_locations`. When the embedding server is down, search keeps working on
the lexical arm and says so — it does not fail the call and does not pretend the vector arm ran.

Fusion contributes **at most one rank term per arm per distinct text**. Adding a term per copy would
score a passage for being duplicated, and the most duplicated text in a real corpus is boilerplate.

Retrieval quality was probed rather than assumed. On an 8-query probe over this project's own source,
with an identical stopword list for both arms, the vector arm returned the correct file first for 5 of
8 questions (MRR@10 0.781) against 3 of 8 for FTS5 alone (MRR@10 0.463). That is a probe, not a
benchmark, and it was measured on code; quality over the artifact corpus — which is 95% JSON tool
output by volume — is **not** measured. Lexical search covers everything indexed from the moment it is
indexed, which is why it is the arm that always runs.

### Automatic reminders before repeated work

The project configuration enables `retrieval.auto_recall` and
`retrieval.continuous_indexing`. Lillia checks proposed actions before executing them, including
before the first context compaction. Executed research, file, and command tools record compact attempts:
its arguments, observed outcome, and original evidence reference. Successful tools, failed tools,
timeouts, and uncertain interruptions remain distinct. A successful search with no useful results
is recorded too; tool success does not imply that the investigation succeeded. Process polls update
the original launch with its terminal outcome instead of counting as new attempts.

Exact attempts from the current request are available immediately from the journal, without an
embedding or a history scan. They survive compaction and resume. Older sessions also retain the
bounded fallback for successfully executed actions in compacted journal messages. When no exact
attempt matches, a bounded embedding and keyword search looks for related passages in this session.

While a turn runs, a background worker discovers its session's new artifacts and embeds up to four
chunks per batch, checking again two seconds after each batch. Compact attempt records have priority
over archive chunks. Compaction archives and other saved artifacts are indexed as well. The worker
starts again on resume and continues from missing vectors, independently of `search_evidence` calls
and `backfill_chunks_per_call`. New embeddings become
available on subsequent lookups without restarting the solver. Indexing is asynchronous: semantic
recall can lag behind a recent tool result, especially with a slow or unavailable embedding server.

An exact repeated action can defer the entire proposed tool batch once, giving the model a short
historical reminder before any dependent changes execute. The terminal labels this **Tool postponed
(historical recall)**. The model can reuse the finding, choose another action, or deliberately submit
the same call again. When a batch includes a previously reviewed action, new matches become advisory
so that adding another call cannot repeatedly block that retry. Status and evidence lookups in the same
batch (`wait_subagent`, `list_subagents`, `task_status`, `work_plan_status`, `poll_process`,
`inspect_evidence`, `read_artifact` and similar) still execute; only the repeated action and the calls
that could depend on it are postponed, and those proceed through normal validation and approvals when
reissued.

Related passages from embedding or keyword search are always advisory: they never postpone tools.
A bounded runtime note accompanies the actual tool results in the model's next turn and requires no
acknowledgement or retry. Attempt passages whose complete, terminal tool results are already in active
context are omitted; truncated results can still receive a reference note. The same source is normally
shown once per request between compactions, even when later
queries use different wording. Reminders do not mark work complete or replace fresh verification;
files and other conditions may have changed.

Reviewed actions survive resume and compaction: an exact repeat is postponed once per request. (Before
2026-10-08 every compaction re-armed the postponement, and one run deferred the same command five
times.) Shown advisory sources survive resume; after compaction advances, an advisory may be shown
again because the earlier one may have left active context. This deduplication uses the latest 500
reminder records. Journal events distinguish `advisory` from `deferred` reminders. Negative lookups
expire with the lookup cooldown; newly indexed material can be found without another compaction.

Attempt reminders label the last observed outcome, including failure or uncertainty. Related artifact
passages may come from other requests in the same session and are reference material, not proof that
the proposed action was completed. Neither path searches every session automatically or guarantees
recall of every earlier action. The legacy exact journal scan is bounded to 2,000 messages; newly
recorded exact attempts use an indexed lookup without that history limit. Automatic vector search
considers at most 8,000 vectors, prioritizing compact attempts so newer archives do not push them out
of the scan. It still depends on existing index coverage.

The project configuration allows up to three references in a 3,500-character reminder, a five-second
retrieval budget (ten seconds locally), and a semantic check on every eligible action proposal. Only
the short query is embedded in the recall path. Background document indexing uses the shared writer
lock, commits each batch, and yields to manual indexing or other workers holding the lock. Index
failures back off for at least 30 seconds and appear as `continuous_index_unavailable` journal events;
successful batches emit `continuous_index_progress`. Pending work survives shutdown; active embedding
requests are cancelled when the turn ends. Exact matching and indexed keyword retrieval remain
available when embeddings are unavailable, and a lookup failure lets normal work continue.

An existing `index` job can keep running. Restart and resume an existing Lillia worker once to load
this implementation; subsequent memory updates need no restart or manual indexing. Set
`retrieval.auto_recall = false` to disable reminders, or `retrieval.continuous_indexing = false` to
disable the background indexer. Both library defaults remain false.

## Project playbooks

The project configuration enables `[playbooks]`. After a parent delivers a completed, verified
task, Lillia can save a compact account of observed file actions, successful checks, and failed command
attempts. Capture uses the journal without another model call. It keeps source references and bounded
command metadata; it does not copy tool stdout, file contents, or private reasoning into the playbook.
Partial tasks, unverified answers, child reports, and outcomes with unresolved or unavailable independent
reviews do not create playbooks.

A new task automatically receives up to two relevant playbooks from the same project. Project identity
is the nearest Git root, or the working directory when no Git root exists. Retrieval uses shared task
terms and recorded file hashes; changed, missing, expired, or unrelated records are skipped. These are
historical suggestions. File matches do not establish that external services, tool versions, or every
dependency remain the same, and historical commands are never executed automatically.

Use `list_playbooks` and `read_playbook` to inspect saved procedures, and `forget_playbook` to remove one.
The assistant can honor a request not to retain work using
`set_playbook_capture(enabled=false, scope="request")`; `scope="project"` also disables automatic reuse
and capture for future tasks in that project. Existing playbooks and the ordinary session journal remain
until explicitly removed. Setting `[playbooks] enabled = false` disables the feature and its tools.

Defaults are two results, 3,500 characters of injected context, 100 retained entries per project, and a
90-day reuse window. Entries and preferences live in the existing session database. Restart to load
configuration changes; no embedding rebuild is needed. The library default is disabled.

## Subagents

Lillia defaults to `delegation.strategy = "aggressive"` with `assessment_enabled = true`.
After three local tool calls, the runtime asks the model a focused delegation question before the
parent continues. Further assessments happen after `assessment_interval` local tool calls (eight by
default), child reports, context compaction, and completed work milestones. Once the parent has declared
an outcome with `finish_task`, staged an answer, or had an answer withheld for reconciliation, automatic
assessment stops for that request, including across compaction and `/continue`; the worker may still call
`spawn_subagent` itself. (Before 2026-10-08 a withheld answer re-armed the planner, which spawned eleven
reviewers after a completed review.) Blocked work items do not count as unfinished work for milestone
assessments. Read-only investigations count even when the parent has not made a work plan. The assessment considers remaining independent
work, existing assignments, capacity, and the user's request. It can choose bounded assignments or
explain why the remaining work should stay local. Ordinary assessments continue when all child slots
are occupied. Progress review considers each worker's stated subgoal, reasoning for its approach,
expected result, reported findings, and linked tool evidence. It can inspect relevant context and ask
the worker a question at its next normal model request. Assessment is advisory: it cannot cancel an
assignment based on its progress judgment. The parent still has `cancel_subagent` for explicit
cancellation and coordination; cancellation preserves evidence and partial effects and does not count
as successful completion.

Before executing a pure batch of built-in reads, the runtime also estimates its output from file
metadata and requested ranges. A batch estimated to add `preflight_read_chars` (8,000 by default), or
at least 2,000 characters while approaching 60% of the effective input ceiling, gets an earlier
assessment. This can happen before the parent's first large read. The planner sees proposed paths,
ranges, estimates and the current request, without loading those files' contents into the parent.
Every assessment also receives `project_layout`, a bounded listing of the working directory's top-level
entries and one level of subdirectories, so assignments name real paths instead of invented modules.
Successful file, artifact and search output also accumulates between assessments. A substantial next
read can trigger assessment when that accumulated output plus the proposed read reaches the threshold,
even during the normal three-tool cooldown. Narrow checks remain available locally.
Estimates are approximate; a broad search may return much less than its requested maximum.

Preflight assignments are read-only. Reads covered by a successfully started/queued assignment receive
`delegated_read_pending` instead of executing twice; other reads continue. The parent receives the
child's scope and can do independent work, then inspect the needed evidence. A valid local decision,
unavailable capacity, or user opt-out leaves ordinary execution available. Narrow checks
stay local. Mixed batches containing writes, shell commands or coordination calls use the ordinary
workflow. Workload fingerprints and a three-local-tool cooldown prevent repeated planning of the same
read batch. Preflight does not replace compaction or enlarge the server context window.

The advertised assignment schema matches the runtime's deliverable rules. Preflight advertises only
read-only assignments, available read-only tools and runtime-saved reports; a report is exactly `{"kind":"report"}` without a
path. A rejected decision receives its bounded proposal and an actionable correction. The last three
distinct diagnostic messages survive subsequent assessments and resume within the current request; they
are dropped when the work ledger changes or a decision succeeds, so a stale rejection (an old ledger
count, for example) is never presented as current state. The planner may issue its two allowed
`inspect_progress` calls in one response: each runs, and inspections beyond the allowance return an error
result instead of rejecting the reply. A read-only assignment whose task tells the child to write or
append to a file is rejected with a correction, because the child could only end blocked.
After a failed assessment, the parent receives a notice that manual delegation and focused reads remain
available. Two consecutive failed assessment rounds cause one large read batch to receive
`read_batch_deferred`, explicitly identifying it as unexecuted and asking the parent to reconsider its
approach. This checkpoint happens once per failure streak: retrying the same read remains possible,
small checks continue, and a valid local or delegation decision resets the streak. Failure never
creates an assignment, counts as completed work, or requires the user to approve a routine read.

For broad reviews, map the project briefly, assign distinct components or questions, then integrate
findings and inspect the selected evidence. Reading all source in the parent before delegating retains
that source in the parent's context. More children alone cannot recover that lost context isolation.

When an assessment selects valid assignments, the runtime submits them immediately. The parent does
not need another turn to remember and perform the promised spawn. The assessment has its own bounded
output and reasoning budgets (4,096 and 1,024 tokens by default), while sharing the turn's model-call
budget and generation slots. The normal parent and child model settings remain unchanged.

Small requests can finish before the first assessment. Explicit requests to work without agents take
precedence, and serial work may remain with the parent. This controller makes delegation an explicit
runtime step; the model still decides which assignments are useful. It does not guarantee that every
useful opportunity will be found, that reports will be correct, or that more agents will finish faster.
Set `strategy = "balanced"` for selective prompting without focused assessments. Set
`assessment_enabled = false` to use work-plan advisories without focused assessments in aggressive mode.
Restart Lillia after changing configuration or updating the running implementation.

### Progress notes and review

Parents and children use `record_work_note` to retain a bounded explanation of their work:
`subgoal`, `rationale`, `next_action`, and `expected_result`, with optional `finding` and `evidence_ids`.
Write a note when choosing an approach, changing direction, or reaching a useful finding. Notes are
not required after every tool call. They survive context compaction and resume and belong to the
current user request; a new request starts a separate progress record.

`inspect_progress` reads the caller's progress by default, or a child's progress with `agent_id`.
Optional `evidence_ids` retrieve relevant original tool evidence. A note's explanation remains an
unverified claim: linking a command result makes its basis inspectable, but does not make its
interpretation correct or satisfy the completion checks automatically.

In interactive chat, `/status` includes a `progress` snapshot with the latest and earlier work notes,
linked evidence, and advisory reviews for the current request. `/agents` and the `list_subagents` tool
show each child's `reported_subgoal`, `reported_next_action`, `advisory_status`, and `advisory_question`.
These compact fields summarize the latest recorded note and review; absent notes are shown as unknown
progress, not a stall. Use `inspect_progress` for the fuller context and evidence behind a child's summary.

Review uses three states: `unknown` when the available context cannot establish progress,
`evidence_observed` when relevant evidence is available, and `suspected_stall` when the approach may
need clarification or a change. These are advisory judgments. Token consumption, frequent tool calls,
new files, elapsed time, or a long-running experiment alone do not establish useful progress or a stall.
A failed experiment can produce a useful reported finding when the worker explains what it learned
and links the evidence. Assessment can request that explanation without stopping the worker.

For a printer problem, a useful note might link a spooler error to a full disk and explain why moving
a large game could free space needed for printing. The next checks would measure free space and print
a test page. Moving the game is then an inspectable prerequisite, while successful printing remains
unverified. If printing still fails, the worker can record that result and revise its hypothesis.

Progress review uses the existing assessment points and each worker's normal requests. Evidence-triggered
recovery checkpoints can also make a small separate review call after repeated outcomes, even when
delegation is disabled. They use no timer or background surveillance loop. Questions and warnings provide
guidance; the advisory reviewer's judgments do not impose a stop or cancellation. A separate configurable
repeated-read safeguard can end a worker as partial after sustained unchanged reads and a recovery opportunity;
see the repeated-work detection policy below.

`record_finding` and `list_findings` maintain a request-scoped ledger through compaction and resume.
Use a stable claim ID and a status of `hypothesis`, `supported`, `refuted`, `inconclusive`, or `superseded`.
Promotion to supported/refuted requires cited completed observations and `test_validity = "valid"`.
Deliberate negative tests can declare `expected_exit_codes` (default `[0]`); a crash, timeout, or unfinished
test cannot refute a hypothesis. The model must still explain validity: these checks validate provenance,
not the truth of its interpretation. Changed assumption revisions or newly observed file hashes mark
dependent findings for rechecking. Unrelated commands do not invalidate the entire findings ledger.
Unobserved external file changes remain unknown until inspected. This ledger does not replace task verification.

### Assignments and the queue

The parent can call `spawn_subagent` directly, or the focused assessment can submit an assignment.
Include concrete paths, constraints, expected results, and checks. `resources` identifies file/resource
ownership; `read_only = true` restricts a child to read tools, plus `fetch_url` under a full-mode parent.
Editing assignments require explicit resources,
either supplied to the call or inherited from the assigned work item. Both delegation paths reject editing
assignments that overlap active parent or queued/running child editing claims, including overlapping directories
and canonical aliases of the same path. Read-only assignments can share resources. A simple spawn creates a
work item automatically, so a separate multi-field plan update is not required just to delegate. An
optional `work_item_id` binds an assignment to an existing ready work item. Incomplete dependencies
must still be reconciled from observed evidence before that item can run.

Declare `required_tools` for new assignments, for example `["read_file", "search_files"]` for a source
review or `["shell"]` for cloning, commands, or executable tests. `read_only = true` excludes `shell`. It keeps
`fetch_url` when the parent runs in full mode, so web research that edits nothing is a read-only assignment
and needs no resources; assisted and observe parents keep read-only children strictly read-only. Admission checks
reject declared requirements missing from the child's effective capabilities before creating a child or
work item. The declaration is optional for compatibility and does not infer requirements from task prose.
Queued tasks are checked again before dispatch and cancelled if a required capability has disappeared;
their capabilities can narrow but cannot expand beyond the original assignment.

Assignments can declare `deliverable = {"kind": "report"}` for an internally saved report, or
`{"kind": "workspace_artifact", "path": "..."}` for a file. Artifact assignments require a writable
owned path and explicitly declared writing tools before admission; a read-only child can submit a report
without file-write access. Optional `recovery_from` names a finished sibling assignment and supplies its
saved findings, evidence references, remaining checks, and next actions to a replacement worker. The
manifest keeps the original evidence provenance; the replacement must observe fresh evidence for its own checks.

With a positive `max_active`, at most that many children run at once. Additional accepted assignments are recorded as `queued`
and start automatically when slots become free. A queued assignment is not a completed or running
child. The pending queue holds at most 20 assignments. `max_spawn_per_turn`, when nonzero, limits
accepted assignments for the whole turn; its default
of zero allows later waves. The queue does not increase model-server concurrency. In this checkout's
local mode, two active children share one worker generation slot while the parent has its own slot.
Cloud mode sets `max_active = 0`, so all accepted independent assignments start immediately, without
an active-child or pending-queue cap, and cloud generation requests are unlimited within Lillia.

Children receive their assignment, explicit reference context, saved memories, and the same working
directory. They do not receive the parent's full conversation or planning tools and cannot spawn nested
children. Their reports describe findings, evidence, changes, checks, uncertainties, and blockers.
Children aim for short conclusions with exact source paths/line ranges and check results, instead of
source or log dumps. Wait results and automatic reports use a compact envelope bounded by
`max_report_chars`, including verification metadata. Full new reports and verification records remain
in the journal and a `saved_output` artifact; omitted report text is marked. `report_output` points to
the narrative alone, suitable for `read_artifact` from offset zero. The parent should use it for findings
and `saved_output` for the full audit. New audit artifacts also place the narrative before verification
metadata. Older saved results gain a narrative artifact lazily while preserving their original audit.
At very small envelope budgets (the supported minimum is 500 characters), only the audit path may fit;
older audit layouts may still need paging. Compaction preserves narrative handles before audit handles
within its bounded reference list. Legacy text already discarded by an older runtime cannot be recovered.

The parent reviews important claims, uncertainty and changed interfaces using targeted source reads or
raw-output artifacts, broadening checks when evidence is missing, stale or contradictory, or the task
requires it. Child evidence IDs refer to the child's journal; parent verification records must cite its
own observation results. Code changes still require executable integration evidence. A child report
does not automatically verify or complete the parent's task.

For substantial work, `update_work_plan(items, decision)` tracks dependencies, ownership, status, and
completion evidence. Updates preserve omitted items; `work_plan_status` returns the current ledger and
the number of archived items. When the 20-item or text limit needs room, completed items move to the
durable journal. Their IDs remain valid completed dependencies; `work_plan_status(item_id="...")` retrieves
an individual current or archived record. Archived IDs cannot be repurposed, and unfinished work is never
archived to admit another assignment. The plan survives compaction and `/continue` and resets on a new
user request. With `max_active = 0` (the cloud profile), the unfinished-item count and aggregate text caps
are disabled so they cannot indirectly limit parallel assignments; completed records are still archived.
Use
`owner="parent", status="active"` for local work and `owner="unassigned", status="pending"` for future
assignments. New items need `id` and `task`; routine fields have defaults. Existing items accept just
their `id` and changed fields, such as `status="done"` and observed `evidence`, after reviewing a result.
Spawning assigns the child owner automatically. Queued and running assignments cannot be reassigned
until their child finishes or is cancelled. When a finished child's report is delivered to the parent,
the runtime marks its item `done` with the report paths as evidence; the report is still not proof of
success, so verify important claims. A child that failed, was cancelled while running or ended partial
moves its item to `review`; a queued assignment superseded by the final answer moves to `blocked`, and
one cancelled because the turn or its capacity ended returns to `pending`. The runtime's completion is
one-shot: a parent that reopens a delivered item (status `review` or `pending`) keeps its verdict. Pending,
active and review items withhold a staged final answer; blocked items record a limitation and do not. A
staged answer is withheld at most twice for unfinished items or newly delivered reports; the next hold
delivers the latest proposal as `partial` with the reason instead of looping. Partial and blocked
outcomes remain available. `plan_task` defines the working verification plan separately.

### Permissions and shared work

- Children inherit the parent's permission mode and exposed tools, including any tool allowlist.
  A read-only child has read capabilities, plus `fetch_url` when the parent runs in full mode; observe
  mode keeps children strictly read-only.
- In full mode, child actions need no per-action approval. Assisted-mode commands, writes, and network
  requests use the parent's serialized approval flow and are denied if no approval callback exists.
- Give implementation assignments separate files or worktrees. Resource ownership coordinates work;
  it is not an enforced filesystem boundary. Commands run with the current user's host permissions.
- The parent coordinates shared installations, services, databases, Git operations, and memory keys.
  Assign exclusive ownership before delegating work on these shared resources.
- Changes take effect immediately. Cancellation does not undo edits or external effects; inspect partial
  work before retrying. Each child has separate process handles and logs and closes its managed jobs
  on completion, failure, or cancellation. Process limits apply per agent; turn and model budgets are shared.

### Work-plan advisories

With `assessment_enabled = false`, aggressive mode uses the saved work plan to suggest reviews after
child reports, compaction, or extended local work. These suggestions ask for context at a normal
checkpoint. They do not withhold shell/file tools, limit the number of review reads, require a recovery
record, or pause work after a fixed number of unanswered or rejected updates. Existing permissions and
the ordinary per-response read-batch limit still apply.

`work_plan_status` reports `review_suggested` and `replan_suggested`; the compatibility fields
`review_required` and `replan_required` are false, and `review_reads_remaining` is `null`. Earlier
journals may retain observation counts and recovery metadata, which are advisory on resume.
`update_work_plan(items=[], decision="...")` can record a decision without changing ownership or
claiming unfinished work is complete.

Optional `recovery.action` records remain available for `delegate`, `milestone`, and `sequential`.
If supplied, their assignment, dependency, and evidence arguments are validated; invalid or stale
evidence cannot be recorded as a valid recovery. Rejecting that optional record does not block the
worker's next ordinary action. Work ownership, completed dependencies, completion evidence, and
verification requirements continue to be enforced regardless of whether a review suggestion is answered.

### Evaluating delegation

The automated tests use scripted model responses to verify runtime mechanics. They do not demonstrate
the live model's willingness to delegate. To measure live behavior, export a conversation and summarize
its delegation metrics:

```bash
# Export metrics from an existing conversation.
./run.sh export SESSION_ID --output /tmp/lillia-run.json
.venv/bin/python -m lillia.evaluation /tmp/lillia-run.json
```

Compare time to first started child, attempted/accepted/queued assignments, later reuse of freed slots,
assessment decisions and rejections, coordination errors, verified outcomes, tokens, and elapsed time.
The metrics also separate parent/child read-output characters, exact cross-agent duplicate file ranges,
assessment triggers, and preflight deferrals. A targeted read of a different range is not counted as an
identical read. Duplicate observations can be legitimate verification; deferral estimates do not prove
net context or token savings. Compare correct outcomes together with actual read and generation costs.
A queue admission is not a child start, and an extra spawn alone is not evidence of improvement. Old
exports without provenance cannot establish which code or model settings produced the result.

Exports also report execution outcomes without counting repeated polls as new experiments, recovery-review
attempts and rejections, retained child submissions, interrupted workers without a submission, finding
revisions, and reported generation tokens. These are diagnostic metrics; they do not grade correctness.
Scripted tests cover the mechanics across debugging, research, data, and operations labels,
but do not establish the live model's recovery rate.

### Scheduling and budgets

The library and this checkout leave child time limits and final report-collection time limits disabled.
The library defaults in `[delegation]` are below. This checkout overrides `model_concurrency` to `1`
to match the Qwen3.8 server's single generation slot, and uses a separate one-slot worker backend.
With `--model Qwen3.6-35B`, the parent generates on the MI50 entry's slot and the children on the P100
worker entry's own slot (since 2026-10-08); the gateway gates them separately by model name.

| Setting | Default | Scope |
| --- | --- | --- |
| `enabled` | `true` | Set false to remove delegation tools |
| `strategy` | `"aggressive"` | Focused assessments by default; balanced uses selective prompting |
| `assessment_enabled` | `true` | Focused runtime delegation assessments in aggressive mode |
| `assessment_interval` | `8` | Local tools between assessments after the initial three-tool assessment |
| `assessment_max_tokens` | `4096` | Output budget for the focused assessment |
| `assessment_reasoning_tokens` | `1024` | Reasoning budget for the focused assessment |
| `preflight_read_chars` | `8000` | Estimated read output triggering assessment before a pure read batch executes |
| `max_active` | `2` | Concurrent active child tasks per parent; zero disables the cap |
| `max_spawn_per_turn` | `0` | Accepted assignments per turn; zero disables the quota |
| `max_steps` | `0` | Rounds per child; zero disables the cap, subject to parent limits |
| `max_seconds` | `0` | Child deadline; zero disables it |
| `max_report_chars` | `6000` | Compact child report envelope, including check metadata; full result remains retrievable |
| `final_wait_seconds` | `0` | Final report-collection time limit across running and queued children; zero disables it |
| `max_model_calls` | `0` | Shared parent, child, assessment, and summary requests; zero disables the cap |
| `model_concurrency` | `2` | Parent backend requests; shared with children when no separate worker model is configured; zero means unlimited |
| `subagent_model_concurrency` | `1` | Worker backend requests when `[subagent_model]` is configured; zero means unlimited |

Without `[subagent_model]`, children use the parent's provider and generation queue. Configuring that
section creates a separate worker provider and queue; children, their summaries, and their recovery
reviews use it. Parent work, delegation assessments, and parent summaries remain on the parent backend.
Both queues share the same `max_model_calls` budget. A worker server failure is reported normally;
requests are not silently rerouted to the parent server.
Generation requests queue within their own backend's capacity. Failed logical requests count toward a
positive shared call cap; transport retries remain part of their original logical request. A continued or
new user turn receives a fresh model-call budget.
Resuming applies the current configured time limits. Setting a limit to zero disables it, including a
previously saved deadline; legacy deadlines inferred from prompt wording are not restored.
While a positive task limit remains enabled, `/continue` keeps its saved absolute deadline rather than
granting a fresh allowance.
Keep model concurrency aligned with the server's actual capacity. These budgets do not enforce a dollar
or cumulative-token spending cap. More agents can use more tokens and take longer on shared hardware.

### Results, cancellation, and recovery

`wait_subagent` waits up to 30 seconds for a selected report. Finished reports also reach the parent
automatically. Before final synthesis, the runtime waits for running children; `final_wait_seconds = 0`
leaves this wait unlimited. Queued assignments that have not started when `finish_task` stages the final
answer are cancelled as superseded rather than started; a plain text answer leaves them queued for the
parent to wait for, cancel or supersede (before 2026-10-08 the final wait kept dispatching the queue,
which held a finished answer for two hours). An assignment cancelled before it started because the turn
ended, capacity or a capability vanished returns to the ready pool for `/continue`; an explicit
`cancel_subagent` leaves it in review for the parent to decide. A positive `final_wait_seconds` opts into a collection
time limit. On expiry it cancels unfinished assignments, preserves their evidence, and delivers
cancellation reports for parent review. Work items remain unfinished until reconciled, and failure,
cancellation, timeout, and step-limit outcomes remain explicitly incomplete. After a collection timeout,
further delegation and assessments stop for that turn; the parent can inspect saved evidence, complete
remaining local work, or report limitations. A subsequent user turn starts with delegation available again.

Ctrl-C cancels the turn and its active/queued assignments. Turn errors/limits, `/clear`, and exit also
close children. After an uncatchable crash, opening the parent marks unfinished assignments interrupted;
it does not silently restart queued work. Completed reports remain available. Child sessions are excluded
from `lillia sessions` and `--resume latest`; resume the parent to continue work.

Children can use `submit_result(outcome, report, evidence_ids, uncertainties)` to commit their report and
verification state in one transaction and finish immediately. Complete submissions require planned,
passed criteria; partial or blocked submissions retain their explanation and remaining gaps. Later parent
cancellation cannot overwrite a committed result. This child-only bookkeeping tool grants no additional
command or file permissions. Existing `finish_task` followed by a final response remains supported.

Use `/agents` or `/status` to inspect child IDs and queue state. `lillia show CHILD_ID` shows a transcript;
exporting the parent includes child records, messages, and events. Child edits already affect the shared
workspace. Reading or collecting a report does not apply or undo those changes.

## Configuration

Edit [config.toml](config.toml), or copy it to the git-ignored `config.local.toml` for personal settings.

Direct `lillia` invocation searches in this order (first match, no merging):

1. `--config PATH`, when explicitly supplied.
2. `config.local.toml` in the current directory.
3. `config.toml` in the current directory.
4. `~/.config/lillia-full/config.toml`.
5. Built-in defaults.

`run.sh` selects this checkout's local or main config explicitly. CLI options override configuration;
these environment variables are also supported:

| Variable | Purpose |
| --- | --- |
| `LILLIA_BASE_URL` | OpenAI-compatible endpoint, including `/v1` |
| `LILLIA_MODEL` | Model identifier |
| `LILLIA_SUBAGENT_BASE_URL` | Override the worker endpoint when `[subagent_model]` is present |
| `LILLIA_SUBAGENT_MODEL` | Override the worker model when `[subagent_model]` is present |
| `LILLIA_API_KEY` | API key; omitted for a local server that needs no key |
| `LILLIA_STATE_DIR` | Override persistent state location |

For another API key variable, set `model.api_key_env`, such as `OPENAI_API_KEY`. Model-specific request
options go in `[model.extra_body]`. They cannot override the message/tool structure or configured token
limit. The default extra options (`reasoning_budget_tokens` and `chat_template_kwargs`) are for
llama.cpp; use an empty `[model.extra_body]` table when switching to a provider that does not accept them.
Per-request reasoning budgets (assessments and compaction) also send a `reasoning_budget_message`, which
llama.cpp appends when it forces the end of reasoning so the model still produces its output; set your own in
`[model.extra_body]` to override it. A per-request budget of zero means no reasoning: it is sent as
`chat_template_kwargs.enable_thinking = false` rather than a zero budget, which llama.cpp does not enforce
(the model would think until `max_tokens`); without `chat_template_kwargs`, a one-token budget is used.
Both depend on the configured extra options: an empty `[model.extra_body]` sends no reasoning control.
For servers that take `reasoning_effort` instead of a token budget (Ollama), put it in `[model.extra_body]`:
explicit per-request budgets then send `"low"`, and a zero budget sends `model.zero_budget_effort`
(default `"none"`). A configured level already below either is never raised.
For APIs that require `max_completion_tokens`, change `model.token_parameter` accordingly.
`model.include_usage = true` requests usage counts in streamed responses. If a backend rejects
`stream_options`, set `model.include_usage = false`; the counter will use labeled estimates instead.

The optional `[subagent_model]` accepts the same settings as `[model]`, including its own endpoint,
credentials, output/context limits, timeouts, and `[subagent_model.extra_body]`. Its context and generation
budget are validated separately and govern child compaction. Set its complete intended model settings;
omitting the whole section reuses the parent provider. `--model`, `--base-url`, `LILLIA_MODEL`, and
`LILLIA_BASE_URL` override the parent configuration and leave an explicit worker configuration unchanged.
The exception is `--model Qwen3.6-35B`, which selects the 35B layout's profile and so also sets the
worker and embedding models.

The provider currently speaks streaming **Chat Completions with function tools**. It can target compatible
local or hosted services; compatibility depends on their actual implementation. Native Responses API,
Anthropic API, GUI/browser automation, a built-in search engine, and scheduled background tasks are
future extensions. `fetch_url` retrieves known URLs; it is not a web search engine.

### Limits and failures

- `startup_models`: check enabled model roles before chat, run, resume, and doctor (`true` in this
  checkout, `false` in the library). Parent is always checked; the worker requires enabled delegation
  and `[subagent_model]`; embeddings require both retrieval and embeddings enabled.
- `startup_timeout_seconds`: positive deadline per model readiness check (`1800` seconds).
- `max_steps`: model/tool rounds per user turn (`0` by default: unlimited).
- `max_consecutive_tool_failures`: optional consecutive-tool-failure stop (`0` by default: disabled).
- `max_turn_seconds`: optional parent turn time limit, also set by `--max-seconds` (`0` by default: unlimited).
- `delegation.max_seconds`: optional child assignment time limit (`0` by default: unlimited); an enabled
  parent deadline also applies to its children.
- `delegation.final_wait_seconds`: optional final report-collection time limit (`0` by default: unlimited).
- `model.max_tokens`: output token limit per model request, including reasoning where applicable.
- `model.context_window_tokens`: server context capacity **per request**, shared by input and output.
  Both local Qwen models use `250000`, matching their configured server windows.
  The built-in default `0` means unknown and retains the character-only input limit.
- `model.timeout_seconds`: maximum model HTTP idle time (`0` by default: unlimited).
- `model.request_timeout_seconds`: total deadline per model request (`0` by default: unlimited).
- `subagent_model`: optional child-specific versions of the model settings above; absent means use the parent.
- `command_timeout_seconds`: default shell-command lifetime and minimum positive lifetime for all command
  tools (`3600`: 60 minutes). Shorter per-call values are raised to this minimum; explicit per-call `0`
  remains unlimited. Setting the configuration to `0` makes the shell default unlimited and allows any
  explicit per-call limit. `execute_program` still requires an explicit `timeout_seconds`.
- `web.timeout_seconds`: HTTP timeout for `fetch_url` and `web_search` (`3600`: 60 minutes).
- `max_context_chars`: approximate input budget, including tools and messages. Both local Qwen profiles
  use **552,000 characters**; the library and cloud profile use **250,000 characters**. A configured token
  window can lower this ceiling. Neither input setting changes the server's KV-cache allocation.
- `parent_context_chars`: additional parent-only working ceiling, including tools and messages.
  The project configuration sets **300,000 characters** (raised from 180,000 on 2026-10-08 after a run
  compacted 34 times in 11.5 hours), inherited by the named and cloud profiles.
  The library default `0` disables this additional ceiling; positive values must be at least 16,000.
  The smaller of this setting, `max_context_chars`, and the model's input allowance governs parent work.
  Children retain their existing budgets. Summary requests retain the larger ingestion allowance so
  existing long sessions can compact when resumed. This is initial tuning, not a measured quality threshold.
- `compaction_trigger_ratio`: fraction of the available message budget at which compaction starts
  (`0.90` locally; `0.80` in the library and cloud profile). Output tokens, tool schemas and runtime state
  are reserved before applying this fraction.
- `compaction_keep_ratio`: fraction of that message budget available for recent complete exchanges after
  compaction (`0.50` locally; one-third in the library and cloud profile). Instructions and the summary
  also need room, so the retained history can be smaller. Both ratios must be between 0 and 1, with
  `compaction_keep_ratio` smaller than `compaction_trigger_ratio`.
- `summary_max_tokens`: separate total generation allowance for compaction (default `4096`, including reasoning).
- `summary_reasoning_tokens`: reasoning allowance within that summary cap (default `512`); retries disable
  reasoning. `0` leaves the server default in place for the first attempt.
- `max_output_chars`: tool-output text budget. Original command logs retain up to 8 MiB per command.
- `max_read_tools_per_round`: maximum built-in file, artifact, directory, and search reads per model response
  (default `4`). Excess reads receive `read_batch_deferred` results without execution; the model can request
  still-needed reads in another response. Other tools and total model-call allowances are unaffected.
- `recovery.enabled`: enable evidence-triggered advisory reviews, independent of delegation (default `true`).
- `recovery.max_tokens` / `reasoning_tokens`: separate review allowances (`2048` / `256` by default).
  Zero reasoning tokens leaves the server default in place; these settings do not cap the main worker.
- `recovery.cooldown_tools`: minimum additional executed tools between review attempts (default `6`).
- `recovery.read_loop_limit`: accumulated unchanged executed reads before the repeated-read safeguard
  checkpoints and finishes partial (default `12`; `0` disables; positive values must be at least `12`).
  Stronger recovery feedback begins at `max(6, read_loop_limit // 2)`, and the worker gets an opportunity
  to change course before stopping. Applies to parents and children even when `recovery.enabled = false`.

Startup displays the loaded source hash and effective output/reasoning settings. `/status` also reports
loaded configuration identity and whether source or configuration changed on disk. A `runtime_loaded`
journal event preserves this provenance in exports without credentials. Changes are not hot-reloaded:
restart with `./run.sh --resume SESSION_ID`, then use `/continue` to apply new code and settings to saved work.

Library defaults and this checkout leave ordinary chat HTTP, parent, child, and final collection
time limits disabled. Command tools default to a 60-minute lifetime and minimum positive timeout.
Web tools, file searches, and embedding requests also allow 60 minutes by default. Embedding settings of
`0` select that bounded default; positive values override it. Process stdin writes use the job's lifetime
instead of a separate short timeout. Startup checks and automatic background recall have their own budgets.
Enabled parent or child time limits cover model generation and context summaries.
Ctrl-C still cancels work.
A 10-second connection-establishment timeout remains for unreachable chat model servers.
Parent/child step caps, total child creation caps, and the shared model-call cap default
to disabled. Context and per-response output-token limits still apply.

Prompt durations are instructions for the model, never automatic runtime cutoffs. This applies to wording
such as "Work on an eight-hour investigation", "Work for eight hours", and "Task time budget: 90 seconds".
To enable a runtime cutoff, set a positive `max_turn_seconds` or pass `--max-seconds`; set either to `0`
to disable it. Children inherit an enabled parent deadline as well as any positive `delegation.max_seconds`
limit, with the earlier deadline taking precedence. On resume, current settings govern time limits:
zero disables a saved limit, and older deadlines inferred from prompt text are discarded.
An enabled deadline cancels remaining work and saves a report with the latest summary and verification
state. It does not mark incomplete work complete.

Choose a character budget comfortably below the model's actual context capacity and leave room for
generation. Context management applies to both the parent and subagents:

- With the agent's configured context window, input reserves its model's `max_tokens` plus 2,048
  safety tokens. Summary input reserves its smaller `summary_max_tokens` allowance plus the same margin.
  The remaining input allowance is estimated at three characters per token and capped by `max_context_chars`.
  Parent work is additionally capped by `parent_context_chars` when it is nonzero.
  With the configured 250,000-token window and 64,000-token worker output allowance, child input is
  limited to **551,856 characters** (183,952 estimated tokens); summary input retains the
  **552,000-character** configured cap,
  before prompt/schema overhead and early compaction.
  This is a sizing estimate, not an exact tokenizer guarantee; token-dense text and different model
  templates can need a smaller character ceiling. Children use `[subagent_model]` limits when present.
  Update the window when changing servers.
- Compaction uses its own 4,096-token generation cap and 512-token reasoning budget, bounded by the
  configured model output limit. Invalid or truncated summaries retry within that same cap with reasoning
  disabled. Worker generation settings remain unchanged.
- Changing time, deadline, delegation, verification, and progress state is appended after conversation
  history in a transient user-role packet explicitly labeled as runtime status, not a new request.
  This boundary lets recurrent model servers checkpoint the unchanged history before the changing status.
  The packet is not stored as user input or included in summary evidence; actual request identity,
  permissions, deadlines, and completion checks remain enforced by the runtime. Stable instructions
  stay in the initial system message. Actual cache reuse still depends on the backend and other prompt changes.
- Historical `reasoning_content` stays in the transcript but is omitted from active requests by default.
  This does not lower the thinking budget or output-token allowance for new responses. Set
  `[model] replay_reasoning = true` for backends/workflows that require replaying previous reasoning.
  The local Qwen template accepts omitted historical reasoning; model quality still needs task-level evaluation.
- Local compaction starts at 90% of the applicable message budget, after accounting for tool schemas
  and runtime state. With the 300,000-character parent ceiling, its trigger is below 270,000 characters
  before that overhead (about 49,000 characters of instructions and tool schemas plus the runtime status
  message); children keep the larger ceiling. The full 64,000-token output allowance and
  2,048-token safety margin remain reserved. Recent complete tool exchanges are retained up to 50% of the message
  budget, leaving space for further work. The cloud profile keeps its 80% trigger and one-third retention.
  If the latest exchange is too large because of tool results, bounded excerpts preserve the calls,
  operational status, and paths to original evidence.
  Exchanges whose call arguments alone are too large still require archival summarization. Long histories
  are summarized in bounded batches, with every request checked against the budget.
- The latest user request remains available after compaction. Summary checkpoints advance only after all
  batches succeed and the final context fits. Original messages, reasoning, and results are never deleted.
  `context_compacted` records before/after character counts, the working ceiling, and the agent role.
  A durable pointer locates the current user request without reloading archived messages and reasoning
  on each model round. Older sessions build this pointer once when first needed; runtime feedback does
  not replace the user's request.
- Summary instructions prioritize completed work, concrete findings, and the next action. Durable work
  notes and a small read ledger remain available independently of what the summarizer retains.
  A bounded list of journal-derived `evidence_id` and `process_id` references also survives summarization
  with distinct labels; process handles are not verification citations.
- Large tool results are represented by marked excerpts with `saved_output` paths. `read_artifact` pages
  through saved evidence by character offset, even for long lines or files larger than `read_file` supports.
  Artifacts live under the state directory's `artifacts/SESSION_ID/`, with private file permissions.
- Summaries include a source archive path; each archive retains the prior notes and original exchanges,
  so earlier evidence can be followed back through the archive chain. Summaries can still lose details;
  the AI is instructed to retrieve evidence before relying on incomplete excerpts.
- File reads and HTTP fetches save oversized returned text before excerpting. Existing collection limits
  still apply: HTTP downloads stop at 1 MB (reported as `download_truncated`), directory/search scans are
  bounded, and command logs retain up to 8 MiB. Artifacts cannot recover data never collected.

The character budget remains an approximation, not server tokenization. System instructions and tool
schemas must themselves fit. Notes are requested within 6,000 characters. Complete notes are accepted as
they are up to a ceiling derived from the budget, `min(9,000, max(6,000, limit / 8 - 600))` characters
(9,000 with the checked-in budget; the 6,000 target itself at small budgets), because a small overrun does
not justify discarding a finished handoff; a `context_summary_oversized` event records any accepted overrun
and each event body reports the ceiling as `limit`. Longer notes are condensed with one short request
(`context_summary_condensed`) before any retry from the source (`context_summary_condense_rejected`).
Empty, truncated, tool-calling, or still-oversized summaries are retried up to three total attempts per
batch, with a 3,000-character target and no reasoning on retry. Rejected responses are saved as artifacts
(including any partial text of a truncated response, for diagnosis only) and recorded in
`context_summary_rejected` events. Retries and condensing count toward the shared model-call budget.
If retries fail, the previous checkpoint and original history are preserved. Restart Lillia with
`./run.sh --resume SESSION_ID`, then enter `/continue` to resume outstanding work.
A configured positive `max_consecutive_tool_failures` pauses work after that many consecutive
failed tools; the default permits continued recovery attempts. Budget-limited sessions remain resumable.

Repeated-work detection warns parents and children. Duplicate successful
observations add to a repeat count; a new observation subtracts two instead of erasing accumulated
repetition. Six remaining repeats trigger a warning asking the worker to explain or reconsider its
approach. Crossing the warning threshold is
retained even if a later read in the same batch returns new evidence. For unchanged files, the guard
tracks the source ranges actually returned; shrinking or shifting into already-read ranges does not
make them new. Artifact reads compare overlapping returned text, including the visible prefix of a saved
file-read result. Repeat counts are tracked per resource. Successful built-in file edits invalidate only
the changed resource's coverage and repeat count, so writing a progress note cannot clear repetition on
unrelated source files. Newly observed file content or source ranges, a new completed experiment outcome,
and the first accepted pass of a verification criterion each subtract two from accumulated repetition.
Completed experiments are distinguished by their signature; repeatedly polling the same completed process
does not earn new credit. Process polling does not add repeats. Commands, failed tools, deferred tool
calls, and coordination do not automatically reset the count. Elapsed time, token consumption, and
model-call counts do not drive it.
Fingerprint and coverage histories are bounded; `/continue` restores them from the original journal
while resetting repeat debt and allowing another recovery opportunity.
A new user request starts fresh. Warnings are recorded as `repeated_work_warning` events. They can
inform an advisory progress review, but do not prove a semantic loop.

The separate `recovery.read_loop_limit` safeguard defaults to 12 remaining repeats. At
`max(6, read_loop_limit // 2)` it gives stronger feedback asking for a concrete change of approach.
That runtime feedback remains present through compaction. After a recovery opportunity, another batch
containing unchanged executed reads that reaches the configured limit saves
a diagnostic checkpoint and finishes the worker as `partial`, preserving evidence and unfinished work.
Stopping is deferred while managed commands or child assignments are still live, and a newly finished
result that has not been collected gets an opportunity to be processed first.
This applies to parents and children independently of `recovery.enabled`, which only controls the
optional advisory reviewer. New observations, experiments, and verified criteria reduce the count, so fresh
investigation can continue.
This policy targets sustained repeated reads; it does not detect every kind of unproductive loop and
can stop intentional repeated reading. Set `recovery.read_loop_limit = 0` to disable it, including when
measuring sampling behavior without this intervention. Positive limits must be at least 12 to leave room
for warning and recovery. It does not impose a general time, token, or model-call limit.

A separate experiment detector notices three completed attempts with repeated outcomes. It uses declared
experiment identity when present, otherwise command identity and output hashes. It tracks each process's
terminal observation once, so repeated polls are not new attempts. Undeclared successful commands with no
output are excluded because silence does not establish repeated effects. Repeated expensive read rounds
can also trigger a review when duplicate observations corroborate them; token use alone never does.
The bounded reviewer receives actual observations, their age and current process state, and proposes one
concrete discriminating test. It has no action tools, cannot declare completion, and cannot cancel work.
Malformed reviews are recorded and skipped; cooldowns prevent immediate repeated reviews. These signals
can miss semantic loops and can flag justified repetition, so the worker may explain disagreement.

A `finish_reason='length'` can mean either the output allowance or the combined input/output context
window was exhausted. Increasing reasoning alone does not fix a full context window. The error reports
token usage when available, and `model_response_incomplete` events retain the request limits and finish
reason for diagnosis. When reported usage confirms output-limit exhaustion and the context window still
has room, each worker retries up to `length_retries` consecutive times (library default 1; a usable
response restores the allowance). The retry is reported explicitly, logged as `model_length_retry`, and
keeps the same configured output allowance and reasoning setting. What it adds depends on how the
generation ended: if an answer or tool call had started, it asks for a smaller next action; if the
generation was reasoning only, that reasoning is first condensed into working notes by a bounded model
call (purpose `reasoning_condensation`, journaled as `model_reasoning_condensed`) and the retry continues
from those notes, so long thinking is carried forward rather than lost; if the reasoning had degenerated
into repetition or raw hex/number enumeration, it is discarded with a note saying why and the retry alone
sends `[model.degenerate_retry_sampling]` (default `presence_penalty = 1.0`; journaled as `sampling_ignored`
when the backend cannot take per-request sampling). A bounded diagnostic copy of every discarded
reasoning (the whole text when 6,000 characters or fewer, otherwise a 3,000-character head and tail) is
saved as an artifact. Completed work remains saved, active children continue during the retry, and
partial tool calls are never executed.
A failure beyond the allowance, a context-limit failure, or an unconfirmed cause follows normal failure
handling. Configured turn, step, and model-call limits still apply.

Degenerate reasoning is also caught while it streams. `lillia/degeneracy.py` watches the trailing
`degenerate_window_chars` (12,000) of reasoning text: a zlib compression ratio below
`degenerate_compression_ratio` (0.12, a loop or cyclic copying) or, on two consecutive checks, fewer than
`degenerate_prose_density` (30) language words per 1,000 characters aborts the stream within minutes
instead of at the output limit, journaled as `model_response_degenerate` and retried as above. Density
counts Latin words and non-ASCII letters on the text outside long hex identifiers, the density rule is
skipped when a tenth of the window is non-ASCII letters, and a line of eight or more words defers the
verdict by one check, so reasoning in any script is never mistaken for enumeration and a captioned table
gets one more step. The thresholds were calibrated on every saved reasoning trace (legitimate
12,000-character-window minimums 0.247 and 53), so genuine thinking of any length is never cut short;
`degenerate_detection = false` disables both the streaming abort and the output-limit classification per
backend.

A completed parent/child work generation containing only reasoning or whitespace, with neither an answer nor
tool calls, gets up to `empty_response_retries = 2` automatic retries. A usable response restores that
allowance; `0` disables it. The runtime asks for the next concrete action without replaying discarded
reasoning or automatically replaying completed tools. Running children and managed commands stay active during
recovery, and normal output/reasoning limits stay unchanged. All retries count against any configured
step, model-call, and task-time limits and require room for the recovery prompt in the context window.
Cancellation still stops work. Repeated failures stop with an explicit error and saved progress.
Ordinary final answers do not need tool calls and are not retried merely for being text.

`model_response_empty` and `model_empty_response_retry` journal events record the failure and recovery
attempts, including token usage and reasoning character count, without saving discarded reasoning.
Other model errors now retain bounded error diagnostics in `model_request_failed` and `turn_error`.

A broken or truncated model stream never executes its partial tool calls. Transient connection failures
and selected HTTP errors have bounded retries. Tool calls themselves are never automatically retried by
the transport.

If interrupted between a tool call and its saved result, resume inserts an **outcome unknown** result;
the model must inspect actual state before retrying. This avoids automatic replay, but cannot provide
exactly-once execution or undo arbitrary external effects. A model's final answer is recorded as
`answered` only after any active completion gate has accepted it; that status is not an independent
certificate of correctness.

### Embedding and retrieval settings

`[embedding_model]` configures the vector transport; `[retrieval]` configures the index and the
`search_evidence` tool and automatic reminders. Setting `embedding_model.enabled = false` keeps
lexical search working, and `retrieval.enabled = false` disables search, reminders, and background indexing.

| Setting | Checked-in value | Scope |
| --- | --- | --- |
| `embedding_model.enabled` | `true` | False disables the vector arm; lexical search continues |
| `embedding_model.base_url` | `http://127.0.0.1:8080/v1` | The llama-swap gateway; validated like any other backend URL |
| `embedding_model.name` | `Qwen3-Embedding-8B` | Model id sent to the endpoint (the gateway route) |
| `embedding_model.index_model` | empty (means `name`) | Identity stored with vectors; set it when `name` is another route to the same weights, pooling and width (the GPU2 route used by `--cloud` and `--model Qwen3.6-35B`) |
| `embedding_model.dimensions` | `1024` | Matryoshka width; `0` keeps the server's native width |
| `embedding_model.batch_size` | `32` | Inputs per embedding request |
| `embedding_model.query_instruction` | see config | Prepended to queries only; documents are embedded bare |
| `retrieval.enabled` | `true` | False removes `search_evidence` and disables automatic reminders |
| `retrieval.chunk_chars` | `1500` | Chunk size; offsets address the artifact directly |
| `retrieval.chunk_overlap` | `200` | Must be smaller than `chunk_chars` |
| `retrieval.max_results` | `10` | Ceiling on returned candidates |
| `retrieval.backfill_chunks_per_call` | `128` | Vectors computed during one search; `0` disables in-call backfill |
| `retrieval.max_scan_chunks` | `40000` | Ceiling on the brute-force cosine scan |
| `retrieval.index_archives` | `true` | Index compaction archives, which are ~76% of artifact volume |
| `retrieval.auto_recall` | `true` | Review exact repeats once and provide advisory related history; library default is `false` |
| `retrieval.continuous_indexing` | `true` | Index new session artifacts during active turns; library default is `false` |
| `retrieval.index_interval_seconds` | `2.0` | Delay after each background batch; indexing errors back off to at least 30 seconds |
| `retrieval.index_batch_chunks` | `4` | Maximum chunks embedded in each background batch |
| `retrieval.recall_max_results` | `3` | Maximum historical references per reminder |
| `retrieval.recall_max_chars` | `3500` | Maximum reminder length |
| `retrieval.recall_timeout_seconds` | `5.0` | Budget for an automatic index lookup, including its query embedding |
| `retrieval.recall_min_similarity` | `0.65` | Minimum cosine score; lower-score candidates also need matching argument text, not just generic tool words |
| `retrieval.recall_cooldown_steps` | `1` | Action proposals between automatic semantic lookups; library default is `4`; exact matching remains available |

`dimensions` is a real tradeoff and was measured rather than guessed. Qwen3-Embedding-8B is natively
4096-wide; truncating to a leading slice and renormalizing kept 4/5 top-1 agreement with the full width
at 1024 dims, for a quarter of the storage (99 MB rather than 396 MB over the current 24,189-chunk
corpus) and a quarter of the scan cost — width sets both linearly. Vectors are only comparable
within one model and width, so the index records a signature and refuses to mix: change `name` or
`dimensions` and `search_evidence` reports a stale index and asks for `lillia index --rebuild` rather
than silently ranking against incompatible vectors. The signature uses `index_model` when it is set,
so a second gateway route to the same weights (the GPU2 entry used by `--cloud` and `--model Qwen3.6-35B`)
keeps using the same index.

The cosine scan is deliberately plain Python over `struct`/`heapq` rather than numpy, so the project
keeps its four runtime dependencies. Measured against the live index, not a synthetic one: 58.2 µs per
vector at 1024 dims, so about 1.4s for the current 24,189-chunk corpus and 2.3s at the
`max_scan_chunks` ceiling. That is why the scan runs in a worker thread rather than on the event loop,
and why it runs newest-first — when the cap bites, the chunks it drops are the oldest, and the result
says so via `scan_capped`.
Index writes take a non-blocking `flock`: when a subagent or a concurrent `lillia index` already holds
it, a search skips its backfill and serves what is already indexed rather than queueing behind it.

## Task-specific verification

Lillia remains general-purpose. Verification profiles supply guidance for the current task; they do
not change `full`, `assisted`, or `observe` permissions, select a different model, or restrict available
work to coding. No manual coding-mode switch is needed. The model selects one or more profiles with
`plan_task`; creating or editing recognized source files also activates the coding profile automatically.

| Profile | Appropriate evidence |
| --- | --- |
| General | Inspect a created document, validate a data transformation, or check the requested result. |
| Research | Inspect sources, including code paths and relevant passages; check dates and freshness when relevant. |
| System | Measure disk usage before/after cleanup, inspect update results, or check service health. |
| Coding | Execute relevant behavioral checks, test component integration, and compare advertised features with the implementation. |

For substantial work, the model declares observable working criteria before acting.
Mixed tasks can combine profiles. Simple conversation has no verification ceremony, and a small file
task can use one readback check. This adapts the acceptance-criteria and executable-demonstration ideas
from the sibling Lillia project without importing its coding-only orchestration or dependencies.

Static code investigation and source review use research or general verification. A coding plan requires
`execute_program` or `shell` capability for executable checks; without either, `plan_task` rejects the plan before adding criteria
and explains the mismatch. Actual source changes still activate executable verification requirements.
Legacy plans keep their criteria: `task_status` explains capability limitations so the agent can report
partial or blocked work honestly rather than erase an unsatisfied requirement.

New requests distinguish acceptance authority from the model's working plan. The actual request and
delivered user updates remain authoritative. Once verification is required, the runtime adds a protected
`outcome` obligation; passing, revising or retiring working criteria does not satisfy it. Source changes
also retain the protected `code_behavior` obligation. Pure conversation still needs no verification.
For a new request, an allowed `plan_task` attempt activates outcome verification even when its arguments
or proposed criteria are rejected.
Existing saved ledgers keep their earlier immutable contract until a new request starts; `/continue`
does not reinterpret them.

Verification tools manage that distinction:

- `plan_task`: add model-authored working criteria and profiles. They cannot become user requirements
  merely through the model's wording or a guessed result count. Existing criteria cannot be silently
  rewritten. A criterion can declare an exact `required_check` before its
  execution. To attach a check to an existing criterion, supply `id` and `required_check` and omit
  `description` to preserve its wording; new criteria require a description. A new user request starts
  a fresh task; `/continue` preserves the ledger.
- `revise_plan`: explicitly replace or retire a working criterion using a reason and fresh quoted
  observations. The journal preserves its prior definition, check, verdict and revision evidence.
  Replacement invalidates the affected pass and the requested-outcome pass; old executions cannot
  satisfy a revised check, even when its command is unchanged. Retirement is not a pass. The model
  cannot retire the requested outcome, runtime code behavior or host-supplied acceptance checks.
  It may correct a model-chosen code-behavior method without changing that protected obligation.
  Supported and refuted Findings can resolve neutral `{id, finding_id}` inquiries; they never replace
  the requested deliverable. Legacy ledgers do not expose plan revision.
- `record_verification`: record passed, failed, or inconclusive checks, their interpretation, and actual
  tool `evidence_id` values. Fabricated IDs, prior-task evidence, failed commands, running commands, and
  stale evidence cannot support a pass. `recall` can verify memory changes; research needs no shell test.
  Prefer `evidence_selections: [{evidence_ref, passage_id, reasoning}]`: observations expose short
  journal references such as `E123` and exact labeled passages such as `P1`. The runtime retrieves the
  original evidence ID and quotation, so the model only selects a passage and explains its relevance.
  References survive compaction and restart and cannot identify another task's or session's evidence.
  The existing `evidence_ids` and `evidence_support` interface remains available:
  `{evidence_id, excerpt, reasoning}` for every fresh cited result.
  Each excerpt must be one contiguous passage copied exactly from that result's observable payload,
  preserving whitespace and `read_file` line numbers across lines. To cite non-adjacent passages, use
  separate support entries with the same `evidence_id`; each quote and its reasoning is checked.
  Joining skipped/reordered lines or mixing commentary into a quote is rejected with correction guidance
  and a bounded sample of observed content. Put interpretation in `reasoning`, explaining its bearing on
  this criterion. Metadata such as `ok` or the evidence ID is not observation content. For silent checks,
  `exit_code=0` can anchor execution status only; the model must explain the actual checked assertions.
  Existing sessions keep their earlier contract until a new user request starts a task.
- `task_status`: retrieve criteria, gaps, recent evidence IDs with eligibility/rejection reasons, and
  profile guidance. `completion_ready` means criteria pass; `completion_declared` means `finish_task`
  recorded an outcome. Other verification tools return compact acknowledgements and affected checks.
- `inspect_evidence`: retrieve a page of exact selectable passages for an `evidence_ref`, with its
  source observation ID and current eligibility. Inspection does not refresh old evidence or turn a
  running process into a completed one. Observation results initially show up to three passages;
  use this tool for additional passages instead of counting character offsets or reconstructing quotes.
- `finish_task`: record complete, partial, or blocked. Complete requires every criterion to have a current
  passing check; partial and blocked require an explanation. An optional `answer` stages the complete
  user-facing result for delivery after the tool batch, without another model generation. It is required
  for complete outcomes after a prior answer was withheld. Later mutations invalidate the staged answer;
  newly delivered child reports and unfinished work items require the parent to revise it before
  delivery, at most twice per request, after which the latest proposal is delivered as partial.

Required checks work for any task with executable validation. For example, a data-export criterion can
include this declaration in `plan_task`:

```json
{
  "id": "export_valid",
  "description": "The exported data has the requested schema and reconciles with the source totals.",
  "required_check": {
    "tool": "execute_program",
    "argv": ["python", "checks/validate_export.py", "output.csv"],
    "cwd": "/work/report",
    "expectation": "Assert the requested columns, row constraints, and totals against the source data."
  }
}
```

The command and directory above are illustrative; the agent selects checks from the actual task's
instructions and configuration. A shell check uses `tool: "shell"` and `command` instead of `argv`.
Omitting `cwd` fixes it to the session's current directory when the check is declared. Each criterion
has at most one required command or assertion suite. Checks are optional for general, research, and
system work; new coding tasks must declare one for `code_behavior`. This also applies when source edits
activate the coding profile automatically. Persisted tasks without an `execution_contract` retain their
earlier behavior until a new user request starts a task.

The tool schema represents shell and direct-execution checks as complete, independent alternatives.
This also supports inference servers that compile each JSON Schema union branch separately; required
fields must not depend on constraints outside the branch. Both forms require `expectation`, and a
shell check requires `command` while a direct-execution check requires `argv`. Invalid calls report
the complete allowed forms without executing anything.

Declaring a check does not run it. `run_required_check(criterion_id, timeout_seconds)` executes the saved
command exactly, through ordinary permissions and process handling. The agent chooses the timeout,
subject to the same configured minimum as direct commands (or `0` for unlimited);
optional `wait_ms` and `verification_effect` have the same behavior as ordinary execution tools.
The transcript keeps the short call while the journal records the concrete command. Inspection,
automatic recall, capability restrictions, and assisted-mode approval see that concrete command.
The agent can also execute the exact command directly. The
runtime matches the declared tool, exact arguments or shell command, and actual working directory to
journaled execution provenance. Only executions launched after declaration qualify, and the latest
matching run must have a fresh, successful terminal result. An earlier success cannot override a later
failure or unfinished run. `record_verification` automatically attaches the qualifying result when
recording a pass; the agent still supplies evidence excerpts and criterion-specific reasoning.
`task_status` exposes `required_check_status`, criterion origins and bounded revision references.
Host-supplied and legacy declarations remain immutable. For new requests, a mistaken model-chosen
method requires an explicit evidence-backed `revise_plan`; changing its text with `plan_task` is rejected.
Completion requires current results for active checks and separate verification of the actual request.
If a saved method changes after a batch's execution preflight, `run_required_check` is deferred to the
next round so inspection and normal permissions apply to its current command and directory.

Applications embedding Lillia can supply explicit acceptance checks through the host API:

```python
await agent.run(
    "Produce output.csv with the agreed export schema and totals.",
    acceptance_checks=[{
        "id": "required_export",
        "description": "The delivered export satisfies the agreed schema and totals.",
        "required_check": {
            "tool": "execute_program",
            "argv": ["python", "checks/validate_export.py", "output.csv"],
            "expectation": "Assert the agreed columns, row constraints and source totals.",
        },
    }],
)
```

This argument is accepted only for a fresh request, validated before the request is recorded, and has
no model tool equivalent or model-controlled trust flag. It declares checks without executing them or
granting permissions. Ordinary natural-language instructions are not automatically converted into
machine-checkable assertions; their coverage still needs evidence and semantic assessment against the
saved request. Quoting a request does not prove that a proposed assertion follows from it.

`task_status`, verification acknowledgements, and rejected `finish_task` calls include
`completion_actions` for each outstanding criterion. `record_verification` means the declared execution
already succeeded and remains current: inspect its saved evidence and record the criterion without
rerunning it. `run_check` means execution is missing or stale; `poll_process` means collect an existing
run; failed or interrupted checks require inspection. These are mechanical next steps, not automatic
certification of the model's conclusions. Historical recall is advisory for an exact successful check
that the verifier now requires again after a possible change. It still reviews other repeated actions,
including failed checks and unrelated work in a mixed batch.

Repeated completion attempts against the same unresolved required check trigger recovery feedback
after two rejections. Another rejection in a later model round saves a diagnostic checkpoint and ends
partial. Unrelated reads, helper scripts, and rewritten explanations do not erase this history. A new
matching check execution or accepted pass restores the opportunity to proceed. Active commands and
children receive time to finish and their results can be collected. This safeguard also works when
advisory recovery reviews are disabled; it adds no general time or token limit. Resume preserves the
warning, while `/continue` after a saved stop grants a fresh recovery opportunity.

Plan checks after inspecting the relevant interfaces, and prefer a test-file or suite entry point over
a long inline program. Test files can be repaired against the requested behavior. A new model-authored
working declaration can be corrected with recorded evidence; a mistaken or impossible host/legacy
declaration requires an honest partial result. Lillia must not change correct application behavior just
to satisfy an incorrect assertion.

Writes, memory changes, shell/process mutations, and URL fetches activate verification even without an
explicit plan. Recognized source-file writes add a behavioral criterion that requires execution evidence.
The model must explicitly select coding for code generated through arbitrary shell commands; the runtime
does not pretend it can reliably infer every shell command's purpose. Pure built-in file reads do not
automatically require a plan, but an explicitly planned inspection still requires its declared checks.

Saved-memory changes (`remember`, `forget`) preserve unrelated workspace, system, and source checks.
Checks citing `recall` retain a fingerprint of the observed memory snapshot and become stale when that
snapshot changes, including writes from another session. Verify memory changes with a new `recall` result;
do not rerun unrelated tests. Saving the same memory value again leaves the snapshot unchanged. Memory-only
tasks still require verification. Existing saved checks retain their original workspace revision and
execution contract on resume; this change does not automatically approve previously stale evidence.

Possible environment mutations invalidate prior passes and a previous completion decision. Shell and direct program
executions default to `verification_effect="mutation"`. The model can explicitly declare
`verification_effect="observation"` when a command only inspects state. This declaration affects evidence
freshness only: it does not sandbox commands, grant permissions, or make commands available to observe-mode
children. Observational processes can overlap without invalidating one another; a mutation during their
execution still makes their evidence stale. Unknown commands and tests that change state must retain the
default. Stopping an already-finished process cannot refresh its evidence. A prior measurement
can remain in the transcript for comparison, but the passing check must cite fresh evidence showing the
comparison/outcome. A pass may cite older IDs alongside at least one fresh ID; the older IDs are kept as
`stale_evidence_ids` for context, and a pass with only stale IDs is rejected with a message naming the stale
IDs, the journal floor, and the freshest usable evidence. Completed process polling can provide evidence;
a still-running job cannot.
Tool argument validation reports the applicable bounds, types, allowed fields, or missing required fields
without echoing submitted values. Invalid calls remain unexecuted and cannot invalidate passing checks.
For example, a `wait_ms` maximum error reports `maximum=10000`; correct the argument and use
`poll_process` if the job is still running. The runtime does not silently rewrite commands or relax schemas.
The same global freshness rule applies to required checks: running several mutating checks separately
invalidates earlier results. Use one assertion suite that propagates any check's failure, or perform
genuinely observational checks after the writes. Declaring a mutating command observational to preserve
an earlier pass is incorrect.
After a CLI restart, lost handles remain unresolved. Inspect the actual machine/process state with a new
command, then use `record_verification.resolved_process_ids` with that fresh successful evidence and an
explanation. Known live handles must be polled instead; bookkeeping never stops a process.

The criteria, checks, and invalidation state are stored in SQLite events and reintroduced independently
of model-generated context summaries. Large ledgers are exposed through a bounded reference and retrievable
artifact. A proposed final answer with incomplete bookkeeping is withheld, archived, and returned to the
model for correction. System-role feedback explicitly says the user has not seen it and names missing
criteria or missing finalization. The first and latest withheld proposals remain addressable through
compaction and resume, including read-only requests without declared criteria. The runtime reminds the
model that these drafts have not been delivered. Recovery normally uses an explicit corrected
`finish_task(answer=...)`, which delivers the answer without another acknowledgement.
After two reminders, if the model still supplies a fresh answer without calling `finish_task` and all
checks and work items are satisfied, the runtime records completion through the same verification gate
and sends that current answer through ordinary independent review and delivery. It never selects an
older saved draft for automatic delivery. `/continue` reopens older sessions stopped solely because
`finish_task` was missing, preserving their evidence and drafts for reconsideration.
If verification or planned work remains incomplete after two correction opportunities without a
successful observation or action, it ends as `partial`, preserving the work for continuation.
Successful observation/action tools restore this
allowance, so isolated slips throughout a long task do not accumulate into an early stop. Failed or
deferred calls and status/bookkeeping calls do not restore it. A tool succeeding still does not prove
completion: the same criteria and verification checks remain required.
Partial and blocked CLI runs return exit code 2. Permission denials remain enforced by the existing tools;
verification never grants new access or prompts for approval by itself.

Subagent reports now distinguish `reported` (an answer was submitted) from `completed` (declared checks
passed), and include `outcome_verified` plus the verification record. `ok` indicates that the child returned
normally, not proof of task success. A child that made changes obliges the parent to verify integration;
cancelled or failed children retain their verification needs because they can leave partial effects.

**Limits:** evidence existence, source-excerpt attribution, declared command identity, execution status,
freshness relative to observed tool actions, and ledger completeness are enforced. The runtime does not
discover the right commands or prove that a declared command implements its `expectation`; that text
describes intended assertions. The model still judges whether checks and evidence are relevant and
adequate. An inadequate command can be declared and misinterpreted, external changes outside journaled
operations are not detected, and a mistaken observation declaration can misclassify effects. This does
not prove that every requirement was identified. For substantial software work the guidance
calls for behavioral/integration checks and useful independent review. The optional outcome review below
adds a separate assessment without automatically running arbitrary project tests. These safeguards reduce premature completion; they are
not a correctness guarantee or a measured improvement in model capability.

### Independent outcome review

The project configuration enables `[outcome_review]`. Before delivering evidence-backed
parent work, a separate model request compares the proposed answer with the actual request, declared
criteria, executable check definitions, and bounded original observations. The reviewer uses a fresh
context on the configured parent model. Bounded earlier conversation excerpts help resolve follow-ups
that refer to a previous proposal; the latest user request controls scope. It can identify omitted requirements or unsupported claims and
suggest a concrete check; it cannot execute tools, change criteria, or certify real-world correctness.

Even a task based on one observation receives review. Pure conversation, child reports, partial/blocked
outcomes, and requests explicitly prohibiting agents skip it. Read-only answers without a declared plan
receive an implicit outcome criterion in the review. A clear verdict requires a support assessment for
every criterion, each citing eligible observed evidence and explaining coverage and applicability.
The review checks the requested entity, time, environment and scope, including the worker's selected
passages and its interpretation. A fresh read of an obsolete document establishes only what it says.
Each assessment separates what a source says from why it applies to the requested situation, using
`claim_scope`, `applicability_basis`, and `applicability_evidence_ids`. Generic source-status words
such as draft, archived, or temporary trigger an additional check: a supported current-state claim
needs a separate eligible observation connecting that reference material to the current situation.
Reading the same source again does not satisfy this check. Explicit questions about historical
documents or source contents remain answerable from those documents. These word-based signals are
conservative and can request unnecessary checks; absence of a signal does not establish authority.
Within the context limit, complete cited observations take priority over uncited search history.

There are at most two outcome packages, one correction of invalid review output, and three review
generations in total per request. One worker repair is allowed for concerns or missing support.
These allowances survive resume and obey the shared model-call budget. A format correction shares
the original review deadline and cannot execute returned tools. Prefer criterion or conversation IDs
over copying quotations; legacy quotation fields are still checked exactly. Review packages and raw
responses are saved as private artifacts, with rejected output linked from `outcome_review_rejected`.
Identical reviewed packages reuse their result.

If review remains unavailable, uncertain, or concerned, the delivered task has **partial** status
with a separate explanatory notice and journal record. The candidate answer retains its requested
format, including JSON/CSV. Such results cannot seed playbooks. Existing completion checks still apply,
including checking evidence freshness after the review finishes. The reviewer is another model judgment;
it can still miss an error or raise a mistaken concern.

Defaults are a 2,048-token generation allowance, 512 reasoning tokens, a 24,000-character review package,
and a 180-second timeout. These apply only to the optional reviewer. Existing task deadlines and shared
model-call limits also apply. Set `[outcome_review] enabled = false` to disable it; the library default
is disabled. An additional review can increase latency, and a second model judgment can be wrong.
`clear` means no material concern was found in the supplied package, not proof of correctness.

## State and privacy

By default, `~/.local/share/lillia-full/` contains:

```text
sessions.sqlite3       Conversations, tool events, session state, and durable memory
input-history         Interactive prompt history
outputs/SESSION_ID/   Bounded command output logs
backups/SESSION_ID/   Original bytes saved by built-in file writes
SESSION_ID.lock       Exclusive session lock
```

The state directory is created with owner-only access and the database/log files with restrictive
permissions. Existing directory permissions are left intact. Sessions and logs can contain sensitive
tool output; they are not automatically redacted. API keys are read from the environment, not written
into model requests' logs by the provider. A hosted provider receives the conversation and selected tool
results. A local gateway may itself route to remote services depending on its configuration.

Individual command logs are capped, but total retained sessions, logs, and backups are not automatically
pruned. Use `--state-dir` for disposable work and manage retention according to your needs.

## Development

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Tests use temporary directories and fake model responses. They exercise real subprocesses, timeouts,
cancellation, file mutation, session locking/recovery, streaming parsing, context compaction, policy
enforcement, delegation isolation/budgets/cancellation, and the full CLI → HTTP → tool → answer path.
Terminal tests also cover the live panel, approvals, cancellation, and clearing the conversation;
usage tests check parent/child totals, resumed history, and replacement of estimates with reported counts.
They do not call a paid model or clean the host.

### Layout

```text
lillia/
  cli.py          Terminal interface, commands, cancellation
  dashboard.py    Live bottom panel for agent phases and session token usage
  telemetry.py    Shared streaming token accounting and durable usage totals
  agent.py        Persistent inspect/act/verify loop
  budget.py       Opt-in configured deadlines, inherited limits, and elapsed-time guidance
  verification.py Task profiles, durable acceptance criteria, evidence and completion gate
  evidence.py     Durable observation references and exact selectable passages
  outcome_review.py Bounded independent review of proposed answers and actual evidence
  playbooks.py    Project-scoped historical procedures and applicability checks
  context.py      Active context, bounded summaries, and retrievable evidence
  recall.py       Automatic historical reminders before possible repeated actions
  attempts.py     Durable attempted actions and observed outcomes for immediate recall
  live_index.py   Background incremental indexing during active turns
  recall_history.py Exact successful action lookup in compacted request journals
  index.py        Lexical (FTS5) + vector recall over saved artifacts, and search_evidence
  embeddings.py   Batched embedding transport for the evidence index
  delegation.py   Subagent permissions, report delivery, shared model-call scheduling
  provider.py     Streaming model adapter
  tools.py        Tool schemas, argument validation, policy, file/web/memory tools
  processes.py    Local command lifecycle, output capture, process groups
  findings.py     Durable claims, dependencies, observed artifact versions
  recovery.py     Evidence-triggered advisory checkpoints
  runtime.py      Loaded code/configuration identity and effective settings
  store.py        SQLite journal, memory, recovery, exclusive session locks
  config.py       Configuration and validation
```

Add a capability by registering a `Tool` with a JSON argument schema, an effect category, and an async
handler in `ToolRegistry`. A second model provider needs `complete(messages, tools, on_text) -> Reply`
and `close()`; keep provider-specific protocol details out of tool execution.

The project was informed by the neighboring Lillia project's agent loop, shell execution, and recovery
design. This implementation is independent: it does not import or modify `../Lillia/`.

Protocol references: [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)
and [HTTPX async streaming](https://www.python-httpx.org/async/).

### Long-run stress tests

Parent/child rounds, total child creation, and model calls remain unlimited unless configured otherwise.
Both the library and this checkout leave parent turns, child assignments, and final report collection
unlimited in time. Prompt wording never enables a runtime deadline. Positive time limits remain available
through `--max-seconds`, `max_turn_seconds`, `delegation.max_seconds`, and `delegation.final_wait_seconds`;
zero disables each limit. The AI can finish earlier when the request is complete or blocked.
Ctrl-C cancels the run and managed child processes; additional positive caps remain available for experiments.
The repeated-read safeguard remains enabled independently of those limits. Set
`recovery.read_loop_limit = 0` for sampling experiments that must observe sustained repeated reading
without this intervention; the advisory reviewer alone still cannot stop work.

Local resource bounds in this checkout remain: two active children, one parent and one child model request at
a time on separate servers (one request in total under `--model Qwen3.6-35B`, where they share a
server). Cloud mode removes Lillia's child-count and generation concurrency caps.
Both modes keep eight active commands per agent, finite context/output/report sizes,
bounded file/web reads and command logs, and no nested
delegation. These constrain simultaneous load and request size without limiting the total number of
sequential work steps. Artifact/transcript storage can grow throughout a long run.
