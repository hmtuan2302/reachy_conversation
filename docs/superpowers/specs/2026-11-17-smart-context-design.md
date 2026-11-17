# Smart context retrieval — design

Status: approved by the owner (2026-11-17). Builds on the memory design
([2026-11-17-memory-design.md](2026-11-17-memory-design.md)) and the context budget
([2026-11-17-context-budget-design.md](2026-11-17-context-budget-design.md)).
Tags: **[owner]** decided by the owner, **[my call]** proposed and open to correction,
**[verify]** unconfirmed (check before relying on it).

## 1. Intent

Memory will run for months. Today the robot gets a fixed briefing (newest/most-used facts) plus
keyword `recall` when the model thinks to call it. Two things break as memory grows:

1. **Relevant memories don't show up unasked.** "I'm stuck on a bug" never brings up "Tuan codes
   with Claude" or last Saturday's bug, because the briefing ignores the topic and the model
   doesn't call `recall` for small remarks.
2. **Keyword search misses paraphrases.** "What AI tool do I code with?" misses "Tuan uses Claude"
   with no shared words, and the consolidator misses the old "uses Copilot" fact, so it is never
   expired (two contradicting facts stay current).

Success: the robot brings up a related memory when it fits, finds memories by meaning in VN or EN,
and the reply does not feel slower.

## 2. Goals, non-goals, anti-goals

- **Goals**
  - Local embeddings with `google/embeddinggemma-300m` **[owner]**.
  - Hybrid search (keywords + vectors + recency) for `recall` and for the consolidator's
    similar-fact lookup.
  - A per-turn **push** of a few related memories, inside the token budget.
  - Built in two steps; step 2 only after step 1's timing is measured **[owner: approach C]**.
- **Non-goals**: an LLM query-rewrite or decomposition step (the agent already writes standalone
  `recall` queries; §7.4); a vector database or `sqlite-vec`; re-ranking with a cross-encoder;
  objects/places (memory Phase 2); summarizing old episodes (memory Phase 3).
- **Anti-goals**
  - The robot bringing up unrelated memories ("speaking of Python, your sister is Mai").
    A quiet robot beats a noisy one.
  - A guest's query surfacing owner-only facts through the vector path.
  - "Forget that" leaving a vector behind.
  - The voice loop stalling while the CPU embeds.

## 3. Constraints

- CPU-only WSL2 laptop, live demo; reliability beats polish (CLAUDE.md).
- `onnxruntime`, `numpy` are core deps; `tokenizers` and `huggingface-hub` are installed today only
  through optional extras (`vieneu`; `huggingface-hub` also via `robot`).
- DeepSeek has no embeddings API; DeepSeek caches the prompt prefix (system + history), so text
  that changes every turn must not go in the system prompt.
- One SQLite file (`MEMORY_DB`), `PRAGMA foreign_keys=ON`, migrations list in `store.py`
  (`MIGRATIONS`).
- Business logic stays free of SDK/network imports; the embedder is injected so tests use a fake.
- `CONTEXT_MAX_TOKENS` / `MEMORY_MAX_TOKENS` already cap each request.

## 4. Approaches

| | Approach | Trade-off |
|---|---|---|
| A | **Push only**: embed what the user said each turn, add the top related memories | Proactive, but `recall` stays keyword-only, time-window and expired facts never pushed, Copilot→Claude invalidation stays broken; adds ~50–100 ms per reply **[verify]** |
| B | **Pull only**: hybrid `recall` + hybrid consolidator lookup | No added delay, least code, fixes paraphrase and invalidation; the model still has to decide to call `recall`, so nothing comes up unasked |
| **C (picked)** | **Both, in two steps**: B first, then the push on top | Best recall and the only one that is proactive *and* finds paraphrases; costs A's delay and two things to tune. Two steps put the latency risk before the code that depends on it |

Rejected: Mem0/Zep/graph stores (heavy, against the single-process rule); embeddings through a
cloud API (none from DeepSeek; adds network latency to every turn); an LLM query rewrite per turn
(0.5–1 s on the hot path).

This reopens the memory design's "vectors only if the eval misses" gate (§4 there): the owner
chose vectors for proactive recall and paraphrase, not because the 22-question eval failed.

## 5. Architecture

```
WRITE (background worker, never on the reply path)
  consolidator ADD/UPDATE fact ─► embed "title: none | text: <fact>" ─► fact_vecs
  session end → episode summary ─► embed                          ─► episode_vecs
  startup / `memory embed`      ─► embed rows that have no vector for the current model

READ 1 — pull (only when the model calls recall), step 1
  recall(query) ─► BM25 list ─┐
                 ─► vector list (embed "task: search result | query: <q>") ─┼─► RRF + recency ─► top k
                                                                           ─┘  (scope enforced in SQL)

READ 2 — push (every user turn, before the first LLM call), step 2
  user text ─► small talk? reuse last set
            ─► embed once, keep vector; query = mean of last ≤3 user-turn vectors
            ─► cosine vs this speaker's current facts + episodes ─► top 3 above threshold
            ─► "[related memories]" lines prepended to the *sent* copy of the user message
```

### 5.1 Components

| File | Responsibility |
|---|---|
| `memory/embed.py` (new) | `Embedder` protocol `embed(texts: list[str], kind: "query"\|"doc") -> np.ndarray` (float32, L2-normalized, 256 dims); `GemmaEmbedder` (ONNX Runtime + `tokenizers`); `load_embedder(env) -> Embedder \| None` |
| `memory/vectors.py` (new) | `VectorIndex`: in-RAM numpy matrix + ids per kind, loaded from SQLite at start, updated on write/delete; `top(query_vec, kind, n, allowed: set[int]) -> list[(id, cosine)]` |
| `memory/store.py` | Migration V4 (tables §6); `set_vec`, `missing_vecs`, `load_vecs`; `allowed_ids(kind, speaker, person_id, subject_text, start, end) -> set[int]` (the same `_scope` and filters as search, ids only); `search_facts`/`search_episodes` take an optional `vec_ids` candidate list |
| `memory/recall.py` | Fuse three ranks (BM25, vector, recency) |
| `memory/consolidate.py` | Embed on ADD/UPDATE; hybrid similar-fact lookup |
| `memory/episodes.py` | Embed the summary |
| `memory/related.py` (new, step 2) | `related_block(store, index, embedder, speaker, user_text) -> str` |
| `memory/briefing.py` (step 2) | Also return the fact and episode ids it showed (`Briefing(text, fact_ids, episode_ids)`), so the push can skip them |
| `memory/session.py` | Owns `Embedder`/`VectorIndex`; caches the briefing ids; keeps the last ≤3 user-turn vectors (RAM only); `async related(text) -> str` |
| `agent/agent.py` (step 2) | Optional `related: Callable[[str], Awaitable[str]]` hook, awaited once per turn; result prepended to the sent copy of the user message, never stored. The request is now built from a copy of the turn on every round (today `turn[0]` is both sent and stored) |
| `agent/cli.py`, `agent/memory_eval.py` | Wiring, `memory embed` subcommand, new eval categories and timing |

## 6. Data model (migration V4)

```sql
CREATE TABLE fact_vecs(
  fact_id INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
  model   TEXT    NOT NULL,          -- EMBED_MODEL (or folder name) + dims, e.g. 'onnx-community/embeddinggemma-300m-ONNX@256'
  vec     BLOB    NOT NULL,          -- 256 × float32 little-endian = 1 KB
  PRIMARY KEY(fact_id, model));
CREATE TABLE episode_vecs(
  episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
  model      TEXT    NOT NULL,
  vec        BLOB    NOT NULL,
  PRIMARY KEY(episode_id, model));
```

- **Two tables, not one**, so each has a real foreign key: `delete_facts` and `forget_person`
  (the paths the `forget` tool uses) delete vectors by cascade, with no extra code; Phase 3
  retention, when built, gets it for free **[my call]**. Retention is not in this change.
- `model` in the key: changing model or size re-embeds; old and new vectors never mix.
- `update_fact` deletes that fact's vectors in the same transaction; the consolidator re-embeds.
  A fact without a vector is still found by keywords.
- User-turn vectors (push) are **not stored**: RAM only, gone at session end (privacy).
- Size: ~1 KB per item; 10 000 items ≈ 10 MB.

## 7. Behaviour

### 7.1 Embedder

- Model: `onnx-community/embeddinggemma-300m-ONNX` **[verify name, files, output names]**,
  quantized variant if the spike shows it agrees with fp32 (cosine ≥ 0.98 on the eval texts)
  **[verify]**. Downloaded once to the HF cache, or a local folder via `EMBED_MODEL`.
- Gated under the Gemma license: the owner accepts it once on Hugging Face; `HF_TOKEN` (already
  in `.env`) authorizes the download **[verify]**.
- Prefixes (model card): query `task: search result | query: {text}`; document
  `title: none | text: {text}` **[verify]**.
- Output 768 dims → keep the first 256 (Matryoshka) → L2-normalize again **[verify]**.
- Inputs cut at 256 tokens (facts and summaries are short).
- Runs through `asyncio.to_thread` so the event loop (audio, turn detection) never blocks;
  every caller on the reply path is therefore async (`recall`'s tool call already is; the push
  hook is async). One warm-up embed at load, so the first turn is not slow.
- **Fail small**: no extra installed, model not downloadable, or load error → log once,
  `embedder = None`; recall and consolidator fall back to keywords, push is off. The agent keeps
  working.

### 7.2 Write path

- Consolidator: after `add_fact`/`update_fact`, embed the fact text (English `text`, not
  `text_original`) and `set_vec`; update `VectorIndex`.
- Episodes: embed `summary` after `add_episode`.
- Backfill: on each worker idle tick, once no turn happened for 60 s, embed up to 200 rows lacking
  a vector for the current model, in batches of 16 **[my call; changed after the final review:
  no startup burst, so it never competes with a live turn, and failures retry next tick]**. `reachy-demo memory embed`
  does all of them now and prints a count.

### 7.3 Hybrid search (step 1)

- Vector candidates: first `store.allowed_ids(...)` with the **same `_scope` and
  person/subject/time filters** as keyword search (ids only, a few ms), then
  `VectorIndex.top(q, kind, n=20, allowed=ids)` masks to those ids **before** taking the top n.
  So a guest never gets an owner fact, and a filtered query is not starved by other people's
  vectors. Rows are then fetched through the same SQL as today. Candidates under cosine 0.3 are
  dropped **[my call, tune]**.
- Fusion in `recall`: RRF of BM25 rank, vector rank and recency rank (`RRF_K=60`, as today); an
  item missing from a list gets no term for it. Dedupe as today.
- Consolidator similar-fact lookup uses the same hybrid search (current facts only), top 5.
- Recall tool description gains one line: *"Write a standalone query (no pronouns). For several
  topics, call recall once per topic."*

### 7.4 Push (step 2)

- Called once per user turn with the user text, before the first LLM call.
- Small talk (the consolidator's `SMALL_TALK` test) → reuse the previous block; no embedding.
- Query vector = mean of the last ≤3 user-turn vectors of this session, re-normalized (so "what
  about her?" carries "my sister Mai").
- Candidates: current, non-sensitive facts and episodes of the **current speaker only**
  (same rule as the briefing, applied through `allowed_ids` before top-n), excluding the ids
  the session's briefing already shows.
- Keep top 3 with cosine ≥ `PUSH_MIN_SIM` (start 0.55, tuned on the eval) **[my call, tune]**;
  often zero lines.
- Rendered like recall lines, each ≤ 240 chars:
  ```
  [related memories — use only if it fits the conversation]
  - fact: Tuan codes with Claude [said, 2 days ago]
  - conversation Sat 15 Nov 20:10: Tuan fixed a websocket reconnect bug.
  ```
- Placement: prepended to the **sent copy** of the user message, not the system prompt (keeps
  DeepSeek's prefix cache) and not stored in history or passed to `on_turn` (the consolidator
  must not extract facts from it; history does not fill with stale blocks).
- Budget: counts as part of the current turn in the agent's `CONTEXT_MAX_TOKENS` trim; at most
  3 × 240 chars ≈ 180 tokens. The CLI startup check reserves 200 tokens for it when push is on.
- Pushed items do not bump `last_used`/`use_count` **[my call]**: pushing is not using, and
  bumping would let the push feed the briefing ranking.

### 7.5 Not built: query rewrite / decomposition

The agent already resolves pronouns and splits topics when it writes `recall` calls; a second LLM
would add 0.5–1 s. The push uses the mean of recent turns instead. Revisit only if the multihop
eval category fails.

## 8. Settings (`.env`)

```
EMBED=on                         # off = keyword-only memory search, no push
EMBED_MODEL=onnx-community/embeddinggemma-300m-ONNX   # HF repo or local folder
MEMORY_PUSH=on                   # step 2: add related memories to each turn (needs EMBED=on)
```

`PUSH_MIN_SIM`, the 256 dims, and the backfill batch stay constants until the eval shows they
need a knob **[my call]**.

## 9. Dependencies

New optional extra `embed = ["tokenizers>=0.20", "huggingface-hub>=0.25"]` **[owner approved
2026-11-17]**. `onnxruntime` and `numpy` are already core. No torch, no
sentence-transformers.

## 10. Build order

**Step 0 — spike (throwaway, `scripts/embed_spike.py`).** Download, load, embed the eval texts;
record: output names, load time, RAM, p50/p95 latency for a 1-sentence query and a 16-fact
batch, fp32 vs quantized agreement, VN↔EN similarity on 5 pairs. Result goes in §13 of this doc.
**Gate**: warm p95 (after the warm-up embed) of one push = query embed + allowed-ids SQL +
cosine + row fetch, measured with 5 000 fake vectors, ≤ 150 ms on the laptop; else step 2 ships
with `MEMORY_PUSH=off` by default **[my call]**.

**Step 1 — hybrid pull** (TDD, fake embedder in unit tests): migration V4 → `embed.py` →
`vectors.py` → write path + backfill → hybrid `search_*` → recall fusion → consolidator lookup →
`memory embed` → eval categories → docs. Ship.

**Step 2 — push**: `related.py` → `MemorySession.related` → `Agent(related=)` → CLI wiring and
budget reserve → eval proactive/noise → docs. Ship.

## 11. Evaluation

Extend `tests/replay/memory_eval.jsonl` (24 questions today) to ≥ 50:

| Category | New | What it proves |
|---|---|---|
| `paraphrase` | 8 | Recall by meaning, no shared words, VN and EN |
| `invalidate` | 4 | "switched to Claude" expires "uses Copilot" with no shared content words |
| `multihop` | +3 | Two chained recalls ("the person I went to Da Lat with") |
| `proactive` (step 2) | 8 | A plain remark; answer mentions the related memory (`expect`) |
| `noise` (step 2) | 8 | An unrelated remark; answer mentions none of the listed memories (new `expect_absent`) |

The report adds warm p50/p95 for query embedding and push time.

Without the embedding model, `memory eval` stops with an error naming the missing piece,
unless `EMBED=off` is set on purpose; then all categories still run and are reported (that run
is the keyword-only baseline that shows the gain). `--baseline` (long context, no memory) also
runs every category, gated on nothing **[my call]**.

## 12. Acceptance

- **Step 1**: existing categories still pass (24/24 expected; record today's score in §13 first); `paraphrase` ≥ 7/8; `invalidate`
  4/4; unit test: a guest's recall never returns an owner fact through the vector path;
  `forget` leaves no row in `fact_vecs`; with `EMBED=off` or no model, everything behaves as
  before; ruff, mypy, pytest green.
- **Step 2**: `proactive` ≥ 6/8; `noise` ≥ 7/8 (no unrelated mention); added warm p95 per
  turn (as measured in the step 0 gate) ≤ 150 ms, else default `MEMORY_PUSH=off` and say so in the docs; history and `on_turn` never
  contain the related block.

## 13. Spike results

Run 2026-11-17 on the owner's laptop (WSL2, 24 logical CPUs), `scripts/embed_spike.py`.

- **Download**: `onnx-community/embeddinggemma-300m-ONNX` downloaded with `HF_TOKEN`, not
  blocked. Files: `onnx/model.onnx` (+ `model.onnx_data`, 1.23 GB), `onnx/model_quantized.onnx`
  (+ 309 MB), also fp16/q4 variants. **The HF cache layout fails**: onnxruntime refuses external
  data reached through the cache's blob symlinks ("External data path escapes model directory").
  So the model is downloaded with `local_dir` into `models/embeddinggemma/` (real files).
  `models/` added to `.gitignore`.
- **Graph**: inputs `input_ids`, `attention_mask` (int64); outputs `last_hidden_state`,
  `sentence_embedding` (pooled; first 256 dims re-normalized are used). Prefixes as in §7.1.
- **Load**: 0.3 s; RSS ≈ 610 MB (fp32).
- **Latency, fp32, warm**: short eval sentences p50/p95 20/25 ms; a 12-word sentence p50/p95
  52/115 ms (high variance); 16-doc batch p50 166 ms. Cosine top-3 over 5 000 × 256 masked to
  2 500: p50/p95 4.5/10 ms.
- **Quantized (int8)**: min cosine vs fp32 0.9955 (quality fine) but **slower on this CPU**:
  query p50/p95 111/125 ms, batch 275 ms; RSS higher.
- **VN↔EN**: matching pairs 0.76–0.89, best non-matching 0.60 → cross-language retrieval works;
  0.6 is the noise floor to stay above.

**Decisions**
- Use `onnx/model.onnx` (fp32), output `sentence_embedding`. Quantized only if RAM matters more
  than speed later.
- Push warm p95 ≈ 115–125 ms (embed + search) ≤ 150 ms → **`MEMORY_PUSH=on` by default**, with
  little margin: re-check in the step 2 eval timing.
- `PUSH_MIN_SIM` starts at 0.55 as planned but the VN↔EN noise floor (0.60) suggests it will
  need to rise; tune in Task 9.

### Step 1 eval (2026-11-17, `deepseek-chat`, one run each, 39 questions)

| Category | Keyword only (`EMBED=off`) | Hybrid |
|---|---|---|
| Old categories (24) | 24/24 | 24/24 |
| `paraphrase` (8) | 8/8 | 8/8 |
| `invalidate` (4) | 4/4 | 4/4 |
| `multihop` (5) | 3/5 | 2/5 (one was a scoring miss: "học y" = medicine; expect list fixed) |
| query embed p50/p95 | — | 17/27 ms |

Acceptance met (old 24/24, paraphrase ≥ 7/8, invalidate 4/4). **But vectors show no gain on this
set**: the agent writes English keyword queries itself and stored facts carry the owner's name, so
FTS finds them. The paraphrase questions are not hard enough to separate the two; harder ones
(no name, no shared stem, VN query on EN fact about a third person) are needed before claiming a
recall gain. Both runs fail the same 2 multihop questions ("Where does my best friend work?",
"Who gave me my corgi?"), which points at extraction, not search. Step 2 (push) is where vectors
must earn their cost.

### Step 2 eval (2026-11-17, one run each, 55 questions)

| Category | Keyword only (`EMBED=off`, no push) | Hybrid + push |
|---|---|---|
| Old categories (24) | 24/24 | 24/24 |
| `paraphrase` / `invalidate` | 8/8 · 4/4 | 8/8 · 4/4 |
| `multihop` (5) | 4/5 | 5/5 |
| `proactive` (8) | 7/8 | 7/8 |
| `noise` (8) | 8/8 | 8/8 |
| overall | 53/55 | 54/55 |
| push p50/p95 | — | 22/25 ms |

Acceptance met: `proactive` ≥ 6/8, `noise` ≥ 7/8, push warm p95 25 ms ≤ 150 ms, so
`MEMORY_PUSH=on` stays the default and `PUSH_MIN_SIM` stays 0.55 (no noise failures to tune
against). **The set still doesn't separate push from no push**: the Nov 13 facts are recent, so
most sit in the 10-fact `[memory]` block already, and the agent calls `recall` on its own. The one
multihop difference is within one-run noise. To measure the push, the eval needs (a) more than
~40 facts spread over weeks so targets fall out of the briefing, and (b) several runs per arm.

## 14. Docs to update (same change as the code; CLAUDE.md workflow)

| Doc | Step | What changes |
|---|---|---|
| `docs/usage/memory.md` | 1, 2 | New section "How Reachy finds what's relevant" (briefing, related memories, recall: when each runs, with the example above); first-time setup (`uv sync --extra embed`, accept the Gemma license, `HF_TOKEN`, download size and where it goes); settings `EMBED`, `EMBED_MODEL`, `MEMORY_PUSH`; `memory embed`; Privacy: vectors stored next to facts, deleted with them, user-turn vectors RAM only; Limits: approximate similarity, push threshold, CPU cost; Errors table: model gated/not downloaded, extra not installed, slow embedding |
| `docs/usage/memory-testing.md` | 1, 2 | New eval categories and `expect_absent`; how to read the timing lines; sqlite helpers `vecs()` (count per model) next to `facts()`; how to tune `PUSH_MIN_SIM` |
| `docs/usage/agent.md` | 2 | Budget paragraph: the related block counts in the current turn |
| `.env.example` | 1, 2 | `EMBED`, `EMBED_MODEL`, `MEMORY_PUSH` with comments |
| `docs/superpowers/specs/2026-11-17-memory-design.md` | 1 | §4: approach C chosen by the owner, link here; §6: V4 tables; §7.11 "As built" note per step |
| `docs/architecture/05-agent-tools-memory.md` | 1 | Update note at §5: hybrid retrieval and push, link here |
| `README.md` | 1 | Mention `--extra embed` where extras are listed (memory.md is already linked) |

## 15. Left to the builder

Exact log fields; batch sizes beyond the defaults above; how `VectorIndex` stores ids
(parallel array vs dict); the `scripts/embed_spike.py` layout; wording of new eval questions
(inside the categories and counts above).
