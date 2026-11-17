# Smart Context Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Local embeddinggemma vectors make `recall` and the consolidator find memories by meaning (step 1), then push up to 3 related memories into each turn (step 2).

**Architecture:** An injected `Embedder` (ONNX, async via `to_thread`) writes 256-dim vectors to two SQLite tables beside facts/episodes; an in-RAM `VectorIndex` answers cosine top-n restricted to ids SQL says the speaker may see. `recall` and the consolidator fuse BM25, vector and recency ranks. Step 2 adds a per-turn related block on the sent copy of the user message.

**Tech Stack:** Python 3.12, numpy, onnxruntime, tokenizers, huggingface-hub (new `embed` extra), SQLite, pytest, mypy strict.

**Spec:** [docs/superpowers/specs/2026-11-17-smart-context-design.md](../specs/2026-11-17-smart-context-design.md) — read §5–§7 and §14 before starting.

## Global Constraints

- Model key `MODEL_ID = "embeddinggemma-300m@256"`; vectors 256 × float32, L2-normalized; inputs cut at 256 tokens.
- Prefixes: query `task: search result | query: {text}`, doc `title: none | text: {text}` (confirm in Task 1).
- Embedding never runs on the event loop: every call goes through `asyncio.to_thread`.
- Fail small: any embedder failure (missing extra, download, load, a single call) → keyword-only for that call, log, never fail the turn.
- Scope: vector candidates are always masked by `store.allowed_ids(...)` (same `_scope` + filters as keyword search) **before** top-n.
- Constants (no env knobs): `VEC_MIN_SIM = 0.3`, `VEC_TOP_N = 20`, `PUSH_MIN_SIM = 0.55`, `PUSH_TOP = 3`, `PUSH_LINE_CHARS = 240`, `BACKFILL_BATCH = 16`, `BACKFILL_PER_TICK = 200`, `PUSH_RESERVE_TOKENS = 200`.
- Env: `EMBED=on|off` (default on), `EMBED_MODEL` (default `onnx-community/embeddinggemma-300m-ONNX`), `MEMORY_PUSH=on|off` (default per Task 1 gate).
- Unit tests use `FakeEmbedder` only; anything loading the real model is `@pytest.mark.integration`.
- Done per task = `uv run ruff format . && uv run ruff check . && uv run mypy && uv run pytest` green. Not a git repo: skip commit steps.
- Each step ships its docs (spec §14) in the same task as the code they describe.

## Review Focus

1. The embedder raises mid-session (ONNX error, OOM) → that recall/consolidation/push falls back to keywords; the turn still answers. (Tests in Tasks 5, 6, 8.)
2. `EMBED_MODEL` or dims change → old vectors are ignored (keyed by model) and backfill re-embeds; no shape error. (Task 2 + Task 3 tests.)
3. Fresh DB / backfill not done / no vectors for a speaker → `top()` returns `[]` on an empty or fully masked matrix, no numpy error. (Task 3 test.)
4. A fact is updated → its old vector is gone from SQLite and the index, so search never matches the old wording. (Task 2 + Task 6 tests.)
5. A guest speaker → neither recall nor push ever returns an owner-only fact through the vector path. (Tasks 5 and 8 tests.)

---

## Step 0

### Task 1: Embedding spike (throwaway)

**Files:**
- Create: `scripts/embed_spike.py`
- Modify: spec §13 (results), spec §8 + this plan's Global Constraints (`MEMORY_PUSH` default)

- [ ] **Step 1:** `uv add --optional embed "tokenizers>=0.20" "huggingface-hub>=0.25"`; `uv sync --extra embed`.
- [ ] **Step 2:** Write the script: download `EMBED_MODEL` with `huggingface_hub.snapshot_download` (uses `HF_TOKEN`), list the repo's `.onnx` files, load fp32 and the quantized file with `onnxruntime.InferenceSession(providers=["CPUExecutionProvider"])`, print input/output names, run the eval texts (`say` lines of `tests/replay/memory_eval.jsonl`) through both with the doc prefix, take the pooled output, cut to 256, re-normalize.
- [ ] **Step 3:** Run `uv run python scripts/embed_spike.py`. Record in spec §13: repo files used, input/output names, load time, RSS, warm p50/p95 for one query and a 16-doc batch, fp32 vs quantized min cosine, cosine for 5 VN↔EN pairs, and the push timing below.
- [ ] **Step 4:** Push timing: 5 000 random normalized vectors in a numpy matrix, mask to 2 500 ids, top 3; time query embed + mask + matmul + top. Record warm p95.
- [ ] **Step 5:** Decide and write down in §13: the ONNX file (quantized if min cosine ≥ 0.98, else fp32) and output name for Task 4; `MEMORY_PUSH` default = `on` if push p95 ≤ 150 ms else `off`.
  If the download fails as gated: stop and ask the owner to accept the Gemma license on the model page.

## Step 1 — hybrid pull

### Task 2: Vector tables and store methods

**Files:**
- Modify: `src/reachy_demo/memory/store.py` (`MIGRATIONS`, `update_fact`, new methods)
- Test: `tests/unit/test_memory_store.py`

**Interfaces:**
- Produces (all on `MemoryStore`):
  - `set_vec(kind: Literal["fact","episode"], item_id: int, model: str, vec: np.ndarray) -> None` (upsert; stores `vec.astype("<f4").tobytes()`)
  - `load_vecs(kind, model: str) -> list[tuple[int, np.ndarray]]`
  - `missing_vecs(kind, model: str, limit: int) -> list[tuple[int, str]]` → `(id, text)`; facts: current (`expired IS NULL`) only; episodes: `summary`
  - `allowed_ids(kind, speaker: Speaker, *, person_id: int | None = None, subject_text: str | None = None, start: float | None = None, end: float | None = None, current_only: bool = True, include_sensitive: bool = True) -> set[int]` — facts: same `_scope`, person/subject and time-window clauses as `search_facts` (current only when no window); episodes: same window clause as `search_episodes`, `importance > 0`, and for a guest only episodes containing `speaker.person_id`
  - `facts_by_ids(ids: list[int]) -> list[Fact]`, `episodes_by_ids(ids: list[int]) -> list[Episode]` (order of `ids`, missing skipped)
- `V4_VECTORS` = the two `CREATE TABLE` statements of spec §6, appended to `MIGRATIONS`.
- Deviation from spec §5.1: instead of a `vec_ids` parameter on `search_*`, vector hits are fetched with `*_by_ids`, because fusion needs the BM25 and vector lists separately. Same scope guarantee (ids come from `allowed_ids`).

- [ ] **Step 1: Write failing tests**
  - `test_vec_roundtrip_by_model`: `set_vec("fact", f, "m1", v)`, `set_vec("fact", f, "m2", w)` → `load_vecs("fact","m1") == [(f, v)]` (np.allclose).
  - `test_delete_facts_cascades_vectors` and `test_forget_person_cascades_vectors`: after delete, `load_vecs` is empty for that id.
  - `test_update_fact_drops_its_vector`: `update_fact(f, "new")` → `load_vecs("fact", m)` has no `f`.
  - `test_missing_vecs_skips_embedded_and_expired`.
  - `test_allowed_ids_guest_sees_own_and_shared_only`: owner fact, guest fact, shared owner fact → guest gets {guest, shared}; owner gets all three.
  - `test_allowed_ids_time_window_includes_expired`: an expired fact is excluded with no window, included when its validity overlaps the window.
- [ ] **Step 2:** `uv run pytest tests/unit/test_memory_store.py -v` → new tests FAIL (`AttributeError`).
- [ ] **Step 3:** Implement. Factor the WHERE-clause building of `search_facts`/`search_episodes` into one private helper each, used by both search and `allowed_ids`, so the filters cannot drift. `update_fact` deletes from `fact_vecs` in the same statement batch.
- [ ] **Step 4:** Same command → PASS; full check green.

### Task 3: VectorIndex and FakeEmbedder

**Files:**
- Create: `src/reachy_demo/memory/vectors.py`, `tests/unit/fake_embedder.py`
- Test: `tests/unit/test_vectors.py`

**Interfaces:**
- Consumes: `store.load_vecs` (Task 2).
- Produces:
  - `Embedder` protocol in `vectors.py` (not `embed.py` as in spec §5.1, so code that only searches never imports the ONNX module): `model: str`; `embed(texts: list[str], kind: Literal["query","doc"]) -> np.ndarray` (shape `(n, 256)`, float32, rows L2-normalized).
  - `VectorIndex(model: str)`; `@classmethod load(store, model) -> VectorIndex`; `put(kind, item_id, vec) -> None`; `drop(kind, item_id) -> None`; `top(query: np.ndarray, kind, n: int, allowed: set[int], min_sim: float = 0.0) -> list[tuple[int, float]]` (highest cosine first).
  - `async embed_one(embedder: Embedder | None, text: str, kind) -> np.ndarray | None` — `to_thread`, returns `None` on `embedder is None` or any exception (logged once per call with `log.warning`).
  - `tests/unit/fake_embedder.py`: `FakeEmbedder(synonyms: dict[str, str] | None = None, fail: bool = False)`, `model = "fake@256"`: lowercase words → synonym map → crc32 → bucket in 256 dims, summed, normalized; raises `RuntimeError` when `fail`.

- [ ] **Step 1: Write failing tests**
  - `test_top_ranks_by_cosine_within_allowed`: three vectors, `allowed` excludes the best → returns the next two in order.
  - `test_top_empty_index_and_empty_allowed_return_empty` (Review Focus 3).
  - `test_put_overwrites_and_drop_removes`.
  - `test_load_ignores_other_models` (Review Focus 2).
  - `test_fake_embedder_synonyms_make_texts_similar`: with `{"copilot": "aitool", "claude": "aitool"}`, cosine("uses Copilot", "uses Claude") > 0.9; unrelated text < 0.3.
  - `test_embed_one_returns_none_on_failure` (FakeEmbedder `fail=True`) and on `None`.
- [ ] **Step 2:** `uv run pytest tests/unit/test_vectors.py -v` → FAIL (import error).
- [ ] **Step 3:** Implement. Store per kind an id list and a stacked matrix rebuilt lazily after `put`/`drop`.
- [ ] **Step 4:** PASS; full check green.

### Task 4: GemmaEmbedder and loader

**Files:**
- Create: `src/reachy_demo/memory/embed.py`
- Modify: `pyproject.toml` (the `embed` extra from Task 1 stays; add `tokenizers`, `huggingface_hub` to the mypy untyped allow-list if needed)
- Test: `tests/unit/test_embed.py`, `tests/integration/test_embed_model.py`

**Interfaces:**
- Consumes: `Embedder` protocol (Task 3); ONNX file and output name from spec §13 (Task 1).
- Produces: `MODEL_ID = "embeddinggemma-300m@256"`; `GemmaEmbedder(path: Path)` implementing `Embedder` (prefixes per Global Constraints, truncation 256, slice to 256 dims, re-normalize; one warm-up embed in `__init__`); `load_embedder(env: Mapping[str, str]) -> tuple[Embedder | None, str]` — second item is a one-line status for the CLI: `"hybrid (embeddinggemma-300m)"`, `"keywords only (EMBED=off)"`, `"keywords only (install: uv sync --extra embed)"`, `"keywords only (model download failed: <reason>)"`.

- [ ] **Step 1: Write failing tests**
  - `test_load_embedder_off`: `{"EMBED": "off"}` → `(None, "keywords only (EMBED=off)")`.
  - `test_load_embedder_missing_extra`: monkeypatch import of `tokenizers` to raise `ImportError` → `None`, status contains `uv sync --extra embed`.
  - `test_load_embedder_download_error`: monkeypatch `snapshot_download` to raise → `None`, status starts with `keywords only (model download failed`.
  - Integration `test_gemma_vn_en_pair_is_close`: "Tôi có con chó tên là Bin" vs "I have a dog named Bin" cosine > the same vs "I live in Saigon".
- [ ] **Step 2:** `uv run pytest tests/unit/test_embed.py -v` → FAIL.
- [ ] **Step 3:** Implement. Imports of `tokenizers`/`huggingface_hub`/`onnxruntime` happen inside `load_embedder`/`GemmaEmbedder` so the module imports without the extra.
- [ ] **Step 4:** Unit PASS; `uv run pytest -m integration tests/integration/test_embed_model.py` PASS on the laptop; full check green.

### Task 5: Hybrid recall

**Files:**
- Modify: `src/reachy_demo/memory/recall.py`, `src/reachy_demo/agent/tools/memory.py` (`MemoryTools.__init__`, `_recall` → async, `SPECS` recall description)
- Test: `tests/unit/test_recall.py`, `tests/unit/test_memory_tools.py`

**Interfaces:**
- Consumes: Tasks 2–3.
- Produces:
  - `recall(store, speaker, query, *, now, person=None, kind="any", when=None, limit=5, index: VectorIndex | None = None, query_vec: np.ndarray | None = None) -> dict` — unchanged output shape.
  - `hybrid_facts(store, speaker, query: str, query_vec: np.ndarray | None, index: VectorIndex | None, *, person_id=None, subject_text=None, start=None, end=None, limit: int) -> list[Fact]` — used by recall **and** the consolidator (Task 6).
  - `MemoryTools(store, consolidator, *, speaker, now=..., embedder: Embedder | None = None, index: VectorIndex | None = None)`.
- Fusion: three rank lists (BM25 order from `search_*`, vector order from `index.top(..., n=VEC_TOP_N, allowed=store.allowed_ids(...), min_sim=VEC_MIN_SIM)` fetched via `*_by_ids`, recency over the union); score = Σ `1/(RRF_K + rank)` over the lists an item appears in. Keep `_dedupe`.
- Recall description gains: `"Write a standalone query (no pronouns). For several topics, call recall once per topic."`

- [ ] **Step 1: Write failing tests** (FakeEmbedder with `{"copilot":"aitool","claude":"aitool","tool":"aitool"}`)
  - `test_recall_finds_paraphrase_without_shared_words`: fact "Tuan codes with Claude"; `recall("which AI tool do I use", query_vec=…)` → found, result mentions Claude. Same call with `index=None` → `found: False` (proves the vector path did it).
  - `test_recall_vector_path_respects_guest_scope` (Review Focus 5): owner-only fact matched by vector; guest speaker → not in results.
  - `test_recall_time_window_returns_expired_by_vector`.
  - `test_recall_tool_falls_back_when_embedder_fails` (Review Focus 1): `MemoryTools(..., embedder=FakeEmbedder(fail=True), index=…)` → keyword result returned, no `error`.
  - `test_recall_spec_asks_for_standalone_queries`: the description contains `"standalone query"`.
- [ ] **Step 2:** `uv run pytest tests/unit/test_recall.py tests/unit/test_memory_tools.py -v` → new tests FAIL.
- [ ] **Step 3:** Implement. `_recall` awaits `embed_one(self._embedder, query, "query")`, then calls sync `recall`.
- [ ] **Step 4:** PASS; full check green.

### Task 6: Write path, backfill, wiring, `memory embed`

**Files:**
- Modify: `src/reachy_demo/memory/consolidate.py` (`Consolidator.__init__`, `_apply`), `src/reachy_demo/memory/episodes.py` (`summarize_session`), `src/reachy_demo/memory/session.py` (`__init__`, `run`, `_process`), `src/reachy_demo/agent/cli.py` (`memory_from_env`), `src/reachy_demo/agent/memory_eval.py` (`add_parser`, `run`), `.env.example`
- Test: `tests/unit/test_consolidate.py`, `tests/unit/test_memory_session.py`, `tests/unit/test_agent_cli.py`, `tests/unit/test_memory_eval.py`

**Interfaces:**
- Consumes: Tasks 2–5 (`hybrid_facts`, `embed_one`, `VectorIndex`, `load_embedder`, `MODEL_ID`).
- Produces:
  - `Consolidator(store, chat, clock, *, embedder: Embedder | None = None, index: VectorIndex | None = None, ...)`. In `_apply`: embed the candidate once as `"doc"`; similar = `hybrid_facts(..., query_vec=vec, limit=5)` (current facts only); after ADD/UPDATE/INVALIDATE store the vector for the written fact (`set_vec` + `index.put`); after UPDATE re-embed the merged text.
  - `summarize_session(..., embedder=None, index=None)`: embed the summary after `_insert`.
  - `MemorySession(..., embedder=None, index=None)`; `async backfill(limit: int) -> int` (batches of `BACKFILL_BATCH` through `to_thread`, returns count); `run()` calls `backfill(BACKFILL_PER_TICK)` once at start and on each idle timeout while the last call returned > 0.
  - `memory_from_env` builds `load_embedder(os.environ)` + `VectorIndex.load(store, MODEL_ID)`, passes both on, prints `  (memory search: <status>)`.
  - `reachy-demo memory embed [--db PATH]`: loops `backfill` until 0, prints `memory embed: N facts, M episodes (embeddinggemma-300m@256)`; exit 2 with the loader status if no embedder.

- [ ] **Step 1: Write failing tests**
  - `test_invalidate_found_by_meaning` (spec §11 `invalidate`): store "Tuan uses Copilot for coding"; candidate "Tuan switched to Claude"; scripted DECIDE replies INVALIDATE for the id it was shown → old fact expired. Assert the DECIDE prompt listed the Copilot fact (it shares no content word).
  - `test_update_reembeds_merged_text` (Review Focus 4): after UPDATE, `load_vecs` holds the vector of the merged text, not the old one.
  - `test_consolidate_without_embedder_unchanged` and with `FakeEmbedder(fail=True)` → fact still written, no vector (Review Focus 1).
  - `test_episode_summary_gets_vector`.
  - `test_backfill_embeds_missing_then_returns_zero`.
  - `test_memory_embed_command_counts` and `test_memory_embed_without_embedder_returns_2` (monkeypatch `load_embedder`).
  - `test_memory_from_env_prints_search_status`.
- [ ] **Step 2:** Run the four test files → new tests FAIL.
- [ ] **Step 3:** Implement. Add to `.env.example` after `MEMORY_MAX_TOKENS`:
  ```
  EMBED=on                         # off = keyword-only memory search (and no push)
  EMBED_MODEL=onnx-community/embeddinggemma-300m-ONNX   # HF repo or local folder; needs `uv sync --extra embed`
  ```
- [ ] **Step 4:** PASS; full check green.

### Task 7: Eval categories (step 1) and step 1 docs

**Files:**
- Modify: `tests/replay/memory_eval.jsonl`, `src/reachy_demo/agent/memory_eval.py` (`evaluate`, `Report.lines`, `run`), `docs/usage/memory.md`, `docs/usage/memory-testing.md`, `docs/superpowers/specs/2026-11-17-memory-design.md` (§4, §6, §7.11), `docs/architecture/05-agent-tools-memory.md` (§5 note), `README.md` (extras list)
- Test: `tests/unit/test_memory_eval.py`

**Interfaces:**
- Consumes: Task 6 wiring (`load_embedder`, `MemoryTools(embedder=, index=)`).
- Produces: `Report.embed_ms: list[float]` printed as `query embed p50/p95: X/Y ms`; eval refuses to run (exit 2, loader status printed) when `load_embedder` returns `None` unless the env has `EMBED=off`. `--baseline` (no memory) never loads the embedder and runs every category.

- [ ] **Step 1: Write failing tests**
  - `test_eval_needs_embedder_unless_embed_off` (both branches, `load_embedder` monkeypatched).
  - `test_baseline_runs_without_embedder`.
  - `test_report_prints_embed_timing`.
- [ ] **Step 2:** `uv run pytest tests/unit/test_memory_eval.py -v` → FAIL.
- [ ] **Step 3:** Implement; then add to the jsonl: 8 `paraphrase` (4 VN, 4 EN, no content word shared with the stored fact), 4 `invalidate` (a "switched/moved/changed" statement, then a question whose only right answer is the new value), 3 `multihop`.
- [ ] **Step 4:** PASS; full check green. Run `uv run reachy-demo memory eval` with `EMBED=off`, then with the model; record both scores and timing in spec §13 and memory-design §7.11. Acceptance (spec §12 step 1): old categories 24/24, `paraphrase` ≥ 7/8, `invalidate` 4/4. If below, tune `VEC_MIN_SIM` once and record it; if still below, stop and report.
- [ ] **Step 5: Docs** — each item of spec §14 marked step 1:
  - `memory.md`: new section "How Reachy finds what's relevant" (briefing / recall, when each runs, the Copilot→Claude example); first-time setup (`uv sync --extra embed`, accept the Gemma license, `HF_TOKEN`, download size from §13 and where it goes); settings `EMBED`, `EMBED_MODEL`; `reachy-demo memory embed`; Privacy (vectors next to facts, deleted with them); Limits (approximate similarity, CPU cost); Errors table rows for each `load_embedder` status.
  - `memory-testing.md`: new categories, the timing line, `vecs()` helper `sqlite3 -header -column "$MEMORY_DB" "select 'fact' kind, model, count(*) n from fact_vecs group by model union all select 'episode', model, count(*) from episode_vecs group by model"`.
  - memory-design §4: approach C chosen by the owner, link to the smart-context spec; §6: V4 tables; §7.11: as-built note.
  - architecture 05 §5: update note linking the smart-context spec.
  - README: `--extra embed` beside the other extras.

## Step 2 — push

### Task 8: Related block in the memory session

**Files:**
- Create: `src/reachy_demo/memory/related.py`
- Modify: `src/reachy_demo/memory/briefing.py`, `src/reachy_demo/memory/session.py`
- Test: `tests/unit/test_related.py`, `tests/unit/test_memory_session.py`, `tests/unit/test_briefing.py`

**Interfaces:**
- Consumes: Tasks 2–3, `worth_keeping` (consolidate.py), `_line`-style rendering from recall.py.
- Produces:
  - `briefing.py`: `@dataclass(frozen=True) Briefing(text: str, fact_ids: frozenset[int], episode_ids: frozenset[int])`; `build_briefing_parts(...same args...) -> Briefing`; `build_briefing(...)` stays and returns `.text`.
  - `related.py`: `related_block(store, index, speaker, query_vec: np.ndarray, *, exclude_facts: frozenset[int], exclude_episodes: frozenset[int], now: datetime) -> str` — candidates via `allowed_ids(..., current_only=True, include_sensitive=False)` minus excludes; facts and episodes ranked together by cosine; top `PUSH_TOP` ≥ `PUSH_MIN_SIM`; lines cut to `PUSH_LINE_CHARS`; header `[related memories — use only if it fits the conversation]`; `""` when none. Does **not** call `touch_facts`.
  - `MemorySession.related(text: str) -> Awaitable[str]` (`async def`): not `worth_keeping(text)` → last block; else `embed_one(..., "query")`, append to a deque(maxlen=3) of turn vectors, query = normalized mean; `None` vector → last block. Deque and last block reset when a new session starts.

- [ ] **Step 1: Write failing tests**
  - `test_related_picks_top_above_threshold_only`; `test_related_empty_when_nothing_close`.
  - `test_related_skips_briefing_ids_and_sensitive`.
  - `test_related_guest_never_sees_owner_facts` (Review Focus 5).
  - `test_related_does_not_touch_facts`: `use_count` unchanged.
  - `test_session_related_small_talk_reuses_last_block`: "ok" after a real turn → same block, embedder called once (count calls on FakeEmbedder).
  - `test_session_related_mean_carries_previous_turn`: "my sister Mai visited", then "what about her" → the Mai fact is in the second block.
  - `test_session_related_embedder_failure_returns_last_block` (Review Focus 1).
  - `test_briefing_parts_ids_match_text`.
- [ ] **Step 2:** Run the three test files → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** PASS; full check green.

### Task 9: Agent hook, CLI, eval, step 2 docs

**Files:**
- Modify: `src/reachy_demo/agent/agent.py` (`Agent.__init__`, `answer`), `src/reachy_demo/agent/cli.py` (`run`, `make_agent`), `src/reachy_demo/agent/memory_eval.py` (`score`, `evaluate`, `Report`), `tests/replay/memory_eval.jsonl`, `.env.example`, `docs/usage/memory.md`, `docs/usage/memory-testing.md`, `docs/usage/agent.md`, spec §13
- Test: `tests/unit/test_agent.py`, `tests/unit/test_agent_cli.py`, `tests/unit/test_memory_eval.py`

**Interfaces:**
- Consumes: `MemorySession.related` (Task 8).
- Produces: `Agent(..., related: Callable[[str], Awaitable[str]] | None = None)`. Awaited once at the start of `answer()`; each round sends `[head, *history, sent_first, *turn[1:]]` where `sent_first = {"role": "user", "content": f"{block}\n\n{text}"}` when the block is non-empty; `turn` (stored, passed to `on_turn`) keeps the plain text. The block counts in the budget's fixed part. A hook exception → log, no block.
  - CLI: `MEMORY_PUSH` (default from Task 1); push needs memory and an embedder, else printed `  (memory push: off: <reason>)`; with push on, the startup check fails when `budget - PUSH_RESERVE_TOKENS <= memory_max`.
  - Eval: `score` supports `"expect_absent": [...]` (pass when none appear); `Report.push_ms` printed as `push p50/p95: X/Y ms`.

- [ ] **Step 1: Write failing tests**
  - `test_related_block_sent_but_not_stored`: sent user content starts with the block; `agent.history` and the `on_turn` turn hold the plain text.
  - `test_related_awaited_once_per_turn_across_tool_rounds`.
  - `test_related_hook_error_still_answers`.
  - `test_related_block_counts_in_budget`: a budget that fits the turn without the block but not with it drops the oldest history turn.
  - `test_run_push_reserve_in_startup_check`.
  - `test_push_off_without_embedder_prints_reason`.
  - `test_score_expect_absent`.
- [ ] **Step 2:** Run the three test files → FAIL.
- [ ] **Step 3:** Implement. `.env.example` after `EMBED_MODEL`:
  ```
  MEMORY_PUSH=on                   # add up to 3 related memories to each turn (needs EMBED=on)
  ```
  (value = Task 1 decision). Add 8 `proactive` and 8 `noise` steps to the jsonl.
- [ ] **Step 4:** PASS; full check green. Run `uv run reachy-demo memory eval`; record scores and push timing in spec §13. Acceptance (spec §12 step 2): `proactive` ≥ 6/8, `noise` ≥ 7/8, push warm p95 ≤ 150 ms. If `noise` fails, raise `PUSH_MIN_SIM` by 0.05 steps (max 3 tries) and record; if timing fails, set the default `MEMORY_PUSH=off` everywhere and say so in the docs.
- [ ] **Step 5: Docs** — spec §14 items marked step 2:
  - `memory.md`: "How Reachy finds what's relevant" gains the related-memories part (when it runs, the example block, that it's never stored, small talk reuse); `MEMORY_PUSH`; Privacy (turn vectors RAM only); Limits (threshold, may stay quiet); Errors (`memory push: off: …`).
  - `memory-testing.md`: `proactive`/`noise`, `expect_absent`, push timing line, how to tune `PUSH_MIN_SIM`.
  - `agent.md`: budget paragraph — the related block counts in the current turn; the 200-token reserve and its startup error.
