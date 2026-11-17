# Memory — design for Phases 1–4

Date: 2026-11-17 · Feature F8 (milestone M7), extended · Research: [docs/research/2026-11-17-robot-memory-research.md](../../research/2026-11-17-robot-memory-research.md) (cited as "research §n")

Status: **Phase 1 built 2026-11-17** (branch `feat/memory-phase1`; as-built notes in §7.11). Phases 2–4 are design only. Calls marked **[my call]** are proposals the owner has not confirmed. **[verify]** = check before relying on it.

---

## 1. Intent

Reachy should remember people, what they said and what happened, across restarts, and use it naturally in conversation: greet the owner by name, follow up on things ("how did the exam go?"), answer "what did I tell you about…", "where did I leave…", and say **"I don't remember"** instead of inventing. Later it should know where things are, learn routines and act on them.

What success looks like: the scripted memory test set (§8) passes its targets, and memory adds ≤ 150 ms p95 to a turn. What failure looks like even if it "works": the robot states a false memory, tells a guest the owner's facts, or a memory write slows or breaks a turn on stage.

## 2. Goals, non-goals, anti-goals

**Goals**
- The six memory types from research §4 (working, person, semantic, episodic, spatial, procedural), built in four shippable phases.
- One local store (`data/memory.db`), no server.
- Memory never blocks a turn and never breaks one: a failed write loses a memory, not the answer.
- Transparency: "what do you know about me?", "forget that", "forget everything".

**Non-goals (now)**: 3D maps, a graph database, storing raw audio/video, fine-tuning, multi-robot shared memory (research §7 "not recommended").

**Anti-goals**: invented memories; guest access to owner data; silent overwrite of facts; anything sensitive (health, finance, passwords, addresses) put in the prompt automatically.

## 3. Constraints (from the codebase, 2026-11-17)

- **Nothing exists yet.** `src/reachy_demo/memory/` is an empty package. The `facts/sightings/people` tables in `docs/architecture/05-agent-tools-memory.md` §5 are a design only. No data to migrate.
- **No identity.** There is no owner gate, `enroll` command or face ID (that is M8). `sherpa-onnx` is only an optional extra.
- **Agent** (`agent/agent.py`): fixed `system_prompt` string, last 10 turns in RAM, `MAX_ROUNDS = 4` LLM calls per turn, tools via `Toolbox` of `ToolSet`s (`specs / needs_response / call`). Tools are only registered in `--robot` mode today.
- **LLM** is DeepSeek (`agent/llm.py`: `chat_stream`, `chat_events`). It has **no embeddings API**.
- **CPU-only WSL2 laptop.** `onnxruntime` is already a dependency.
- Python's SQLite is 3.53.1 with **FTS5** available; `unicode61 remove_diacritics 2` matches `tuan` to `Tuấn` (checked 2026-11-17).
- `robot/motion.py` keeps `Motion.pose` (last head pose sent): the source of head yaw/pitch for spatial memory.
- Rules from `src/CLAUDE.md`: permissions in code, fail small, inject dependencies, mypy strict, a test for every non-trivial branch, ask before adding dependencies, usage doc per feature.

## 4. Approaches considered

| | Approach | Trade-off |
|---|---|---|
| A | **Long context only**: put the whole recent history in the prompt, no memory store. | Zero code; a cited study shows long context can beat fact memory on recall. But cost and latency grow every turn, nothing survives a context limit, no "forget", no spatial or procedural memory. |
| B | **SQLite + FTS5 + LLM consolidator, vectors later** **[my call]** | No new dependency; keyword search is exact and explainable. Weak on paraphrase ("my pet" vs "dog"); handled by storing facts in normalized English and letting the agent write the query. Vectors (`sqlite-vec` + ONNX embeddings) added only if the eval misses. |
| C | SQLite + FTS5 + `sqlite-vec` + local embeddings from day one (research §6.2–6.3 as written). | Best paraphrase and cross-language recall; two new dependencies, a model file, re-embedding on model change, ~10–20 ms per sentence **[verify]**. |
| D | Mem0 / Zep / Graphiti as a library or service. | Proven, but brings a vector DB or graph server, heavy deps, and cloud-shaped defaults. Against the lean single-process rule. |

> **Update 2026-11-17:** the owner chose **C** (local embeddings) for proactive recall and paraphrase, not because the eval failed. It uses `embeddinggemma-300m` through ONNX Runtime and two plain SQLite tables with numpy cosine search instead of `sqlite-vec`. Design: [2026-11-17-smart-context-design.md](2026-11-17-smart-context-design.md).

**Pick: B**, with C as a planned upgrade gated by the eval (Phase 1, step 1.8). A is kept as the **baseline** the eval must beat. D is ruled out by the dependency and single-process rules.

**Identity in Phase 1 [my call: single user].** Every turn is attributed to the owner (`speaker = Speaker(person_id=OWNER, role="owner")`). All memory code takes a `Speaker` so M8 only has to fill it in. The alternative (build voice enrolment + speaker ID inside Phase 1) roughly doubles Phase 1 and pulls M8 forward.

---

## 5. Architecture (all phases)

### 5.1 Components

```
src/reachy_demo/memory/
  store.py        MemoryStore: SQLite connection, schema + migrations, all SQL. No LLM, no network.
  timeparse.py    "yesterday", "hôm qua", "last Tuesday", "sáng nay" → (start, end). Pure.
  briefing.py     build_briefing(store, speaker, now) → str (~600 tokens), once per session. Pure over the store.
  consolidate.py  Consolidator: extract → retrieve similar → decide → apply. Uses an injected chat fn.
  session.py      MemorySession: session boundaries, cached briefing, turn log, bounded job queue +
                  background worker, crash recovery, preload of last turns, close/backup.
  episodes.py     summarize_session (P1), task episodes (P2).
  prompts.py      Extract / decide / summarize prompts + ask_json (JSON-only call, one retry).
  spatial.py      Bearing math, landmarks, freshness policy (P2).
  jobs.py         Cold-path jobs: compress, dedupe, retention, reflections, routines (P3/P4).
  scheduler.py    Reminders and proactive triggers (P4).
src/reachy_demo/agent/tools/memory.py   MemoryTools (ToolSet): remember, recall, forget (+ later tools)
src/reachy_demo/agent/memory_eval.py    `reachy-demo memory eval` (P1). It wires the real agent, so it lives in agent/.
                                        `memory list|show|forget|export` come in P3.
```

Dependency direction: `agent` → `memory`; `memory` never imports `agent`, `robot`, `voice` or `perception`. Chat functions, clock and the robot pose are injected (constructor arguments), so every component tests with fakes.

### 5.2 Three paths (research §6.1)

```
 HOT (inside the turn, no extra LLM call, ≤ 150 ms)
   turn text ─► Speaker ─► briefing (built at session start) ─► system prompt + history ─► DeepSeek
                                                    │ model may call recall/remember/forget
                                                    ▼
                                               answer ─► TTS
 WARM (after the turn, background asyncio task)
   (user text, answer, tool calls) ─► event_log ─► importance gate ─► extract (LLM)
        ─► retrieve 5 similar (FTS5) ─► decide ADD/UPDATE/INVALIDATE/NOOP (LLM) ─► one transaction
 COLD (session end, idle, startup)
   episode summary + follow-ups (P1) · compress/dedupe/retention/reflection (P3) · routines (P4)
```

Rules:
1. **The hot path never calls an LLM for memory.** The briefing is SQL only. `recall` is a normal tool round the model chooses.
2. **The warm path is fire-and-forget.** Bounded queue (64 items); when full, drop the oldest and log `memory_dropped`. LLM errors log and drop; they never reach the user.
3. **The cold path is idempotent.** Every job can re-run from `event_log` without duplicating rows (it records what it has processed).
4. **Permissions in code**, never only the prompt: writes and deletes need `speaker.role == "owner"`; reads are filtered to the speaker's scope.

### 5.3 Agent integration (the only change to `agent/`)

- `Agent.__init__` gains `context: Callable[[], str] | None = None`. The system message is `system_prompt + "\n\n" + context()`; the CLI's `context` returns the briefing cached for the current session (rebuilt by `MemorySession` at session start, §7.3), so calling it each round costs nothing.
- `Agent` gains `preload(messages)` to seed its history with the last raw turns at startup; the CLI passes `MemorySession.preload()` (§7.3 block 5).
- `Agent` gains `on_turn: Callable[[list[dict]], None] | None = None`, called with the finished turn (user, assistant, tool messages) where `self._turns` is updated today. The CLI passes `MemorySession.submit`.
- `MemoryTools` is registered in **both** robot and no-robot modes (it needs no robot). In no-robot mode the CLI builds a `Toolbox(MemoryTools(...))` and passes tool specs to the LLM, as robot mode does now.
- New env vars (in `.env.example` and the usage doc): `MEMORY_DB` (default `data/memory.db`), `MEMORY=off` to disable memory entirely (demo escape hatch), `MEMORY_MODEL` (default `deepseek-chat`, used by warm/cold LLM calls; non-reasoning for speed, same reason as the stop check).

### 5.4 Speaker and scope

```python
@dataclass(frozen=True)
class Speaker:
    person_id: int  # people.id; OWNER_ID = 1 is created on first start
    role: Literal["owner", "guest"]
```
P1: always the owner. M8: the gate sets it per turn; an unknown voice gets a session-only guest row. Read scope: owner sees everything; a guest sees only facts with `subject_person_id` = the guest and `shared = 1` facts (e.g. the robot's own persona facts). Writes: a guest's own statements are consolidated into facts with `subject_person_id` = the guest (no name, follow-ups or description); nothing a guest says is ever written under the owner. *(Changed 2026-11-17 at the user's request: previously guests wrote nothing.)*

---

## 6. Data model (full schema; each phase creates only its tables)

> **Schema v4 (2026-11-17):** `fact_vecs(fact_id → facts ON DELETE CASCADE, model, vec)` and `episode_vecs(episode_id → episodes ON DELETE CASCADE, model, vec)`, 256 × float32 per row. See the smart-context design §6.

`store.py` holds a list of migrations; `meta(key, value)` stores `schema_version`. Each migration runs in one transaction; the DB file is copied to `memory.db.bak-<version>` before a migration runs. WAL mode, `foreign_keys=ON`, all times as UTC epoch seconds (REAL).

```sql
-- P1 ------------------------------------------------------------------------
meta(key TEXT PRIMARY KEY, value TEXT)
people(id INTEGER PRIMARY KEY, role TEXT CHECK(role IN ('owner','guest')) NOT NULL,
       name TEXT, aliases_json TEXT DEFAULT '[]',                          -- current + earlier names
       consent_recognize INTEGER DEFAULT 0, created REAL, last_seen REAL,
       retention_until REAL)                                               -- NULL = keep
facts(id INTEGER PRIMARY KEY, subject_person_id INTEGER REFERENCES people(id),
      subject_text TEXT NOT NULL,            -- "Tuan", "Tuan's sister Mai", "Reachy"
      text TEXT NOT NULL,                    -- one atomic sentence, English, third person
      text_original TEXT NOT NULL DEFAULT '', -- as said, any language (for read-back)
      category TEXT CHECK(category IN ('bio','preference','relation','plan','setting','other')),
      source TEXT CHECK(source IN ('said','seen','inferred')) NOT NULL,
      confidence REAL DEFAULT 1.0, importance REAL DEFAULT 0.5,   -- 0..1
      sensitive INTEGER DEFAULT 0, shared INTEGER DEFAULT 0,
      valid_from REAL, valid_to REAL,        -- world time; NULL = unknown / still true
      created REAL NOT NULL, expired REAL,   -- system time; expired = replaced or invalidated
      replaced_by INTEGER REFERENCES facts(id),
      last_used REAL, use_count INTEGER DEFAULT 0,
      evidence_json TEXT DEFAULT '[]')       -- event_log ids
CREATE VIRTUAL TABLE facts_fts USING fts5(text, subject_text, text_original,
      content='facts', content_rowid='id', tokenize='unicode61 remove_diacritics 2')
      -- + the three standard triggers keeping facts_fts in sync
episodes(id INTEGER PRIMARY KEY, session_id TEXT,  -- UNIQUE when set: one episode per session
         kind TEXT CHECK(kind IN ('conversation','task','visual','moment')) NOT NULL,
         start REAL NOT NULL, end REAL NOT NULL, person_ids_json TEXT DEFAULT '[]',
         summary TEXT NOT NULL, topics_json TEXT DEFAULT '[]', importance REAL DEFAULT 0.5,
         outcome TEXT CHECK(outcome IN ('success','failure','partial')),   -- task only
         details_json TEXT,                                                -- task: goal, steps, reason
         compressed_level INTEGER DEFAULT 0, source_event_max INTEGER)    -- last event_log id covered
CREATE VIRTUAL TABLE episodes_fts USING fts5(summary, topics_json, content='episodes', ...)
follow_ups(id INTEGER PRIMARY KEY, person_id INTEGER REFERENCES people(id), text TEXT NOT NULL,
           due REAL, created REAL NOT NULL, episode_id INTEGER REFERENCES episodes(id), done REAL)
event_log(id INTEGER PRIMARY KEY, t REAL NOT NULL, kind TEXT NOT NULL,   -- 'turn' | 'tool' | 'sighting' | 'system'
          person_id INTEGER, session_id TEXT, payload_json TEXT NOT NULL, consolidated INTEGER DEFAULT 0)
-- indexes: facts(subject_person_id, expired), facts(expired, importance), episodes(start),
--          follow_ups(person_id, done), event_log(t), event_log(consolidated)

-- P2 ------------------------------------------------------------------------
landmarks(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, yaw_deg REAL, pitch_deg REAL,
          radius_deg REAL DEFAULT 12, created REAL)
objects(id INTEGER PRIMARY KEY, label TEXT NOT NULL, name TEXT,          -- "mug" / "Tuan's blue mug"
        owner_person_id INTEGER REFERENCES people(id), description TEXT,
        ref_thumb BLOB, created REAL)
sightings(id INTEGER PRIMARY KEY, object_id INTEGER REFERENCES objects(id), label TEXT NOT NULL,
          az_deg REAL, el_deg REAL, box_norm_json TEXT, landmark_id INTEGER REFERENCES landmarks(id),
          where_text TEXT, thumb BLOB, confidence REAL, seen REAL NOT NULL,
          source TEXT CHECK(source IN ('camera','find_object','user')))
relations(id INTEGER PRIMARY KEY, src_kind TEXT, src_id INTEGER, rel TEXT, dst_kind TEXT,
          dst_id INTEGER, valid_from REAL, valid_to REAL, created REAL, expired REAL)

-- P3 ------------------------------------------------------------------------
reflections(id INTEGER PRIMARY KEY, person_id INTEGER, text TEXT, evidence_json TEXT,
            confidence REAL, created REAL, expired REAL)
job_runs(job TEXT, ran REAL, cursor TEXT)                                  -- idempotency

-- P4 ------------------------------------------------------------------------
reminders(id INTEGER PRIMARY KEY, person_id INTEGER, text TEXT NOT NULL, due REAL,
          deliver_when TEXT CHECK(deliver_when IN ('at_time','when_present')), done REAL, created REAL)
routines(id INTEGER PRIMARY KEY, person_id INTEGER, pattern_json TEXT, action TEXT,
         confidence REAL, proposed REAL, enabled INTEGER DEFAULT 0, last_fired REAL)
settings(person_id INTEGER, key TEXT, value TEXT, updated REAL, PRIMARY KEY(person_id, key))
macros(id INTEGER PRIMARY KEY, person_id INTEGER, phrase TEXT UNIQUE, steps_json TEXT, created REAL)
```

Why these choices:
- **`expired` + `replaced_by` instead of DELETE** keeps history ("you used to live in Saigon") and makes the consolidator undoable. A user `forget` is a **hard delete** (privacy wins over history), including the FTS row and `evidence_json` events.
- **`text` in English, `text_original` as said**: one language for FTS matching across VN/EN without embeddings **[my call]**; the original is used when reading facts back.
- **`source`** lets replies hedge correctly: "you told me…", "I saw…", "I think…".
- **Thumbnails as small BLOBs** (≤ 320 px JPEG, ~20 KB) keep everything in one file; retention purges them (P3).

---

## 7. Phase 1 — People, facts, episodes

### 7.1 Use cases (research §5)
1 greets the owner by name (name from memory; recognition comes with M8) · 2 remembers preferences (language, reply length, how to be addressed) · 3 continues past conversations ("how did the exam go?") · 4 (partial) household facts as plain facts ("Mai is your sister") · 7 birthdays stated → stored as `plan`/`bio` facts (spoken reminder waits for P4) · 16 "what did we talk about yesterday?" · 26 "what do you know about me?" / "forget that" / "forget everything".

### 7.2 Features

| # | Feature | Behaviour |
|---|---|---|
| 1.1 | Store + schema P1 | `MemoryStore(path, clock)`; creates the owner row (`id=1`, name NULL) on first start. Backup copy on clean shutdown (`memory.db.bak`). |
| 1.2 | Event log + writer queue | Every finished turn → `event_log(kind='turn')` (user text, answer, tool names), then queued for consolidation. |
| 1.3 | Importance gate | Rules first, no LLM (`worth_keeping`): keep anything with "remember/nhớ/my name/tên tôi là"; skip turns under 3 words and turns made only of small-talk or robot-command words ("hi", "cảm ơn nhé", "turn left", "look up again"). Everything else is kept (the extractor can return nothing). |
| 1.4 | Consolidator | §7.4. One extract call, one decide call (merged into one when there are no similar facts). |
| 1.5 | `recall` tool | §7.5. FTS5 + filters + scoring + abstention. |
| 1.6 | `remember` / `forget` tools | Explicit memory. `remember` goes through the same consolidator synchronously (with a 3 s timeout) so it dedupes; `forget` §7.6. |
| 1.7 | Briefing | §7.3: once per session, SQL only; ≥ 5 last conversations; last raw turns reloaded after a short gap. |
| 1.8 | Episodes + follow-ups | §7.7. Session ends after 10 min without a turn, or at exit; summarized then. |
| 1.9 | Eval harness | `reachy-demo memory eval` runs the scripted set (§12) against the real LLM; also runs the **long-context baseline** (approach A). |

### 7.3 Briefing (built once per session)

**When.** `build_briefing(store, speaker, now)` runs **once when a session starts** (first turn after 10 min of silence, or the first turn after startup) and again only when the speaker changes (M8). Inside a session anything new is already in the raw turn history, so rebuilding every turn adds nothing; a stable block also keeps DeepSeek's prompt-prefix cache warm **[verify DeepSeek caching applies to our requests]**. SQL only, no LLM.

**What, in priority order** (cap **~600 tokens**, estimate 1 token ≈ 4 chars; when over the cap, trim from the bottom of this list up: facts first (keep ≥ 5), then follow-ups; blocks 1, 2, 4 and 7 are never cut):

| # | Block | Content | Size |
|---|---|---|---|
| 1 | Who + now | Name, role, language, how to address them; current date/time; "last talked 2 days ago" | 1–2 lines |
| 2 | Preferences / settings | `preference`/`setting` facts about the speaker: reply length, language, behaviour corrections | ≤ 5 lines |
| 3 | Follow-ups due | Not done, due within ±2 days | ≤ 3 lines |
| 4 | Last 5 conversations | `episodes(kind='conversation')` with this person, newest first, each cut to ~60 tokens, with a time label | ≤ 5 lines |
| 5 | Last raw turns | Last 5 turns from `event_log`, **only if the previous session ended < 2 h ago**. Loaded into the agent's **history** (not the block), so a restart or crash continues where it stopped | ≤ 5 turns, counted outside the cap |
| 6 | Top facts | Not expired, not sensitive, in scope; top 10 by `score = 0.5·recency + 0.3·importance + 0.2·log1p(use_count)/log1p(50)`, recency = `exp(-age_days/20)` (≈ 14-day half-life) | ≤ 10 lines |
| 7 | Rule line | "Use recall for anything else. Never invent memories." | 1 line |

Later phases add: **P3** ≤ 2 reflection hints (tone only, hedged); **P4** reminders due this session and active privacy rules. **P2 adds nothing**: object locations go stale fast and come only through `recall` / `find_object`.

**Never loaded:** sensitive facts, expired facts, other people's facts when a guest speaks, task episodes (recall only), the event log beyond block 5.

Example:
```
[memory] Now: Mon 17 Nov 2026 19:05. You are talking to Tuan (owner; earlier name: Ha); last talked 2 days ago.
Preferences:
- Tuan wants answers in Vietnamese. (said, 2 weeks ago)
Follow-up: ask how the exam on Fri 14 Nov went.
Recent conversations:
- Sat 20:10: exam stress, trip plan to Da Lat.
- Thu 09:30: asked you to dance, talked about the dog Bin.
Known about Tuan: sister Mai (said, 3 weeks ago); dog named Bin (said, 1 week ago).
Use recall for anything else. Never invent memories: if recall finds nothing, say you don't remember.
```

No name known yet → block 1 says "You don't know the owner's name yet; if they tell you, remember it." Prompt rules are added to `SYSTEM_PROMPT` once (§7.8).

### 7.4 Consolidator (warm path)

Input: one turn (user text, answer, speaker, event id). Steps:

1. **Extract** (LLM, `MEMORY_MODEL`, JSON only, temperature 0). Prompt gives today's date and the speaker's name and asks for:
   ```json
   {"facts": [{"subject": "Tuan", "text": "Tuan's exam is on 2026-11-21.",
               "text_original": "thứ sáu này tôi thi", "category": "plan",
               "valid_from": null, "valid_to": "2026-11-21", "importance": 0.7, "sensitive": false}],
    "follow_ups": [{"text": "Ask how the exam went.", "due": "2026-11-21"}],
    "name": null}
   ```
   Rules in the prompt: atomic, third person, English, resolve "I/tôi" to the speaker's name, absolute dates, skip small talk, return `{"facts": []}` when nothing is worth keeping, mark health/finance/passwords/addresses `sensitive`. `name` is set when the speaker states their name.
2. **Retrieve similar**: for each candidate, FTS5 `MATCH` on its key words (stopwords removed) over non-expired facts with the same `subject_person_id`, top 5 by BM25.
3. **Decide** (LLM, JSON): given the candidate and the ≤ 5 retrieved facts with ids, return one of `ADD`, `UPDATE {id, text}` (same fact, better wording or new detail), `INVALIDATE {id, valid_to}` (contradicted: old fact expires, candidate is added, `replaced_by` set), `NOOP` (already known). **Any id not in the retrieved set is rejected** (falls back to ADD). Skipped when 0 similar facts (→ ADD).
4. **Policy in code** before applying: guest speaker → facts only, stored under the guest; owner → after a non-NOOP `bio`/`preference` fact, one DESCRIBE call rewrites `people.description` (≤ 3 sentences, from the current non-sensitive bio/preference facts only, so it can't drift from them; schema v3, added 2026-11-17); `sensitive` → written only if the turn was an explicit `remember` call, stored `sensitive=1` (never in the briefing, read back only on direct request); `name` → `people.name` for the speaker (the replaced name goes to `aliases_json`); a fact about the speaker's own name is dropped, and any old one is closed.
5. **Apply** in one transaction; set `evidence_json`, mark `event_log.consolidated=1`.

Failure handling: invalid JSON → one retry with the error message, then drop and log `memory_extract_failed`. LLM timeout 10 s. The worker catches everything; a crash restarts the worker (supervisor rule) and the turn is retried once at next startup via `consolidated=0`.

Cost: ≤ 2 `deepseek-chat` calls per kept turn; the gate skips most chit-chat.

### 7.5 `recall` (tool)

```json
{"name": "recall",
 "description": "Search your long-term memory about people, what they told you and past conversations. Write query in English keywords. Use before saying you don't know something personal.",
 "parameters": {"type": "object", "properties": {
   "query": {"type": "string"},
   "person": {"type": ["string", "null"], "description": "'me' for the speaker, a name, or null"},
   "kind": {"type": "string", "enum": ["fact", "episode", "any"], "default": "any"},
   "when": {"type": ["string", "null"], "description": "time words as the user said them, e.g. 'yesterday', 'hôm qua', 'last Tuesday'"},
   "limit": {"type": "integer", "default": 5}},
   "required": ["query"]}}
```

Algorithm:
1. Resolve `person` (`me` → speaker; name → `people.name`/aliases; unknown name → keep as text filter on `subject_text`). Apply scope (§5.4).
2. `when` → `timeparse.parse(when, now)` → `(start, end)`; parsing is code, not the LLM (VN + EN: today/hôm nay, yesterday/hôm qua, this morning/sáng nay, last week/tuần trước, weekday names/thứ hai…chủ nhật, "N days ago"/"N ngày trước", ISO dates). Unparsed → ignored and noted in the result.
3. FTS5 over facts and/or episodes with the filters; facts match non-expired by default, but a time filter also searches expired facts whose validity overlaps (answers "where did I live last year?").
4. Score = BM25 rank fused with recency (reciprocal rank fusion, k = 60); drop near-duplicates (same subject and > 0.8 token overlap).
5. **Abstention**: no match, or best BM25 below threshold → `{"found": false}`.
6. Return compact lines, update `last_used`/`use_count`:
   ```json
   {"found": true, "results": [
     "fact: Tuan's sister is Mai. [said, 3 weeks ago]",
     "episode 2026-11-16 20:10: talked about the exam and a trip to Da Lat."]}
   ```
`needs_response` = true. Not owner-only (guests get scoped results). Timeout 1 s.

### 7.6 `remember` / `forget` (tools, owner-only)

- `remember(fact: str)`: runs extract + decide synchronously on that sentence (3 s timeout; on timeout it is queued and the tool still says ok). Returns `{"ok": true, "stored": "Tuan's favourite drink is iced coffee."}` or `{"ok": true, "already_known": true}`.
- `forget(query: str | null, ids: list[int] | null, everything: bool = false, confirmed: bool = false)`:
  - `query` → recall over facts; 1 match → delete; several → return `{"matches": [{"id": 12, "text": ...}]}` and the model asks which; then `forget(ids=[12])`.
  - `everything=true` → refused unless `confirmed=true`; the tool description tells the model to ask "are you sure?" first. Deletes facts, episodes, follow-ups and event_log for that person; keeps the people row (name cleared).
- Both refuse guests in code: `{"error": "not_owner"}`.

### 7.7 Episodes and follow-ups

- `MemorySession`: a session starts at the first turn (and rebuilds the briefing, §7.3); it ends after **10 min** with no turn (checked at the next turn and every 30 s by the worker), or at app exit. At construction, before any turn, sessions in `event_log` without an episode and unconsolidated turns of the last 24 h are queued (crash recovery).
- A session where no turn passes the importance gate gets an episode "Only small talk." with importance 0, without an LLM call; the briefing skips importance-0 episodes, and recovery doesn't retry it.
- On end: one LLM call over the session's turns (capped at the last 40 turns / 6k tokens) → `{"summary": "...", "topics": [...], "importance": 0.0–1.0, "follow_ups": [...]}` → `episodes(kind='conversation', session_id)`; the unique `session_id` makes it idempotent.
- Follow-ups from both the consolidator and the summary go to `follow_ups`; the briefing shows due ones; the model marks one done via `remember` ("Tuan said the exam went well") → consolidator sets `done` when the fact matches the follow-up **[my call: word overlap between the follow-up and the new fact, ≥ 2 shared content words (or all of them if fewer); simple, may miss]**. *(2026-11-17: also checked on every turn against the user's words and the robot's reply, before the small-talk gate, so a bare "yes" whose reply settles it, or the robot asking it, closes it. Ignored words: the owner's name, "owner", "ask", "robot", "whether". Also 2026-11-17: the EXTRACT call now gets the open follow-ups (id: text) and returns `"answered": [ids]`; those close first, and word overlap is the fallback. Ids it wasn't given are ignored; string ids are accepted.)*
- At exit the summary runs with a 5 s timeout; if it fails, the startup recovery does it next time.

### 7.8 Prompt rules (added to `SYSTEM_PROMPT`)

"You have a long-term memory. Facts in [memory] are what you know. For anything personal not listed there, call recall before answering. If recall finds nothing, say you don't remember; never guess or invent memories. When the user tells you something worth keeping, you don't need to call remember: you remember automatically. Call remember only when they ask you to remember something. Only call forget when they ask you to forget."

### 7.9 Build workflow (TDD, in order)

1. `store.py` schema + migrations + backup — tests: fresh DB, re-open is a no-op, migration failure rolls back.
2. `timeparse.py` — table tests, VN and EN, fixed clock.
3. Fact CRUD + FTS sync triggers + scope filters — tests including guest scope and diacritics.
4. `recall` scoring + abstention + time filters — tests with a seeded DB.
5. `briefing.py` — tests: ~600-token cap with trim order (facts first, ≥ 5 kept; episodes never cut), 5 last conversations, raw turns only when the gap is < 2 h, no sensitive facts, no-name line, rebuilt only at session start.
6. `consolidate.py` with a fake chat fn returning canned JSON — tests for each decision, invalid id rejected, invalid JSON retry, guest no-write, sensitive policy.
7. `session.py` queue + session boundaries with a fake clock — tests: drop-oldest, 10-min boundary, crash recovery.
8. `MemoryTools` — tests: owner-only, forget flows, confirmation for everything.
9. Agent hooks (`context`, `on_turn`) — extend `test_agent.py`; CLI wiring, `MEMORY=off`.
10. `memory eval` harness + test set; run baseline A, then B. **Gate**: if B misses the targets on paraphrase or VN/EN categories, plan approach C (ask before adding `sqlite-vec` + embedding model).
11. Usage doc `docs/usage/memory.md`, link from README; update `docs/architecture/05-agent-tools-memory.md` §5 to point here.

### 7.10 Acceptance
- After restart, the robot uses the owner's name and a fact told in the previous session.
- Eval (§12) P1 categories meet their targets; abstention 100%.
- Added turn latency ≤ 150 ms p95 (briefing build measured in-process); warm path ≤ 2 LLM calls per kept turn.
- `MEMORY=off` runs the agent exactly as before.
- ruff, mypy strict, pytest green.


### 7.11 As built (2026-11-17)
- **Single source of truth (schema v2).** Identity (name, earlier names) lives only in `people`; everything else, including language, is a fact (`category='setting'`). `people.language` was dropped. Migration v2 moves an existing language into a setting fact and turns old "the owner's name is X" facts into `people.name` / `aliases` and closes them. Reason: the name is a key (recall's person filter, M8 voice ID), so it needs a structured column; free-text facts can't be a reliable key.
- All of §7.2 is built and green (ruff, mypy strict, 287 unit tests). No new dependencies.
- **Eval gate (step 10):** approach B passed, so no vectors for now. `memory eval`, `deepseek-chat`, one run, 22 questions: memory **22/22**; long-context baseline 20/22 (fails both temporal questions). The set has 22 questions, not the planned 40. Grow it before trusting small differences.
- **Latency:** the briefing is built once per session from SQL in under 1 ms (eval p95 0.5 ms). That is far inside the 150 ms budget. The p95 added turn latency on the real robot isn't measured yet **[verify]**.
- **Background cost:** 1.7 memory LLM calls per turn in the eval (extract, plus decide when similar facts exist; summaries per session).
- `forget` also deletes the logged turns a fact came from (`evidence_json`). Episode summaries that mention a forgotten fact stay until "forget everything" (limit, P3).
- **Smart context, step 1 (2026-11-17):** hybrid `recall` and consolidator lookup (keywords + embeddinggemma vectors + recency), schema v4 vector tables, background backfill, `memory embed`. Eval results: smart-context design §13.
- Not built in P1, as designed: identity (M8), the `memory list/show/export` CLI (P3), the LLM judge in the eval (keyword scoring, marked as a shortcut).
---

## 8. Phase 2 — Objects and places

### 8.1 Depends on
- `find_object` (F5, M5) and a detector/`WorldState` from perception (research §6.6). Until they exist, 2.1–2.4 work with the existing `camera` tool only.
- Camera horizontal FOV for bearing math **[verify on the robot; store in config]**.

### 8.2 Use cases
8 "where did I leave my keys?" · 9 (partial) "when did the mug appear?" · 10 ownership "my mug" vs "Mai's mug" · 11 "what's on my desk?" · 25 show-and-tell "this is my lucky coin" · 21 (partial) search experience via task episodes.

### 8.3 Features

| # | Feature | Behaviour |
|---|---|---|
| 2.1 | Sightings as bearings + landmarks | Bearing of an object = head yaw/pitch at capture (`Motion.pose`) + box centre offset × FOV → absolute `(az, el)` in the base frame (body yaw included when the body can turn). Tool `name_place(name)` (owner-only): "this is my desk" → landmark at the current head bearing. Each sighting gets the nearest landmark within its radius; `where_text` = "on the desk" or "about 30° to my left". |
| 2.2 | Camera writes sightings | The `camera` tool asks the VLM for its answer **plus** `objects: [{label, box}]` as JSON (one call, same latency) **[verify both providers follow it]**; each object → `sightings(source='camera')`. |
| 2.3 | Memory-aware `find_object` | Before searching, query the latest sighting of the label/name: age < 10 min and confidence ≥ 0.7 → look there, confirm one frame; age < 1 day → look there first, then partial sweep; else full scan. Always write the new sighting. Policy in `spatial.py`, tested with a fake clock. |
| 2.4 | Ownership + custom objects | Tool `name_object(name, owner?)` (owner-only): "this is my lucky coin" → crop of the centre box → `objects(name, owner_person_id, ref_thumb)`. Recall and `find_object` resolve "my X" to the speaker's object first. |
| 2.5 | Usual places | Computed on read, no job: the landmark with ≥ 3 of the last 10 sightings of that object = usual place. Lets the robot say "it's not where you usually leave it". Written to `relations(object, usually_on, landmark)` only when it changes. |
| 2.6 | Task episodes | When a multi-step tool (`find_object` first) finishes, success or failure, write `episodes(kind='task', outcome, details_json={goal, steps:[{pose, landmark, result}], reason})` from the tool result (template, no LLM). Single-step moves are never logged. `find_object` reads failed attempts for the same object in the last 24 h and skips the places already checked unless their sighting is newer. |
| 2.7 | Keyframe verification | `recall(kind="sighting")` can return the stored thumb; for "are you sure?" the model calls `camera` to re-check live (STaR evidence re-check). |

`recall` gains `kind: "sighting" | "task"` and `place: landmark name`.

### 8.4 Acceptance
"Where is my phone?" answered from memory in ≤ 3 s when seen < 10 min ago; full search still ≤ 15 s; correct landmark ≥ 80%; a failed search followed by "did you find my keys earlier?" is answered from the task episode; no duplicate searches of the same place within one task.

---

## 9. Phase 3 — Consolidation, reflection, forgetting

### 9.1 When jobs run
`jobs.py` runs at startup (after crash recovery), after 30 min idle, and nightly if the app is up (03:00 local). Each job records its cursor in `job_runs` and is safe to re-run. Jobs run one at a time in a background task with a CPU-friendly pause between batches; any turn in progress has priority (jobs yield when a turn starts).

### 9.2 Features

| # | Feature | Behaviour |
|---|---|---|
| 3.1 | Dedupe/merge facts | For each subject, pairs of non-expired facts with high FTS overlap → the same decide prompt (UPDATE/INVALIDATE/NOOP) over the pair. Bounded to 50 pairs per run. |
| 3.2 | Episode compression | Conversation episodes older than 7 days → one daily summary (`compressed_level=1`); older than 60 days → weekly (`2`). Originals deleted after the merge succeeds; task episodes kept 30 days then dropped. |
| 3.3 | Decay | No job: activation `ln(Σ (age_of_use_i)^-0.5)` over `last_used` history (approximated by `use_count` and `last_used`) is part of the briefing score; facts below a floor and older than 6 months drop out of the briefing but stay searchable. |
| 3.4 | Reflections | Nightly, per person with ≥ 3 episodes that day: one LLM call → ≤ 3 insights with evidence episode ids → `reflections` (inferred, confidence ≤ 0.6). Used in the briefing as tone hints ("may be stressed this week"), never stated as fact without hedging. |
| 3.5 | Retention | Event log 30 days; thumbnails 7 days; guest rows and their data at session end unless `consent_recognize=1` (then 30 days). Limits are `settings` keys, owner-changeable by voice. |
| 3.6 | Idle-glance captions (opt-in) | Off by default (`MEMORY_GLANCE_MIN=0`). When on and idle, every N minutes: one frame → VLM caption → `episodes(kind='visual')` with bearing and object labels. Rate-limited; skipped when privacy settings forbid the camera. |
| 3.7 | Memory CLI | `reachy-demo memory list [--kind] [--person]`, `show <id>`, `forget <id>`, `export > memory.json`, `stats`. For debugging and user trust. |
| 3.8 | Emotional continuity (if F3 exists) | Mood per episode from the emotion context notes → `episodes.importance` and a weekly reflection; opt-in, hedged (use case 6). |

### 9.3 Acceptance
After two weeks of simulated logs (generated by the eval harness): briefing ≤ 600 tokens; no duplicate facts after a 50-fact update script; knowledge-update questions ≥ 85%; DB growth ≤ 5 MB/week without thumbnails; jobs never add latency to a turn.

---

## 10. Phase 4 — Proactive and procedural

### 10.1 Use cases
13 reminders · 14 routines · 15 proactive context · 16 daily/weekly recap · 17 habit tracking (user-confirmed counts) · 18 learns liked moves · 19 corrections ("speak slower", "don't nod so much") · 20 voice macros · 22 language-tutor spaced repetition (optional) · 27 privacy rules as memory · 7 birthday reminders.

### 10.2 Features

| # | Feature | Behaviour |
|---|---|---|
| 4.1 | Reminders | Tool `set_reminder(text, when, deliver_when)` (owner-only; `when` via `timeparse`). `scheduler.py` checks every 30 s. `at_time`: spoken at the time if a conversation is active or someone is present (WorldState face); else held. `when_present`: spoken at the next turn or next face detection. Delivered = `done`. The agent loop gets a `[event] reminder: …` input, same convention as other events. |
| 4.2 | Routine proposals | Weekly job: per weekday, cluster session start times and stated habits; a pattern holding ≥ 3 of the last 4 weeks → `routines(proposed)`; the robot asks once ("I noticed you take a break around 3 pm. Want me to remind you to stretch?"). Enabled only on a yes. |
| 4.3 | Behaviour preferences | Tool `set_preference(key, value)` (owner-only) with an enum of keys the code reads: `speech_rate` (TTS), `reply_length` (prompt), `nod_amplitude` and `dance_ok` (motion/head tools), `language`. Corrections like "speak slower" map to keys; unknown requests become plain facts. |
| 4.4 | Voice macros | Tools `define_macro(phrase, steps)` and `run_macro(phrase)` (owner-only). Steps are a list of existing tool calls with arguments, validated against tool specs at definition time; run sequentially through the same `Toolbox` with the same permission checks. |
| 4.5 | Privacy rules in code | `settings` keys like `camera_quiet_hours="22:00-07:00"`, `no_recording=true`. The `camera` tool, idle glances and sighting writes check them and refuse with a reason. The prompt mentions them; the code enforces them. |
| 4.6 | Recap | `recall(kind="episode", when="today")` + one answer; weekly recap on request. Tutor mode (optional): `facts(category='other', topic='vocab')` with a due date per word; spaced repetition intervals 1/3/7/14 days. |
| 4.7 | Liked moves (if F3 exists) | After a dance, a smile/laugh within 5 s → +1 for that move in `settings` (per person); the head tool prefers higher-scored dances when the user just says "dance". |

### 10.3 Acceptance
A reminder set by voice is delivered within 60 s of its time when the owner is present, or at the next turn otherwise; no routine is ever enabled without a spoken yes; a defined macro replays its steps with the same permission checks as direct calls; a camera quiet-hours rule blocks the camera tool in a test.

---

## 11. Identity seam (M8, not a memory phase)

When the owner gate lands: `enroll` stores voice (and later face) embeddings in `person_embeddings(person_id, kind, vec, quality, created)`; the turn manager sets `Speaker` per turn; unknown voices get a session guest row. Memory code needs no change beyond reading the `Speaker` it already takes. Guest consent ("can I remember you?") sets `consent_recognize`.

## 12. Evaluation (`reachy-demo memory eval`)

A JSONL script of sessions with simulated time: each step is a user utterance (told to the real agent) or a question with the expected answer. Categories and targets from research §8:

| Category | Phase | Target |
|---|---|---|
| Single-session recall | 1 | ≥ 95% |
| Cross-session recall (restart between) | 1 | ≥ 90% |
| Knowledge update ("moved to Hanoi") | 1 | ≥ 90% |
| Temporal ("what did we talk about last Tuesday?") | 1 | ≥ 80% |
| Multi-hop ("what does my sister's dog eat?") | 1 | ≥ 75% |
| Abstention (never mentioned) | 1 | 100% |
| VN told / EN asked and the reverse | 1 | ≥ 85% |
| Object location | 2 | ≥ 80% landmark correct |
| Person isolation (guest asks owner facts) | M8 | 100% refusal |

Names and dates: exact match. Open answers: an LLM judge (`deepseek-chat`) with the expected answer. Output: per-category score, added latency p50/p95, LLM calls per turn, cost. The set starts at 40 questions (≥ 4 per P1 category) in `tests/replay/memory_eval.jsonl`. The same run with memory off and full history in context gives the long-context baseline.

Unit tests (no network) cover every non-trivial branch listed in the build workflows; the eval is `@pytest.mark.integration`.

## 13. Privacy and safety
Local-only DB; cloud calls send only the turn or session text needed for extraction and summaries (DeepSeek), never the whole memory **[verify: owner accepts cloud extraction — research §10 Q2]**. Owner-only writes/deletes in code. Scoped reads. Sensitive facts never in the briefing. Hard delete on `forget`. Export on request. No pickle anywhere; JSON only.

## 14. Left to the builder
- Exact prompt wording for extract/decide/summary (keep them in `memory/prompts.py`, tested with canned outputs).
- BM25 abstention threshold: tune on the eval set.
- FTS stopword list (small, EN + VN).
- Token estimate method for the briefing cap.
- Whether `decide` and `extract` merge into one call (research §6.4 says it can) — measure on the eval.

## 15. Open decisions for the owner
1. **Identity in Phase 1**: single user (my call) or build voice enrolment now?
2. **Cloud extraction**: OK to send turn text to DeepSeek for memory extraction and summaries?
3. **Target setting**: home companion, demo booth, or both (changes guest retention and proactivity defaults)?
4. **Idle-glance captions** (3.6): wanted at all, and how often?

export MEMORY_DB=data/memory-test.db
alias run='uv run reachy-demo agent --type --no-speak'
alias facts='sqlite3 -header -column $MEMORY_DB "select id, text, category, sensitive, expired is not null as old from facts"'
unalias facts eps fus 2>/dev/null
facts() { sqlite3 -header -column "$MEMORY_DB" "select id, text, category, sensitive, expired is not null as old from facts"; }
eps()   { sqlite3 -header -column "$MEMORY_DB" "select session_id, datetime(start, 'unixepoch', 'localtime') as start, importance, summary from episodes"; }
fus()   { sqlite3 -header -column "$MEMORY_DB" "select id, text, datetime(due, 'unixepoch', 'localtime') as due, done is not null as done from follow_ups"; }