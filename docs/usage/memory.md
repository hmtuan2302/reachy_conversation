# Memory: Reachy remembers you across restarts

Tell Reachy things about yourself and it remembers them the next time you talk, even after the app restarts. It brings them up when they fit, finds them when you ask, and says it doesn't remember instead of guessing.

An example (the exact words vary):

```
$ uv run reachy-demo agent --type
  (memory on: data/memory.db)
you: Hi, I'm Tuấn. Thứ sáu này tôi có bài thi toán.
bot: Chào Tuấn! Chúc bạn thi tốt nhé.
^D
$ uv run reachy-demo agent --type          ← next day, new run
  (memory on: data/memory.db)
you: Hello!
bot: Chào Tuấn! Bài thi toán hôm thứ sáu thế nào rồi?
you: What did we talk about yesterday?
bot: Yesterday you told me about your maths exam on Friday.
you: What's my mother's name?
bot: I don't remember you telling me that.
```

Memory is **on by default** for `reachy-demo agent`, with or without `--robot`. Nothing to set up: it uses the same `DEEPSEEK_API_KEY` as the agent.

## What it remembers

| Kind | Example | Written when |
|---|---|---|
| **Facts** about you and your life | "Tuan has a sister named Mai", "Tuan prefers short answers" | In the background, after each turn worth keeping |
| **Your name** (and earlier names) | "tên tôi là Tuấn", "call me Tuan" | Same |
| **Preferences and settings**, including the language you ask for | "please speak Vietnamese" → "Tuan wants answers in Vietnamese" | Same |
| **Conversations**: one short summary per conversation | "Sat 20:10: exam stress, trip plan to Da Lat" | When the conversation ends (10 min silence, or exit) |
| **Follow-ups**: things to ask about later | "Ask how the exam went (due Fri)" | From facts and summaries |

A **conversation** ends after 10 minutes without a turn, or when the app exits.

Small talk and robot commands ("hi", "thanks", "turn left", "dance") are logged but never sent to the LLM for memory.

**One source of truth for each thing.** Who you are (your name, and the names you used before) lives only on your person record, in the `people` table. Everything else about you lives only in facts, including the language you want. The newest "my name is …" or "call me …" wins: Reachy switches without asking you to confirm, and the old name is kept as an earlier name ("did I go by another name?" still works). A fact about your own name is never stored.

**Changes are tracked, not overwritten.** "I moved to Hanoi" closes the old "lives in Saigon" fact instead of deleting it, so "where did I live before?" still works.

## Voice commands

Everything is automatic. You can also be explicit:

| You say | What happens |
|---|---|
| "Remember that my wifi password is …" | Stored (`remember` tool), even sensitive things. Sensitive facts are never brought up on their own |
| "What do you know about me?", "What did we talk about yesterday?", "hôm qua mình nói gì?" | Searched (`recall` tool), by keywords, person and time ("yesterday", "last Tuesday", "sáng nay", "tuần trước") |
| "Forget that I have a cat" | Deleted. With several matches, Reachy asks which one |
| "Forget everything about me" | Reachy asks "are you sure?" first, then deletes all your facts, conversations, follow-ups and logged turns |

Sensitive things (health, money, passwords, ID numbers, exact addresses) are only stored when you explicitly ask Reachy to remember them.

## What Reachy knows at the start of each conversation

When a conversation starts, a `[memory]` block (about 600 tokens at most) is added to the prompt. It's built from the database in under a millisecond, with no LLM call:

```
[memory] Now: Mon 17 Nov 2026 19:05. You are talking to Tuan (owner; earlier name: Ha); last talked 2 days ago.
Preferences:
- Tuan wants answers in Vietnamese. (said, 2 weeks ago)
Follow up when it fits:
- Ask how the exam went. (due Fri 21 Nov)
Recent conversations:
- Sat 15 Nov 20:10: Tuan talked about exam stress and a trip to Da Lat with Mai.
Known about Tuan:
- Tuan has a sister named Mai. (said, 3 weeks ago)
Use recall for anything else. Never invent memories: if recall finds nothing, say you don't remember.
```

It holds your name, preferences, due follow-ups, the **last 5 conversation summaries** and up to 10 top facts. When it gets too long, facts are trimmed first, keeping at least 5.

If you restart within **2 hours**, the last 5 turns are also reloaded into the conversation, so Reachy picks up where it left off.

## Settings (`.env`)

```
# MEMORY=off                  # turn memory off entirely: the agent works exactly as before
# MEMORY_DB=data/memory.db    # where memory lives (one SQLite file; data/ is git-ignored)
# MEMORY_MODEL=deepseek-chat  # LLM for memory writes; keep a fast non-reasoning model
```

## Files

- `data/memory.db`: everything, in one SQLite file. Older files upgrade themselves on first open (schema v2 on 2026-11-17 moved names out of facts and the language into a fact). Copied to `data/memory.db.bak` on every clean exit, and to `memory.db.bak-<n>` before a schema upgrade.
- **To start over**, quit the app, then:
  ```bash
  uv run reachy-demo memory reset          # deletes MEMORY_DB (default data/memory.db); asks first
  uv run reachy-demo memory reset --all    # also deletes the .bak backups
  uv run reachy-demo memory reset -y       # no question (for scripts)
  uv run reachy-demo memory reset --db data/memory-test.db
  ```
  It lists the files before deleting them. Without `--all` the backups stay, so `cp data/memory.db.bak data/memory.db` undoes a reset. By voice, "forget everything about me" clears what Reachy knows about you but keeps the file.

## Testing memory (`memory eval`)

To test by hand, step by step, see [memory-testing.md](memory-testing.md).


A scripted test set tells the agent things over simulated days, restarts it, and asks questions:

```bash
uv run reachy-demo memory eval              # memory on (tests/replay/memory_eval.jsonl)
uv run reachy-demo memory eval --baseline   # memory off, whole history kept in context
uv run reachy-demo memory eval my_set.jsonl # your own script
```

Result on 2026-11-17 (`deepseek-chat`, one run, 22 questions):

| Category | Memory | Baseline (long context) |
|---|---|---|
| single-session, cross-session, knowledge update, multi-hop, VN/EN mixed | 18/18 | 18/18 |
| temporal ("what did we talk about last Tuesday?") | 2/2 | 0/2 |
| abstention (never mentioned) | 4/4 | 4/4 |
| **overall** | **22/22** | **20/22** |

Memory used 1.7 background LLM calls per turn, and building the briefing took under 1 ms. The baseline has no timestamps, so it can't answer "when" questions. It also sends the whole history on every turn, which grows without limit.

Script format, one JSON object per line:

```
{"t": "2026-11-01T20:00:00+07:00", "say": "I have a sister named Mai."}
{"restart": true}
{"t": "2026-11-09T19:00:00+07:00", "ask": "What's my sister's name?", "expect": ["Mai"], "category": "cross"}
{"t": "2026-11-09T19:01:00+07:00", "ask": "What's my mother's name?", "expect_none": true, "category": "abstain"}
```

An answer counts as correct when it contains any `expect` word (case and accents ignored). `expect_none` counts as correct when the answer says it doesn't know. The run uses a temporary database, so your `data/memory.db` is never touched.

## Privacy

- The database stays on this laptop.
- The text of each turn worth keeping, and each conversation transcript, goes to DeepSeek for fact extraction and summaries. That's the same provider that already answers every question.
- Phase 1 treats **every speaker as the owner**. There's no voice ID yet (milestone M8), so anyone talking to Reachy can read and change its memory.

## Limits

- **Keyword search, no embeddings.** Facts are stored in English, and the agent searches in English keywords, so Vietnamese and English work together. A very different paraphrase can still miss ("my pet" vs. "dog"). If the eval shows misses, the next step is vector search.
- **The test set is small** (22 questions, one run) and scored by keyword match. 100% means no regressions on these cases, not that memory is perfect.
- **Follow-ups close by word overlap.** "thi ổn rồi" closes "Ask how the exam went" only if the new fact shares words with it, so occasionally Reachy may ask twice.
- **Background memory:** a memory write can lag a few seconds behind the conversation. When the app is killed (not a clean exit), the rest is finished at the next start.
- **Forget** deletes facts and their logged turns. A conversation summary that mentioned the forgotten thing stays until you "forget everything".
- Not built yet (later phases): object locations, reminders, routines, forgetting old memories over time, and the `memory list/show/export` commands. See the [design](../superpowers/specs/2026-11-17-memory-design.md).

## Errors

| What you see | Meaning / fix |
|---|---|
| `  (memory off: cannot open data/memory.db: …)` | The file is damaged or not writable. The agent runs without memory. Restore `data/memory.db.bak` or delete the file |
| Reachy says "I don't remember" about something you told it | Memory is written in the background a few seconds after the turn. Check the logs for `memory extract failed` (DeepSeek error or timeout) |
| `config error: DEEPSEEK_API_KEY is required` from `memory eval` (exit 2) | Add the key to `.env` |

Code: [src/reachy_demo/memory/](../../src/reachy_demo/memory/) (`store.py` SQLite and queries, `briefing.py`, `recall.py`, `consolidate.py`, `episodes.py`, `session.py`, `timeparse.py`, `prompts.py`), [src/reachy_demo/agent/tools/memory.py](../../src/reachy_demo/agent/tools/memory.py) (tools), [src/reachy_demo/agent/memory_eval.py](../../src/reachy_demo/agent/memory_eval.py) (eval).
