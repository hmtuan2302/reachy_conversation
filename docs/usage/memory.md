# Memory: Reachy remembers you across restarts

Tell Reachy things about yourself and it remembers them the next time you talk, even after the app restarts. It brings them up when they fit, finds them when you ask, and says it doesn't remember instead of guessing.

An example (the exact words vary):

```
$ uv run reachy-demo agent --type
  (memory search: loading the embedding model...)
  (memory search: hybrid (embeddinggemma-300m))
  (memory on: data/memory.db)
  (memory push: on)
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

Memory is **on by default** for `reachy-demo agent`, with or without `--robot`. It uses the same `DEEPSEEK_API_KEY` as the agent. Search by meaning needs a one-time setup ([below](#first-time-setup-search-by-meaning)); without it, memory still works with keyword search.

## First-time setup: search by meaning

Reachy finds memories by meaning with a local embedding model, [embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m). It runs on the laptop CPU through ONNX Runtime (no torch, no GPU), and nothing leaves the laptop.

1. Install the extra (adds `tokenizers` and `huggingface-hub`):
   ```bash
   uv sync --extra embed            # add your other extras too, e.g. --extra vieneu --extra mic
   ```
   `uv sync` installs exactly the extras you list, so list all of them. Leaving out `--extra vieneu` uninstalls VieNeu.
2. Make sure `HF_TOKEN` is in `.env` (the same token as speech-to-text). The model is under the Gemma license: if the download says it's gated, accept the license on the model page once.
3. Start the agent. The first start downloads about **1.2 GB** into `models/onnx-community--embeddinggemma-300m-ONNX/` (git-ignored); later starts load it in under a second, offline.

Memories you already had get their vectors in the background, 200 at a time, once nobody has talked to Reachy for a minute (so it never slows a reply). If an embedding fails, it's retried at the next idle moment. To do them all at once: `uv run reachy-demo memory embed`.

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

**Your description.** When you tell Reachy about yourself (who you are, your personality, likes and dislikes), it saves each as a fact and then rewrites your `description` in `people` from all of those facts: up to 3 sentences, such as "Tuan is a calm, curious software engineer who loves black coffee and is a bit of a night owl. Tuan hates crowded places." It goes at the top of Reachy's memory each session. Plans, pets and family don't change it. Correct a fact ("actually I prefer tea now") and the description follows at the next rewrite.

**Other people.** When a guest talks about themselves, only facts are saved, under the guest, never under you. Guests get no name change, follow-ups or description. Guests are only told apart while [owner mode](agent.md#owner-mode-speaker-verification) is on, and then their turns are dropped. So for now everything goes to you.

**Changes are tracked, not overwritten.** "I moved to Hanoi" closes the old "lives in Saigon" fact instead of deleting it, so "where did I live before?" still works.

## Voice commands

Everything is automatic. You can also be explicit:

| You say | What happens |
|---|---|
| "Remember that my wifi password is …" | Stored (`remember` tool), even sensitive things. Sensitive facts are never brought up on their own |
| "What do you know about me?", "What did we talk about yesterday?", "hôm qua mình nói gì?" | Searched (`recall` tool) by meaning and keywords, person and time ("yesterday", "last Tuesday", "sáng nay", "tuần trước") |
| "Forget that I have a cat" | Deleted. With several matches, Reachy asks which one |
| "Forget everything about me" | Reachy asks "are you sure?" first, then deletes all your facts, conversations, follow-ups and logged turns |

Sensitive things (health, money, passwords, ID numbers, exact addresses) are only stored when you explicitly ask Reachy to remember them.

## What Reachy knows at the start of each conversation

When a conversation starts, a `[memory]` block (at most `MEMORY_MAX_TOKENS`, default 600) is added to the prompt. It's built from the database in under a millisecond, with no LLM call:

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

## How Reachy finds what's relevant

Three things bring memories into a reply:

| | When | What |
|---|---|---|
| **`[memory]` block** | Once per conversation | Name, preferences, follow-ups, last 5 conversations, top 10 facts (above) |
| **Related memories** | Every turn, before Reachy answers | Up to 3 memories close in meaning to what you just said |
| **`recall`** | When the agent decides to search | Anything else: questions about the past, names, dates |

### Related memories (every turn)

Before each answer, Reachy embeds what you just said (with your previous 2 turns, so "what about her?" still means Mai) and looks for memories close in meaning. Up to 3 that are clearly related are added in front of your message:

```
[related memories — use only if it fits the conversation]
- fact: Tuan is allergic to shrimp. [said, 4 days ago]
- conversation Sat 15 Nov 2026 20:10: Tuan planned a seafood dinner with Mai.
you: I'm thinking about cooking seafood for dinner tonight.
```

So Reachy can say "careful, no shrimp for you" without being asked. Details:

- Often there are **none**: a memory must be similar enough (0.55), so unrelated chat ("what's 17 × 23?") gets nothing.
- Small talk ("ok", "haha", "thanks") reuses the previous turn's memories, with no new embedding.
- Memories already in the `[memory]` block, sensitive facts and other people's memories are never added.
- It's **never stored**: not in the conversation history, not in the turn the memory writer reads. So it can't turn into a new fact, and old blocks don't pile up.
- It costs one embedding per turn, about 20 ms (up to ~125 ms for a long sentence), before the answer starts.
- Being pushed doesn't count as "used", so it doesn't change which facts the `[memory]` block shows.
- To see them, set `CONTEXT_DEBUG=on`: before each answer the console prints the `[memory]` block (once per conversation), the related memories and every `recall` result ([agent.md](agent.md#debugging-the-context)).

### `recall`

The `[memory]` block and related memories cover a few memories. Everything else is found with **`recall`**, which the agent calls when you ask about the past or mention something personal it doesn't see.

`recall` combines three rankings and keeps the best 5:

| Ranking | Finds |
|---|---|
| **Meaning** (vectors) | "Which AI assistant do I use when I code?" → "Tuan uses Claude for programming", with no word in common. Works across Vietnamese and English |
| **Keywords** (SQLite full-text) | Exact names and rare words: "Sapiens", "VNG", "Bin" |
| **Recency** | Newer memories first when the other two tie |

The same search runs when a new fact is saved, to find the old fact it replaces. So "I switched to Claude for programming" closes "Tuan writes code with GitHub Copilot" even though the two share no words, and Reachy stops mentioning Copilot.

Each memory's vector is computed **once**, when it's saved (in the background, not while Reachy answers). Each `recall` embeds only its own short query, about 20–100 ms on this laptop.

Guests: the vector search only looks at memories the speaker is allowed to see, the same rule as keyword search.

## Settings (`.env`)

```
# MEMORY=off                  # turn memory off entirely: the agent works exactly as before
# MEMORY_DB=data/memory.db    # where memory lives (one SQLite file; data/ is git-ignored)
# MEMORY_MODEL=deepseek-chat  # LLM for memory writes; keep a fast non-reasoning model
# MEMORY_MAX_TOKENS=600       # cap for the [memory] block (approx, chars/4); must be below CONTEXT_MAX_TOKENS (agent.md)
# EMBED=off                   # keyword-only search (no model download, no vectors)
# EMBED_MODEL=onnx-community/embeddinggemma-300m-ONNX   # HF repo, or a local folder with tokenizer.json and onnx/model.onnx
# MEMORY_PUSH=off             # no related memories per turn (recall still searches by meaning)
```

With `MEMORY_PUSH` on, the agent keeps 200 tokens of `CONTEXT_MAX_TOKENS` free for the related memories ([agent.md](agent.md)).

## Files

- `data/memory.db`: everything, in one SQLite file. Older files upgrade themselves on first open (schema v2 on 2026-11-17 moved names out of facts and the language into a fact). Copied to `data/memory.db.bak` on every clean exit, and to `memory.db.bak-<n>` before a schema upgrade.
- Vectors live in the same file (tables `fact_vecs`, `episode_vecs`, 1 KB per memory, schema v4). They're deleted with the memory they belong to.
- `models/onnx-community--embeddinggemma-300m-ONNX/`: the embedding model (1.2 GB). Delete it to free space; it downloads again on the next start.
- **After changing `EMBED_MODEL`**, run `uv run reachy-demo memory embed` once. Vectors are stored per model (the `EMBED_MODEL` value, or the folder name for a local folder), so old ones are simply ignored and every memory is embedded again:
  ```
  $ uv run reachy-demo memory embed
  memory embed: 41 facts, 9 episodes (onnx-community/embeddinggemma-300m-ONNX@256)
  ```
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

Two description questions were added afterwards ("What kind of person am I?" after a personality turn, and the drink after "I prefer green tea now"). With them, memory scored 24/24 on 2026-11-17. The baseline wasn't rerun.

**Search by meaning (2026-11-17, 39 questions, one run each).** 15 questions were added: `paraphrase` (no word shared with the stored fact, VN and EN), `invalidate` (a changed value must win) and 3 more `multihop`. Keyword-only and hybrid scored the same on paraphrase (8/8) and invalidate (4/4), and both missed 2 multi-hop questions. So on this set the vectors didn't change recall: the agent already writes good English keywords. The query embedding took 17 ms (p95 27 ms).

**Related memories (2026-11-17, 55 questions, one run each).** 16 more: `proactive` (a plain remark; the answer should bring up a stored memory) and `noise` (an unrelated question; no memory may come up). With related memories: 54/55, proactive 7/8, noise 8/8, and the whole per-turn step took 22 ms (p95 25 ms). Keyword-only without related memories: 53/55, proactive also 7/8. So this set shows related memories are fast and quiet, but not yet that they help: most target facts were recent enough to be in the `[memory]` block anyway. A fair test needs many more facts spread over weeks. Results: [smart-context design §13](../superpowers/specs/2026-11-17-smart-context-design.md#13-spike-results).

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
- To find related memories, each thing you say is embedded on the laptop. Those vectors stay in RAM for the current conversation only and are never saved.
- Vectors are computed on the laptop; no text goes anywhere for them. A vector is a fingerprint of the text, so it's deleted together with its fact or conversation ("forget that", "forget everything").
- The text of each turn worth keeping, and each conversation transcript, goes to DeepSeek for fact extraction and summaries. That's the same provider that already answers every question.
- Phase 1 treats **every speaker as the owner**. There's no voice ID yet (milestone M8), so anyone talking to Reachy can read and change its memory.

## Limits

- **Search by meaning is approximate.** It finds close wording and translations, but a vague question can still pull a related-but-wrong memory, and a hit below similarity 0.3 is dropped. Keywords still catch exact names.
- **Related memories can stay quiet or miss.** The 0.55 threshold favours silence over noise; a memory phrased very differently from what you say may not come up. `recall` still finds it when you ask.
- **CPU cost.** Loading the model takes about 0.6 GB of RAM. Each `recall` adds 20–100 ms (up to ~125 ms for a long sentence). Saving a memory adds a background embedding of the same size.
- **Without the extra or the model**, memory falls back to keyword search only, and says so at start (`memory search: keywords only (...)`).
- **The test set is small** (55 questions, one run) and scored by keyword match. 100% means no regressions on these cases, not that memory is perfect.
- **How follow-ups close.** Two checks:
  - The LLM that already reads each turn for facts also sees your open follow-ups, and says which ones the turn answered. It understands other wording and languages: "thi ổn rồi" or "the test went fine" closes "Ask how the math exam went". Just mentioning the topic ("I'll study for the exam tonight") doesn't close it.
  - Short turns like "yes" skip that LLM call, so a word match is the fallback. A follow-up closes when your words or Reachy's reply share at least 2 topic words with it. That's why "yes" → "I'm back to my own voice" closes "Ask whether … switch back to its own voice", and also why a follow-up closes once Reachy has asked it.
- **Background memory:** a memory write can lag a few seconds behind the conversation. When the app is killed (not a clean exit), the rest is finished at the next start.
- **Forget** deletes facts and their logged turns. A conversation summary that mentioned the forgotten thing stays until you "forget everything".
- Not built yet (later phases): object locations, reminders, routines, forgetting old memories over time, and the `memory list/show/export` commands. See the [design](../superpowers/specs/2026-11-17-memory-design.md).

## Errors

| What you see | Meaning / fix |
|---|---|
| `  (memory off: cannot open data/memory.db: …)` | The file is damaged or not writable. The agent runs without memory. Restore `data/memory.db.bak` or delete the file |
| Reachy says "I don't remember" about something you told it | Memory is written in the background a few seconds after the turn. Check the logs for `memory extract failed` (DeepSeek error or timeout) |
| `config error: DEEPSEEK_API_KEY is required` from `memory eval` (exit 2) | Add the key to `.env` |
| `  (memory search: keywords only (install: uv sync --extra embed))` | The `embed` extra isn't installed. Run the [setup](#first-time-setup-search-by-meaning), listing all your extras |
| `  (memory search: keywords only (model download failed: …))` | No network, wrong `EMBED_MODEL`, or the model is gated: check `HF_TOKEN` and accept the Gemma license on the model page |
| `  (memory search: keywords only (model load failed: …))` | `EMBED_MODEL` points at a folder without `tokenizer.json` and `onnx/model.onnx`, or the download was cut off: delete the folder under `models/` and restart |
| `  (memory search: keywords only (EMBED=off))` | You turned it off in `.env` |
| `  (memory push: off: no embedding model)` | Related memories need the embedding model; see the `memory search` line above it |
| `  (memory push: off: MEMORY_PUSH=off)` | You turned it off in `.env` |
| `config error: related memories (MEMORY_PUSH) need ~200 more tokens …` (exit 2) | Raise `CONTEXT_MAX_TOKENS`, lower `MEMORY_MAX_TOKENS`, or set `MEMORY_PUSH=off` |
| `config error: memory eval needs the embedding model: …` (exit 2) | `memory eval` refuses to run keyword-only by accident. Fix the model, or set `EMBED=off` to test keyword-only on purpose |
| `memory embed: no embedder: …` (exit 2) | Same causes as the `memory search: keywords only` lines above |

Code: [src/reachy_demo/memory/](../../src/reachy_demo/memory/) (`store.py` SQLite and queries, `briefing.py`, `recall.py`, `embed.py` the model, `vectors.py` the vector index, `related.py` related memories, `consolidate.py`, `episodes.py`, `session.py`, `timeparse.py`, `prompts.py`), [src/reachy_demo/agent/tools/memory.py](../../src/reachy_demo/agent/tools/memory.py) (tools), [src/reachy_demo/agent/memory_eval.py](../../src/reachy_demo/agent/memory_eval.py) (eval).
