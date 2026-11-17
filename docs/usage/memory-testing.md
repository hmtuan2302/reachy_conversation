# Testing memory by hand

Scenarios to check [memory](memory.md) yourself, with the expected result for each. They use typed input and a separate test database, so your real `data/memory.db` stays clean.

## Setup (each new terminal)

Run from the repo folder:

```bash
export MEMORY_DB=data/memory-test.db
alias run='uv run reachy-demo agent --type --no-speak'

# Look inside the memory database
facts() { sqlite3 -header -column "$MEMORY_DB" "select id, text, category, sensitive, expired is not null as old from facts"; }
eps()   { sqlite3 -header -column "$MEMORY_DB" "select session_id, datetime(start, 'unixepoch', 'localtime') as start, importance, summary from episodes"; }
fus()   { sqlite3 -header -column "$MEMORY_DB" "select id, text, datetime(due, 'unixepoch', 'localtime') as due, done is not null as done from follow_ups"; }
who()   { sqlite3 -header -column "$MEMORY_DB" "select id, role, name, aliases_json as earlier_names from people"; }
```

| Helper | Shows | Columns to watch |
|---|---|---|
| `who` | People: your current name and earlier names | Names live **only** here, never in `facts` |
| `facts` | Stored facts | `old = 1`: replaced by a newer fact. `sensitive = 1`: kept out of the prompt |
| `eps` | One summary per conversation | `importance = 0`: small talk only, not shown to the robot |
| `fus` | Follow-ups ("Ask how the exam went") | `done = 1`: answered, won't be asked again |

`sqlite3` reads double quotes as column names, so the times use single quotes (`'unixepoch'`). That's why these are shell functions and not aliases. Wait 2–3 seconds after a turn before you check, because memory is written in the background.

**Reset between experiments** (quit the app first):

```bash
uv run reachy-demo memory reset          # deletes $MEMORY_DB (here data/memory-test.db); asks first
uv run reachy-demo memory reset --all -y # also the backups, without asking
```

## What "restart" means

Many tests say **restart**: quit the app and start it again, so the answer can't come from the current conversation and has to come from the memory database.

1. Press **Ctrl-D** on an empty line. This is end-of-input: Reachy finishes the answer it's giving, then exits cleanly. It writes the conversation summary, finishes pending memory writes (it waits up to 5 s), and copies the database to `.bak`.
2. Wait for the shell prompt.
3. Type `run` again.

| Key | What it does | Memory |
|---|---|---|
| **Ctrl-D** (empty line) | Ends the input; clean exit after the last answer | Summary and backup written now |
| **Ctrl-C** | Quits at once, even mid-answer | Same: summary and backup written on the way out |

Both keys save the same things. Ctrl-C was checked 2026-11-17: after one turn, the summary, follow-up and backup were all written. The exceptions are pressing Ctrl-C twice and a summary call that takes over 5 s; then that work is done at the next start.

If you've typed text without pressing Enter, the first Ctrl-D only sends that text, so press it again. With voice input (`--mic` / `--stdin`) there's no end of input, so use Ctrl-C.

**Catch:** if you restart within 2 hours, the app reloads your **last 5 turns** into the conversation, so it may answer from those turns and not from memory. To be sure it came from memory, say 5 or more other things after the test sentence before quitting, or check with `facts`.

## A. Core

| # | Steps | Expected |
|---|---|---|
| 1 | **Name.** `run` → "Hi, my name is Tuấn." → restart → "Hello!" | It greets you by name. `who` shows the name; `facts` has no name fact |
| 1b | **Name change.** "My name is Ha." → restart → "What's my name? Did I go by another name?" | "Ha", and it knows you were Tuấn before. `who` shows `name = Ha`, `earlier_names = ["Tuấn"]`. It switches without asking you to confirm |
| 1c | **Language.** "Please always answer in Vietnamese." → restart → "What's the weather like in winter?" | Answers in Vietnamese. `facts` shows a `setting` fact; `who` has no language |
| 2 | **Fact across restart.** "I have a dog named Bin, he's a corgi." → restart → "What's my dog's name?" | "Bin". `facts` shows one English fact |
| 3 | **Vietnamese told, English asked.** "Em gái tôi tên là Mai, 20 tuổi." → restart → "How old is my sister?" | "20". `facts` shows the fact in English, and the original words are kept |
| 4 | **Update.** "I live in Saigon." → restart → "I moved to Hanoi last week." → restart → "Where do I live?", then "Where did I live before?" | "Hanoi", then "Saigon". `facts` shows Saigon with `old = 1` |
| 5 | **Says it doesn't know.** "What's my mother's name?" / "Tôi sinh năm nào?" | "I don't remember" or "Mình không nhớ", never a made-up answer |
| 6 | **Multi-hop.** "Mai has a cat called Mochi that only eats fish." → restart → "What does my sister's cat eat?" | "Fish" |

## B. Conversations and follow-ups

| # | Steps | Expected |
|---|---|---|
| 7 | **Summary at exit.** Talk about 3 topics → Ctrl-D → `eps` → `run` → "What did we talk about today?" | `eps` shows one summary. The answer lists the topics |
| 8 | **Follow-up.** "Ngày mai tôi có bài thi toán, hơi lo." → Ctrl-D → `fus` → `run` → "Hi" | `fus` shows "Ask how the exam went", due tomorrow. On the next run Reachy may ask about the exam. The window is ±2 days, so it works today and tomorrow |
| 9 | **Follow-up closes.** After 8: "The exam went well." → wait 3 s → `fus` | `done = 1`. If it stays 0, the word overlap missed it (known limit) |
| 10 | **Reload after restart (< 2 h).** "My favourite colour is green." → Ctrl-D → `run` → "What did I just tell you?" | It knows, from the reloaded turns, without calling `recall` |
| 11 | **Small talk only.** Fresh run: "hi" → "thanks" → Ctrl-D → `eps` | An "Only small talk." row with `importance = 0`. The next run doesn't mention it |

## C. Explicit tools and privacy

| # | Steps | Expected |
|---|---|---|
| 12 | **Sensitive.** "My bank PIN is 1234." → `facts` → "Remember that my bank PIN is 1234." → `facts` → restart → "Hi" → "What's my PIN?" | Not stored the first time. Stored with `sensitive = 1` the second time. Not in the greeting, but answered when you ask |
| 13 | **Forget one.** "Forget that I have a dog." | It confirms. `facts` shows no dog, and the turn it came from is deleted too |
| 14 | **Forget, ambiguous.** Store "I like green tea" and "Mai likes black tea" → "Forget the tea thing." | It asks which one, then deletes only that one |
| 15 | **Forget everything.** "Forget everything about me." → it asks → "Yes." → restart → "What's my name?" | It asks before deleting. Afterwards `facts`, `eps` and `fus` are empty and it doesn't know your name |
| 16 | **Commands not stored.** `uv run reachy-demo agent --type --robot --no-speak` → "turn left", "dance", "what do you see?" | The robot moves. `facts` gains nothing from these turns |

## D. Robustness

| # | Steps | Expected |
|---|---|---|
| 17 | **Memory off.** `MEMORY=off run` → "What's my dog's name?" | No `(memory on…)` line; it doesn't know |
| 18 | **Broken DB.** `echo junk > data/broken.db` → `MEMORY_DB=data/broken.db run` | `(memory off: cannot open …)`, and the chat still works. Delete `data/broken.db` afterwards |
| 19 | **Killed mid-conversation.** Tell it a fact → in another terminal: `pkill -9 -f "reachy-demo agent"` → `run` → ask about the fact | Remembered: the turn was logged and gets processed at the next start. `eps` gains the old session's summary |
| 20 | **10-minute gap.** Talk → stay idle 11 min (don't exit) → say something → `eps` | The first conversation is summarized without exiting, and a new one starts |
| 21 | **By voice.** Repeat 1–5 with `--mic` or `--stdin` (see [agent.md](agent.md)), mixing Vietnamese and English | Same results. Speech-to-text mistakes in names ("Tuấn" → "Tuan") are fine |

## E. Automatic

Days-apart questions ("yesterday", "last Tuesday") are slow to test by hand. The scripted eval runs them on a simulated clock (about 1.5 min, real DeepSeek calls):

```bash
uv run reachy-demo memory eval              # memory on
uv run reachy-demo memory eval --baseline   # memory off, whole history in context
```

Add your own cases to `tests/replay/memory_eval.jsonl` ([format](memory.md#testing-memory-memory-eval)).

## Most likely to fail

- **Case 9:** follow-ups close by simple word overlap.
- **Case 8:** whether Reachy opens with the exam is up to the LLM, so it may not happen every time.
- **Case 21:** with laptop speakers, Reachy can hear its own answers, and they get stored as your turns. Use a headset.
