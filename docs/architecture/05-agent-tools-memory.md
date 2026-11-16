# 05 — Agent, tools and memory (F1, F3, F5, F8, F9, F10)

> **Update 2026-11-16:** the owner chose a **pipeline** (speech → text → LLM → text → speech) instead of the speech-to-speech Realtime model in §1. Stage 1 is built: `reachy-demo agent` (S2T on the Reachy HF server, which decides the end of turn; DeepSeek as the LLM; text out). See `docs/superpowers/specs/2026-11-16-voice-agent-design.md` and `docs/usage/agent.md`. The tool layer, permissions and memory below still apply; they plug into `agent/agent.py` when tools are added.

## 1. Agent model

The "agent" is the OpenAI Realtime model plus a tool registry. The app does not run a second planner: the model decides which tool to call, while code enforces permission, timeouts and side effects **[arch]**. Multi-step tasks (search → confirm → look → report) live *inside* one tool, so the model stays out of tight loops and latency stays bounded.

Tool execution:
```
function_call(name, args, call_id)
  → registry.get(name) → validate args (JSON Schema, stdlib check of required/types)
  → permission check (owner_only vs current turn's speaker)       ── refuse → {"error":"not_owner"}
  → asyncio.wait_for(tool(ctx, **args), tool.timeout_s)           ── timeout → {"error":"timeout"}
  → exception → log + {"error":"internal"}
  → send function_call_output → response.create (unless tool.silent)
```

## 2. Permission policy (F9) **[req]**

- Each turn carries `speaker ∈ {owner, guest}` from the gate. A tool call is attributed to the turn that triggered the response.
- `owner_only=True` tools refuse for guests in code. The system prompt also tells the model "only the owner can ask for actions; politely decline guests". That is a UX nicety, never the control.
- Autonomous behaviours (FaceTrack, emotion mirroring, Investigate) are not tools and are not gated.

## 3. Tool catalog

| Tool | Args | owner_only | Timeout | Does |
|---|---|---|---|---|
| `introduce_self` | — | no | 3 s | Plays the `greet` move, looks at the speaker; returns persona facts for the model to speak (F1). |
| `play_move` | `name` (enum of moves.py) | yes | 5 s | Expressive move on request ("dance", "nod"). |
| `express` | `mood` (enum) | no | 2 s | Mood offset matching the reply tone (F3). The model calls it itself; it's not a user command. |
| `look_at` | `target` ("me" / object label) | yes | 2 s to start | Starts `LookAt` / switches to FaceTrack on the owner (F4). |
| `find_object` | `query`, `coco_hint?` | yes | 15 s | Search behaviour + cloud confirm + memory sighting (F5). |
| `mimic` | `on: bool` | yes | 1 s | Start/stop Mimic (F6). |
| `look_at_pointed` | `mode ∈ {identify, read_text}` | yes | 10 s | PointFollow → crop → `describe()` (F10). |
| `look_around` | — | no | 8 s | Describe the scene (wide frame → `describe`). Harmless, so open to all. |
| `remember` | `fact`, `subject?` | yes | 1 s | Store a fact (F8). |
| `recall` | `query` | no | 1 s | Search facts + sightings. |
| `forget` | `query` | yes | 1 s | Delete matching facts. |
| `stop` | — | yes | 1 s | Cancel the current Command / Mimic, return to FaceTrack. |

Emotion context (F3): every 2 s while the conversation is active, and at each commit, the turn manager adds a compact note `[context] owner looks happy (0.8)` only when the dominant expression changed. The model adapts tone and may call `express`. No extra round trips.

## 4. Prompting (`prompts.py`)

The system prompt is assembled at session start and on reconnect:
1. Persona: name, short self-intro facts, voice/tone guidance, keep answers ≤ 2 sentences unless asked.
2. Rules: speaker tags; owner-only actions; say a filler before slow tools; never invent locations; use `recall` for "where is/was…".
3. Memory summary: the owner's name, ≤ 15 most recent/important facts, the last-known location of ≤ 10 objects.
4. Event conventions: `[event] …` and `[context] …` items are observations, not user speech.

## 5. Memory (F8)

SQLite file `data/memory.db` (stdlib `sqlite3`, WAL mode) **[arch]**:

```sql
facts(id INTEGER PK, subject TEXT, text TEXT, source TEXT, created REAL, last_used REAL)
sightings(id INTEGER PK, label TEXT, where_text TEXT, yaw_deg REAL, pitch_deg REAL, seen REAL)
people(id INTEGER PK, role TEXT CHECK(role IN ('owner','guest')), name TEXT, first_seen REAL)
```

- `recall(query)`: keyword match (`LIKE` + simple token overlap) across facts and the latest sighting per label, ranked by recency. Good enough for ~100s of rows **[arch]**. Shortcut: no embeddings; upgrade to RoboMemory or a vector index only if recall quality fails the acceptance test.
- The owner's name is stored on "remember that I'm X" (via `remember`, owner-only). A guest's name goes to `people` only for the session (not persisted) **[arch: privacy]**.
- Survives restart (F8 acceptance). A backup copy is written on clean shutdown.
