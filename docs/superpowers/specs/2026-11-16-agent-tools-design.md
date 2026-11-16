# Agent tools: one agent, a tool registry, tools ported from the reference app

## Intent
Give `reachy-demo agent` tools. The user asked for the tools of `reachy_mini_conversation_app`. Of the three options (copy as reference, **port**, copy verbatim and run), they picked **port**: rewrite each tool against our own small registry, using the reference's logic and descriptions. This keeps the "own lean code, ideas not code wholesale" decision. Architecture: **one agent with all tools**, grouped by topic in code (decided in conversation 2026-11-16: ~12–20 tools, latency budget F1 ≤ 1.5 s, cross-topic context, one owner of the head).

## Spike (2026-11-16)
DeepSeek `deepseek-chat`, streaming, `tools` in OpenAI format: one Vietnamese turn produced two parallel calls, `get_time {"timezone":"Asia/Tokyo"}` and `remember {"fact":"Tên của người dùng là Tuấn."}`, `finish_reason: tool_calls`. Calls stream as `delta.tool_calls[{index, id, function{name, arguments-fragment}}]`.

## Port inventory (reference `tools/`, 16 files)

| Phase | Tools | Needs |
|---|---|---|
| **1 (this design)** | `remember`, `forget` (+ memory store, facts injected into the prompt), `get_time` (the reference's remote time tool, ported as local code with stdlib `zoneinfo`) | nothing new |
| 2 | `search_web`, `get_weather` | a web/weather source: the reference calls Pollen's HF Spaces over MCP (new `mcp` dependency) or a plain HTTP API. Decide then |
| 3 (robot milestone) | `dance`, `stop_dance`, `play_emotion`, `stop_emotion`, `move_head`, `sweep_look`, `head_tracking`, `idle_do_nothing`, `camera`, `volume_control`, `robot_status`, `go_to_sleep` | robot adapter, motion loop, camera (06-operations milestones) |
| with long-running tools | `task_status`, `task_cancel` | background task manager |

## Approaches (tool loop)
- A. **Own tool loop over the DeepSeek chat API (PICKED).** The agent streams; on `tool_calls` it runs them through the registry, appends results, and asks again (max 4 rounds), then streams the final text. Full control over permissions and timeouts. ~100 lines.
- B. An agent framework (OpenAI Agents SDK / LangGraph): new dependencies and hidden control flow, for a loop this small. No.
- C. The reference's background tool manager (tools run async, results injected later): needed for slow robot tools, not for phase 1. Revisit in phase 3.

## Design
```
src/reachy_demo/agent/
  registry.py   Tool(name, description, parameters, run, owner_only=False, timeout_s=5)
                Registry.specs() -> DeepSeek "tools" list; Registry.call(name, args_json, speaker) -> dict
  tools/
    memory.py   remember(fact), forget(query)           ← reference tools/remember.py, forget.py, memory.py
    clock.py    get_time(timezone?, compare_to?)         ← reference time tool
  memory.py     MemoryStore(path): add / forget / list / prompt_block()   JSON file data/memory.json
  llm.py        chat_stream(..., tools=None) now also yields ToolCall(id, name, arguments)
  agent.py      answer(): tool loop (below)
```
- **Registry.call** checks that the name is known, parses the JSON arguments, checks `required` keys and basic types, enforces `owner_only` against the turn's speaker (everyone is "owner" until the voice gate exists; 05 §2), and runs under `asyncio.wait_for(timeout_s)`. Errors become `{"error": "unknown_tool" | "bad_arguments: …" | "not_owner" | "timeout" | "internal"}`, so the model can recover and the agent never crashes on a tool.
- **Tool loop:** messages = system (+ memory block) + history + user. Stream: text pieces are yielded to the terminal; collected `tool_calls` are run concurrently, then `assistant{tool_calls}` + `tool{tool_call_id, content=json}` messages are appended and the loop asks again. Max 4 rounds, then a short apology. History stores the user turn, the tool exchanges and the final answer (trimmed to the last 10 turns as whole turns).
- **Terminal:** each tool call prints a dim line `  [tool] remember(fact="…") → saved`, so you see the agent working. TTS speaks only the final text.
- **Memory (my call, simpler than 05's SQLite):** port the reference: a JSON file of facts (id, text, created), deduplicated case-insensitively, max 50, atomic write; the facts go into the system prompt at each turn ("Known facts about the user: …"), so no `recall` tool is needed yet. 05's `sightings` table comes with `find_object` (phase 3). File: `data/memory.json` (git-ignored); env `MEMORY_PATH` overrides it.
- **get_time:** `{timezone?: IANA, compare_to?: IANA}` → local time when empty (machine zone), ISO time, weekday, and the offset difference when comparing. Bad zone → `{"error": "unknown timezone"}`.
- **Prompt:** add the reference's tool-use guidance: remember stable facts silently, don't say "I will remember that", don't save sensitive data, use get_time for any time question.

## Non-goals
Remote tools (phase 2), robot tools (phase 3), the background task manager, the speaker gate.

## Tests
Registry (unknown tool, bad args, owner_only refusal, timeout, exception → internal), the tool loop with a fake LLM (one round, two parallel calls, max-rounds stop, a tool error passed back), streamed tool-call parsing in llm.py (fragments joined by index), memory store (add/dedupe/forget/limit/corrupt file → empty + warning), get_time (fixed clock, comparison, bad zone). A manual run: "Mấy giờ rồi ở Tokyo? Nhớ là tôi tên Tuấn" → time answer + fact saved; restart → "Tôi tên gì?" → "Tuấn".

## Docs
`docs/usage/agent.md` (tools section and memory file), a new `docs/usage/tools.md` (catalog + how to add a tool), `docs/architecture/05` (registry and memory decisions), README.

## Licensing
Ported descriptions and logic come from an Apache-2.0 project: add `NOTICE` crediting `reachy_mini_conversation_app` (Pollen Robotics), and a header line in each ported file.
