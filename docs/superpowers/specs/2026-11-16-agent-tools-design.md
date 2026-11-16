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
- **Tool loop:** messages = system (+ memory block) + history + user. Stream: text pieces are yielded to the terminal; collected `tool_calls` are run **one at a time, in order** (changed 2026-11-16: parallel head moves would fight over the one head pose), then `assistant{tool_calls}` + `tool{tool_call_id, content=json}` messages are appended and the loop asks again. Max 4 rounds, then a short apology. History stores the user turn, the tool exchanges and the final answer (trimmed to the last 10 turns as whole turns).
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

## Update 2026-11-16: head tools pulled forward
`turn_head` / `tilt_head` / `roll_head` (`agent/tools/head.py`) shipped before phase 1, behind `reachy-demo agent --robot`. Each tool sets one axis to a fixed angle (yaw ±20°, pitch ±15°, roll ±15°, or 0) and keeps the other axes. There is no generic `Registry` yet: `Agent` takes any object with `call(name, arguments)`, and `HeadTools` validates its own arguments. Build the registry when the second tool group (memory, clock) arrives. Moves go through `robot/adapter.py` `goto_target` directly. The motion loop and arbiter (01-overview rule 2) replace that at the robot milestone.

## Update 2026-11-16: head queue, silent moves, stop
- **Queue:** `HeadTools.call` validates, plans the target from the pose after all queued moves, puts it on an `asyncio.Queue` (max 4, else `busy`) and returns `started …` at once. One worker task runs the moves in order through `asyncio.to_thread(goto)`, so the head never gets two moves at once while the agent and TTS go on. Measured with a fake 1 s head: a joke's text was ready 1 s sooner, during the move.
- **Silent moves:** when every call in a round is a move without `reply: true` and none returned an error, the agent ends the turn with no follow-up LLM call (`Tools.needs_response(name, arguments)`). `reply: true` asks for the follow-up, for mixed requests like "turn left and tell me a joke". Without the flag, DeepSeek dropped the joke in 6 of 6 runs.
- **Stop:** the barge-in stop calls `converse(on_stop=tools.stop)`, and a `stop_head` tool covers "stop" when no answer is running. Both drop queued moves; the move in progress finishes.
- **Failures** happen after the tool returned, so the model can't report them: the worker drops the queue, resets the planned pose and prints `(head error: …)`.
- **Shutdown:** `HeadTools.close()` waits for the queue (5 s cap) inside the event loop, before `connect_head` sends the head to neutral, so the neutral move never overlaps a queued one.

## Update 2026-11-16: one motion loop, dances, busy rules
Replaces the `HeadTools` worker queue above (which called blocking `goto_target`).
- **`robot/motion.py` `Motion`** is the only sender of head poses (01-overview rule 1): one queue of `Move(name, long, duration_s, pose_at(t, start), end)`, one asyncio loop playing the current move by calling `set_target` at 50 Hz through `robot/adapter.py`. No SDK import. Every pose is clamped (yaw ±25°, pitch ±20°, roll ±20°).
- **Moves:** `goto` (short, 1 s, smoothstep to a fixed target) and `dance` (long; `bob`, `sway`, 4 s per pass, eases in from the start pose and out to neutral; `repeat` clamped so a dance is ≤ 15 s). Two built-in dances instead of `reachy_mini_dances_library`, so no new dependency; the library's moves can become `Move`s later.
- **Rules:** while a dance runs or waits, every other move is refused with `busy: dancing, say stop first`; the error triggers the follow-up LLM call, so Reachy says it. Stop is always accepted: queued moves are dropped, a dance ends on the next tick and the head eases to neutral (0.5 s), a short move in progress finishes so it ends on its fixed angle. The dance slot frees at once, so `stop_head` + `dance` in one reply works.
- **Edge cases handled:** (1) a crash in the loop clears the queue and the busy state in `finally`, reports `(head error: …)`, and the next `submit` restarts the loop. (2) `converse(moving=…)`: while the robot moves and no answer runs, each turn goes through the fast stop classifier first, so "thôi"/"enough" stops the dance without relying on the agent calling `stop_head`. (3) Moves start from the last pose sent (the loop is the only sender, so that is the real pose), and a stopped dance eases back to neutral. (4) dance length capped. (5) `close()` ends whatever runs at once, eases to neutral within 3 s, then the adapter goes neutral and disables the motors. On startup the adapter eases to neutral so the loop starts from a known pose.
- Live check (DeepSeek, fake robot): "dance and tell me a joke" → joke during the dance; "turn left" while dancing → refused and explained aloud; "stop dancing" → stopped; "turn left" → accepted.

## Update 2026-11-16: dances from reachy_mini_dances_library
The user asked for the reference `tools/dance.py` behaviour: `reachy-mini-dances-library>=0.2.1` is now in the `robot` extra (approved), replacing our two built-in dances.
- `robot/dances.py` `load_dances()` wraps each `AVAILABLE_MOVES` entry as `Dance(description, pass_s, offset(t))`: the library's move function is evaluated at `t * 114 / 60` beats (its `DanceMove.default_bpm`) and converted to mm and degrees. It calls the move functions directly, not `DanceMove`, because `DanceMove.__init__` mutates the library's shared parameter dicts. No library → `{}` and no `dance` tool.
- The motion loop's pose grew from yaw/pitch/roll to the full set: x/y/z (mm), roll/pitch/yaw, two antenna joints (degrees, `set_target` order). `adapter.set_pose(pose)` sends head + antennas. Neutral antennas are the SDK's `INIT_ANTENNAS_JOINT_POSITIONS` (±10°).
- Clamp limits raised to the widest range the library uses (sampled: yaw 45°, pitch 30°, roll 25°, x 20 mm, y/z 40 mm, antennas 60° + 10°), so moves play unclipped; a test samples all 20 moves against them. Head tools keep their fixed ±15–20° angles.
- `dance` tool, as in the reference: `move` enum with each move's description, omit for random, `repeat`; plus our `reply` flag. Length cap 15 s.
- Live check (DeepSeek, fake robot): "dance and tell me a joke" → `groovy_sway_and_roll`, reply true, joke during the dance; "do a dizzy dance three times" → `dizzy_spin` repeat 3 (6.3 s); "nhảy đi" → `side_to_side_sway`.
