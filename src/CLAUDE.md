# CLAUDE.md — Reachy Mini Agentic Demo

Single-process Python 3.12 app that drives a **Reachy Mini Wireless** over Wi‑Fi from a CPU-only WSL2 laptop. It is a **live demo**, so reliability on stage beats polish. Ten features, built in this order: greeting → realtime voice → emotion-aware → tracking/attention → find object → mimic → sound DoA → memory → owner-only voice → point-and-recognize.

## Read first (source of truth)

- Requirements, MVP/future split, acceptance criteria, risks: [docs/brainstorming/2026-11-16-reachy-agentic-demo-design.md](docs/brainstorming/2026-11-16-reachy-agentic-demo-design.md)
- Architecture: [docs/architecture/README.md](docs/architecture/README.md) → `01-overview` … `06-operations`. Tags there: **[req]** fixed, **[arch]** decided, **[verify]** unconfirmed (check before relying on it).
- Reference app (ideas only, don't copy wholesale): `../opensource/apps/reachy_mini_conversation_app`. Robot SDK lives in `~/reachy_mini/reachy_mini_env/.../reachy_mini`.

If code and docs disagree, stop and ask; don't silently pick one. Update the doc when a decision changes.

## Commands

```bash
uv sync                       # core + dev deps (extras: --extra robot|perception|speaker)
uv run ruff format . && uv run ruff check .
uv run mypy                   # strict
uv run pytest                 # unit + integration (no robot needed)
uv run pytest -m "not integration"
uv run reachy-demo check-config
```

Done = ruff, mypy and pytest all green. `unset VIRTUAL_ENV` if uv warns about a conda env.

## Layout (`reachy_demo/`)

`config.py` env → frozen `Config` · `ops/logging.py` JSONL + trace id · `robot/` (only SDK import) · `audio/` · `voice/` (only Realtime import) · `perception/` (only vision-API import) · `attention/` · `agent/` · `memory/` · `ops/`. Tests mirror it under `tests/{unit,integration,replay}`; `models/` and `data/` are git-ignored. Currently **scaffold only**: config, logging, CLI `check-config`. Feature modules are empty packages awaiting their milestone (06-operations §9).

## Architecture rules (do not break)

1. **One owner per resource**: one component each for mic, speaker, camera, head. Only `motion` calls `set_target()`; only `robot/adapter.py` imports `reachy_mini`.
2. **One behaviour owns the head**, chosen by the arbiter (priority: Command > Investigate > PointFollow > Mimic > FaceTrack > Idle). Expressive moves are bounded offsets, never owners.
3. **Permissions are enforced in code**: tools marked `owner_only` refuse guests. The prompt is a courtesy, not the control. Guests can chat.
4. **Fail small**: workers run under a supervisor, and a dead model degrades one feature, not the conversation. Stale perception data (>500 ms) counts as absent.
5. **Latest-value world**: perception writes `WorldState`, readers take snapshots. Events only for things that happen.
6. Models run only while leased by a behaviour/tool (CPU budget).
7. Server turn detection is off. The app decides turn end, tags the speaker, then commits.

## Engineering standards

- SOLID where it earns its keep; simple over clever; no abstraction without a second user. Business logic stays free of SDK/network imports, which lets tests use fakes.
- Inject dependencies (robot, realtime client, clock) through constructors; no module-level singletons.
- **No hardcoded credentials.** Secrets come from `.env` (git-ignored; template in `.env.example`) and are masked in logs. `Config.openai_api_key` is excluded from `repr`.
- Log with `logging.getLogger(__name__)`, structured fields via `extra={...}`. No `print` outside the CLI. Set `trace_id` per user turn.
- Type everything (mypy strict). Third-party libs without stubs are allow-listed in `pyproject.toml`.
- Write a test for every non-trivial branch (arbiter, permissions, turn state machine, onset detector, memory). Hardware/network tests get `@pytest.mark.integration`.
- Shortcuts with a known limit get a comment: `shortcut: <limit>, <when to upgrade>`.

## Gotchas

- Under WSL2 connect to the robot by **IP** with mirrored networking; mDNS and WebRTC may fail (R12). The first task of M0 is the connection spike.
- Direction of arrival: 0 = left, π/2 = front, π = right; front half-plane only [verify mapping].
- Realtime event names are isolated in `voice/realtime.py`; confirm against current OpenAI docs before coding [verify].
- Audio is 16 kHz from the robot and 24 kHz for the Realtime API, so resample at the boundary.
- Always return the robot to neutral and disable motors on exit (`try/finally`).

## Workflow

Every user-facing feature ships with a usage doc in `docs/usage/<feature>.md`, linked from README.md.

Build one milestone at a time (M0…M10 in `06-operations.md` §9) and check it against its acceptance criteria before the next. Don't start a milestone's feature code without a failing test or a spike result. Ask before adding dependencies.
