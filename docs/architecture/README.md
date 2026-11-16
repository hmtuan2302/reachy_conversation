# Reachy Mini Agentic Demo — Architecture

Date: 2026-11-16 · Source of requirements: [../brainstorming/2026-11-16-reachy-agentic-demo-design.md](../brainstorming/2026-11-16-reachy-agentic-demo-design.md)

"Production-ready" here means **show-ready**. The app must survive a 30+ minute live run, contain every failure, be observable, and be testable without the robot. It is still a single-laptop demo (see the requirements' non-goals). There is no cloud deployment, no multi-tenant support and no auto-update.

| Doc | What it covers |
|---|---|
| [01-overview.md](01-overview.md) | Principles, system context, runtime/thread model, module layout, data flow, key contracts |
| [02-voice-and-audio.md](02-voice-and-audio.md) | Mic → VAD → speaker gate → OpenAI Realtime → playback, barge-in, echo, sound events (F2, F7, F9) |
| [03-perception.md](03-perception.md) | Camera loop, CPU models, World State, cloud vision (F3, F4, F5, F6, F10) |
| [04-motion-and-attention.md](04-motion-and-attention.md) | Attention arbiter, behaviours, motion loop and safety (F4–F7, F10) |
| [05-agent-tools-memory.md](05-agent-tools-memory.md) | Tool catalog, owner-only permissions, prompts, memory store (F1, F5, F8, F9, F10) |
| [06-operations.md](06-operations.md) | Config, WSL2 setup, startup, fault handling, logging/metrics, operator console, testing, security |

Conventions: **[req]** = fixed by the requirements doc, **[arch]** = an architecture decision made here, **[verify]** = an assumption to confirm in the Step 0 spike.
