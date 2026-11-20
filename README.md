# reachy-demo

All features at a glance: [docs/FEATURES.md](docs/FEATURES.md)

Usage docs per feature: [docs/usage/](docs/usage/)

- [Live transcribe: speech → text (mic / WAV)](docs/usage/transcribe.md)
- [Voice agent: ask by voice, hear the answer (mic → DeepSeek → Gemini speech)](docs/usage/agent.md)
- [Speak: text → speech (Gemini, streamed, saved as WAV)](docs/usage/speak.md)
- [Sound direction: "what was that?", "turn to me" (robot mic array DoA)](docs/usage/sound-direction.md)
- [Face tracking: "track me", follows you with head and body until you say stop](docs/usage/face-tracking.md)
- [Object tracking: "track my iPhone", follows a phone, cup, bottle... until you say stop](docs/usage/object-tracking.md) · needs `uv sync --inexact --extra perception` ([setup](docs/usage/object-tracking.md#setup-once))
- [Find an object: "find my phone", looks around the room and says where it is](docs/usage/find-object.md)
- [Mimic: "mimic me", copies your head moves like a mirror](docs/usage/mimic.md) · needs `uv sync --inexact --extra perception` ([setup](docs/usage/mimic.md#setup-once))
- [Memory: Reachy remembers you across restarts](docs/usage/memory.md) · [hand tests](docs/usage/memory-testing.md) · search by meaning needs `uv sync --extra embed` ([setup](docs/usage/memory.md#first-time-setup-search-by-meaning))
- VLA dataset recording (`scripts/track_record.py`): [faces](docs/usage/track-record-face.md) · [objects](docs/usage/track-record.md)
- [VLA face labelling: auto-label recorded face episodes (`scripts/track_label.py`)](docs/usage/track-label.md) · needs `uv sync --inexact --extra perception` · needs `uv sync --inexact --extra robot --extra perception`
