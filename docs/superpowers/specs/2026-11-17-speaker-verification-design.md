# Speaker verification and owner mode: design

Status: approved by the owner on 2026-11-17. Size: small feature (short design).

## Intent

Reachy should obey only its owner when the owner asks it to. The owner says "only follow my command" to turn **owner mode** on, and "stop following only me" (any wording) to turn it off. While owner mode is on, each spoken turn is checked against the owner's voiceprint. When owner mode is off, Reachy behaves as it does today: everyone counts as the owner.

A face check for unsure voice matches comes later (a separate feature). Until then, an unsure match makes Reachy ask the speaker to say it again.

## Goals

- Enroll the owner from a WAV file: `reachy-demo enroll --file data/my_voice1.wav` (12.8 s, 48 kHz mono).
- Score each spoken turn against the voiceprint on CPU, fast enough not to delay the answer (target under 100 ms).
- Owner mode on:
  - Guest turns are dropped before the LLM sees them. There is no reply, only a log line.
  - Unsure turns get the fixed reply "I'm not sure that's you, please say it again". They do not reach the LLM.
  - `dance` and `look_toward_sound` refuse non-owners in code with `not_owner`.
- Only a verified owner can turn owner mode on or off.
- Every score is printed to the log, so the thresholds can be tuned.

## Non-goals

- Face recognition fallback (next feature).
- Using the real verdict while owner mode is off. Memory and voice tools keep treating everyone as the owner then, as today.
- Remembering owner mode across restarts. It always starts off.
- Several enrolled people, or naming guests.

## Anti-goals

- Owner mode must never lock the owner out. Unsure asks again; it never ignores silently.
- Must not add PyTorch or other large dependencies.

## Approaches

1. **sherpa-onnx + 3D-Speaker CAM++ ONNX (chosen).** Small pip wheel with no PyTorch. It does the fbank features and the embedding. CAM++ (`3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx`, about 28 MB, about 7M params) runs in about 30 ms per sentence on CPU. The project already lists `sherpa-onnx` as the `speaker` extra.
2. **onnxruntime + the same ONNX, hand-written fbank.** Saves one dependency, but a small feature mismatch quietly ruins accuracy. Rejected.
3. **3D-Speaker repo or SpeechBrain ECAPA (PyTorch).** About 2 GB installed and a slow start on a CPU laptop. Rejected.

The owner picked 1.

## Design

- **Model:** CAM++ ONNX in `models/`, path from `SPEAKER_MODEL`. Audio is resampled from 24 kHz (the app's rate) or the WAV's rate to 16 kHz.
- **Voiceprint:** `enroll` saves the mean L2-normalised embedding to `data/voice/owner.npy`. It also prints a self-score (e.g. first half vs second half of the clip) to help pick thresholds.
- **Per-turn scoring:** the mic tap (`_tap` in `agent/cli.py`) also feeds a speaker buffer that keeps speech audio (VAD speaking) since the last turn. When the transcript of a turn arrives, that audio is scored once and the buffer is cleared. Verdict:
  - cosine ≥ `SPEAKER_HIGH`: owner
  - cosine < `SPEAKER_LOW`: guest
  - between the two, or under 1 s of speech: unsure
- **Typed input (`--type`):** no audio, so turns count as the owner.
- **Gate:** a wrapper around the turn stream, placed before `converse`. While owner mode is on, it drops guest turns and handles unsure turns. Because it sits before `converse`, a guest saying "stop" can't cancel the owner's answer either.
- **Speaker hook:** `speaker()` returns `OWNER` when owner mode is off, otherwise the last turn's verdict. Head and sound tools take it the way memory and voice tools already do (`speaker=lambda: OWNER` today).
- **Mode switch:** an `owner_mode(on: bool)` tool. The LLM calls it on "only follow my command" or "stop following only me". Turning it on needs a voiceprint and a model; otherwise it returns `not_enrolled`. The tool refuses unless the current turn is a verified owner.
- **Config:** `SPEAKER_MODEL`, `SPEAKER_HIGH`, `SPEAKER_LOW`. The builder sets the defaults from real scores of `my_voice1.wav` vs other voices.
- **Failure:** if sherpa-onnx or the model is missing, the agent still runs, and `owner_mode` returns an error telling the user to run `enroll`.

## Left to the builder

- Default thresholds (measure them).
- Whether scoring runs in a thread (`asyncio.to_thread`) or inline, after measuring its cost.
- Timing edge: the transcript may arrive before VAD's hangover closes the last segment. Include the audio up to that moment and check on real recordings.
- Docs: a FEATURES.md row and a usage doc section.
