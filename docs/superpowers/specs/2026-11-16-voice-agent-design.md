# Voice agent, stage 1: speech → text → LLM answer (text out)

Small change on top of `transcribe`. Text-to-speech comes in a later stage.

## Intent
Speak a question; the agent answers it. This stage builds the pipeline **S2T → LLM** with a text boundary between stages, so **T2S** plugs in later without touching the first two. The job is answering questions: no tools, no robot.

## Decided by the user
- Pipeline approach (S2T, then LLM, then T2S), not speech-to-speech.
- LLM: DeepSeek API, key in `.env` as `DEEPSEEK_API_KEY`.
- Input: mic (and the existing sources). Output: text only for now.

## Spike (2026-11-16)
`GET https://api.deepseek.com/models` → `deepseek-flash`, `deepseek-v4-pro`. Streaming `POST /chat/completions` (OpenAI format) answered a Vietnamese question correctly: first token 0.35 s, done 0.5 s.

## Approaches (LLM stage)
- A. DeepSeek chat API over httpx, streaming (PICKED by the user). Fast, cheap, swappable model, function calling available later.
- B. Pollen HF server's built-in LLM (text in/out works, ~1.2 s): free, but a black box and tied to that server's TTS (no Vietnamese voice).
- C. HF Inference Providers: the token has no credits (402).

## Architecture
```
 source (--mic | --stdin | --file)
   → S2T   transcribe_stream()            existing, yields one sentence of text
   → Agent agent.answer(text)             new, yields answer text pieces (streamed)
   → Sink  terminal printer               new; T2S becomes a second sink later
```
- `src/reachy_demo/agent/llm.py`: `chat_stream(messages, *, api_key, model, base_url, transport=None) -> AsyncIterator[str]`. httpx, SSE parsing, 30 s timeout. HTTP/auth/network errors → `LLMError`. Provider details live only here.
- `src/reachy_demo/agent/agent.py`: `Agent(llm, system_prompt, max_turns=10)` with `answer(text) -> AsyncIterator[str]`. Keeps the conversation history (last 10 Q/A pairs) so follow-ups work ("and its population?"). It adds the answer to the history only when it completes.
- System prompt: answer in the user's language (vi/en), short (≤ 3 sentences unless asked), plain text with no markdown (it will be spoken later), and say so when unsure.
- `src/reachy_demo/agent/cli.py`: `reachy-demo agent (--mic | --stdin | --file PATH)`. Prints `you: <transcript>`, then `bot: <answer streamed>`.
- **End of turn** (user requirement: send to the LLM only when the user has finished speaking): the server decides, as in the reference app (its `REALTIME_VOICE_CONVERSATION.md` §5.2: "server VAD is the authority for conversational turns"). The server runs smart-turn detection: a spike with a sentence split by a 0.7 s and by a 1.2 s mid-sentence pause gave **one** turn each time ("The birch canoe slid on the smooth planks."), with `smart_status: "complete"` and a VAD decision in about 0.1 s. So each `conversation.item.input_audio_transcription.completed` (what `transcribe_stream` yields) is one complete user turn, and goes to the LLM once. No client-side turn assembly is needed; add it only if real use shows turns getting split.
- One answer at a time. Turns that complete while an answer is printing wait in order (asyncio queue). Barge-in comes with T2S.
- Fail small: an LLM error prints `bot: (error: …)` and listening continues. A missing `DEEPSEEK_API_KEY` or `HF_TOKEN` exits 2. S2T errors behave as in `transcribe`.
- Config via env: `DEEPSEEK_API_KEY` (required), `LLM_MODEL` (default `deepseek-chat`: non-thinking mode, fastest first token; `deepseek-flash`/`deepseek-v4-pro` think first, and their hidden reasoning is never printed), `LLM_BASE_URL` (default `https://api.deepseek.com`). Added to `.env.example`.

## Non-goals (this stage)
TTS, tools/function calling, long-term memory, robot actions, speaker gate. The `Agent` is where a tool loop goes later, using the 05-agent-tools-memory.md tool layer.

## Docs to update
New `docs/usage/agent.md` (linked from README). `docs/architecture/05` and `02` currently say the agent is the speech-to-speech Realtime model; I'll add a note that the pipeline approach is now chosen, from this stage onward.

## Tests
Fake DeepSeek via httpx MockTransport (SSE parsing, error → LLMError), Agent history and trimming, and pipeline order with fake S2T and LLM (queueing, error keeps listening). Then one manual run against the real API and `--file`.

## Risks
- The API key was pasted in chat; rotate it at platform.deepseek.com when convenient.
- Latency: turn end → text (~0.1–0.5 s) + first DeepSeek token (~0.4 s) ≈ 1 s before the answer starts.
- Smart-turn behavior was measured on English. Vietnamese turn-taking is untested.
