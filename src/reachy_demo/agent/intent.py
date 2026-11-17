"""Did the user, speaking while Reachy talks, ask it to stop? One LLM call decides."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from dataclasses import dataclass

from reachy_demo.agent.agent import ChatFn
from reachy_demo.agent.llm import LLMError

log = logging.getLogger(__name__)

PROMPT = (
    "A voice assistant robot is talking. While it talks, the user said the text below "
    "(from speech recognition, Vietnamese or English, may contain mistakes). "
    "Decide whether the user wants the robot to stop talking now, in any phrasing "
    '(e.g. "stop", "wait", "enough", "be quiet", "dừng lại", "thôi", "im đi"). '
    "Background talk, fillers, agreement or a question without a stop request are NOT a stop. "
    'If it is a stop, "rest" is any new request said after the stop command, else "". '
    'Reply with JSON only: {"stop": true or false, "rest": "..."}'
)


@dataclass(frozen=True)
class Intent:
    stop: bool
    rest: str


NOT_STOP = Intent(False, "")

# A turn that is only a stop phrase ("Stop, stop!", "Thôi đi", "Dừng lại.") skips the LLM,
# saving its 0.4-1.3 s while Reachy keeps talking. Anything longer still goes to the LLM.
_STOP = r"(?:stop|wait|enough|thoi|dung lai|dung|du roi|im di|im lang)"
_FILLER = r"(?:ok|okay|please|di|nhe|nha|reachy)"
_BARE_STOP = re.compile(rf"(?:{_FILLER} )*{_STOP}(?: (?:{_STOP}|{_FILLER}))*")


def _fold(text: str) -> str:
    """Lowercase, no accents, letters only: "Thôi, dừng lại!" -> "thoi dung lai"."""
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    plain = "".join(c for c in decomposed if not unicodedata.combining(c)).lower()
    return " ".join(re.sub(r"[^a-z]+", " ", plain).split())


def is_bare_stop(text: str) -> bool:
    return _BARE_STOP.fullmatch(_fold(text)) is not None


async def classify(text: str, chat: ChatFn, timeout_s: float = 2.0) -> Intent:
    """Classify `text`. Any failure is NOT_STOP: never cancel an answer by mistake."""
    if is_bare_stop(text):
        return Intent(True, "")
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": text}]
    try:
        async with asyncio.timeout(timeout_s):
            reply = "".join([piece async for piece in chat(messages)])
        data = json.loads(reply.strip().removeprefix("```json").strip("`").strip())
        stop, rest = data["stop"], data.get("rest") or ""
        if not isinstance(stop, bool) or not isinstance(rest, str):
            raise ValueError(f"bad types: {data!r}")
    except (LLMError, TimeoutError, ValueError, KeyError, TypeError) as e:
        log.warning("stop-intent classify failed: %s", e)
        return NOT_STOP
    return Intent(stop, rest.strip()) if stop else NOT_STOP
