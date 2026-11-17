"""Memory LLM prompts and a JSON-only call helper (one retry on bad JSON, never raises)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

log = logging.getLogger(__name__)

# Same shape as agent.agent.ChatFn; memory/ must not import agent/.
ChatFn = Callable[[list[dict[str, str]]], AsyncIterator[str]]

EXTRACT = """You extract long-term memories for a home robot from one conversation turn.
Today is {today}. The speaker is {speaker}.
Reply with JSON only:
{{"facts": [{{"subject": "...", "text": "...", "text_original": "...",
  "category": "bio|preference|relation|plan|setting|other",
  "valid_from": "YYYY-MM-DD or null", "valid_to": "YYYY-MM-DD or null",
  "importance": 0.0-1.0, "sensitive": true|false}}],
 "follow_ups": [{{"text": "...", "due": "YYYY-MM-DD or null"}}],
 "name": "speaker's own name or null"}}
Rules:
- Keep only what the speaker states about themselves, people and pets in their life, their things,
  plans and events, or how they want the robot to behave (category "setting").
- Each fact is one short sentence in English, third person, using {who} instead of I/me/tôi.
  "text_original" is the speaker's own words for it. "subject" is who the fact is about.
- Turn relative dates into absolute dates using today's date.
- If something changed ("I moved to Hanoi"), state the new fact only.
- Skip greetings, small talk, questions, requests to the robot, and anything the robot said.
- sensitive = true for health, money, passwords, ID numbers and exact addresses.
- follow_ups: things a friend would ask about later (an exam, a trip, being sick), as
  "Ask how ..." with the due date on the day it happens or ends.
- name: when the speaker says their own name or what to call them ("call me Tuan").
  Never also add it as a fact.
- A language they ask the robot to use is a fact with category "setting", e.g.
  "{who} wants answers in Vietnamese."
- Nothing worth keeping: {{"facts": [], "follow_ups": [], "name": null}}"""

DECIDE = """You maintain a robot's long-term memory. A new fact arrived. Compare it with the
existing facts (id: text) and reply with JSON only, one of:
{"action": "ADD"}  - new information
{"action": "NOOP"}  - already known with the same meaning
{"action": "UPDATE", "id": N, "text": "..."}  - the same fact with a new detail or better
  wording; text is the merged sentence
{"action": "INVALIDATE", "id": N}  - the new fact contradicts or replaces that one (it is no
  longer true, e.g. the person moved or changed their mind)
Use only ids from the list."""

SUMMARIZE = """Summarize this conversation between a home robot and {speaker} for the robot's
long-term memory. Today is {today}. Reply with JSON only:
{{"summary": "1-2 sentences in English, what was talked about or happened, with names",
 "topics": ["2-5 short topics"], "importance": 0.0-1.0,
 "follow_ups": [{{"text": "Ask how ...", "due": "YYYY-MM-DD or null"}}]}}
follow_ups: only open things worth asking about in a later conversation. Small talk only:
importance 0.1 and no follow-ups."""


def parse_json(reply: str) -> dict[str, Any]:
    text = reply.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the reply")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    return data


async def ask_json(
    chat: ChatFn, system: str, user: str, *, timeout_s: float = 10.0
) -> dict[str, Any] | None:
    """One JSON object from the LLM, or None. Bad JSON gets one retry with the error."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(2):
        try:
            async with asyncio.timeout(timeout_s):
                reply = "".join([piece async for piece in chat(messages)])
        except Exception as e:  # LLM/network/timeout: memory fails small
            log.warning("memory LLM call failed: %s", e)
            return None
        try:
            return parse_json(reply)
        except ValueError as e:  # json.JSONDecodeError is a ValueError
            log.warning("memory LLM sent bad JSON (attempt %d): %s", attempt + 1, e)
            messages += [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": f"That was not valid JSON ({e}). Reply with the JSON object only.",
                },
            ]
    return None
