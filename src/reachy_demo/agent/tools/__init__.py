"""Tools the agent can call."""

from __future__ import annotations

from typing import Any, Protocol


class ToolSet(Protocol):
    def specs(self) -> list[dict[str, Any]]: ...

    def needs_response(self, name: str, arguments: str) -> bool: ...

    async def call(self, name: str, arguments: str) -> dict[str, Any]: ...


class Toolbox:
    """Several tool sets as one: each call goes to the set that offers that tool name."""

    def __init__(self, *sets: ToolSet) -> None:
        self._specs = [spec for s in sets for spec in s.specs()]
        self._owner: dict[str, ToolSet] = {}
        for s in sets:
            for spec in s.specs():
                name = spec["function"]["name"]
                if name in self._owner:
                    raise ValueError(f"duplicate tool name: {name}")
                self._owner[name] = s

    def specs(self) -> list[dict[str, Any]]:
        return list(self._specs)

    def needs_response(self, name: str, arguments: str) -> bool:
        owner = self._owner.get(name)
        return owner is None or owner.needs_response(name, arguments)

    async def call(self, name: str, arguments: str) -> dict[str, Any]:
        owner = self._owner.get(name)
        if owner is None:
            return {"error": f"unknown_tool: {name}"}
        return await owner.call(name, arguments)
