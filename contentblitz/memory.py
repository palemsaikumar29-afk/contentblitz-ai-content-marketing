"""Conversation memory — context preservation across turns.

Each chat session gets its own :class:`ConversationMemory`, holding the
turn history plus the evolving content brief. The router and all agents
read from it so multi-turn conversations stay coherent ("make it
shorter", "now turn that into a LinkedIn post", ...).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    ts: float = field(default_factory=time.time)


@dataclass
class ConversationMemory:
    """Session-scoped memory: turns + working brief + last outputs."""

    turns: list[Turn] = field(default_factory=list)
    brief: dict = field(default_factory=dict)
    last_outputs: dict = field(default_factory=dict)  # {"blog": ..., "linkedin": ...}
    last_research: list = field(default_factory=list)

    # -- turns ----------------------------------------------------------
    def add_turn(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))

    def history(self, n: int | None = None) -> list[dict]:
        turns = self.turns if n is None else self.turns[-n:]
        return [{"role": t.role, "content": t.content} for t in turns]

    def history_text(self, n: int = 6) -> str:
        """Compact transcript for LLM context windows."""
        lines = [
            f"{t.role.upper()}: {t.content[:600]}" for t in self.turns[-n:]
        ]
        return "\n".join(lines)

    # -- brief ----------------------------------------------------------
    def update_brief(self, **fields) -> dict:
        self.brief.update({k: v for k, v in fields.items() if v is not None})
        return dict(self.brief)

    # -- outputs --------------------------------------------------------
    def remember_output(self, kind: str, content: str) -> None:
        self.last_outputs[kind] = content

    def last_output(self, kind: str) -> str | None:
        return self.last_outputs.get(kind)

    def remember_research(self, results: list[dict]) -> None:
        self.last_research = results

    def clear(self) -> None:
        self.turns.clear()
        self.brief.clear()
        self.last_outputs.clear()
        self.last_research = []

    def __len__(self) -> int:
        return len(self.turns)
