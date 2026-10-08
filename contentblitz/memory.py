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
    """Session-scoped memory: turns + working brief + last outputs.

    ``last_topic`` is the real topic the most recent draft was written
    about. Follow-up turns ("make it punchier", "turn that into a LinkedIn
    post") carry it forward so the follow-up text never becomes the topic.
    """

    turns: list[Turn] = field(default_factory=list)
    brief: dict = field(default_factory=dict)
    last_outputs: dict = field(default_factory=dict)  # {"blog": ..., "linkedin": ...}
    last_research: list = field(default_factory=list)
    last_topic: str = ""  # real topic of the latest draft (follow-up anchor)

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

    def last_draft(self, kind: str | None = None) -> tuple[str | None, str | None]:
        """Most recent draft as (kind, content).

        With ``kind`` set, returns that draft if it exists, else (None, None).
        Without it, returns the latest draft across all kinds.
        """
        if kind:
            draft = self.last_outputs.get(kind)
            return (kind, draft) if draft else (None, None)
        if not self.last_outputs:
            return None, None
        latest = next(reversed(self.last_outputs.keys()))
        return latest, self.last_outputs[latest]

    def remember_topic(self, topic: str | None) -> None:
        """Anchor the real topic so follow-ups can't overwrite it."""
        if topic and topic.strip():
            self.last_topic = topic.strip()

    def remember_research(self, results: list[dict]) -> None:
        self.last_research = results

    def clear(self) -> None:
        self.turns.clear()
        self.brief.clear()
        self.last_outputs.clear()
        self.last_research = []
        self.last_topic = ""

    def __len__(self) -> int:
        return len(self.turns)
