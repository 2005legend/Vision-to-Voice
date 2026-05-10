from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List


@dataclass
class BoardStep:
    id: int
    text: str


@dataclass
class BoardState:
    topic: str
    board_steps: List[BoardStep] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({
            "topic": self.topic,
            "board_steps": [{"id": s.id, "text": s.text} for s in self.board_steps],
            "equations": self.equations,
        })

    @classmethod
    def from_json(cls, data: str) -> "BoardState":
        d = json.loads(data)
        return cls(
            topic=d["topic"],
            board_steps=[BoardStep(id=s["id"], text=s["text"]) for s in d["board_steps"]],
            equations=d["equations"],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoardState):
            return False
        return (
            self.topic == other.topic
            and self.board_steps == other.board_steps
            and self.equations == other.equations
        )


@dataclass
class ChangeDelta:
    added_steps: List[BoardStep]
    changed_topic: str | None
    added_equations: List[str]


@dataclass
class StudentProfile:
    grade_level: int
    skip_count: int = 0
    replay_count: int = 0
    interrupt_count: int = 0
    repeat_count: int = 0
    followup_count: int = 0
    explanation_count: int = 0
    silence_duration: float = 0.0
    preferred_detail: str = "medium"   # "brief" | "medium" | "detailed" | "slow"

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, data: str) -> "StudentProfile":
        d = json.loads(data)
        return cls(**d)
