from typing import Literal

from pydantic import BaseModel


class AgentResult(BaseModel):
    """
    Structured result returned by a specialist agent after an investigation.
    It creates a standard result format that both specialist agents must follow when they report back to the orchestrator.
    """
    agent: Literal[
        "data_evaluation",
        "model_code",
    ]

    status: Literal[
        "completed",
        "needs_more_information",
        "failed",
    ]

    summary: str
    evidence: list[str]

class FileSelection(BaseModel):
    """
    Files selected by an agent for closer inspection.
    """

    files: list[str]
    reason: str