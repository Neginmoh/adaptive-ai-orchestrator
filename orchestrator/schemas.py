from typing import Literal

from pydantic import BaseModel, Field




class RoutingDecision(BaseModel):
    # What overall routing strategy are we using
    route: Literal[
        "direct",
        "single_specialist",
        "multi_specialist",
    ]

    # Which actual agent(s) should be called
    selected_agents: list[
        Literal[
            "data_evaluation",
            "model_code",
        ]
    ]

    # This tells the system whether it needs to inspect the user's codebase/project files.
    requires_project_access: bool = Field(
        description="True when resolving the issue requires inspecting actual project files, source code, datasets, logs, configurations, or experiment results. False when the issue can be answered from the description alone."
    )

    reason: str

class ReevaluationDecision(BaseModel):
    """
    Decide what the orchestrator should do after reviewing
    results from the specialists that have already run.
    
    stop
    enough evidence; investigation can finish

    call_specialist
    new evidence shows another specialist is needed

    replan
    findings changed the problem enough that the investigation
    strategy needs to be reconsidered
    """

    action: Literal[
        "stop",
        "call_specialist",
        "replan",
    ]

    selected_agents: list[
        Literal[
            "data_evaluation",
            "model_code",
        ]
    ]

    reason: str



class InvestigationPlan(BaseModel):
    """
    A revised investigation plan created after new evidence changes
    the understanding of the problem.
    """

    selected_agents: list[
        Literal[
            "data_evaluation",
            "model_code",
        ]
    ]

    objective: str = Field(
        description=(
            "The specific investigation objective that should guide "
            "the specialists selected by this revised plan."
        )
    )

    reason: str

class DirectAnswer(BaseModel):
    """
    An answer produced directly by the orchestrator when no
    specialist investigation is required.
    """

    answer: str



class FinalAnswer(BaseModel):
    """
    Final user-facing answer produced after the investigation is complete.
    """

    status: Literal[
        "resolved",
        "needs_more_information",
    ]

    answer: str

    evidence: list[str]


class SynthesisResult(BaseModel):
    """
    Combined conclusion produced by comparing specialist findings.
    """

    status: Literal[
        "resolved",
        "needs_more_information",
    ]
    answer: str