from typing import Literal, TypedDict

from agents.schemas import AgentResult
from orchestrator.schemas import (
    DirectAnswer,
    FinalAnswer,
    InvestigationPlan,
    ReevaluationDecision,
    RoutingDecision,
)


class InvestigationState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes during an investigation.
    
    total=False means not every field has to exist from the beginning and it gets added later.
    """

    # Original user request and project information.
    issue: str
    project_path: str
    synthesis_mode: Literal["never", "auto", "always"]

    # Decisions produced during orchestration.
    routing_decision: RoutingDecision
    reevaluation: ReevaluationDecision
    investigation_plan: InvestigationPlan

    # Specialist findings accumulated during the investigation.
    results: list[AgentResult]
    completed_agents: list[str]

    # Answer produced when the router chooses the direct path.
    direct_answer: DirectAnswer

    # Distinguishes a successful conclusion from an investigation
    # that must stop even though more investigation is still needed.
    termination_status: Literal[
        "resolved",
        "needs_more_information",
    ]

    # Final user facing result produced by the finalizer.
    final_answer: FinalAnswer