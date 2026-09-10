from agents.data_evaluation import run_data_evaluation
from agents.model_code import run_model_code
from agents.schemas import AgentResult


def run_selected_agents(
    issue: str,
    project_path: str,
    selected_agents: list[str],
    objective: str | None = None,
) -> list[AgentResult]:
    """
    Dispatch the selected specialist agents by the orchestrator to their 
    corresponding functions, collect and return their results.

    Args:
        issue: The ML issue to investigate.
        project_path: Path to the ML project being inspected.
        selected_agents: list of names of the specialist agents to run.
        objective: Optional investigation objective for the specialists.
            can be either a string or None. if nothing is provided, it starts with None as a default value.

    Returns:
        A list of AgentResult objects produced by the selected specialists.
    """
    # Store the result returned by each selected specialist.
    results = []

    for agent_name in selected_agents:

        if agent_name == "data_evaluation":

            result = run_data_evaluation(
                issue=issue,
                project_path=project_path,
                objective=objective,
            )

        elif agent_name == "model_code":

            result = run_model_code(
                issue=issue,
                project_path=project_path,
                objective=objective,
            )
       
        else:
            # Raise an error if an unsupported agent name is provided.
            # RoutingDecision, ReevaluationDecision and InvestigationPlan already restrict the allowed agent names. 
            raise ValueError(f"Unknown agent: {agent_name}")

        results.append(result)

    # Return the list of AgentResult objects produced by specialist agents is returned.
    return results