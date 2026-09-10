from unittest.mock import Mock, call

import pytest

from agents.schemas import AgentResult
from orchestrator import graph
from orchestrator.schemas import (
    DirectAnswer,
    InvestigationPlan,
    ReevaluationDecision,
    RoutingDecision,
)


@pytest.mark.integration
def test_direct_route_returns_answer(monkeypatch):
    fake_router = Mock(return_value=RoutingDecision(
        route="direct",
        selected_agents=[],
        requires_project_access=False,
        reason="A conceptual question.",
    ))
    fake_answer = Mock(return_value=DirectAnswer(
        answer="Overfitting means poor generalization."
    ))
    fake_dispatcher = Mock()
    fake_reevaluation = Mock()

    monkeypatch.setattr(graph, "route_issue", fake_router)
    monkeypatch.setattr(graph, "answer_direct", fake_answer)
    monkeypatch.setattr(graph, "run_selected_agents", fake_dispatcher)
    monkeypatch.setattr(graph, "reevaluate_issue", fake_reevaluation)

    state = graph.build_graph().invoke({
        "issue": "What is overfitting?",
        "project_path": ".",
        "synthesis_mode": "never",
        "results": [],
        "completed_agents": [],
    })

    fake_router.assert_called_once_with("What is overfitting?")
    fake_answer.assert_called_once_with("What is overfitting?")
    fake_dispatcher.assert_not_called()
    fake_reevaluation.assert_not_called()

    assert state["final_answer"].status == "resolved"
    assert state["final_answer"].answer == (
        "Overfitting means poor generalization."
    )
    assert state["final_answer"].evidence == []


@pytest.mark.integration
@pytest.mark.parametrize(
    "action, requested_agent, expected_status",
    [
        ("stop", None, "resolved"),
        ("call_specialist", "data_evaluation", "resolved"),
        ("replan", "data_evaluation", "resolved"),
        ("call_specialist", "model_code", "needs_more_information"),
        ("replan", "model_code", "needs_more_information"),
    ],
)
def test_investigation_follows_decision(
    monkeypatch, action, requested_agent, expected_status
):
    issue = "Validation results are unreliable."

    model_result = AgentResult(
        agent="model_code",
        status="completed",
        summary="Training code inspected.",
        evidence=["Training uses the configured split."],
    )
    data_result = AgentResult(
        agent="data_evaluation",
        status="completed",
        summary="Data split inspected.",
        evidence=["Training and validation indices overlap."],
    )

    fake_router = Mock(return_value=RoutingDecision(
        route="single_specialist",
        selected_agents=["model_code"],
        requires_project_access=True,
        reason="Inspect training first.",
    ))

    requested_agents = (
        [] if requested_agent is None else [requested_agent]
    )

    first_decision = ReevaluationDecision(
        action=action,
        selected_agents=requested_agents,
        reason="Next investigation step.",
    )
    stop_decision = ReevaluationDecision(
        action="stop",
        selected_agents=[],
        reason="Enough evidence.",
    )

    fake_reevaluation = Mock(
        side_effect=[first_decision, stop_decision]
    )
    fake_dispatcher = Mock(
        side_effect=[[model_result], [data_result]]
    )

    plan = InvestigationPlan(
        selected_agents=requested_agents,
        objective="Trace the validation split.",
        reason="Follow the new evidence.",
    )
    fake_replan = Mock(return_value=plan)
    fake_direct = Mock()

    monkeypatch.setattr(graph, "route_issue", fake_router)
    monkeypatch.setattr(graph, "reevaluate_issue", fake_reevaluation)
    monkeypatch.setattr(graph, "run_selected_agents", fake_dispatcher)
    monkeypatch.setattr(graph, "replan_issue", fake_replan)
    monkeypatch.setattr(graph, "answer_direct", fake_direct)

    state = graph.build_graph().invoke({
        "issue": issue,
        "project_path": ".",
        "synthesis_mode": "never",
        "results": [],
        "completed_agents": [],
    })

    expected_calls = [
        call(
            issue=issue,
            project_path=".",
            selected_agents=["model_code"],
        )
    ]
    expected_results = [model_result]

    if requested_agent == "data_evaluation":
        follow_up_arguments = {
            "issue": issue,
            "project_path": ".",
            "selected_agents": ["data_evaluation"],
        }

        if action == "replan":
            follow_up_arguments["objective"] = plan.objective

        expected_calls.append(call(**follow_up_arguments))
        expected_results.append(data_result)

    assert fake_dispatcher.call_args_list == expected_calls
    fake_direct.assert_not_called()

    if action == "replan":
        fake_replan.assert_called_once_with(
            issue=issue,
            previous_results=model_result.model_dump_json(indent=2),
        )
    else:
        fake_replan.assert_not_called()

    expected_reevaluations = [
        call(
            issue=issue,
            previous_results=model_result.model_dump_json(indent=2),
        )
    ]

    if len(expected_results) == 2:
        expected_reevaluations.append(call(
            issue=issue,
            previous_results="\n\n".join(
                result.model_dump_json(indent=2)
                for result in expected_results
            ),
        ))

    assert (
        fake_reevaluation.call_args_list
        == expected_reevaluations
    )
    assert state["results"] == expected_results
    assert state["completed_agents"] == [
        result.agent for result in expected_results
    ]
    assert state["final_answer"].status == expected_status
    assert state["final_answer"].evidence == [
        evidence
        for result in expected_results
        for evidence in result.evidence
    ]