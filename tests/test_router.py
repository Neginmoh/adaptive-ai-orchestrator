from unittest.mock import Mock

import pytest

from orchestrator import router
from orchestrator.schemas import (
    DirectAnswer,
    InvestigationPlan,
    ReevaluationDecision,
    RoutingDecision,
)


@pytest.mark.unit
def test_valid_issue_returns_routing_decision(monkeypatch):
    expected = RoutingDecision(
        route="single_specialist",
        selected_agents=["model_code"],
        requires_project_access=True,
        reason="The training code needs inspection.",
    )

    fake_model = Mock()
    fake_model.invoke.return_value = expected
    monkeypatch.setattr(router, "routing_model", fake_model)

    issue = "My training loop crashes."
    result = router.route_issue(issue)

    fake_model.invoke.assert_called_once_with([
        ("system", router.SYSTEM_PROMPT),
        ("human", issue),
    ])
    assert result is expected


@pytest.mark.unit
def test_findings_reach_reevaluation(monkeypatch):
    expected = ReevaluationDecision(
        action="stop",
        selected_agents=[],
        reason="The findings explain the issue.",
    )

    fake_model = Mock()
    fake_model.invoke.return_value = expected
    monkeypatch.setattr(router, "reevaluation_model", fake_model)

    issue = "Validation results are unreliable."
    findings = "Training and validation indices overlap."

    result = router.reevaluate_issue(issue, findings)

    fake_model.invoke.assert_called_once()
    human_message = fake_model.invoke.call_args.args[0][1][1]

    assert issue in human_message
    assert findings in human_message
    assert result is expected


@pytest.mark.unit
def test_replan_removes_duplicate_agents(monkeypatch):
    expected = InvestigationPlan(
        selected_agents=[
            "data_evaluation",
            "model_code",
            "data_evaluation",
        ],
        objective="Trace how the split is used during training.",
        reason="The connection remains unclear.",
    )

    fake_model = Mock()
    fake_model.invoke.return_value = expected
    monkeypatch.setattr(router, "replanning_model", fake_model)

    issue = "Validation results are unreliable."
    findings = "The split file exists, but its use is unclear."

    result = router.replan_issue(issue, findings)

    fake_model.invoke.assert_called_once()
    human_message = fake_model.invoke.call_args.args[0][1][1]

    assert issue in human_message
    assert findings in human_message
    assert result.selected_agents == ["data_evaluation", "model_code"]
    assert result.objective == "Trace how the split is used during training."
    assert result is expected


@pytest.mark.unit
def test_question_returns_direct_answer(monkeypatch):
    expected = DirectAnswer(
        answer="Overfitting occurs when a model fits training data "
        "but generalizes poorly."
    )

    fake_model = Mock()
    fake_model.invoke.return_value = expected
    monkeypatch.setattr(router, "direct_model", fake_model)

    issue = "What is overfitting?"
    result = router.answer_direct(issue)

    fake_model.invoke.assert_called_once_with([
        ("system", router.DIRECT_PROMPT),
        ("human", issue),
    ])
    assert result is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "function_name, model_name, extra_args",
    [
        ("route_issue", "routing_model", ()),
        ("reevaluate_issue", "reevaluation_model", ("Findings",)),
        ("answer_direct", "direct_model", ()),
    ],
)
def test_blank_issue_raises(
    monkeypatch, function_name, model_name, extra_args
):
    fake_model = Mock()
    monkeypatch.setattr(router, model_name, fake_model)

    function = getattr(router, function_name)

    with pytest.raises(ValueError, match="The issue cannot be empty"):
        function("   ", *extra_args)

    fake_model.invoke.assert_not_called()


@pytest.mark.live
def test_split_issue_selects_data_specialist():
    result = router.route_issue(
        "Inspect my project's train and validation split code "
        "to determine whether the same customer IDs appear in both sets. "
        "Focus only on data leakage in the split."
    )

    assert result.route == "single_specialist"
    assert result.selected_agents == ["data_evaluation"]
    assert result.requires_project_access is True