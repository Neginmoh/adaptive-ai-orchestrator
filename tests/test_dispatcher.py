import pytest

import orchestrator.dispatcher as dispatcher

from unittest.mock import Mock


@pytest.mark.unit
def test_unknown_agent_raises():
    with pytest.raises(ValueError, match="Unknown agent"):
        dispatcher.run_selected_agents(
            issue="Test issue",
            project_path=".",
            selected_agents=["unsupported_agent"],
        )

@pytest.mark.unit
def test_no_agents_returns_empty_list():
    results = dispatcher.run_selected_agents(
        issue="Test issue",
        project_path=".",
        selected_agents=[],
    )

    assert results == []



@pytest.mark.unit
@pytest.mark.parametrize(
    "agent_name, function" \
    "_name",
    [
        ("model_code", "run_model_code"),
        ("data_evaluation", "run_data_evaluation"),
    ],
)
def test_selected_agent_calls_specialist(monkeypatch, agent_name, function_name):
    fake_specialist = Mock(return_value={})
    monkeypatch.setattr(dispatcher, function_name, fake_specialist)

    results = dispatcher.run_selected_agents(
        issue="Test issue",
        project_path=".",
        selected_agents=[agent_name],
    )

    fake_specialist.assert_called_once_with(
        issue="Test issue",
        project_path=".",
        objective=None,
    )

    assert results == [{}]


@pytest.mark.unit
def test_multiple_agents_preserve_result_order(monkeypatch):
    data_result = {"agent": "data_evaluation"}
    model_result = {"agent": "model_code"}

    fake_data = Mock(return_value=data_result)
    fake_model = Mock(return_value=model_result)

    monkeypatch.setattr(dispatcher, "run_data_evaluation", fake_data)
    monkeypatch.setattr(dispatcher, "run_model_code", fake_model)

    results = dispatcher.run_selected_agents(
        issue="Test issue",
        project_path=".",
        selected_agents=["data_evaluation", "model_code"],
        objective="Check training quality",
    )

    fake_data.assert_called_once_with(
        issue="Test issue",
        project_path=".",
        objective="Check training quality",
    )

    fake_model.assert_called_once_with(
        issue="Test issue",
        project_path=".",
        objective="Check training quality",
    )

    assert results == [data_result, model_result]