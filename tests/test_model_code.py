from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents import model_code
from agents.schemas import AgentResult
from orchestrator import dispatcher


@pytest.mark.integration
def test_selected_file_reaches_analyzer(monkeypatch, tmp_path):
    # Create a real file in a temporary project directory.
    source_code = "learning_rate = 0.01\n"
    source_file = tmp_path / "train.py"
    source_file.write_text(source_code, encoding="utf-8")

    # Make the file selector choose that file.
    fake_selector = Mock()
    fake_selector.invoke.return_value = SimpleNamespace(
        files=["train.py"],
    )

    # Prepare the analyzer's response.
    expected_result = AgentResult(
        agent="model_code",
        status="needs_more_information",
        summary="More training code is needed to investigate.",
        evidence=[],
    )

    fake_analyzer = Mock()
    fake_analyzer.invoke.return_value = expected_result

    monkeypatch.setattr(model_code, "file_selector", fake_selector)
    monkeypatch.setattr(model_code, "code_analyzer", fake_analyzer)

    # Run the real dispatcher and specialist.
    results = dispatcher.run_selected_agents(
        issue="Training is unstable",
        project_path=str(tmp_path),
        selected_agents=["model_code"],
    )

    # Check that the selector received the available filename.
    fake_selector.invoke.assert_called_once()
    selection_messages = fake_selector.invoke.call_args.args[0]
    assert "train.py" in selection_messages[1][1]

    # Check that the analyzer received the actual file contents.
    fake_analyzer.invoke.assert_called_once()
    analysis_messages = fake_analyzer.invoke.call_args.args[0]
    assert "File: train.py" in analysis_messages[1][1]
    assert source_code in analysis_messages[1][1]

    # Check that the result reached the dispatcher output.
    assert results == [expected_result]


@pytest.mark.live
def test_model_code_returns_structured_result(tmp_path):
    source_file = tmp_path / "train.py"
    source_file.write_text(
        "import torch\n"
        "\n"
        "model = torch.nn.Linear(2, 1)\n"
        "optimizer = torch.optim.SGD(model.parameters(), lr=0.01)\n",
        encoding="utf-8",
    )

    results = dispatcher.run_selected_agents(
        issue="Identify the model architecture and optimizer.",
        project_path=str(tmp_path),
        selected_agents=["model_code"],
    )

    assert len(results) == 1

    result = results[0]
    assert isinstance(result, AgentResult)
    assert result.agent == "model_code"
    assert result.summary.strip()