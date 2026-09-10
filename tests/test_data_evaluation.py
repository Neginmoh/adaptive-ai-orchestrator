from unittest.mock import Mock

import pytest

from agents import data_evaluation
from agents.schemas import AgentResult, FileSelection


@pytest.mark.unit
def test_valid_selection_returns_analysis(monkeypatch):
    file_listing = Mock(return_value=["data_split.py"])
    file_reader = Mock(return_value="validation_fraction = 0.2\n")

    fake_selector = Mock()
    fake_selector.invoke.return_value = FileSelection(
        files=["data_split.py", "missing.py"],
        reason="Inspect the data split.",
    )

    expected_result = AgentResult(
        agent="data_evaluation",
        status="needs_more_information",
        summary="More code is needed to assess the data split.",
        evidence=[],
    )

    fake_analyzer = Mock()
    fake_analyzer.invoke.return_value = expected_result

    monkeypatch.setattr(
        data_evaluation, "list_project_files", file_listing
    )
    monkeypatch.setattr(
        data_evaluation, "read_project_file", file_reader
    )
    monkeypatch.setattr(
        data_evaluation, "file_selector", fake_selector
    )
    monkeypatch.setattr(
        data_evaluation, "data_eval_analyzer", fake_analyzer
    )

    result = data_evaluation.run_data_evaluation(
        issue="Validation results are unreliable",
        project_path=".",
        objective="Inspect the data split",
    )

    file_listing.assert_called_once_with(".")
    file_reader.assert_called_once_with(
        project_path=".",
        relative_file_path="data_split.py",
    )

    fake_selector.invoke.assert_called_once()
    fake_analyzer.invoke.assert_called_once()

    selection_prompt = fake_selector.invoke.call_args.args[0][1][1]
    analysis_prompt = fake_analyzer.invoke.call_args.args[0][1][1]

    for prompt in [selection_prompt, analysis_prompt]:
        assert "Validation results are unreliable" in prompt
        assert "Inspect the data split" in prompt

    assert "File: data_split.py" in analysis_prompt
    assert "validation_fraction = 0.2" in analysis_prompt
    assert "missing.py" not in analysis_prompt
    assert result is expected_result


@pytest.mark.unit
def test_invalid_selection_returns_incomplete_result(monkeypatch):
    fake_selector = Mock()
    fake_selector.invoke.return_value = FileSelection(
        files=["missing.py"],
        reason="Inspect this file.",
    )

    file_reader = Mock()
    fake_analyzer = Mock()

    monkeypatch.setattr(
        data_evaluation,
        "list_project_files",
        Mock(return_value=["data_split.py"]),
    )
    monkeypatch.setattr(
        data_evaluation, "read_project_file", file_reader
    )
    monkeypatch.setattr(
        data_evaluation, "file_selector", fake_selector
    )
    monkeypatch.setattr(
        data_evaluation, "data_eval_analyzer", fake_analyzer
    )

    result = data_evaluation.run_data_evaluation(
        issue="Validation results are unreliable",
        project_path=".",
    )

    assert result.agent == "data_evaluation"
    assert result.status == "needs_more_information"
    assert result.evidence == []

    file_reader.assert_not_called()
    fake_analyzer.invoke.assert_not_called()



@pytest.mark.live
def test_overlapping_split_returns_completed_analysis(tmp_path):
    source_file = tmp_path / "data_split.py"
    source_file.write_text(
        "train_indices = set(range(100))\n"
        "validation_indices = set(range(80, 120))\n"
        "overlap = train_indices & validation_indices\n",
        encoding="utf-8",
    )

    result = data_evaluation.run_data_evaluation(
        issue="Check whether training and validation indices overlap.",
        project_path=str(tmp_path),
        objective="Inspect data_split.py for overlap between the two sets.",
    )

    assert isinstance(result, AgentResult)
    assert result.agent == "data_evaluation"
    assert result.status == "completed"
    assert result.summary.strip()
    assert result.evidence