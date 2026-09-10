from unittest.mock import Mock

import pytest

from agents.schemas import AgentResult
from orchestrator import finalizer
from orchestrator.schemas import DirectAnswer, SynthesisResult


@pytest.mark.unit
def test_direct_answer_skips_synthesis(monkeypatch):
    fake_synthesis = Mock()
    monkeypatch.setattr(finalizer, "synthesize_results", fake_synthesis)

    result = finalizer.finalize_results(
        issue="What is overfitting?",
        result=DirectAnswer(answer="Poor generalization to unseen data."),
        synthesis_mode="always",
    )

    assert result.status == "resolved"
    assert result.answer == "Poor generalization to unseen data."
    assert result.evidence == []
    fake_synthesis.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    "second_status, expected_status",
    [
        ("completed", "resolved"),
        ("needs_more_information", "needs_more_information"),
    ],
)
def test_specialist_results_are_combined(second_status, expected_status):
    findings = [
        AgentResult(
            agent="data_evaluation",
            status="completed",
            summary="The split contains overlap.",
            evidence=["Shared observation", "Split observation"],
        ),
        AgentResult(
            agent="model_code",
            status=second_status,
            summary="The training code was inspected.",
            evidence=["Shared observation", "Training observation"],
        ),
    ]

    result = finalizer.build_final_answer(findings)

    assert result.status == expected_status
    assert result.answer == (
        "The split contains overlap.\n\n"
        "The training code was inspected."
    )
    assert result.evidence == [
        "Shared observation",
        "Split observation",
        "Training observation",
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode, result_count, should_synthesize",
    [
        ("never", 2, False),
        ("auto", 1, False),
        ("auto", 2, True),
        ("always", 1, True),
    ],
)
def test_synthesis_mode_controls_final_answer(
    monkeypatch, mode, result_count, should_synthesize
):
    findings = [
        AgentResult(
            agent="data_evaluation",
            status="completed",
            summary="Data findings.",
            evidence=["Split overlap exists."],
        ),
        AgentResult(
            agent="model_code",
            status="completed",
            summary="Model findings.",
            evidence=["Training uses the split."],
        ),
    ][:result_count]

    synthesis = SynthesisResult(
        status="needs_more_information",
        answer="Further evidence is needed to establish the cause.",
    )
    fake_synthesis = Mock(return_value=synthesis)
    monkeypatch.setattr(finalizer, "synthesize_results", fake_synthesis)

    result = finalizer.finalize_results(
        issue="Validation is unreliable.",
        result=findings,
        synthesis_mode=mode,
    )

    if should_synthesize:
        fake_synthesis.assert_called_once_with(
            issue="Validation is unreliable.",
            results=findings,
        )
        assert result.answer == synthesis.answer
        assert result.status == synthesis.status
    else:
        fake_synthesis.assert_not_called()
        assert result.answer == "\n\n".join(
            finding.summary for finding in findings
        )
        assert result.status == "resolved"

    assert result.evidence == [
        item
        for finding in findings
        for item in finding.evidence
    ]


@pytest.mark.unit
def test_findings_reach_synthesis_model(monkeypatch):
    finding = AgentResult(
        agent="data_evaluation",
        status="completed",
        summary="The split contains overlap.",
        evidence=["Indices 80 through 99 appear in both sets."],
    )
    expected = SynthesisResult(
        status="resolved",
        answer="The training and validation sets overlap.",
    )

    fake_model = Mock()
    fake_model.invoke.return_value = expected
    monkeypatch.setattr(
        finalizer, "structured_synthesis_model", fake_model
    )

    result = finalizer.synthesize_results(
        issue="Check for split overlap.",
        results=[finding],
    )

    fake_model.invoke.assert_called_once()
    messages = fake_model.invoke.call_args.args[0]

    assert messages[0] == ("system", finalizer.SYNTHESIS_PROMPT)
    assert "Check for split overlap." in messages[1][1]
    assert finding.model_dump_json(indent=2) in messages[1][1]
    assert result is expected