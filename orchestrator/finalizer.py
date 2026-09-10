from typing import Literal

from langchain_ollama import ChatOllama

from agents.schemas import AgentResult
from orchestrator.schemas import DirectAnswer, FinalAnswer, SynthesisResult
from config import MODEL_NAME, TEMPERATURE, EXTENDED_REASONING



def build_final_answer(
    result: DirectAnswer | list[AgentResult],
) -> FinalAnswer:
    """
    Convert a direct answer or specialist results into the common FinalAnswer format
    without making an additional LLM call.

    Args:
        result: Either a DirectAnswer or a list of AgentResult objects.

    Returns:
        A FinalAnswer containing the overall status, combined answer, and collected evidence.
    """

    # Format DirectAnswer into FinalAnswer object and return directly
    # because no specialists inspected project files, there is no project evidence.
    if isinstance(result, DirectAnswer):
        return FinalAnswer(
            status="resolved",
            answer=result.answer,
            evidence=[],
        )

    # For specialist result:
    # If every specialist completed successfully, `resolved` becomes True, else False
    resolved = all(
        agent_result.status == "completed"
        for agent_result in result
    )

    # Set investigation `status` according to `resolved`.
    if resolved:
        status = "resolved"
    else:
        status = "needs_more_information"

    # Combine specialist summaries
    answer = "\n\n".join(
        agent_result.summary
        for agent_result in result
    )

    # Collect evidence from all specialists.
    evidence = []
    seen_evidence = set()

    for agent_result in result:
        for item in agent_result.evidence:
            # Avoid adding the exact same evidence more than once.
            if item not in seen_evidence:
                evidence.append(item)
            seen_evidence.add(item)

    # build and return FinalAnswer object for specialists result
    return FinalAnswer(
        status=status,
        answer=answer,
        evidence=evidence,
    )



SYNTHESIS_PROMPT = """
Synthesize the findings from multiple machine-learning troubleshooting
specialists into one accurate, internally consistent conclusion.

You are given:
1. The original user issue.
2. Structured findings and evidence from all specialists that investigated it.

Your job is to reason across the specialist results as a whole, not simply
repeat or concatenate their individual conclusions.

Rules:

- Compare all specialist findings with each other before forming the final
  conclusion.

- Treat each specialist's conclusion as a domain-specific finding, not
  automatically as the overall root cause.

- Do not favor an earlier specialist's conclusion merely because it was
  produced first. Consider all specialist findings together.

- Distinguish between:
  1. discovering that a problem exists somewhere in the project, and
  2. establishing that the problem caused the behavior reported by the user.

- Before making a causal claim, verify that the evidence connects the
  discovered problem to the relevant execution path, data flow, evaluation
  logic, or metric involved in the user's reported issue.

- A problem found in a file should not be described as the confirmed cause
  of the user's reported behavior if the available evidence does not show
  that the file, code path, data split, or configuration is actually used
  in producing that behavior.

- If a discovered problem could plausibly cause the reported behavior but
  the connection is not established, describe it as a potential or relevant
  cause and clearly state what connection remains unverified.

- Do not say that a discovered problem is "not connected" to the reported
  behavior merely because the connection is missing from the available
  evidence. Say that the connection is "not established" or "not verified"
  unless there is evidence showing that no connection exists.

- Do not claim causality unless the combined evidence supports it.

- If one specialist's findings qualify, limit, or contradict another
  specialist's conclusion, the final answer must reflect that relationship.

- If relevant problems are found but the available evidence does not establish
  that they caused the reported behavior, explain that distinction clearly.

- If the user's issue contains multiple symptoms, evaluate each symptom
  separately.

- A finding may plausibly explain one symptom without explaining the others.
  Do not dismiss a finding merely because it does not explain every part of
  the user's issue.

- When a finding plausibly explains one of multiple reported symptoms, state
  that relationship explicitly even if the overall issue remains unresolved.

- Clearly separate:
  - what was directly found,
  - what those findings could explain,
  - and what remains unverified.

- Use status "resolved" only when the combined findings provide enough
  evidence to answer the user's issue with a supported conclusion.

- Use status "needs_more_information" when important evidence is still
  missing, especially when the connection between a discovered problem and
  the reported behavior cannot be established from the available project
  evidence.

- If status is "needs_more_information", do not state a definitive root cause
  anywhere in the answer.

- The answer must not first state that something definitely caused the issue
  and later state that the causal connection is uncertain or unconfirmed.

- Do not invent evidence, files, code paths, metrics, configurations, or
  technical details.

- Do not include an Evidence section. The application preserves and displays
  the original specialist evidence separately.

Return one coherent final conclusion and the overall investigation status.
"""

synthesis_model = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    reasoning=EXTENDED_REASONING,
)

structured_synthesis_model = (
    synthesis_model.with_structured_output(
        SynthesisResult
    )
)


def synthesize_results(
    issue: str,
    results: list[AgentResult],
) -> SynthesisResult:
    """
    Synthesize multiple specialist findings into one consistent overall conclusion.

    Args:
        issue: The original ML issue being investigated.
        results: A list of specialist AgentResult objects.

    Returns:
        A SynthesisResult containing the overall status and synthesized answer.
    """

    specialist_results = "\n\n".join(
        result.model_dump_json(indent=2)
        for result in results
    )

    # Produce a SynthesisResult after LLM reasoning over the already produced specialist results.
    synthesis = structured_synthesis_model.invoke(
        [
            (
                "system",
                SYNTHESIS_PROMPT,
            ),
            (
                "human",
                f"Original issue:\n{issue}\n\n"
                f"Specialist findings:\n{specialist_results}",
            ),
        ]
    )

    return synthesis



def finalize_results(
    issue: str,
    result: DirectAnswer | list[AgentResult],
    synthesis_mode: str = "auto",
) -> FinalAnswer:
    """
    Convert direct or specialist results into a final answer and optionally
    synthesize multiple specialist findings with an LLM.

    Args:
        issue: The original ML issue being investigated.
        result: Either a DirectAnswer or a list of AgentResult objects.
        synthesis_mode: Controls LLM synthesis of specialist results.
            Must be "never", "auto", or "always". Defaults to "auto".     
            - never: Do not make an additional LLM synthesis call.
            - auto: Synthesize only when multiple specialist results must be reconciled into one conclusion.
            - always: Synthesize specialist findings even when only one specialist ran.

    Returns:
        A FinalAnswer containing the final status, answer, and collected evidence.
    """

    # Build a FinalAnswer object from a DirectAnswer or  a List of AgentResult objects,
    # containing status, answer, evidence.
    final_answer = build_final_answer(
        result
    )

    # if result is DirectAnswer, return final_answer directly, without specialist synthesis.
    if isinstance(result, DirectAnswer):
        return final_answer


    # if result is specialist result, decide whether synthesis is needed
    if synthesis_mode == "always":
        should_synthesize = True

    elif synthesis_mode == "auto" and len(result) > 1:
        should_synthesize = True

    else:
        should_synthesize = False


    # if synthesis isn't needed, return final_answer without LLM call,
    if not should_synthesize:
        return final_answer

    # if synthesis is needed:
    # Produce one consistent conclusion using specialists findings and LLM call.
    # Return SynthesisResult object containing status and answer
    synthesis = synthesize_results(
        issue=issue,
        results=result,
    )

    # Update final_answer according to synthesis result
    # Only replace the answer and status fields
    # Evidence already stored in final_answer remains unchanged,
    return final_answer.model_copy(
        update={
            "status": synthesis.status,
            "answer": synthesis.answer,
        }
    )