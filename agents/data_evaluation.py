from langchain_ollama import ChatOllama
from agents.schemas import AgentResult, FileSelection
from tools.project_files import list_project_files, read_project_file
from time import perf_counter
from config import MODEL_NAME, TEMPERATURE, EXTENDED_REASONING


# select model without extended reasoning.
fast_model = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    reasoning=False,
)

# select model with deeper reasoning.
reasoning_model = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    reasoning=EXTENDED_REASONING,
)

# file_selector responses follow the FileSelection schema with files and reason.
# reasoning off
file_selector = fast_model.with_structured_output(
    FileSelection
)

# data_eval_analyzer investigation results follow the AgentResult schema with agent, status, summary and evidence.
# reasoning on
data_eval_analyzer = reasoning_model.with_structured_output(
    AgentResult
)


def run_data_evaluation(
    issue: str,
    project_path: str,
    objective: str | None = None,
) -> AgentResult:
    """
    Investigate data and evaluation related causes of an ML issue by selecting
    relevant project files and analyzing them with an LLM.

    Args:
        issue: The ML issue to investigate.
        project_path: Path to the ML project being inspected.
        objective: Optional investigation objective. Must be a string or None.
            defaults to None.

    Returns:
        An AgentResult object containing the specialist (agent, status, summary and evidence).
    """

    # prepare the text that will later be sent to the LLM.
    investigation_context = f"Issue:\n{issue}"

    # Add the replanned investigation objective when the replan path provides one.
    if objective:
        investigation_context += (
            f"\n\nCurrent investigation objective:\n{objective}"
        )

    # scan the project directory and find the files that system allows the agents to inspect.
    project_files = list_project_files(project_path)


    # timing file selection
    print("[data_evaluation] Selecting files...")
    start = perf_counter()

    # Ask the LLM which files are most relevant to the data/evaluation investigation.
    # the contents of those files still have not been read.
    selection = file_selector.invoke([
        (
            "system",
            "You are a data and evaluation troubleshooting specialist. "
            "Select only the project files that are likely relevant to "
            "investigating datasets, preprocessing, data splits, leakage, "
            "labels, metrics, or evaluation logic.",
        ),

        (
            "human",
            f"{investigation_context}\n\n"
            f"Available project files:\n{project_files}",
        ),
    ])
    # timing file selection
    print(
        f"[data_evaluation] File selection completed in "
        f"{perf_counter() - start:.1f} seconds."
    )

    # Keep only files that actually exist in the available project-file list.
    selected_files = [
        file_path
        for file_path in selection.files
        if file_path in project_files
    ]

    if not selected_files:
        return AgentResult(
            agent="data_evaluation",
            status="needs_more_information",
            summary="No valid project files were selected for inspection.",
            evidence=[],
        )

    selected_file_contents = []

    # Read and save the files selected by the LLM.
    for file_path in selected_files:
        content = read_project_file(
            project_path=project_path,
            relative_file_path=file_path,
        )

        selected_file_contents.append(
            f"File: {file_path}\n\n{content}"
        )

    # timing file analyzing
    print("[data_evaluation] Analyzing selected files for data/evaluation issues...")
    start = perf_counter()



    # Analyze the actual project evidence.
    analysis = data_eval_analyzer.invoke([
        (
            "system",
            "You are a data and evaluation troubleshooting specialist. "
            "Analyze the provided project files for evidence related to "
            "the reported issue. "
            "Base all conclusions only on evidence actually present in "
            "the provided files. "
            "Do not guess about code or data that is not shown. "
            "If there is not enough evidence, set status to "
            "'needs_more_information'. "
            "Only set status to 'completed' when the available evidence "
            "supports a meaningful conclusion. "
            "Set agent to 'data_evaluation'. "
            "Evidence must contain only concrete observations directly "
            "supported by the inspected project files. "
            "Do not put causal conclusions, interpretations, recommendations, "
            "or speculation in the evidence list. Put those in the summary. "
            "For numeric claims such as index ranges, sample counts, dimensions, "
            "or overlaps, verify the values directly from the code before "
            "reporting them. "

            "Do not report line numbers unless line numbers are explicitly "
            "included in the provided file contents. "
            "When reporting that a configuration or behavior is absent, state "
            "the absence as an observation from the inspected files rather than "
            "inventing a location or line reference. "

            "Do not claim that a file, split, configuration, or code path is "
            "used by another part of the project unless the inspected files "
            "establish that connection. "
            "If a finding could explain the user's reported behavior but the "
            "connection is not established, state that uncertainty in the "
            "summary rather than presenting the causal relationship as evidence.",
        ),

        (
            "human",
            f"{investigation_context}\n\n"
            f"Selected project files:\n\n"
            + "\n\n".join(selected_file_contents),
        ),
    ])


    # timing file analyzing
    print(
        f"[data_evaluation] Analysis completed in "
        f"{perf_counter() - start:.1f} seconds."
    )

    return analysis