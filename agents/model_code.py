from langchain_ollama import ChatOllama
from agents.schemas import AgentResult, FileSelection
from tools.project_files import list_project_files, read_project_file
from config import MODEL_NAME, TEMPERATURE, EXTENDED_REASONING
from time import perf_counter


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
    #num_predict=1024, #includes reasoning and output tokens
)


# file_selector responses follow the FileSelection schema with files and reason.
# reasoning off
file_selector = fast_model.with_structured_output(
    FileSelection
)

# code_analyzer investigation results follow the AgentResult schema with agent, status, summary and evidence.
# reasoning on
code_analyzer = reasoning_model.with_structured_output(
    AgentResult,
    # include_raw=True, #False is default, for True we need to modify code below to analysis result and analysis = analysis_result["parsed"]
)



def run_model_code(
    issue: str,
    project_path: str,
    objective: str | None = None,
) -> AgentResult:
    """
    Investigate model and code related causes of an ML issue by selecting
    relevant project files and analyzing them with an LLM.

    Args:
        issue: The ML issue to investigate.
        project_path: Path to the ML project being inspected.
        objective: Optional investigation objective. Must be a string or None.
            defaults to None.

    Returns:
        An AgentResult object containing the specialist (agent, status, summary and evidence).
    """

    investigation_context = f"Issue:\n{issue}"

    if objective:
        investigation_context += (
            f"\n\nCurrent investigation objective:\n{objective}"
        )

    project_files = list_project_files(project_path)

    # timing file selection
    print("[model_code] Selecting files...")
    start = perf_counter()


    selection = file_selector.invoke([
        (
            "system",
            "You are a model and code troubleshooting specialist. "
            "Select only the project files that are likely relevant to "
            "investigating the reported issue.",
        ),
        (
            "human",
            f"{investigation_context}\n\n"
            f"Available project files:\n{project_files}",
        ),
    ])


    # timing file selection
    print(
        f"[model_code] File selection completed in "
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
            agent="model_code",
            status="needs_more_information",
            summary="No valid project files were selected for inspection.",
            evidence=[],
        )

    # Read the contents of the files selected by the LLM.
    selected_file_contents = []

    for file_path in selected_files:
        content = read_project_file(
            project_path=project_path,
            relative_file_path=file_path,
        )

        # Store each filename together with its contents so the LLM knows which code came from which file.
        selected_file_contents.append(
            f"File: {file_path}\n\n{content}"
        )


    # timing code analyzing
    print("[model_code] Analyzing selected files for data/evaluation issues...")
    start = perf_counter()


    # Analyze the actual project evidence.
    analysis = code_analyzer.invoke([
        (
            "system",
            "You are a model and code troubleshooting specialist. "
            "Analyze the provided project files for evidence related to "
            "the reported issue. "
            "Focus on model architecture, forward-pass logic, tensor shapes, "
            "training behavior, randomness, optimization, and related code issues. "

            "Data splitting, leakage, dataset sizes, class distributions, and "
            "evaluation-data integrity are primarily the responsibility of the "
            "data_evaluation specialist. "
            "If data-related files are inspected, use them only when necessary to "
            "understand how they connect to model or training code. "
            "Do not report independent data-split calculations or data-quality findings "
            "unless they are directly necessary for the model/code investigation. "


            "Base all conclusions only on evidence actually present in "
            "the provided files. "
            "Do not guess about code or behavior that is not shown. "
            "If there is not enough evidence, set status to "
            "'needs_more_information'. "
            "Only set status to 'completed' when the available evidence "
            "supports a meaningful conclusion. "
            "Set agent to 'model_code'. "
            "Evidence must contain only concrete observations directly "
            "supported by the inspected project files. "
            "Do not put causal conclusions, interpretations, recommendations, "
            "or speculation in the evidence list. Put those in the summary. "
            "For numeric claims such as tensor shapes, dimensions, parameter "
            "values, sample counts, or index ranges, verify the values directly "
            "from the code before reporting them. "

            "Do not report line numbers unless line numbers are explicitly "
            "included in the provided file contents. "
            "When reporting that a configuration or behavior is absent, state "
            "the absence as an observation from the inspected files rather than "
            "inventing a location or line reference. "


            "Do not claim that a file, model component, configuration, or code "
            "path is used by another part of the project unless the inspected "
            "files establish that connection. "
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

    # timing code analyzing
    print(
        f"[model_code] Analysis completed in "
        f"{perf_counter() - start:.1f} seconds."
    )    

    
    return analysis