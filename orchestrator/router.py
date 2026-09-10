from langchain_ollama import ChatOllama

from orchestrator.schemas import RoutingDecision, ReevaluationDecision, InvestigationPlan, DirectAnswer
from config import MODEL_NAME, TEMPERATURE, EXTENDED_REASONING



# Fast model for simpler tasks.
fast_model = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    reasoning=False,
)

# Reasoning model for decisions that require deeper logic.
reasoning_model = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    reasoning=EXTENDED_REASONING,
)



# Initial routing requires deeper reasoning about the appropriate investigation path.
routing_model = reasoning_model.with_structured_output(
    RoutingDecision
)

# Reevaluation chooses the next action from already collected specialist findings.
reevaluation_model = fast_model.with_structured_output(
    ReevaluationDecision
)

# Replanning requires reasoning about how the investigation should change.
replanning_model = reasoning_model.with_structured_output(
    InvestigationPlan
)

# Direct conceptual answers do not need extended reasoning.
direct_model = fast_model.with_structured_output(
    DirectAnswer
)



SYSTEM_PROMPT = """
You are an adaptive orchestrator for troubleshooting machine-learning projects.

Choose the smallest route that can reliably handle the current issue.

Available routes:

- direct:
  A conceptual or advisory question that can be answered reliably from the
  user's description alone, without inspecting project-specific evidence.

- single_specialist:
  The current request can be investigated meaningfully by one specialist.
  Other domains may contain possible causes, but possible causes alone do
  not justify selecting additional specialists.

- multi_specialist:
  The current request already requires investigation of more than one
  specialist domain based on the user's request or currently available
  evidence. Do not choose this route merely because multiple domains could
  theoretically explain the problem.

Do not choose a more complicated route unless it is necessary.

Important consistency rules:

- Choose single_specialist when one specialist can perform the first complete
  investigation, even when several possible causes exist.

- Do not select additional specialists merely because another domain could
  theoretically contain a possible cause.

- Choose multi_specialist only when the current issue already requires
  investigation by more than one specialist domain.

- Choose direct for conceptual or advisory questions that can be answered
  reliably from the user's description alone without inspecting project
  evidence.

- Do not choose direct for a concrete failure in the user's project when
  determining the actual cause requires inspecting project files or code,
  even if a clarifying question could provide additional information.

- Treat a user's report of an actual crash, exception, failure, or unexpected
  project behavior as a request to diagnose that concrete problem.

- The user does not need to explicitly say "inspect my code" for such a
  concrete problem to require project access.

- Do not choose direct merely because a concrete symptom has a familiar or
  likely explanation. A plausible diagnosis is not confirmed project evidence.
  If verifying the cause requires inspecting the user's code, data,
  configuration, or results, select a specialist route.
  
The selected route must agree with the explanation.

Available specialist agents:

- data_evaluation:
  Handles datasets, preprocessing and feature transformations,
  train/validation/test splits, leakage, class imbalance, labels,
  metrics, prediction/label handling during evaluation, evaluation logic,
  and other data or evaluation-pipeline issues.

- model_code:
  Handles model architecture, forward-pass logic, tensor shapes,
  training loops, loss functions, optimization, model initialization,
  training randomness and random seeds, inference implementation,
  model loading, model behavior, and other model/code implementation errors.


Agent selection rules:

- direct must use selected_agents = []

- single_specialist must select exactly one agent.

- multi_specialist must select both agents.

- Select agents based on what must actually be investigated now, not on every
  domain that could possibly contain a cause.

- Select agents based on domain responsibility, not file type. Metrics,
  label and prediction handling, dataset splits, and evaluation logic belong
  to data_evaluation even when they are implemented in source-code files.
  
- Distinguish prediction generation from prediction evaluation:
  problems in how the model produces predictions belong to model_code;
  problems in how predictions are matched with labels, aggregated into
  metrics, or scored during evaluation belong to data_evaluation.

- Distinguish training randomness from data-split randomness:
  model initialization, minibatch order, and training reproducibility belong
  to model_code; random split construction, dataset partitioning, and
  partition integrity belong to data_evaluation.
  
Project access rules:

- Set requires_project_access = true when resolving the user's specific issue
  requires inspecting actual project files, source code, datasets, logs,
  configurations, tracebacks, experiment results, or other project evidence.

- Set requires_project_access = false when the question can be answered
  reliably from the user's description alone.

- If requires_project_access = true, route cannot be direct.

Examples of project access:

- "What commonly causes tensor shape mismatches?"
  requires_project_access = false

- "My model crashes with a tensor shape mismatch. Inspect my code and find
  the exact cause."
  requires_project_access = true

Routing examples:

- "What is data leakage and why is it harmful?"
  -> direct
  -> selected_agents = []

- "My model training results change significantly between runs.
  Inspect the model training code and determine what may be causing it."
  -> single_specialist
  -> selected_agents = ["model_code"]

  Even though data issues could theoretically cause inconsistent results,
  the current requested investigation is specifically model/training code.
  Possible alternative causes alone do not justify calling another agent.

- "I suspect duplicate customer IDs appear in both my training and validation
  sets. Inspect the split and determine whether there is leakage."
  -> single_specialist
  -> selected_agents = ["data_evaluation"]

- "My results change significantly between runs. Inspect both the data
  splitting process and the model training code."
  -> multi_specialist
  -> selected_agents = ["data_evaluation", "model_code"]

  Both specialist domains are explicitly required by the current request.


"""



REEVALUATION_PROMPT = """
You are reevaluating an ongoing machine-learning troubleshooting investigation.

You are given:
1. The original user issue.
2. Results and evidence from specialists that have already investigated it.

Decide what should happen next.

Available actions:

- stop:
  The current investigation provides enough evidence to answer the issue.

- call_specialist:
  The current findings show that another specialist domain must now be
  investigated.

- replan:
  The findings substantially change the understanding of the problem and
  require the investigation strategy itself to be reconsidered.

Rules:

- Base the decision only on the original issue and the actual evidence
  returned by the specialists.

- This system investigates and diagnoses problems. It does not fix, repair,
  retrain, or modify the user's project.

- Choose stop when the collected evidence is sufficient to answer the user's
  issue, even if the diagnosed problem still needs to be fixed by the user.

- Do not choose replan merely because a discovered problem needs to be fixed.

- Do not call another specialist merely because that specialist's domain
  could theoretically be relevant.

- Choose call_specialist when the existing findings specifically identify
  another specialist investigation that is required to answer the issue.

- Do not select a specialist that has already completed an investigation
  during the current run.
  
- Choose replan only when the findings substantially change the current
  understanding of the problem such that the investigation strategy itself
  must change.

- Replan is not a synonym for continuing the investigation or fixing a
  discovered problem.

- If the available evidence already identifies the relevant issue or issues
  well enough to answer the user, choose stop rather than replan.

Agent selection rules:

- stop must use selected_agents = []

- call_specialist must select only the specialist or specialists that are
  now required.

- replan may select agents only when they are part of the revised
  investigation strategy.
"""



REPLANNING_PROMPT = """
You are revising an ongoing machine-learning troubleshooting investigation.

The investigation has produced new evidence that changed what still needs to
be investigated.

You are given:
1. The original user issue.
2. All specialist findings collected so far.

Create a revised investigation plan.

The plan must contain:

- selected_agents:
  Select only the specialist or specialists required for the revised
  investigation.

- objective:
  State the specific unresolved investigation objective that the selected
  specialist should pursue next.

- reason:
  Explain why the collected evidence requires this revised investigation.

Rules:

- Keep the original user issue as the overall goal.

- Derive the objective from the actual findings collected so far.

- The objective must identify what specifically remains unresolved after the
  previous investigation.

- Do not merely repeat the original user issue.

- Do not use a generic objective such as "inspect the data" or
  "investigate the model."

- Make the objective specific enough to guide what project evidence the
  specialist should inspect next.

- Select only specialists justified by the current evidence.

- Do not select specialists merely because they could theoretically be useful.

- Do not repeat an investigation that has already been completed.

- This system diagnoses problems. The revised objective should investigate
  and establish evidence, not fix, modify, or retrain the user's project.
"""



DIRECT_PROMPT = """
You answer machine-learning questions that do not require specialist
investigation or project inspection.

Answer the user's question directly and clearly.

Do not claim to have inspected project files, code, data, logs, or experiment
results when none were inspected.
"""




def route_issue(issue: str) -> RoutingDecision:

    """
    Routes an issue through the LLM orchestrator and returns
    a structured routing decision.
    """

    # Reject empty or whitespace-only issues.
    if not issue.strip():
        raise ValueError("The issue cannot be empty.")

    # Get a structured RoutingDecision output from LLM.
    decision = routing_model.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", issue),
    ])

    return decision


def reevaluate_issue(
    issue: str,
    previous_results: str,
) -> ReevaluationDecision:
    """
    Decide what should happen after reviewing specialist findings.
    """

    if not issue.strip():
        raise ValueError("The issue cannot be empty.")

    # Give the orchestrator both the original problem and the new evidence
    # produced by specialists that have already investigated it.
    decision = reevaluation_model.invoke([
        
        (
            "system",
            REEVALUATION_PROMPT,
        ),
        (
            "human",
            f"Original issue:\n{issue}\n\n"
            f"Investigation results:\n{previous_results}",
        ),
    ])

    return decision



def replan_issue(
    issue: str,
    previous_results: str,
) -> InvestigationPlan:
    """
    Create a revised investigation plan using evidence collected so far.
    """

    # Give the planner the original issue together with all findings
    # discovered during the investigation.
    plan = replanning_model.invoke([
        (
            "system",
            REPLANNING_PROMPT,
        ),
        (
            "human",
            f"Original issue:\n{issue}\n\n"
            f"Investigation results:\n{previous_results}",
        ),
    ])

    # Remove duplicate agent names returned by the LLM.
    unique_agents = []

    for agent in plan.selected_agents:
        if agent not in unique_agents:
            unique_agents.append(agent)

    # Replace the original agent list with the cleaned list.
    plan.selected_agents = unique_agents

    return plan


def answer_direct(issue: str) -> DirectAnswer:
    """
    Answer an issue directly when no specialist investigation is required.
    """

    if not issue.strip():
        raise ValueError("The issue cannot be empty.")

    # Ask the LLM to answer the question without starting an investigation.
    answer = direct_model.invoke([
        (
            "system",
            DIRECT_PROMPT,
        ),
        (
            "human",
            issue,
        ),
    ])

    return answer
