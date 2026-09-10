from langgraph.graph import StateGraph, START, END

from orchestrator.dispatcher import run_selected_agents
from orchestrator.finalizer import finalize_results
from orchestrator.router import (
    route_issue,
    answer_direct,
    reevaluate_issue,
    replan_issue,
)
from orchestrator.state import InvestigationState
import time


def routing_node(
    state: InvestigationState,
) -> dict:
    """
    Run the router and store its RoutingDecision
    in the shared LangGraph state.

    Args:
        state: current shared state following the InvestigationState schema.
    Returns:
        dictionary containing the state fields it wants to add or update.
        LangGraph merges them into the existing shared state.
        It does not return the whole state.
    """

    # Extract the issue from shared graph state and pass it to router.
    # `decision` follows RoutingDecision schema
    decision = route_issue(
        state["issue"]
    )

    return {
        "routing_decision": decision
    }


def direct_answer_node(
    state: InvestigationState,
) -> dict:
    """
    Answer a direct question without specialist investigation.
    """

    # it does not need to retrieve state["routing_decision"]
    # Generate a DirectAnswer object for the current issue without running specialist agents.
    answer = answer_direct(
        state["issue"]
    )

    # returns a state update. langGraph merges that into the shared state.
    # later the finalizer would know what kind of result it received.
    return {
        "direct_answer": answer
    }


def specialists_execution_node(
    state: InvestigationState,
) -> dict:
    """
    Run the specialists selected by the initial router.
    """

    # Retrieve the routing decision produced by routing_node()
    # to access decision.selected_agents
    decision = state["routing_decision"]

    # Run all specialists selected by the router.
    # `results` is a list of AgentResult objects produced by each specialist.
    results = run_selected_agents(
        issue=state["issue"],
        project_path=state["project_path"],
        selected_agents=decision.selected_agents,
    )

    # Store which specialists have already completed an investigation.
    completed_agents = [
        result.agent
        for result in results
    ]

    # Return the state updates
    # No previous specialist history need to be preserved on this first specialist run.
    return {
        "results": results,
        "completed_agents": completed_agents,
    }


def reevaluation_execution_node(
    state: InvestigationState,
) -> dict:
    """
    Reevaluate the investigation after all specialist results are collected
    and determine the next investigation action.

    Args:
        state: shared InvestigationState

    Returns:
        a dictionary state update containing a ReevaluationDecision with:
        - action: The next investigation step (stop, call_specialist, replan, unresolved)
        - selected_agents: Specialist agents requested for additional investigation.
        - reason: Explanation for the decision.
    """


    previous_results_list = []
    # iterating through each AgentResult element of state["results"]
    for result in state["results"]:

        result_json = result.model_dump_json(indent=2)
        previous_results_list.append(result_json)

    previous_results = "\n\n".join(previous_results_list)


    # timing reevaluating
    print("\n[orchestrator] Reevaluating specialist findings...")
    start_time = time.perf_counter()

    # Decide whether to stop, call another specialist or replan.
    reevaluation = reevaluate_issue(
        issue=state["issue"],
        previous_results=previous_results,
    )

    # timing
    elapsed_time = time.perf_counter() - start_time
    print(
        f"[orchestrator] Reevaluation completed in "
        f"{elapsed_time:.1f} seconds."
    )


    # Show which adaptive branch LangGraph will consider next.
    print(
        f"[orchestrator] Reevaluation action: "
        f"{reevaluation.action}"
    )

    # LangGraph adds/replaces state["reevaluation"] with this latest decision.
    return {
        "reevaluation": reevaluation
    }


def follow_up_specialists_execution_node(
    state: InvestigationState,
) -> dict:
    """
    Run additional specialists requested by the reevaluator.
    """

    reevaluation = state["reevaluation"]

    # Only run specialists that have not already completed.
    new_agents = [
        agent
        for agent in reevaluation.selected_agents
        if agent not in state["completed_agents"]
    ]

    # technically if no new agent is selected we must have already gone through the unresolved path.
    if not new_agents:
        raise RuntimeError(
            "follow_up_specialists_execution_node reached with no new agents."
        )

    # Run only the newly requested specialists.
    follow_up_results = run_selected_agents(
        issue=state["issue"],
        project_path=state["project_path"],
        selected_agents=new_agents,
    )

    # Preserve earlier findings and add new findings.
    # Later reevaluator sees the accumulated history. Also final answer is based on all collected specialist findings.
    # both state["results"] and follow_up_results are list of AgentResult objects
    updated_results = (
        state["results"]
        + follow_up_results
    )


    # Preserve the completed_agents history and add new_agents
    updated_completed_agents = (
        state["completed_agents"] + new_agents
    )

    # LangGraph replaces two state values with the combined versions
    return {
        "results": updated_results,
        "completed_agents": updated_completed_agents,
    }


def replanning_execution_node(
    state: InvestigationState,
) -> dict:
    """
    Create a revised investigation plan using all findings collected so far.
    """

    # Convert all collected specialist findings into the text format
    # expected by the replanning function.
    previous_results = "\n\n".join(
        result.model_dump_json(indent=2)
        for result in state["results"]
    )

    # timing replanning
    print("\n[orchestrator] Replanning investigation...")
    start_time = time.perf_counter()

    # Build a revised investigation plan from accumulated evidence.
    # `plan` is InvestigationPlan object includes objective, selected_agents, and reason.
    # objective is a more focused investigation goal.
    plan = replan_issue(
        issue=state["issue"],
        previous_results=previous_results,
    )

    # timing replanning
    elapsed_time = time.perf_counter() - start_time
    print(
        f"[orchestrator] Replanning completed in "
        f"{elapsed_time:.1f} seconds."
    )

    # Show which specialists the revised plan selected.
    print(
        f"[orchestrator] Replanned specialists: "
        f"{plan.selected_agents}"
    )

    # add new investigation_plan to state
    return {
        "investigation_plan": plan
    }


def replanned_specialists_execution_node(
    state: InvestigationState,
) -> dict:
    """
    Run specialists selected by the revised investigation plan
    using the plan’s more specific investigation objective.
    """

    # Retrieve the revised plan.
    plan = state["investigation_plan"]

    # Only run specialists that have not already completed.
    new_agents = [
        agent
        for agent in plan.selected_agents
        if agent not in state["completed_agents"]
    ]

    # if new_agents is empty, choose_replan() should already have routed to go_unresolved
    if not new_agents:
        raise RuntimeError(
            "replanned_specialists_execution_node reached with no new agents."
        )

    # run specialist, this time with objective added 
    new_results = run_selected_agents(
        issue=state["issue"],
        project_path=state["project_path"],
        selected_agents=new_agents,
        objective=plan.objective,
    )

    # Preserve earlier findings and append the new findings.
    updated_results = (
        state["results"]
        + new_results
    )

    # Preserve completed agent history and add the specialists that just finished.
    # could also used new_agents instead
    updated_completed_agents = (
        state["completed_agents"]
        + [
            result.agent
            for result in new_results
        ]
    )

    return {
        "results": updated_results,
        "completed_agents": updated_completed_agents,
    }



def finalization_execution_node(
    state: InvestigationState,
) -> dict:
    """
    Build the final user facing answer from the graph's collected results.

    Uses either the direct answer or accumulated specialist results, applies
    the configured synthesis behavior, and preserves an unresolved graph
    outcome when additional investigation could not be completed.

    Args:
        state: Shared InvestigationState.

    Returns:
        A state update containing the FinalAnswer.
    """

    # timing
    print("\n[orchestrator] Finalizing answer...")
    start_time = time.perf_counter()

    # Direct routes produce a DirectAnswer.
    if "direct_answer" in state:
        result = state["direct_answer"]
    # Specialist routes produce a list of AgentResult objects.
    else:
        result = state["results"]

    # Build FinalAnswer object based on issue, result and synthesis_mode
    final_answer = finalize_results(
        issue=state["issue"],
        result=result,
        synthesis_mode=state["synthesis_mode"],
    )

    # The normal finalizer may mark completed specialist results as resolved.
    # Override that status when the graph actually stopped because it
    # could not perform the additional investigation that was requested.
    
    # Only unresolved_node adds "termination_status": "needs_more_information"
    # For a normal resolved path state.get("termination_status") return None
    if state.get("termination_status") == "needs_more_information":
        final_answer = final_answer.model_copy(
            update={
                "status": "needs_more_information"
            }
        )

    # timing
    elapsed_time = time.perf_counter() - start_time
    print(
        f"[orchestrator] Finalization completed in "
        f"{elapsed_time:.1f} seconds."
    )


    return {
        "final_answer": final_answer
    }


def unresolved_execution_node(
    state: InvestigationState,
) -> dict:
    """
    Mark the investigation as incomplete when more investigation
    is needed but no new specialist is available to run.
     
    """
    # add flag to state to signal that graph is stopping because it cannot perform additional investigation.
    return {
        "termination_status": "needs_more_information"
    }


def choose_route(
    state: InvestigationState,
) -> str:
    """
    Choose which graph path should run after initial routing.
    
    Args:
        state: shared InvestigationState

    Returns:
        a string routing label that will later be mapped to a destination node through conditional edge mapping/path map 
    """
    # retrieves the RoutingDecision that was added by routing_node().
    decision = state["routing_decision"]

    # Conceptual or simple issues can be answered directly.
    # "go_direct" is not a node name, it is a routing label.
    # conditional edge mapping will map this routing label into destination node "direct_node"
    if decision.route == "direct":
        return "go_direct"

    # Both single specialist and multi specialist routes use the same node, specialists_node.
    return "go_specialists"


def choose_reevaluation(
    state: InvestigationState,
) -> str:
    """
    Choose which graph path should run after reevaluation.
    """

    reevaluation = state["reevaluation"]

    # The reevaluator explicitly determined that the collected
    # evidence is sufficient to finish the investigation.
    if reevaluation.action == "stop":
        return "go_stop"

    if reevaluation.action == "call_specialist":

        # Find requested specialists that have not already run.
        new_agents = []

        for agent in reevaluation.selected_agents:
            if agent not in state["completed_agents"]:
                new_agents.append(agent)

        # The reevaluator still wants more investigation, but every
        # requested specialist has already been used. This is not
        # considered a successful resolution.
        if not new_agents:
            return "go_unresolved"

        return "go_call_specialist"

    # The remaining valid action is replan.
    return "go_replan"


def choose_replan(
    state: InvestigationState,
) -> str:
    """
    Choose what should happen after a revised investigation plan is created.
    """

    # Retrieve the revised plan created by replanning_execution_node().
    plan = state["investigation_plan"]

    # Find specialists in the revised plan that have not already run.
    new_agents = [
        agent
        for agent in plan.selected_agents
        if agent not in state["completed_agents"]
    ]

    # Continue if the new plan contains specialist work
    # that has not already been completed.
    if new_agents:
        return "go_replanned_specialists"

    # If the revised plan contains no new specialists (new_agents=[]),
    # there is no additional specialist investigation to perform.
    # So the graph cannot make additional progress.
    return "go_unresolved"


def build_graph():
    """
    Build the LangGraph orchestration graph.
    """
    # Create a LangGraph workflow whose nodes communicate using this shared InvestigationState structure.
    graph = StateGraph(
        state_schema = InvestigationState
    )

    # Register LangGraph node names and their Python functions.
    graph.add_node(
        "route_node",
        routing_node,
    )

    # This node will handle questions that don't need project investigation.
    graph.add_node(
        "direct_node",
        direct_answer_node,
    )

    # This is the node that runs whichever specialist agents the router initially selected.
    graph.add_node(
        "specialists_node",
        specialists_execution_node,
    )

    # This one examines the specialist findings afterward and decides whether more work is needed.
    graph.add_node(
        "reevaluation_node",
        reevaluation_execution_node,
    )

    # This is used when reevaluation says: need another specialist.
    graph.add_node(
        "follow_up_specialists_node",
        follow_up_specialists_execution_node,
    )

    # The original investigation direction is no longer enough. create a revised plan.
    graph.add_node(
        "replan_node",
        replanning_execution_node,
    )

    # This executes specialists selected by that new plan.
    graph.add_node(
        "replanned_specialists_node",
        replanned_specialists_execution_node,
    )

    # This converts whatever happened during the investigation into the final user facing answer.
    graph.add_node(
        "finalizer_node",
        finalization_execution_node,
    )


    # This is used when the system knows more investigation is needed, but it has no new specialist available to perform that investigation.
    graph.add_node(
        "unresolved_node",
        unresolved_execution_node,
    )  

    # Every investigation begins with routing.
    graph.add_edge(
        START,
        "route_node",
    )

    # Initial routing chooses between a direct answer and specialist investigation.
    graph.add_conditional_edges(
        "route_node",
        choose_route,
        {
            "go_direct": "direct_node",
            "go_specialists": "specialists_node",
        },
    )

    # Direct answers pass through the common finalization step.
    graph.add_edge(
        "direct_node",
        "finalizer_node",
    )

    # Initial specialist findings are always reevaluated.
    graph.add_edge(
        "specialists_node",
        "reevaluation_node",
    )

    # # Reevaluation decides whether to finish, call another specialist, replan, or mark the investigation as unresolved.
    graph.add_conditional_edges(
        "reevaluation_node",
        choose_reevaluation,
        {
            "go_stop": "finalizer_node",
            "go_call_specialist": "follow_up_specialists_node",
            "go_replan": "replan_node",
            "go_unresolved": "unresolved_node",
        },
    )

    # New specialist findings return to reevaluation.
    graph.add_edge(
        "follow_up_specialists_node",
        "reevaluation_node",
    )

    # A revised plan either runs new specialists
    # or finishes if no new specialist work remains.
    graph.add_conditional_edges(
        "replan_node",
        choose_replan,
        {
            "go_replanned_specialists": "replanned_specialists_node",
            "go_unresolved": "unresolved_node",
        },
    )

    # Findings produced by a revised plan return to reevaluation.
    graph.add_edge(
        "replanned_specialists_node",
        "reevaluation_node",
    )

    # Even if the investigation couldn't fully resolve the issue, you still want to produce a proper final response
    # Unresolved still goes through the finalizer
    graph.add_edge(
        "unresolved_node",
        "finalizer_node",
    )

    # Every completed investigation ends after finalization.
    graph.add_edge(
        "finalizer_node",
        END,
    )

    # Compile the graph definition into an executable LangGraph application.
    return graph.compile()