import argparse

from orchestrator.graph import build_graph


def main():
    """
    Run the adaptive ML troubleshooting system from the command line.
    """

    parser = argparse.ArgumentParser(
        description="Adaptive multi-agent ML troubleshooting orchestrator."
    )

    # Which ML codebase should the agents inspect
    parser.add_argument(
        "--project",
        required=True,
        help="Path to the ML project that should be investigated.",
    )

    # Actual problem the user wants diagnosed.
    parser.add_argument(
        "--issue",
        required=True,
        help="Description of the ML issue to investigate.",
    )

    # Controls whether an LLM should polish/synthesize the final result.
    parser.add_argument(
        "--synthesis",
        choices=["never", "auto", "always"],
        default="auto",
        help="Control whether the final answer is synthesized by an LLM.",
    )

    args = parser.parse_args()


    app = build_graph()

    # Initialize the shared state that follows InvestigationState schema
    # results and completed_agent history start empty and may change later.
    # other optional state fields are added later as the graph executes.
    initial_state = {
        "issue": args.issue,
        "project_path": args.project,
        "synthesis_mode": args.synthesis,
        "results": [],
        "completed_agents": [],
    }

    # Execute the graph from START until it reaches END.
    # final_state follows InvestigationState schema
    final_state = app.invoke(
        initial_state
    )

    # Store the user facing final_answer in state
    # final_answer follows FinalAnswer schema
    final_answer = final_state["final_answer"]

    print("\nFinal answer:")
    print(final_answer.answer)

    if final_answer.evidence:
        print("\nEvidence:")

        for evidence in final_answer.evidence:
            print(
                f"- {evidence}"
            )

    print(
        f"\nStatus: {final_answer.status}"
    )


if __name__ == "__main__":
    main()

