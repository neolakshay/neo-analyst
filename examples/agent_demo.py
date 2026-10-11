# examples/agent_demo.py
"""
Terminal demonstration of the Phase 2 LLM-powered data analysis agent.

Loads the employee analytics dataset, accepts a natural-language question,
executes reasoning and analytical tool calls, and displays the answer.
If a chart is generated, it is saved as an interactive HTML file.

Run from project root:
    python -m examples.agent_demo

Or with a direct question argument:
    python -m examples.agent_demo "Which department has the highest average salary?"
"""

import os
import pathlib
import sys
import pandas as pd
from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file
load_dotenv(PROJECT_ROOT / ".env")

from app.agent import AnalysisAgent


def main():
    print("=" * 70)
    print("  neo-applied-ml: LLM Data Analysis Agent Demo (Phase 2, Step 1)")
    print("=" * 70)

    # 1. Load the employee dataset
    dataset_path = PROJECT_ROOT / "examples" / "employee_analytics.csv"
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    print(f"\n[+] Loaded dataset: {dataset_path.name}")
    print(f"    Dimensions : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"    Columns    : {list(df.columns)}")

    # 2. Check OpenAI API configuration
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("\n[!] Error: OPENAI_API_KEY is not set.")
        print("    Please set OPENAI_API_KEY in your environment or in a .env file at the project root:")
        print("        OPENAI_API_KEY=your_api_key_here")
        print("        OPENAI_MODEL=gpt-4o-mini  # optional (default: gpt-4o-mini)")
        sys.exit(1)

    # 3. Get user question
    default_question = (
        "Which department has the highest average salary, and how many employees does it have?"
    )
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
    else:
        try:
            prompt_text = f"\nEnter your question (press Enter for default: '{default_question}'):\n> "
            user_input = input(prompt_text).strip()
            question = user_input if user_input else default_question
        except (EOFError, KeyboardInterrupt):
            print("\nUsing default question.")
            question = default_question

    print(f"\n[?] Question: {question}")
    print("\n[*] Initializing AnalysisAgent and running tool-calling loop...")

    # 4. Run the Agent
    try:
        agent = AnalysisAgent()
        result = agent.run(df, question)
    except Exception as exc:
        print(f"\n[!] Agent execution failed: {exc}")
        sys.exit(1)

    # 5. Display results
    print("\n" + "=" * 70)
    print("  AGENT ANSWER")
    print("=" * 70)
    print(result.answer)

    print("\n" + "-" * 70)
    print("  EXECUTION SUMMARY")
    print("-" * 70)
    print(f"Model used       : {result.model}")
    print(f"Reasoning rounds : {result.rounds}")
    print(f"Tool calls made  : {len(result.tool_calls)}")

    for i, tc in enumerate(result.tool_calls, start=1):
        print(f"\n  [{i}] Tool: {tc.tool_name}")
        print(f"      Arguments: {tc.arguments}")
        if isinstance(tc.output, dict):
            status = tc.output.get("status", "completed")
            print(f"      Status   : {status}")
            if "matched_rows" in tc.output:
                print(f"      Matched  : {tc.output['matched_rows']} rows")
            elif "results" in tc.output:
                print(f"      Results  : {len(tc.output['results'])} group items returned")

    # 6. Save figures if generated
    if result.figures:
        output_dir = PROJECT_ROOT / "examples"
        for idx, fig in enumerate(result.figures, start=1):
            out_file = output_dir / f"agent_chart_{idx}.html"
            fig.write_html(str(out_file))
            print(f"\n[+] Saved interactive visualization to: {out_file}")

    if result.error:
        print(f"\n[!] Notice / Error: {result.error}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
