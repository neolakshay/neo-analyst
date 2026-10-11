# app/ui/app.py
"""
app.ui.app
----------
Main Streamlit application for the BYTE Applied ML Tabular Data Analysis Agent.

Run from repository root:
    streamlit run app/ui/app.py
"""

import json
import os
import sys
from pathlib import Path

# Ensure the repository root is at the front of sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

# Remove the script directory (app/ui) from sys.path so that 'app.py' does not shadow
# the top-level 'app' package when launched via 'streamlit run app/ui/app.py'.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
while _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)

# If 'app' was registered in sys.modules as a standalone module without __path__,
# unregister it so Python resolves the actual 'app' package from the project root.
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

import streamlit as st
import pandas as pd

from app.agent import AnalysisAgent, GroundingStatus
from app.ui.utils import (
    load_csv,
    get_dataset_summary,
    validate_question,
    format_tool_call_summary,
    is_api_key_configured,
    run_analysis_query,
    EXAMPLE_DATASETS,
    resolve_example_path,
)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BYTE Applied ML — Tabular Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_session_state() -> None:
    """Initialize necessary Streamlit session state keys."""
    defaults = {
        "df": None,
        "dataset_name": None,
        "dataset_key": None,
        "load_error": None,
        "last_result": None,
        "last_question": None,
        "history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset_analysis_results() -> None:
    """Clear query results and history when switching datasets."""
    st.session_state["last_result"] = None
    st.session_state["last_question"] = None
    st.session_state["history"] = []


def main() -> None:
    _init_session_state()

    # ── Sidebar: Data Source & API Configuration ──────────────────────────────
    st.sidebar.title("📁 Dataset Input")

    data_source_mode = st.sidebar.radio(
        "Choose Data Source:",
        ["Example Dataset", "Upload CSV File"],
        key="data_source_mode",
    )

    current_dataset_key = None
    loaded_df = None
    loaded_name = None
    load_error_msg = None

    if data_source_mode == "Example Dataset":
        example_choice = st.sidebar.selectbox(
            "Select an Example Dataset:",
            list(EXAMPLE_DATASETS.keys()),
            key="example_choice",
        )
        if example_choice:
            current_dataset_key = f"example:{example_choice}"
            rel_path = EXAMPLE_DATASETS[example_choice]
            abs_path = resolve_example_path(rel_path)
            loaded_name = example_choice
            loaded_df, load_error_msg = load_csv(abs_path)

    else:  # Upload CSV File
        uploaded_file = st.sidebar.file_uploader(
            "Upload a CSV file:",
            type=["csv"],
            key="uploaded_file_widget",
        )
        if uploaded_file is not None:
            current_dataset_key = f"upload:{uploaded_file.name}:{uploaded_file.size}"
            loaded_name = uploaded_file.name
            loaded_df, load_error_msg = load_csv(uploaded_file)
        else:
            current_dataset_key = None

    # Handle dataset switching or state update
    if current_dataset_key != st.session_state["dataset_key"]:
        st.session_state["dataset_key"] = current_dataset_key
        _reset_analysis_results()
        st.session_state["df"] = loaded_df
        st.session_state["dataset_name"] = loaded_name
        st.session_state["load_error"] = load_error_msg
    else:
        # Keep updated if load error occurred on existing key
        if load_error_msg:
            st.session_state["load_error"] = load_error_msg

    st.sidebar.divider()

    # ── Sidebar: System & API Status ──────────────────────────────────────────
    st.sidebar.subheader("⚙️ System Status")
    api_configured = is_api_key_configured()
    if api_configured:
        st.sidebar.success("OpenAI API Key: Configured")
    else:
        st.sidebar.warning(
            "OpenAI API Key: Not configured.\n"
            "Please set `OPENAI_API_KEY` in your `.env` file to run analysis."
        )

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    st.sidebar.caption(f"Configured Model: `{model_name}`")

    if st.session_state["history"]:
        if st.sidebar.button("🗑️ Clear Analysis History"):
            _reset_analysis_results()
            st.rerun()

    # ── Main Header ───────────────────────────────────────────────────────────
    st.title("📊 BYTE Applied ML — Tabular Analysis Agent")
    st.markdown(
        "An AI-powered data analysis assistant that accepts natural-language questions, "
        "executes local analytical tools via function calling, provides deterministic "
        "output grounding validation, and renders interactive Plotly visualizations."
    )

    # ── Handle Dataset Loading Errors or Missing Data ─────────────────────────
    if st.session_state["load_error"]:
        st.error(f"❌ Error loading dataset: {st.session_state['load_error']}")
        return

    df: Optional[pd.DataFrame] = st.session_state["df"]

    if df is None:
        st.info(
            "👈 **No dataset loaded.** Please choose an example dataset or upload a CSV file "
            "using the sidebar to begin analysis."
        )
        return

    # ── Dataset Overview ──────────────────────────────────────────────────────
    st.subheader(f"📋 Dataset Overview: {st.session_state['dataset_name']}")

    summary = get_dataset_summary(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{summary['rows']:,}")
    col2.metric("Total Columns", f"{summary['columns']:,}")
    col3.metric("Missing Cells", f"{summary['total_nulls']:,}")

    with st.expander("🔍 Inspect Columns & Data Preview", expanded=False):
        tab_preview, tab_columns = st.tabs(["Data Preview (Top 10 Rows)", "Columns & Data Types"])
        with tab_preview:
            st.dataframe(summary["preview"], use_container_width=True)
        with tab_columns:
            st.dataframe(summary["column_details"], use_container_width=True)

    st.divider()

    # ── Natural-Language Analysis Section ─────────────────────────────────────
    st.subheader("💬 Ask an Analytical Question")

    # Example question suggestions based on current dataset
    st.markdown("**Sample Questions:**")
    sample_cols = st.columns(3)
    if sample_cols[0].button("📊 Overview & Column Types", use_container_width=True):
        st.session_state["question_input"] = "What columns are in this dataset and what are their data types?"
    if sample_cols[1].button("📈 Summary Statistics", use_container_width=True):
        st.session_state["question_input"] = "Provide summary statistics for the numeric columns."
    if sample_cols[2].button("👥 Group & Aggregate", use_container_width=True):
        st.session_state["question_input"] = "Which group or category has the highest average values?"

    question_text = st.text_area(
        "Enter your question:",
        key="question_input",
        placeholder="e.g. Which department has the highest average salary, and how does it compare to the overall mean?",
        height=100,
    )

    analyze_clicked = st.button("🚀 Analyze", type="primary")

    if analyze_clicked:
        is_valid, err_msg = validate_question(question_text)
        if not is_valid:
            st.error(f"⚠️ {err_msg}")
        elif not api_configured:
            st.error(
                "❌ Cannot execute query: `OPENAI_API_KEY` is not configured. "
                "Please configure your API key in the project `.env` file and restart the application."
            )
        else:
            with st.spinner("🤖 Agent is analyzing dataset using OpenAI tool calling..."):
                try:
                    result = run_analysis_query(df, question_text)
                    st.session_state["last_result"] = result
                    st.session_state["last_question"] = question_text
                    st.session_state["history"].append({
                        "question": question_text,
                        "result": result,
                    })
                except Exception as exc:
                    st.error(f"❌ Agent execution encountered an unexpected error: {str(exc)}")

    # ── Display Analysis Results ──────────────────────────────────────────────
    last_result = st.session_state["last_result"]
    last_question = st.session_state["last_question"]

    if last_result is not None:
        st.markdown(f"### 🎯 Results for: *\"{last_question}\"*")

        # 1. Execution Errors / Notices
        if last_result.error and not last_result.answer:
            st.error(f"❌ **Execution Error:** {last_result.error}")
        elif last_result.error:
            st.warning(f"⚠️ **Execution Notice:** {last_result.error}")

        # 2. Main Agent Answer
        if last_result.answer:
            st.markdown("#### 💡 Answer")
            st.markdown(last_result.answer)

        # 3. Grounding Status
        if last_result.grounding is not None:
            g_status = last_result.grounding.status
            if g_status == GroundingStatus.VERIFIED:
                st.success(
                    "✅ **Grounding Status: VERIFIED**\n\n"
                    "Specific verifiable claims examined in the answer were confirmed by the executed analytical tools. "
                    "*(Note: Verification confirms examined numerical/factual claims against tool artifacts; it does not guarantee subjective statements.)*"
                )
                if last_result.grounding.claims_checked:
                    with st.expander("🔍 Verified Claims Details", expanded=False):
                        for claim in last_result.grounding.claims_checked:
                            st.write(f"- {claim}")

            elif g_status == GroundingStatus.FLAGGED:
                st.error(
                    "⚠️ **Grounding Status: FLAGGED**\n\n"
                    "The output grounding validator detected potential contradictions, unsupported numerical claims, "
                    "or unverified statistical/causal assertions. This answer should not be treated as fully reliable."
                )
                if last_result.grounding.reasons:
                    with st.expander("⚠️ Grounding Flags & Inconsistencies", expanded=True):
                        for reason in last_result.grounding.reasons:
                            st.write(f"- {reason}")

            elif g_status == GroundingStatus.UNVERIFIABLE:
                st.info(
                    "ℹ️ **Grounding Status: UNVERIFIABLE**\n\n"
                    "The deterministic validator could not verify all claims against executed tool outputs "
                    "(e.g., qualitative, speculative, or derived reasoning). This does not necessarily mean the answer is wrong."
                )
                if last_result.grounding.reasons:
                    with st.expander("ℹ️ Grounding Validation Notes", expanded=False):
                        for reason in last_result.grounding.reasons:
                            st.write(f"- {reason}")

        # 4. Interactive Visualizations
        if last_result.figures:
            st.markdown("#### 📈 Generated Visualizations")
            for idx, fig in enumerate(last_result.figures, start=1):
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"plotly_fig_{idx}_{len(last_result.figures)}",
                )

        # 5. Tool Execution Details
        if last_result.tool_calls:
            with st.expander(
                f"🛠️ Tool Execution Details ({len(last_result.tool_calls)} call(s), {last_result.rounds} round(s))",
                expanded=False,
            ):
                for i, tc in enumerate(last_result.tool_calls, start=1):
                    summary_info = format_tool_call_summary(tc)
                    badge = "✅" if not summary_info["is_error"] else "❌"
                    st.markdown(f"**{i}. `{tc.tool_name}`** — {badge} {summary_info['summary']}")
                    st.caption(f"**Arguments:** `{json.dumps(tc.arguments)}`")
                    if summary_info["is_error"]:
                        st.error(f"Error: {summary_info['summary']}")
                    st.divider()

    # ── Display Session History (if multiple queries were made) ───────────────
    if len(st.session_state["history"]) > 1:
        with st.expander("📜 Session Query History (Independent Requests)", expanded=False):
            st.caption(
                "Note: Each query is evaluated independently by the agent without persistent conversational memory."
            )
            for idx, item in enumerate(reversed(st.session_state["history"][:-1]), start=1):
                st.markdown(f"**Previous Query {idx}:** *{item['question']}*")
                if item["result"].answer:
                    st.write(item["result"].answer)
                st.divider()


if __name__ == "__main__":
    main()
