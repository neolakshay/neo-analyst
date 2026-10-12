# app/ui/app.py
"""
app.ui.app
----------
Main Streamlit application for neo-analyst.

Run from repository root:
    streamlit run app/ui/app.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

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

from app.agent import AnalysisAgent
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
    page_title="neo-analyst",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design System & Custom CSS ────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* CSS Variables & Palette */
:root {
    --bg-workspace: #0a0d13;
    --bg-center: #0d111a;
    --bg-side-panel: #121622;
    --bg-card: #171c2b;
    --bg-input: #0f131e;
    --border-subtle: #1f2738;
    --border-hover: #313d56;
    --border-focus: #6366f1;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent: #6366f1;
    --accent-hover: #4f46e5;
    --accent-glow: rgba(99, 102, 241, 0.22);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
}

/* Base Body & App View */
html, body {
    background-color: var(--bg-workspace) !important;
    color: var(--text-primary) !important;
}

/* App block container: full wide utilization without artificial max-width constraints */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1.25rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}

/* Hide default streamlit header bar */
header[data-testid="stHeader"] {
    display: none !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

/* Main Heading */
h1 {
    font-size: 2rem !important;
    font-weight: 750 !important;
    letter-spacing: -0.03em !important;
    color: #ffffff !important;
    margin: 0 0 0.4rem 0 !important;
    line-height: 1.2 !important;
    text-align: center !important;
}

/* Bordered Containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border-subtle) !important;
    background-color: var(--bg-side-panel) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--border-hover) !important;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.55rem 0.75rem !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}

/* Form Controls & Inputs */
.stTextArea textarea {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.stTextArea textarea:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background-color: var(--bg-input) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
}

/* Buttons */
button[kind="primary"] {
    background-color: var(--accent) !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius-md) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s ease !important;
}

button[kind="primary"]:hover {
    background-color: var(--accent-hover) !important;
    box-shadow: 0 4px 14px var(--accent-glow) !important;
}

button[kind="secondary"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-secondary) !important;
    transition: all 0.2s ease !important;
}

button[kind="secondary"]:hover {
    border-color: var(--border-hover) !important;
    color: var(--text-primary) !important;
}

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    background-color: var(--bg-card) !important;
    overflow: hidden;
}

/* Dataframe containers */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    border: 1px solid var(--border-subtle) !important;
}

/* Alerts and banners */
.stAlert {
    border-radius: var(--radius-md) !important;
}

/* Section dividers */
hr {
    border-color: var(--border-subtle) !important;
    margin: 0.85rem 0 !important;
}

/* Welcome state */
.chat-welcome {
    text-align: center;
    padding: 2.5rem 1rem 1rem 1rem;
}

.chat-welcome h3 {
    margin: 0 0 0.4rem 0 !important;
    color: #f1f5f9 !important;
    font-size: 1.3rem !important;
}

.chat-welcome p {
    color: var(--text-secondary);
    max-width: 540px;
    margin: 0 auto 1.5rem auto;
    font-size: 0.95rem;
    line-height: 1.5;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
        "question_input": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset_analysis_results() -> None:
    """Clear query results and history when switching datasets."""
    st.session_state["last_result"] = None
    st.session_state["last_question"] = None
    st.session_state["history"] = []


def _set_sample_question(question_str: str) -> None:
    """Callback for starter buttons: safely populates question_input before widget instantiation."""
    st.session_state["question_input"] = question_str


def main() -> None:
    _init_session_state()

    # Two-column workspace layout: Left (Controls + Dataset Overview), Centre (Expanded Chat Workspace)
    col_left, col_center = st.columns([1.0, 2.6], gap="medium")

    # ── Left Column: Dataset Controls, System Status & Dataset Overview ───────
    with col_left:
        with st.container(border=True):
            st.markdown("### 📁 Dataset Input")

            data_source_mode = st.radio(
                "Choose Data Source:",
                ["Example Dataset", "Upload CSV File"],
                key="data_source_mode",
            )

            current_dataset_key = None
            loaded_df = None
            loaded_name = None
            load_error_msg = None

            if data_source_mode == "Example Dataset":
                example_choice = st.selectbox(
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
                uploaded_file = st.file_uploader(
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
                if load_error_msg:
                    st.session_state["load_error"] = load_error_msg

            if st.session_state["load_error"]:
                st.error(f"❌ Error loading dataset: {st.session_state['load_error']}")

            st.divider()

            # System & API Status
            st.markdown("### ⚙️ System Status")
            api_configured = is_api_key_configured()
            if api_configured:
                st.success("OpenAI API Key: Configured")
            else:
                st.warning(
                    "OpenAI API Key: Not configured.\n"
                    "Please set `OPENAI_API_KEY` in your `.env` file to run analysis."
                )

            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            st.caption(f"Configured Model: `{model_name}`")

            st.divider()

            # Dataset Overview (placed directly below Dataset Input & System Status)
            st.markdown("### 📋 Dataset Overview")
            df: pd.DataFrame | None = st.session_state["df"]
            if df is not None:
                st.caption(f"Active Dataset: **{st.session_state['dataset_name']}**")
                summary = get_dataset_summary(df)

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Rows", f"{summary['rows']:,}")
                m2.metric("Total Columns", f"{summary['columns']:,}")
                m3.metric("Missing Cells", f"{summary['total_nulls']:,}")

                st.markdown("##### 📄 Data Preview (Top 10 Rows)")
                st.dataframe(summary["preview"], use_container_width=True, height=200)

                with st.expander("🔍 Columns & Data Types", expanded=False):
                    st.dataframe(summary["column_details"], use_container_width=True)
            else:
                st.info("No dataset currently loaded.")

            if st.session_state["history"]:
                st.divider()
                if st.button("🗑️ Clear Conversation", use_container_width=True):
                    _reset_analysis_results()
                    st.rerun()

    # ── Centre Column: Expanded Chat Workspace ────────────────────────────────
    with col_center:
        # Main Application Heading
        st.title("neo-analyst")

        # Native scrollable conversation container with increased height to fill available viewport
        messages_container = st.container(height=660)

        # Bottom-anchored chat input and send action
        with st.container():
            input_col, btn_col = st.columns([5, 1], vertical_alignment="bottom")
            with input_col:
                question_text = st.text_area(
                    "Enter your analytical question:",
                    key="question_input",
                    placeholder="Ask a question about the dataset...",
                    height=68,
                    label_visibility="collapsed",
                )
            with btn_col:
                analyze_clicked = st.button("🚀 Analyze", type="primary", use_container_width=True)

        # Handle analytical query submission
        submission_error = None
        if analyze_clicked:
            is_valid, err_msg = validate_question(question_text)
            if not is_valid:
                submission_error = ("validation", err_msg)
            elif not api_configured:
                submission_error = (
                    "api_key",
                    "Cannot execute query: `OPENAI_API_KEY` is not configured. "
                    "Please configure your API key in the project `.env` file and restart the application.",
                )
            elif df is None:
                submission_error = ("no_data", "No dataset loaded to analyze.")
            else:
                with st.spinner("Analyzing dataset..."):
                    try:
                        result = run_analysis_query(df, question_text)
                        st.session_state["last_result"] = result
                        st.session_state["last_question"] = question_text
                        st.session_state["history"].append({
                            "question": question_text,
                            "result": result,
                        })
                    except Exception as exc:
                        submission_error = (
                            "exception",
                            f"Agent execution encountered an unexpected error: {str(exc)}",
                        )

        # Render conversation content inside the scrollable container
        with messages_container:
            if submission_error:
                err_type, err_text = submission_error
                if err_type == "validation":
                    st.error(f"⚠️ {err_text}")
                elif err_type == "api_key":
                    st.error(f"❌ {err_text}")
                else:
                    st.error(f"❌ {err_text}")

            if df is None:
                st.info(
                    "👈 **No dataset loaded.** Please choose an example dataset or upload a CSV file "
                    "using the left sidebar to begin analysis."
                )
            elif not st.session_state["history"] and not submission_error:
                # Sensible welcome / empty state before any questions
                st.markdown(
                    f"""
                    <div class="chat-welcome">
                        <h3>Ready to analyze {st.session_state['dataset_name']}</h3>
                        <p>
                            Ask questions in plain English to explore trends, calculate summary statistics,
                            perform group aggregations, or generate visual charts.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("Suggested starters:")
                sample_cols = st.columns(3)
                sample_cols[0].button(
                    "📊 Overview & Column Types",
                    use_container_width=True,
                    on_click=_set_sample_question,
                    args=("What columns are in this dataset and what are their data types?",),
                )
                sample_cols[1].button(
                    "📈 Summary Statistics",
                    use_container_width=True,
                    on_click=_set_sample_question,
                    args=("Provide summary statistics for the numeric columns.",),
                )
                sample_cols[2].button(
                    "👥 Group & Aggregate",
                    use_container_width=True,
                    on_click=_set_sample_question,
                    args=("Which group or category has the highest average values?",),
                )
            else:
                # Chronological chat conversation history
                for turn_idx, turn in enumerate(st.session_state["history"], start=1):
                    # User Message
                    with st.chat_message("user"):
                        st.markdown(turn["question"])

                    # Agent Response
                    with st.chat_message("assistant"):
                        res = turn["result"]

                        # 1. Execution errors or notices
                        if res.error and not res.answer:
                            st.error(f"❌ **Execution Error:** {res.error}")
                        elif res.error:
                            st.warning(f"⚠️ **Execution Notice:** {res.error}")

                        # 2. Main Conversational Agent Answer
                        if res.answer:
                            st.markdown(res.answer)

                        # 3. Interactive Visualizations (Plotly Figures)
                        if res.figures:
                            for fig_idx, fig in enumerate(res.figures, start=1):
                                fig.update_layout(
                                    template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                )
                                st.plotly_chart(
                                    fig,
                                    use_container_width=True,
                                    key=f"plotly_fig_{turn_idx}_{fig_idx}",
                                )

                        # 4. Tool Execution Details (Collapsible Expander)
                        if res.tool_calls:
                            with st.expander(
                                f"🛠️ Tool Execution Details ({len(res.tool_calls)} call(s), {res.rounds} round(s))",
                                expanded=False,
                            ):
                                for i, tc in enumerate(res.tool_calls, start=1):
                                    summary_info = format_tool_call_summary(tc)
                                    badge = "✅" if not summary_info["is_error"] else "❌"
                                    st.markdown(f"**{i}. `{tc.tool_name}`** — {badge} {summary_info['summary']}")
                                    st.caption(f"**Arguments:** `{json.dumps(tc.arguments)}`")
                                    if summary_info["is_error"]:
                                        st.error(f"Error: {summary_info['summary']}")
                                    if i < len(res.tool_calls):
                                        st.divider()


if __name__ == "__main__":
    main()
