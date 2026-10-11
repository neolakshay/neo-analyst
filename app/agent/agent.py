# app/agent/agent.py
"""
app.agent.agent
---------------
Core LLM-powered data analysis agent that orchestrates reasoning and tool
execution using the official OpenAI Python SDK.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional
import json
import os
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from app.agent.tools import TOOL_SCHEMAS, execute_tool, to_json_safe

# Load .env file automatically for local development
load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_ROUNDS = 6


@dataclass
class ToolCallRecord:
    """Record of an executed tool call."""
    tool_name: str
    arguments: dict
    output: Any
    raw_output: Any = None


@dataclass
class AgentResult:
    """The complete result returned by the analysis agent."""
    answer: str
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    figures: List[go.Figure] = field(default_factory=list)
    rounds: int = 0
    model: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert the result to a JSON-compatible dictionary."""
        return {
            "answer": self.answer,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "output": tc.output,
                }
                for tc in self.tool_calls
            ],
            "figures_count": len(self.figures),
            "rounds": self.rounds,
            "model": self.model,
            "error": self.error,
        }


def build_dataset_metadata(df: pd.DataFrame, preview_rows: int = 2) -> dict:
    """Generate concise dataset metadata without transmitting entire datasets."""
    n_rows, n_cols = df.shape
    missing_summary = {
        col: int(cnt)
        for col, cnt in df.isnull().sum().items()
        if cnt > 0
    }
    sample_preview = df.head(preview_rows).to_dict(orient="records")

    return {
        "rows": n_rows,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": missing_summary,
        "preview_sample": to_json_safe(sample_preview),
    }


class AnalysisAgent:
    """LLM data analysis agent that selects and executes analytical tools locally."""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the AnalysisAgent.

        Parameters
        ----------
        client : Optional[OpenAI]
            Pre-configured OpenAI client (useful for testing and dependency injection).
        api_key : Optional[str]
            API key if client is not provided. If omitted, read from OPENAI_API_KEY.
        model : Optional[str]
            Model name. If omitted, read from OPENAI_MODEL (defaults to gpt-4o-mini).
        max_rounds : int, default 6
            Maximum tool-calling rounds permitted per query.
        system_prompt : Optional[str]
            Custom system prompt override.
        """
        self.model = (
            model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        ).strip()
        self.max_rounds = max(1, max_rounds)
        self.system_prompt_override = system_prompt

        if client is not None:
            self.client = client
        else:
            resolved_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
            if not resolved_key:
                self.client = None  # Will raise clear error on run() if called without key
            else:
                self.client = OpenAI(api_key=resolved_key)

    def _ensure_client(self) -> OpenAI:
        """Verify the OpenAI client is configured, raising a clear error otherwise."""
        if self.client is not None:
            return self.client

        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise ValueError(
                "OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable "
                "or configure it in a local .env file."
            )
        self.client = OpenAI(api_key=api_key)
        return self.client

    def _create_system_prompt(self, df: pd.DataFrame) -> str:
        """Construct the system instructions with dataset schema and bounds."""
        metadata = build_dataset_metadata(df)
        base_prompt = self.system_prompt_override or (
            "You are an expert, technically sound data analysis assistant. "
            "You answer questions about the provided dataset by invoking the registered analytical tools. "
            "Guidelines:\n"
            "1. Ground all numbers and factual assertions strictly in tool results. Never guess numerical outcomes.\n"
            "2. Always check the dataset structure or summary stats before making assumptions.\n"
            "3. If a visualization is requested, use generate_visualization and inform the user of the chart created.\n"
            "4. When answering, be clear, concise, and provide actionable context.\n"
            "5. Never attempt to run arbitrary code or shell commands; only use registered analytical tools."
        )

        metadata_text = (
            f"\n\nDataset Summary:\n"
            f"- Total Rows: {metadata['rows']}\n"
            f"- Columns & Dtypes: {metadata['dtypes']}\n"
            f"- Columns with Missing Values: {metadata['missing_counts'] or 'None'}\n"
            f"- Sample Preview: {json.dumps(metadata['preview_sample'])}"
        )

        return f"{base_prompt}{metadata_text}"

    def run(self, df: pd.DataFrame, question: str) -> AgentResult:
        """Run the analytical agent on *df* to answer *question*.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset to analyze (remains unmutated).
        question : str
            The natural-language question or analytical task.

        Returns
        -------
        AgentResult
            Container with final answer, executed tool records, generated figures, and round metrics.
        """
        # ── 1. Input Validation ───────────────────────────────────────────────
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected a pandas DataFrame, got {type(df).__name__}."
            )
        if len(df.columns) == 0:
            raise ValueError("The DataFrame has no columns — nothing to analyze.")

        if not isinstance(question, str):
            raise TypeError(
                f"Question must be a string, got {type(question).__name__}."
            )
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")

        client = self._ensure_client()

        # ── 2. Message Setup ──────────────────────────────────────────────────
        messages: List[dict] = [
            {"role": "system", "content": self._create_system_prompt(df)},
            {"role": "user", "content": clean_question},
        ]

        tool_call_records: List[ToolCallRecord] = []
        figures: List[go.Figure] = []
        round_count = 0
        final_answer = ""
        error_msg: Optional[str] = None

        # ── 3. Tool-Calling Reasoning Loop ────────────────────────────────────
        while round_count < self.max_rounds:
            round_count += 1

            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                )
            except OpenAIError as oai_err:
                error_msg = f"OpenAI API error: {str(oai_err)}"
                break
            except Exception as api_err:
                error_msg = f"Unexpected error during model invocation: {str(api_err)}"
                break

            choice = response.choices[0]
            message = choice.message

            # Check if model made tool calls
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                # Model provided its final text response
                final_answer = message.content or ""
                break

            # Append the assistant message with tool calls
            assistant_msg_dict: dict = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_msg_dict)

            # Execute each requested tool call
            for tc in tool_calls:
                tool_name = tc.function.name
                raw_args = tc.function.arguments

                try:
                    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    if not isinstance(parsed_args, dict):
                        parsed_args = {}
                except json.JSONDecodeError as json_err:
                    model_out = {
                        "status": "error",
                        "error": f"Invalid JSON arguments for tool '{tool_name}': {str(json_err)}",
                    }
                    raw_out = None
                    parsed_args = {"raw_arguments": raw_args}
                else:
                    model_out, raw_out = execute_tool(df, tool_name, parsed_args)

                # Track figures if a chart was produced
                if isinstance(raw_out, go.Figure):
                    figures.append(raw_out)

                tool_call_records.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        arguments=parsed_args,
                        output=model_out,
                        raw_output=raw_out,
                    )
                )

                # Return tool result message to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(model_out),
                })

        # ── 4. Handle Loop Exit / Limit Reached ────────────────────────────────
        if round_count >= self.max_rounds and not final_answer and not error_msg:
            error_msg = f"Exhausted maximum tool-calling rounds ({self.max_rounds}) without final answer."
            final_answer = (
                "The analysis could not be completed within the permitted number of reasoning steps."
            )

        return AgentResult(
            answer=final_answer,
            tool_calls=tool_call_records,
            figures=figures,
            rounds=round_count,
            model=self.model,
            error=error_msg,
        )
