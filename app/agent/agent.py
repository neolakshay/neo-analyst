# app/agent/agent.py
"""
app.agent.agent
---------------
Core LLM-powered data analysis agent that orchestrates reasoning and tool
execution using the official OpenAI Python SDK.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
import json
import os
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from app.agent.tools import TOOL_SCHEMAS, REGISTERED_TOOL_NAMES, execute_tool, to_json_safe
from app.agent.validator import GroundingResult, GroundingStatus, validate_grounding

# Load .env file automatically for local development
load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_ROUNDS = 6

# finish_reason values that indicate the model produced a normal text response
_FINISH_REASON_STOP = "stop"
# finish_reason when the model requested tool calls
_FINISH_REASON_TOOL_CALLS = "tool_calls"
# finish_reason values that indicate non-recoverable termination
_FINISH_REASON_LENGTH = "length"
_FINISH_REASON_CONTENT_FILTER = "content_filter"


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
    grounding: Optional[GroundingResult] = None

    @property
    def execution_success(self) -> bool:
        """Return True if execution completed with a non-empty answer and no execution error."""
        return self.error is None and bool(str(self.answer or "").strip())

    @property
    def is_success(self) -> bool:
        """Return True only if execution succeeded AND the answer was not flagged by grounding.

        Semantics:
        - UNVERIFIABLE answers with successful execution remain is_success=True
          (they are not presumed false).
        - FLAGGED answers are never presented as successful (is_success=False).
        - Execution failures (errors) are always is_success=False.
        - To check execution alone, inspect result.execution_success or result.error is None.
        - To check grounding specifically, inspect result.grounding.status.
        """
        if not self.execution_success:
            return False
        if self.grounding is not None and self.grounding.is_flagged:
            return False
        return True

    @property
    def is_verified(self) -> bool:
        """Return True only if specific claims were verified by grounding validation."""
        return self.is_success and self.grounding is not None and self.grounding.is_verified

    @property
    def is_flagged(self) -> bool:
        """Return True if grounding validation detected a concrete contradiction or unsupported claim."""
        return self.grounding is not None and self.grounding.is_flagged

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
            "is_success": self.is_success,
            "execution_success": self.execution_success,
            "is_verified": self.is_verified,
            "is_flagged": self.is_flagged,
            "grounding": self.grounding.to_dict() if self.grounding else None,
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


def _parse_tool_arguments(raw_args: Any, tool_name: str) -> Tuple[dict, Optional[dict]]:
    """Parse and validate tool call arguments from the model.

    Parameters
    ----------
    raw_args : Any
        The raw arguments string or object from the model's tool call.
    tool_name : str
        Name of the tool being called (used in error messages).

    Returns
    -------
    Tuple[dict, Optional[dict]]
        (parsed_args, error_output) where:
        - If parsing succeeds, parsed_args is the dict and error_output is None.
        - If parsing fails, parsed_args is the best-effort dict (may be empty)
          and error_output is a structured error dict to send back to the model.
    """
    # Case 1: raw_args is already a dict (pre-parsed, e.g. in tests or some SDK versions)
    if isinstance(raw_args, dict):
        return raw_args, None

    # Case 2: raw_args is a string — attempt JSON decode
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError as json_err:
            error_out = {
                "status": "error",
                "error": f"Invalid JSON arguments for tool '{tool_name}': {str(json_err)}",
            }
            return {}, error_out

        # Decoded successfully — must be a JSON object (dict) to be valid tool args
        if not isinstance(parsed, dict):
            error_out = {
                "status": "error",
                "error": (
                    f"Tool '{tool_name}' arguments must be a JSON object ({{...}}), "
                    f"got {type(parsed).__name__}: {str(raw_args)[:120]}"
                ),
            }
            return {}, error_out

        return parsed, None

    # Case 3: Unexpected type (None, list, int, etc.)
    error_out = {
        "status": "error",
        "error": (
            f"Tool '{tool_name}' received arguments of unexpected type "
            f"{type(raw_args).__name__}; expected a JSON object string or dict."
        ),
    }
    return {}, error_out


def _validate_tool_call_entry(tc: Any) -> Tuple[Optional[str], Optional[str], Any, Optional[str]]:
    """Validate a single tool-call entry from the model's response.

    Extracts tool_name, call_id, and raw_args defensively without allowing
    AttributeError or TypeError to escape.

    Parameters
    ----------
    tc : Any
        A tool call object from the model's response.

    Returns
    -------
    Tuple[Optional[str], Optional[str], Any, Optional[str]]
        (tool_name, call_id, raw_args, error_string)

        On success: tool_name and call_id are strings, raw_args is the argument
        payload, and error_string is None.

        On failure: tool_name may be a fallback string, call_id may be None or a
        fallback, raw_args is None, and error_string describes the problem.
    """
    # Validate function attribute
    fn = getattr(tc, "function", None)
    call_id = getattr(tc, "id", None)

    # call_id must be a non-empty string for valid tool message routing
    if not isinstance(call_id, str) or not call_id:
        call_id_repr = repr(call_id)
        return (
            "<unknown>",
            call_id if isinstance(call_id, str) else "<invalid-id>",
            None,
            f"Malformed tool-call entry: 'id' is missing or not a string (got {call_id_repr}).",
        )

    if fn is None:
        return (
            "<unknown>",
            call_id,
            None,
            f"Malformed tool-call entry (id={call_id!r}): 'function' attribute is missing.",
        )

    tool_name = getattr(fn, "name", None)
    if not isinstance(tool_name, str) or not tool_name:
        tool_name_repr = repr(tool_name)
        return (
            "<unknown>",
            call_id,
            None,
            f"Malformed tool-call entry (id={call_id!r}): function name is missing or not a string "
            f"(got {tool_name_repr}).",
        )

    raw_args = getattr(fn, "arguments", None)
    return tool_name, call_id, raw_args, None


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
            "5. Never attempt to run arbitrary code or shell commands; only use registered analytical tools.\n"
            "6. In group_and_aggregate, 'group_by' columns and 'aggregations' keys MUST strictly be existing dataset column names from the provided schema. Aggregation operations specify what to calculate. Never use derived or target metric names as aggregation keys (e.g. to count entities, map an existing source column to ['count']; never use '{column}_count' as a key).\n"
            "7. If a tool call returns an error indicating an unknown column or invalid argument, inspect the available columns in the error output, correct the arguments to use existing columns, and retry the tool call."
        )

        metadata_text = (
            f"\n\nDataset Summary:\n"
            f"- Total Rows: {metadata['rows']}\n"
            f"- Columns & Dtypes: {metadata['dtypes']}\n"
            f"- Columns with Missing Values: {metadata['missing_counts'] or 'None'}\n"
            f"- Sample Preview: {json.dumps(metadata['preview_sample'])}"
        )

        return f"{base_prompt}{metadata_text}"

    def _extract_response_choice(
        self, response: Any
    ) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
        """Safely extract the first choice and its finish_reason from a completion response.

        Returns
        -------
        Tuple[Optional[Any], Optional[str], Optional[str]]
            (message, finish_reason, error_string)

            On success: message is the response message object, finish_reason is a
            string (possibly None if not present), and error_string is None.

            On failure: message is None, finish_reason is None, and error_string
            describes the structural problem.
        """
        choices = getattr(response, "choices", None)
        if not choices:
            return None, None, (
                "Malformed API response: 'choices' is missing or empty. "
                "The model did not return any completion choices."
            )

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            return None, None, (
                "Malformed API response: choice[0].message is None. "
                "Cannot extract model response."
            )

        # finish_reason may be absent on some providers; treat it as unknown if so
        finish_reason = getattr(choice, "finish_reason", None)
        return message, finish_reason, None

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

            # ── 3a. Validate response structure and extract finish_reason ─────
            message, finish_reason, resp_error = self._extract_response_choice(response)
            if resp_error is not None:
                error_msg = resp_error
                break

            # ── 3b. Route on finish_reason ────────────────────────────────────
            #
            # "stop"        → model produced a complete text answer; extract it.
            # "tool_calls"  → model wants to invoke tools; fall through to dispatch.
            # "length"      → output was cut off by the token limit; treat as failure.
            # "content_filter" → output was blocked by safety policy; treat as failure.
            # None / other  → unknown/missing; inspect tool_calls to decide defensively.
            #
            if finish_reason == _FINISH_REASON_LENGTH:
                error_msg = (
                    "Model response was truncated because it hit the token limit "
                    "(finish_reason='length'). The answer may be incomplete. "
                    "Consider asking a more focused question or reducing the dataset size."
                )
                break

            if finish_reason == _FINISH_REASON_CONTENT_FILTER:
                error_msg = (
                    "Model response was blocked by the content safety filter "
                    "(finish_reason='content_filter'). The question or dataset content "
                    "may have triggered a policy restriction."
                )
                break

            # For "stop", "tool_calls", None, or any unknown value: inspect tool_calls
            # to determine the correct path. This is the most robust approach because:
            #   - Some providers omit finish_reason entirely.
            #   - "stop" with no tool_calls → final text answer.
            #   - "tool_calls" always comes with tool_calls populated.
            #   - Unknown finish_reason: follow tool_calls presence as the ground truth.
            tool_calls = getattr(message, "tool_calls", None)

            if not tool_calls:
                # No tool calls → model produced its final text response.
                # Accept this for "stop", None, and unknown finish_reason values.
                if finish_reason not in (_FINISH_REASON_STOP, _FINISH_REASON_TOOL_CALLS, None):
                    # Unknown non-error finish_reason with no tool_calls and no text — warn
                    # but still capture whatever content was returned.
                    error_msg = (
                        f"Model finished with unexpected finish_reason={finish_reason!r} "
                        "and no tool calls. Captured available response content."
                    )
                final_answer = message.content or ""
                break

            # ── 3c. Build assistant message dict safely ───────────────────────
            # Validate each tool-call entry before building the assistant message,
            # so structural problems are caught before we touch tc.function.* fields.
            valid_tcs = []
            fatal_tc_error: Optional[str] = None

            for tc in tool_calls:
                tool_name, call_id, raw_args, tc_error = _validate_tool_call_entry(tc)
                if tc_error is not None:
                    # A malformed entry cannot be safely routed back to the model
                    # (we have no valid call_id for the tool message). Abort the round.
                    fatal_tc_error = tc_error
                    break
                valid_tcs.append((tc, tool_name, call_id, raw_args))

            if fatal_tc_error is not None:
                error_msg = fatal_tc_error
                break

            # All entries are structurally valid — build the assistant message
            assistant_msg_dict: dict = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": raw_args if isinstance(raw_args, str)
                                         else json.dumps(raw_args or {}),
                        },
                    }
                    for _tc, tool_name, call_id, raw_args in valid_tcs
                ],
            }
            messages.append(assistant_msg_dict)

            # ── 3d. Execute each validated tool call ──────────────────────────
            for _tc, tool_name, call_id, raw_args in valid_tcs:
                # Parse arguments — always produces a usable dict and optional error
                parsed_args, arg_parse_error = _parse_tool_arguments(raw_args, tool_name)

                if arg_parse_error is not None:
                    # Argument parsing failed: send error back to the model for recovery
                    model_out = arg_parse_error
                    raw_out = None
                else:
                    # Execute the tool locally (execute_tool rejects unregistered tools)
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
                    "tool_call_id": call_id,
                    "content": json.dumps(model_out),
                })

        # ── 4. Handle Loop Exit / Limit Reached ────────────────────────────────
        # The loop exits in one of four states:
        #   a) final_answer set     → model produced a complete text response (success)
        #   b) error_msg set        → API/structural/policy error broke the loop (failure)
        #   c) round_count >= max_rounds, no final_answer, no prior error
        #      → model kept requesting tool calls and hit the round limit (failure)
        #   d) round_count >= max_rounds with an error already set (handled in loop)
        if round_count >= self.max_rounds and not final_answer and not error_msg:
            error_msg = (
                f"Exhausted maximum tool-calling rounds ({self.max_rounds}) without a final answer. "
                "The model continued requesting tool calls past the allowed limit."
            )

        # ── 5. Output Grounding Validation ──────────────────────────────────
        grounding_result: Optional[GroundingResult] = None
        if final_answer and not error_msg:
            grounding_result = validate_grounding(
                final_answer,
                tool_call_records=tool_call_records,
                figures=figures,
            )

        return AgentResult(
            answer=final_answer,
            tool_calls=tool_call_records,
            figures=figures,
            rounds=round_count,
            model=self.model,
            error=error_msg,
            grounding=grounding_result,
        )
