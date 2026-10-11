# app/agent/validator.py
"""
app.agent.validator
-------------------
Deterministic Output Grounding Validator for the Applied ML agent.

Evaluates whether the model's final natural-language answer is grounded in,
consistent with, or contradictory to the executed analytical tools.

Distinguishes three outcomes:
  1. VERIFIED: Specific verifiable claims examined in the answer are supported
     by the relevant analytical tool outputs.
  2. FLAGGED: Concrete inconsistencies, false success claims, unperformed statistical
     tests, causal claims on correlation, or contradictions were detected.
  3. UNVERIFIABLE: The answer cannot be deterministically verified or refuted using
     the available tool outputs (e.g. speculative, qualitative, or derived reasoning).
     UNVERIFIABLE is NOT equated with incorrect.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Dict, List, Optional, Tuple


class GroundingStatus(str, Enum):
    """The grounding validation status of an agent's answer."""
    VERIFIED = "VERIFIED"
    FLAGGED = "FLAGGED"
    UNVERIFIABLE = "UNVERIFIABLE"


@dataclass
class GroundingResult:
    """The structured result of output grounding validation."""
    status: GroundingStatus
    reasons: List[str] = field(default_factory=list)
    claims_checked: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_verified(self) -> bool:
        """Return True if specific claims were verified against tool outputs."""
        return self.status == GroundingStatus.VERIFIED

    @property
    def is_flagged(self) -> bool:
        """Return True if an explicit inconsistency or unsupported claim was flagged."""
        return self.status == GroundingStatus.FLAGGED

    @property
    def is_unverifiable(self) -> bool:
        """Return True if the answer could not be verified or refuted deterministically."""
        return self.status == GroundingStatus.UNVERIFIABLE

    def to_dict(self) -> dict:
        """Convert the result to a JSON-compatible dictionary."""
        return {
            "status": self.status.value,
            "reasons": self.reasons,
            "claims_checked": self.claims_checked,
            "details": self.details,
            "is_verified": self.is_verified,
            "is_flagged": self.is_flagged,
            "is_unverifiable": self.is_unverifiable,
        }


# ── Pattern Definitions ───────────────────────────────────────────────────────

# Registered analytical tools in neo-applied-ml
REGISTERED_TOOLS = {
    "get_dataset_overview",
    "get_summary_statistics",
    "filter_data",
    "analyze_correlations",
    "generate_visualization",
    "group_and_aggregate",
}

# Statistical significance claims (unsupported: none of our 6 tools computes p-values)
_STAT_SIG_PATTERNS = [
    re.compile(r'\bstatistically\s+significant\b', re.IGNORECASE),
    re.compile(r'\bstatistical\s+significance\b', re.IGNORECASE),
    re.compile(r'\bp\s*<\s*0?\.\d+', re.IGNORECASE),
    re.compile(r'\bp\s*=\s*0?\.\d+', re.IGNORECASE),
    re.compile(r'\bp-value\b', re.IGNORECASE),
    re.compile(r'\bsignificant\s+at\s+p\b', re.IGNORECASE),
    re.compile(r'\bsignificant\s+at\s+the\s+0?\.\d+\b', re.IGNORECASE),
]

# Causal claims based solely on correlation
_CAUSAL_PATTERNS = [
    re.compile(r'\bproves?\s+(?:that\s+)?(?:\w+\s+)*causes?\b', re.IGNORECASE),
    re.compile(r'\bproves?\s+causation\b', re.IGNORECASE),
    re.compile(r'\bestablishes?\s+(?:a\s+)?causal\b', re.IGNORECASE),
    re.compile(r'\bdemonstrates?\s+(?:that\s+)?(?:\w+\s+)*causes?\b', re.IGNORECASE),
    re.compile(r'\bshows?\s+(?:that\s+)?(?:\w+\s+)*causes?\b', re.IGNORECASE),
    re.compile(r'\bcauses?\s+(?:higher|lower|increased|decreased|an?\s+increase|a\s+decrease|salary\s+increase)\b', re.IGNORECASE),
]

# Visualization generation claims
_VIZ_CLAIM_PATTERNS = [
    re.compile(r'\b(?:have\s+|has\s+been\s+)?generated\s+(?:a|the)\s+(?:\w+\s+)*(?:chart|plot|graph|visualization|figure)\b', re.IGNORECASE),
    re.compile(r'\bcreated\s+(?:a|the)\s+(?:\w+\s+)*(?:chart|plot|graph|visualization|figure)\b', re.IGNORECASE),
    re.compile(r'\bhere\s+is\s+the\s+(?:\w+\s+)*(?:chart|plot|graph|visualization|figure)\b', re.IGNORECASE),
    re.compile(r'\bvisualization\s+(?:\w+\s+)*generated\s+successfully\b', re.IGNORECASE),
    re.compile(r'\b(?:pie|bar|line|scatter|box|histogram)\s+chart\s+(?:was\s+)?(?:created|generated)\b', re.IGNORECASE),
    re.compile(r'\bgenerated\s+a\s+(?:pie|bar|line|scatter|box|histogram)\s+chart\b', re.IGNORECASE),
]

# Claim that a specific registered tool ran
_CLAIMED_TOOL_RUN_RE = re.compile(
    r'\b(?:ran|executed|called|invoked|used)\s+(?:the\s+)?(get_dataset_overview|get_summary_statistics|filter_data|analyze_correlations|generate_visualization|group_and_aggregate)\b',
    re.IGNORECASE,
)

# Number extraction pattern (handles currency symbols, commas, decimals, negative numbers)
_NUMBER_TOKEN_RE = re.compile(
    r'[-+]?[\$£€]?\s*(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)'
)


# ── Helper Functions ──────────────────────────────────────────────────────────

def _safe_get_tool_name(tc: Any) -> str:
    """Safely extract tool_name from a ToolCallRecord or dict representation."""
    if isinstance(tc, dict):
        return str(tc.get("tool_name", "") or "")
    return str(getattr(tc, "tool_name", "") or "")


def _safe_get_tool_output(tc: Any) -> Any:
    """Safely extract output from a ToolCallRecord or dict representation."""
    if isinstance(tc, dict):
        return tc.get("output")
    return getattr(tc, "output", None)


def _is_negated_in_clause(text: str, match_start: int, window: int = 80) -> bool:
    """Check if the text in the same clause immediately preceding match_start contains a negation.

    Rules:
    - Bounded by sentence boundaries (., !, ?, newline) and contrastive conjunctions
      (however, but, although, while, whereas, nevertheless, yet, ;).
    - Allows 0 to 3 intervening words/adverbs (e.g. 'not necessarily prove', 'does not directly prove').
    - Does NOT allow negation in an earlier sentence or prior contrasting clause to bleed over.
    """
    start = max(0, match_start - window)
    prefix = text[start:match_start]
    clause_splits = list(re.finditer(
        r'[\n.!?]|;\s*|\b(?:however|but|although|while|whereas|nevertheless|yet)\b',
        prefix,
        re.IGNORECASE,
    ))
    if clause_splits:
        clause_start = clause_splits[-1].end()
        local_prefix = prefix[clause_start:].strip()
    else:
        local_prefix = prefix.strip()

    neg_pattern = re.compile(
        r'\b(?:cannot|does\s+not|doesn\'t|could\s+not|fails?\s+to|did\s+not|not|without|no\s+evidence\s+(?:that\s+)?(?:\w+\s+)?)(?:\s+(?:necessarily|directly|automatically|always|definitively|strictly|clearly))?\s*$'
        r'|\bno\s*$',
        re.IGNORECASE,
    )
    return bool(neg_pattern.search(local_prefix))


# Alias for backwards compatibility
_is_negated_in_preceding_text = _is_negated_in_clause


def _column_matches_text(col_name: str, text: str) -> bool:
    """Return True if col_name or its key constituent words appear in text."""
    clean_col = col_name.lower()
    text_lower = text.lower()
    if clean_col in text_lower:
        return True
    # Check underscore-separated words, e.g. 'experience_years' -> 'experience'
    words = [w for w in re.split(r'[_]+', clean_col) if len(w) > 3]
    if words and any(w in text_lower for w in words):
        return True
    return False


def _clean_and_parse_number(token: str) -> Optional[float]:
    """Parse a numeric string token into a float, stripping symbols and commas."""
    cleaned = re.sub(r'[\$£€,\s]', '', str(token))
    try:
        val = float(cleaned)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except (ValueError, TypeError):
        return None


def _extract_numbers(text: str) -> List[Tuple[float, str]]:
    """Extract numeric values and their raw representations from a string."""
    results = []
    for m in _NUMBER_TOKEN_RE.finditer(str(text)):
        raw = m.group(0).strip()
        val = _clean_and_parse_number(raw)
        if val is not None:
            results.append((val, raw))
    return results


def _is_close(v1: float, v2: float, rel_tol: float = 0.02, abs_tol: float = 0.02) -> bool:
    """Return True if two floats are close within tolerance."""
    if math.isnan(v1) or math.isnan(v2) or math.isinf(v1) or math.isinf(v2):
        return False
    return math.isclose(v1, v2, rel_tol=rel_tol, abs_tol=abs_tol)


# ── Main Validator Function ───────────────────────────────────────────────────

def validate_grounding(
    answer: str,
    tool_call_records: Optional[List[Any]] = None,
    figures: Optional[List[Any]] = None,
) -> GroundingResult:
    """Perform deterministic output grounding validation on the agent's answer.

    Parameters
    ----------
    answer : str
        The model's final natural-language response.
    tool_call_records : Optional[List[ToolCallRecord]]
        List of ToolCallRecord objects containing executed tool names and outputs.
    figures : Optional[List[go.Figure]]
        List of Plotly Figure objects produced by the agent.

    Returns
    -------
    GroundingResult
        Structured result indicating VERIFIED, FLAGGED, or UNVERIFIABLE status,
        along with reasons and list of claims checked.
    """
    records = tool_call_records or []
    figs = figures or []
    flags: List[str] = []
    claims_checked: List[str] = []
    details: Dict[str, Any] = {}

    if answer is None or not isinstance(answer, str):
        clean_answer = str(answer or "").strip() if not isinstance(answer, (list, dict, set)) else ""
    else:
        clean_answer = answer.strip()

    if not clean_answer:
        return GroundingResult(
            status=GroundingStatus.UNVERIFIABLE,
            reasons=["Answer is empty or invalid — nothing to validate."],
            details={"empty_answer": True},
        )

    # ── 1. Check Claims of Tool Execution When No Tools Were Called ───────────
    executed_tool_names = {_safe_get_tool_name(tc) for tc in records}

    for match in _CLAIMED_TOOL_RUN_RE.finditer(clean_answer):
        claimed_tool = match.group(1).lower()
        if not _is_negated_in_clause(clean_answer, match.start()):
            if claimed_tool not in {t.lower() for t in executed_tool_names}:
                flags.append(
                    f"Answer claims tool '{match.group(1)}' was executed, "
                    "but no such tool call exists in the execution records."
                )

    # ── 2. Check Unsupported Statistical Significance Claims ─────────────────
    for pattern in _STAT_SIG_PATTERNS:
        match = pattern.search(clean_answer)
        if match and not _is_negated_in_clause(clean_answer, match.start()):
            flags.append(
                f"Answer claims statistical significance ('{match.group(0)}'), "
                "but no statistical significance test was performed by any analytical tool."
            )
            break

    # ── 3. Check Causal Claims Based on Correlation ───────────────────────────
    correlation_executed = "analyze_correlations" in executed_tool_names
    has_correlation_text = bool(re.search(r'\bcorrelation\b', clean_answer, re.IGNORECASE))

    if correlation_executed or has_correlation_text:
        for pattern in _CAUSAL_PATTERNS:
            match = pattern.search(clean_answer)
            if match and not _is_negated_in_clause(clean_answer, match.start()):
                flags.append(
                    f"Answer asserts a causal relationship ('{match.group(0)}') "
                    "based solely on correlation analysis, which cannot establish causation."
                )
                break

    # ── 4. Check Visualization Claims Against Actual Figures ──────────────────
    has_viz_claim = False
    for pattern in _VIZ_CLAIM_PATTERNS:
        match = pattern.search(clean_answer)
        if match and not _is_negated_in_clause(clean_answer, match.start()):
            has_viz_claim = True
            break

    if has_viz_claim:
        if len(figs) == 0:
            flags.append(
                "Answer claims a visualization was generated, but no figure was produced."
            )
        else:
            claims_checked.append("Visualization generation verified (figure object is present)")

    # ── 5. Track Unresolved Tool Failures vs Recovered Retries ─────────────────
    tool_history: Dict[str, List[Any]] = {}
    for tc in records:
        name = _safe_get_tool_name(tc)
        out = _safe_get_tool_output(tc)
        tool_history.setdefault(name, []).append(out)

    unresolved_failures: Dict[str, str] = {}
    for tool_name, outputs in tool_history.items():
        if not outputs:
            continue
        last_out = outputs[-1]
        if isinstance(last_out, dict) and last_out.get("status") == "error":
            unresolved_failures[tool_name] = str(last_out.get("error", "Error"))

    # Check if answer falsely claims success for an unresolved failure
    for failed_tool, err_msg in unresolved_failures.items():
        acknowledged = any(
            w in clean_answer.lower()
            for w in ("error", "failed", "could not", "unable", "not supported", "failure")
        )

        if failed_tool == "filter_data":
            false_success_match = re.search(
                r'\b(?:filter(?:ing)?\s+(?:completed\s+|was\s+)?successfully|filter(?:ing)?\s+succeeded|found\s+\d+\s+(?:\w+\s+)*(?:matching|matched))\b',
                clean_answer,
                re.IGNORECASE,
            )
            if false_success_match and not _is_negated_in_clause(clean_answer, false_success_match.start()):
                flags.append(
                    f"Answer claims filter operation succeeded, but filter_data failed with error: {err_msg}"
                )
            elif acknowledged:
                claims_checked.append("Answer acknowledged failure in filter_data")

        elif failed_tool == "generate_visualization":
            # Handled by Section 4 if viz was claimed
            if acknowledged:
                claims_checked.append("Answer acknowledged failure in generate_visualization")

        else:
            false_success_match = re.search(
                rf'\b(?:{failed_tool}\s+succeeded|operation\s+succeeded)\b',
                clean_answer,
                re.IGNORECASE,
            )
            if false_success_match and not acknowledged:
                flags.append(
                    f"Answer claims operation succeeded, but tool '{failed_tool}' failed with error: {err_msg}"
                )
            elif acknowledged:
                claims_checked.append(f"Answer acknowledged failure in {failed_tool}")

    # ── 6. Numerical & Output Consistency Checks ──────────────────────────────
    for tc in records:
        tool_name = _safe_get_tool_name(tc)
        output = _safe_get_tool_output(tc)

        if not isinstance(output, dict) or output.get("status") == "error":
            continue

        # 6a. Check group_and_aggregate metric consistency
        if tool_name == "group_and_aggregate":
            results_list = output.get("results")
            if isinstance(results_list, list):
                # Identify maximum group for numerical metrics (e.g. salary_mean)
                max_metrics: Dict[str, Tuple[str, float]] = {}
                for row in results_list:
                    if not isinstance(row, dict):
                        continue
                    g_name = None
                    for k, v in row.items():
                        if isinstance(v, str):
                            g_name = v
                        elif isinstance(v, (int, float)) and not isinstance(v, bool):
                            if not math.isnan(v) and not math.isinf(v):
                                if k not in max_metrics or v > max_metrics[k][1]:
                                    max_metrics[k] = (str(g_name or ""), float(v))

                # Check explicit highest claims in the answer
                _HIGHEST_PATTERNS = [
                    re.compile(r'\b([A-Za-z]+)\s+(?:also\s+)?(?:has|had)\s+the\s+highest\s+(?:average|mean)\s+(?:salary|score|price|amount)\b', re.I),
                    re.compile(r'\b(?:highest\s+(?:average|mean)\s+(?:salary|score|price|amount)\s+(?:department|category|group)?\s*(?:is|was)\s*(?:in\s+)?([A-Za-z]+))\b', re.I),
                    re.compile(r'\b(?:department|category|group)\s+with\s+(?:the\s+)?highest\s+(?:average|mean)\s+(?:salary|score|price|amount)\s+(?:is|was)\s+([A-Za-z]+)\b', re.I),
                ]
                _STOP_WORDS = {"also", "it", "which", "that", "this", "and", "but", "the", "a", "an"}
                claimed_highest_groups = []
                for h_pat in _HIGHEST_PATTERNS:
                    for hm in h_pat.finditer(clean_answer):
                        if not _is_negated_in_clause(clean_answer, hm.start()):
                            word = hm.group(1).strip()
                            if word.lower() not in _STOP_WORDS:
                                claimed_highest_groups.append(word)

                if claimed_highest_groups:
                    mean_metric = next((k for k in max_metrics if "mean" in k or "salary" in k or "avg" in k), None)
                    if mean_metric and max_metrics[mean_metric][0]:
                        actual_max_dept, actual_max_val = max_metrics[mean_metric]
                        unique_claimed = list(dict.fromkeys(claimed_highest_groups))
                        if len(unique_claimed) > 1:
                            flags.append(
                                f"Contradictory claims: Multiple departments claimed to have the highest average salary ({', '.join(unique_claimed)})."
                            )
                        else:
                            claimed_dept = unique_claimed[0]
                            if claimed_dept.lower() == actual_max_dept.lower():
                                claims_checked.append(
                                    f"Department '{actual_max_dept}' verified as having the highest {mean_metric} ({actual_max_val})"
                                )
                            else:
                                flags.append(
                                    f"Contradiction: Answer claims '{claimed_dept}' has the highest average salary, "
                                    f"but '{actual_max_dept}' has the highest ({actual_max_val})."
                                )

                # Validate per-row metrics
                for row in results_list:
                    if not isinstance(row, dict):
                        continue

                    # Find group name (string) and numeric metrics
                    group_str = None
                    metrics: Dict[str, float] = {}
                    for k, v in row.items():
                        if isinstance(v, str):
                            group_str = v
                        elif isinstance(v, (int, float)) and not isinstance(v, bool):
                            if not math.isnan(v) and not math.isinf(v):
                                metrics[k] = float(v)

                    if not group_str or group_str.lower() not in clean_answer.lower():
                        continue

                    # The answer discusses this group! Inspect sentences mentioning it
                    sentences = [
                        s.strip()
                        for s in re.split(r'[.\n!?;]+', clean_answer)
                        if group_str.lower() in s.lower()
                    ]

                    all_answer_numbers = _extract_numbers(clean_answer)

                    for sentence in sentences:
                        numbers_in_sentence = _extract_numbers(sentence)
                        for metric_name, expected_val in metrics.items():
                            metric_kw = [kw for kw in metric_name.lower().split('_') if kw]
                            is_discussed = (
                                any(kw in sentence.lower() for kw in metric_kw)
                                or "average" in sentence.lower()
                                or "mean" in sentence.lower()
                                or "salary" in sentence.lower()
                            )
                            if not is_discussed:
                                continue

                            matched = any(
                                _is_close(n, expected_val)
                                for n, _raw in numbers_in_sentence
                            )
                            # Check for contradictory numbers in the sentence that do not match ANY metric for this group
                            contradictory_numbers = [
                                (n, raw)
                                for n, raw in numbers_in_sentence
                                if not any(_is_close(n, m_val) for m_val in metrics.values())
                                and (n > 100 or "$" in raw or "£" in raw or "€" in raw)
                            ]

                            if matched and not contradictory_numbers:
                                claims_checked.append(
                                    f"Group '{group_str}' {metric_name} ({expected_val}) matches answer"
                                )
                            elif contradictory_numbers:
                                found_str = ", ".join(raw for _n, raw in contradictory_numbers)
                                flags.append(
                                    f"Numerical contradiction: Group '{group_str}' {metric_name} reported as {found_str}, "
                                    f"expected {expected_val}."
                                )
                            elif not matched and numbers_in_sentence:
                                expected_in_answer = any(
                                    _is_close(n, expected_val)
                                    for n, _raw in all_answer_numbers
                                )
                                if not expected_in_answer:
                                    found_str = ", ".join(raw for _n, raw in numbers_in_sentence)
                                    flags.append(
                                        f"Numerical contradiction: Group '{group_str}' {metric_name} reported as {found_str}, "
                                        f"expected {expected_val}."
                                    )

        # 6b. Check filter_data row counts
        elif tool_name == "filter_data":
            matched_rows = output.get("matched_rows")
            if isinstance(matched_rows, (int, float)) and not math.isnan(matched_rows):
                match_count_pattern = re.search(
                    r'\b(?:found|matched|there\s+(?:are|were))\s+(\d+)\s+(?:matching\s+)?(?:employees|rows|records|items)\b',
                    clean_answer,
                    re.IGNORECASE,
                )
                if match_count_pattern and not _is_negated_in_clause(clean_answer, match_count_pattern.start()):
                    claimed_count = float(match_count_pattern.group(1))
                    if _is_close(claimed_count, float(matched_rows)):
                        claims_checked.append(f"Filter matched rows ({int(matched_rows)}) matches answer")
                    else:
                        flags.append(
                            f"Numerical contradiction: Answer claims {int(claimed_count)} matching rows, "
                            f"but filter_data returned {int(matched_rows)}."
                        )
                elif matched_rows == 0:
                    # Tool returned 0 rows: verify answer does not claim positive matches
                    pos_match = re.search(r'\bfound\s+(\d+)\s+employees\b', clean_answer, re.IGNORECASE)
                    if pos_match and float(pos_match.group(1)) > 0:
                        flags.append(
                            f"Numerical contradiction: Answer claims {pos_match.group(1)} matching rows, "
                            "but filter_data returned 0 matching rows."
                        )
                    elif re.search(r'\b(?:0|no|zero)\s+(?:matching\s+)?(?:rows|employees|records)\b', clean_answer, re.IGNORECASE):
                        claims_checked.append("Filter matched rows (0) verified in answer")

        # 6c. Check analyze_correlations values and direction
        elif tool_name == "analyze_correlations":
            matrix = output.get("matrix")
            if isinstance(matrix, dict):
                # Check for explicit correlation value citation
                corr_val_match = re.search(
                    r'\b(?:correlation(?:\s+coefficient)?\s+(?:of|is|was|between\s+[\w\s]+\s+is)\s*[:=]?\s*|r\s*=\s*)([-+]?0?\.\d+|\b[-+]?1(?:\.0+)?\b)',
                    clean_answer,
                    re.IGNORECASE,
                )
                if corr_val_match and not _is_negated_in_clause(clean_answer, corr_val_match.start()):
                    claimed_r = _clean_and_parse_number(corr_val_match.group(1))
                    raw_str = corr_val_match.group(1)
                    if claimed_r is not None:
                        # Find matching column pair discussed
                        found_pair_match = False
                        for col1, row_dict in matrix.items():
                            if not isinstance(row_dict, dict):
                                continue
                            for col2, exp_r in row_dict.items():
                                if col1 != col2 and isinstance(exp_r, (int, float)):
                                    if math.isnan(exp_r) or math.isinf(exp_r):
                                        continue
                                    if _column_matches_text(col1, clean_answer) and _column_matches_text(col2, clean_answer):
                                        dec_places = len(raw_str.split('.')[-1]) if '.' in raw_str else 0
                                        is_one_dec = dec_places == 1
                                        matches_precision = (
                                            _is_close(claimed_r, float(exp_r), abs_tol=0.01)
                                            or (is_one_dec and claimed_r == round(float(exp_r), 1))
                                        )
                                        if matches_precision:
                                            claims_checked.append(
                                                f"Correlation between {col1} and {col2} ({exp_r}) matches answer"
                                            )
                                            found_pair_match = True
                                            break
                                        else:
                                            flags.append(
                                                f"Numerical contradiction: Correlation between {col1} and {col2} "
                                                f"reported as {claimed_r}, expected {exp_r}."
                                            )
                                            found_pair_match = True
                                            break
                            if found_pair_match:
                                break

                # Check direction (positive vs negative)
                for col1, row_dict in matrix.items():
                    if not isinstance(row_dict, dict):
                        continue
                    for col2, exp_r in row_dict.items():
                        if col1 != col2 and isinstance(exp_r, (int, float)) and not math.isnan(exp_r):
                            if _column_matches_text(col1, clean_answer) and _column_matches_text(col2, clean_answer):
                                if exp_r < -0.2 and re.search(r'\bpositive\s+correlation\b', clean_answer, re.IGNORECASE):
                                    pos_idx = clean_answer.lower().find("positive correlation")
                                    if not _is_negated_in_clause(clean_answer, pos_idx):
                                        flags.append(
                                            f"Directional contradiction: Tool returned negative correlation ({exp_r}) "
                                            f"between {col1} and {col2}, but answer asserts positive correlation."
                                        )
                                elif exp_r > 0.2 and re.search(r'\bnegative\s+correlation\b', clean_answer, re.IGNORECASE):
                                    neg_idx = clean_answer.lower().find("negative correlation")
                                    if not _is_negated_in_clause(clean_answer, neg_idx):
                                        flags.append(
                                            f"Directional contradiction: Tool returned positive correlation ({exp_r}) "
                                            f"between {col1} and {col2}, but answer asserts negative correlation."
                                        )

        # 6d. Check get_dataset_overview dimensions
        elif tool_name == "get_dataset_overview":
            rows = output.get("rows")
            cols = output.get("columns")
            if isinstance(rows, (int, float)) and not math.isnan(rows):
                dim_match = re.search(
                    r'\b(?:dataset\s+has|data\s+contains|has)\s+([\d,]+)\s+rows\b',
                    clean_answer,
                    re.IGNORECASE,
                )
                if dim_match:
                    claimed_rows = _clean_and_parse_number(dim_match.group(1))
                    if claimed_rows is not None and _is_close(claimed_rows, float(rows)):
                        claims_checked.append(f"Dataset overview row count ({int(rows)} rows) matches answer")
                    else:
                        flags.append(
                            f"Numerical contradiction: Answer claims {dim_match.group(1)} rows, "
                            f"but dataset overview returned {int(rows)} rows."
                        )

    # ── 7. Synthesize Validation Result ───────────────────────────────────────
    details["flags_count"] = len(flags)
    details["claims_checked_count"] = len(claims_checked)
    details["unresolved_failures"] = list(unresolved_failures.keys())

    if flags:
        status = GroundingStatus.FLAGGED
        reasons = flags
    elif claims_checked:
        status = GroundingStatus.VERIFIED
        reasons = claims_checked
    else:
        status = GroundingStatus.UNVERIFIABLE
        reasons = [
            "Answer contains qualitative, derived, or speculative statements "
            "that cannot be deterministically verified against executed tool outputs."
        ]

    return GroundingResult(
        status=status,
        reasons=reasons,
        claims_checked=claims_checked,
        details=details,
    )
