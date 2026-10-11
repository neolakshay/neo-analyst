# tests/test_validator.py
"""
Comprehensive tests for the Output Grounding Validator (Phase 2 Iteration 1).

Covers all 16 Phase 5 test scenarios:
1. Numerical claim consistent with relevant tool output.
2. Contradictory numerical claim.
3. Numeric formatting differences (commas, decimals, currency).
4. Number appearing in an unrelated tool result.
5. Structured tool error followed by a false success claim.
6. Failed tool followed by an honest explanation of the failure.
7. Failed operation followed by a successful retry.
8. Successful partial result alongside an unrelated failed tool.
9. Invented statistical significance claim when no significance test was performed.
10. Causal claim based only on correlation.
11. Claim that a visualization exists when generation failed / no figure exists.
12. Answer that cannot be confidently validated (UNVERIFIABLE).
13. Preservation of the original answer.
14. Preservation of tool-call records and figures.
15. Correct interaction between grounding status and AgentResult.is_success.
16. Malformed or unexpected tool-output structures.
"""

import math
from unittest.mock import MagicMock
import pandas as pd
import plotly.graph_objects as go
import pytest

from app.agent import (
    AnalysisAgent,
    AgentResult,
    ToolCallRecord,
    GroundingStatus,
    GroundingResult,
    validate_grounding,
)


# ── Fixtures & Helpers ────────────────────────────────────────────────────────

def make_tc_record(name: str, output: any, args: dict = None) -> ToolCallRecord:
    """Helper to create a ToolCallRecord."""
    return ToolCallRecord(
        tool_name=name,
        arguments=args or {},
        output=output,
        raw_output=None,
    )


def make_agent_completion(content: str = None, tool_calls: list = None, finish_reason: str = None):
    """Helper to create a mocked completion for AnalysisAgent."""
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls or []
    choice.finish_reason = finish_reason
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: 16 CORE GROUNDING VALIDATION SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputGroundingValidator:
    """Test suite covering deterministic output grounding validation."""

    def test_01_numerical_claim_consistent_with_tool_output(self):
        """1. Numerical claim consistent with relevant tool output is marked VERIFIED."""
        tc = make_tc_record(
            "group_and_aggregate",
            {
                "status": "success",
                "results": [
                    {"department": "Engineering", "salary_mean": 2273750.0},
                    {"department": "HR", "salary_mean": 942000.0},
                ],
            },
        )
        answer = "The average salary in Engineering is $2,273,750."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.VERIFIED
        assert res.is_verified is True
        assert res.is_flagged is False
        assert any("Engineering" in c and "2273750" in c for c in res.claims_checked)

    def test_02_clearly_contradictory_numerical_claim(self):
        """2. A clearly contradictory numerical claim is marked FLAGGED."""
        tc = make_tc_record(
            "group_and_aggregate",
            {
                "status": "success",
                "results": [
                    {"department": "Engineering", "salary_mean": 2273750.0},
                ],
            },
        )
        answer = "The average salary for Engineering is $50,000."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.FLAGGED
        assert res.is_flagged is True
        assert res.is_verified is False
        assert any("Engineering" in r and "50,000" in r for r in res.reasons)

    @pytest.mark.parametrize("formatted_val", [
        "$2,273,750",
        "2273750",
        "2,273,750.00",
        "$2273750.0",
        "2,273,750",
    ])
    def test_03_numeric_formatting_differences(self, formatted_val):
        """3. Numeric formatting differences (commas, decimals, currency) are normalized."""
        tc = make_tc_record(
            "group_and_aggregate",
            {
                "status": "success",
                "results": [{"department": "Engineering", "salary_mean": 2273750.0}],
            },
        )
        answer = f"The Engineering average salary was calculated as {formatted_val}."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.VERIFIED
        assert res.is_verified is True
        assert res.is_flagged is False

    def test_04_number_appearing_in_unrelated_tool_result(self):
        """4. Avoid matching a number merely because it appears in an unrelated tool result.

        Here, get_dataset_overview returned rows: 50.
        The model claims Engineering average salary is $50.
        The validator must NOT treat 50 as verified because it appeared in rows count;
        it must flag the Engineering salary contradiction.
        """
        tc_ov = make_tc_record(
            "get_dataset_overview",
            {"rows": 50, "columns": 6},
        )
        tc_agg = make_tc_record(
            "group_and_aggregate",
            {
                "status": "success",
                "results": [{"department": "Engineering", "salary_mean": 2273750.0}],
            },
        )
        answer = "The average salary in Engineering is $50."
        res = validate_grounding(answer, tool_call_records=[tc_ov, tc_agg])

        assert res.status == GroundingStatus.FLAGGED
        assert res.is_flagged is True
        assert any("Engineering" in r and "50" in r for r in res.reasons)

    def test_05_structured_tool_error_followed_by_false_success_claim(self):
        """5. A structured tool error followed by a false success claim is FLAGGED."""
        tc = make_tc_record(
            "filter_data",
            {"status": "error", "error": "Unknown operator 'bad_op'"},
        )
        answer = "Filtering completed successfully! Found 25 matching employees."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.FLAGGED
        assert res.is_flagged is True
        assert any("filter_data failed" in r for r in res.reasons)

    def test_06_failed_tool_followed_by_honest_explanation(self):
        """6. A failed tool followed by an honest explanation of failure is NOT flagged."""
        tc = make_tc_record(
            "filter_data",
            {"status": "error", "error": "Unknown operator 'bad_op'"},
        )
        answer = "The filter operation failed with an error: Unknown operator 'bad_op'. Please check the operator."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status != GroundingStatus.FLAGGED
        assert res.is_flagged is False
        assert any("acknowledged failure" in c for c in res.claims_checked)

    def test_07_failed_operation_followed_by_successful_retry(self):
        """7. A failed operation followed by a successful retry is NOT flagged for prior failure."""
        tc_fail = make_tc_record(
            "filter_data",
            {"status": "error", "error": "Invalid syntax"},
        )
        tc_success = make_tc_record(
            "filter_data",
            {"status": "success", "matched_rows": 14, "total_rows": 50},
        )
        answer = "Filtering succeeded, found 14 matching employees."
        res = validate_grounding(answer, tool_call_records=[tc_fail, tc_success])

        assert res.status == GroundingStatus.VERIFIED
        assert res.is_verified is True
        assert res.is_flagged is False
        assert any("14" in c for c in res.claims_checked)

    def test_08_successful_partial_result_alongside_unrelated_failed_tool(self):
        """8. A successful partial result alongside an unrelated failed tool is not falsely flagged."""
        tc_ov = make_tc_record(
            "get_dataset_overview",
            {"rows": 50, "columns": 6},
        )
        tc_fail = make_tc_record(
            "filter_data",
            {"status": "error", "error": "Unknown column 'bonus'"},
        )
        answer = (
            "The dataset has 50 rows. However, filtering on column 'bonus' failed "
            "because the column does not exist."
        )
        res = validate_grounding(answer, tool_call_records=[tc_ov, tc_fail])

        assert res.status == GroundingStatus.VERIFIED
        assert res.is_verified is True
        assert res.is_flagged is False
        assert any("50 rows" in c for c in res.claims_checked)

    def test_09_invented_statistical_significance_claim(self):
        """9. An invented statistical significance claim is FLAGGED."""
        tc = make_tc_record(
            "analyze_correlations",
            {"matrix": {"salary": {"experience_years": 0.8525}}},
        )
        answer = "The correlation is 0.85 and is statistically significant at p < 0.001."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.FLAGGED
        assert res.is_flagged is True
        assert any("statistical significance" in r.lower() for r in res.reasons)

    def test_10_causal_claim_based_only_on_correlation(self):
        """10. A causal claim based only on correlation is FLAGGED."""
        tc = make_tc_record(
            "analyze_correlations",
            {"matrix": {"salary": {"experience_years": 0.8525}}},
        )
        answer = "The correlation of 0.85 proves that gaining experience causes higher salary."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.FLAGGED
        assert res.is_flagged is True
        assert any("causal" in r.lower() for r in res.reasons)

    def test_11_claim_that_visualization_exists_when_generation_failed(self):
        """11. A claim that a visualization exists when figures list is empty is FLAGGED."""
        tc = make_tc_record(
            "generate_visualization",
            {"status": "error", "error": "Unknown chart type 'pie'"},
        )
        answer = "I have generated a pie chart of departments for you."
        res = validate_grounding(answer, tool_call_records=[tc], figures=[])

        assert res.status == GroundingStatus.FLAGGED
        assert res.is_flagged is True
        assert any("visualization" in r.lower() for r in res.reasons)

    def test_12_answer_that_cannot_be_confidently_validated_is_unverifiable(self):
        """12. Qualitative or speculative statements without matching data are UNVERIFIABLE."""
        answer = "By 2030, tech salaries will likely increase due to automation and demand for AI skills."
        res = validate_grounding(answer, tool_call_records=[])

        assert res.status == GroundingStatus.UNVERIFIABLE
        assert res.is_unverifiable is True
        assert res.is_flagged is False
        assert res.is_verified is False

    def test_13_preservation_of_original_answer(self, clean_df):
        """13. The original model answer string is strictly preserved."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_agent_completion(
            content="Original unmutated text.", finish_reason="stop"
        )
        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Question")

        assert result.answer == "Original unmutated text."

    def test_14_preservation_of_tool_call_records_and_figures(self, clean_df):
        """14. Tool call records and Plotly figures are preserved in AgentResult."""
        mock_client = MagicMock()
        tc = MagicMock()
        tc.id = "c1"
        tc.type = "function"
        tc.function.name = "get_dataset_overview"
        tc.function.arguments = "{}"
        resp1 = make_agent_completion(tool_calls=[tc], finish_reason="tool_calls")
        resp2 = make_agent_completion(content="Overview complete.", finish_reason="stop")
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Overview")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "get_dataset_overview"
        assert result.grounding is not None

    def test_15_interaction_between_grounding_status_and_is_success(self):
        """15. Correct interaction between grounding status and AgentResult.is_success.

        Requirements:
        - Concrete grounding failure -> is_success is False.
        - UNVERIFIABLE answer -> is_success remains True (not presumed false).
        - VERIFIED answer -> is_success is True.
        - Execution error -> is_success is False, error is set.
        """
        # Clean verified
        res_v = AgentResult(
            answer="Verified",
            grounding=GroundingResult(status=GroundingStatus.VERIFIED),
        )
        assert res_v.is_success is True
        assert res_v.is_verified is True
        assert res_v.is_flagged is False
        assert res_v.execution_success is True

        # Unverifiable
        res_u = AgentResult(
            answer="Speculative",
            grounding=GroundingResult(status=GroundingStatus.UNVERIFIABLE),
        )
        assert res_u.is_success is True       # Not presumed false!
        assert res_u.is_verified is False
        assert res_u.is_flagged is False
        assert res_u.execution_success is True

        # Flagged
        res_f = AgentResult(
            answer="Contradictory",
            grounding=GroundingResult(status=GroundingStatus.FLAGGED),
        )
        assert res_f.is_success is False      # Flagged failure cannot be presented as success!
        assert res_f.is_verified is False
        assert res_f.is_flagged is True
        assert res_f.execution_success is True  # Execution ran, but semantic content is flagged
        assert res_f.error is None            # Conflation avoided: execution error is None!

        # Execution error
        res_err = AgentResult(
            answer="",
            error="API Connection Error",
            grounding=None,
        )
        assert res_err.is_success is False
        assert res_err.execution_success is False

    def test_16_malformed_or_unexpected_tool_output_structures(self):
        """16. Malformed or unexpected tool-output structures do not crash the validator."""
        tc_none = make_tc_record("custom_tool", None)
        tc_str = make_tc_record("custom_tool", "not a dict")
        tc_empty = make_tc_record("filter_data", {})
        tc_bad_res = make_tc_record("group_and_aggregate", {"results": "not a list"})

        answer = "Some general response about data."
        res = validate_grounding(
            answer, tool_call_records=[tc_none, tc_str, tc_empty, tc_bad_res]
        )

        assert res.status == GroundingStatus.UNVERIFIABLE
        assert res.is_unverifiable is True
        assert res.is_flagged is False


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: INTEGRATION WITH ANALYSIS_AGENT
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentGroundingIntegration:
    """Test grounding validator integration inside AnalysisAgent.run()."""

    def test_agent_run_populates_grounding_result(self, clean_df):
        """AnalysisAgent.run() automatically attaches GroundingResult."""
        mock_client = MagicMock()
        tc = MagicMock()
        tc.id = "c1"
        tc.type = "function"
        tc.function.name = "get_dataset_overview"
        tc.function.arguments = "{}"
        resp1 = make_agent_completion(tool_calls=[tc], finish_reason="tool_calls")
        resp2 = make_agent_completion(
            content="The dataset has 5 rows and 4 columns.", finish_reason="stop"
        )
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "How many rows?")

        assert result.grounding is not None
        assert isinstance(result.grounding, GroundingResult)
        assert result.grounding.status == GroundingStatus.VERIFIED
        assert result.is_success is True
        assert result.is_verified is True

    def test_agent_run_contradiction_is_flagged_and_sets_is_success_false(self, clean_df):
        """When the model outputs contradictory text, grounding is FLAGGED and is_success is False."""
        mock_client = MagicMock()
        tc = MagicMock()
        tc.id = "c1"
        tc.type = "function"
        tc.function.name = "get_dataset_overview"
        tc.function.arguments = "{}"
        resp1 = make_agent_completion(tool_calls=[tc], finish_reason="tool_calls")
        resp2 = make_agent_completion(
            content="The dataset has 9999 rows.", finish_reason="stop"
        )
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Rows?")

        assert result.grounding is not None
        assert result.grounding.status == GroundingStatus.FLAGGED
        assert result.is_flagged is True
        assert result.is_success is False
        # execution error is None, preventing conflation
        assert result.error is None
        assert result.execution_success is True

    def test_agent_result_to_dict_includes_grounding_fields(self, clean_df):
        """AgentResult.to_dict() serializes grounding, is_verified, and is_flagged."""
        mock_client = MagicMock()
        resp = make_agent_completion(content="Done.", finish_reason="stop")
        mock_client.chat.completions.create.return_value = resp

        agent = AnalysisAgent(client=mock_client)
        result = agent.run(clean_df, "Simple question")

        d = result.to_dict()
        assert "grounding" in d
        assert "is_verified" in d
        assert "is_flagged" in d
        assert "execution_success" in d
        assert d["grounding"]["status"] in ("VERIFIED", "FLAGGED", "UNVERIFIABLE")


# ══════════════════════════════════════════════════════════════════════════════
# TESTS: ADVERSARIAL REGRESSION PASS (6 CORE RISK AREAS)
# ══════════════════════════════════════════════════════════════════════════════

class TestValidatorAdversarialRegression:
    """Adversarial regression test suite focused on specific grounding risks."""

    # ── 1. Correlation Precision ──────────────────────────────────────────────

    def test_correlation_precision_exact_value(self):
        """Exact correlation value (0.8525) is marked VERIFIED."""
        tc = make_tc_record(
            "analyze_correlations",
            {"status": "success", "matrix": {"salary": {"experience_years": 0.8525}}},
        )
        answer = "The correlation between salary and experience_years is 0.8525."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.VERIFIED
        assert res.is_verified is True
        assert res.is_flagged is False

    @pytest.mark.parametrize("valid_r", ["0.85", "0.853", "0.8525"])
    def test_correlation_precision_valid_roundings(self, valid_r):
        """Correct rounded values (0.85, 0.853) are marked VERIFIED."""
        tc = make_tc_record(
            "analyze_correlations",
            {"status": "success", "matrix": {"salary": {"experience_years": 0.8525}}},
        )
        answer = f"The correlation between salary and experience is {valid_r}."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.VERIFIED
        assert res.is_verified is True
        assert res.is_flagged is False

    def test_correlation_precision_column_stem_matching(self):
        """Natural-language column stem 'experience' matches column 'experience_years'."""
        tc = make_tc_record(
            "analyze_correlations",
            {"status": "success", "matrix": {"salary": {"experience_years": 0.8525}}},
        )
        answer = "The correlation between salary and experience is 0.85."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.VERIFIED
        assert res.is_verified is True
        assert any("salary and experience_years" in c for c in res.claims_checked)

    @pytest.mark.parametrize("flawed_r", ["0.90", "0.81", "0.99", "-0.85", "0.20"])
    def test_correlation_precision_clearly_incorrect_coefficients(self, flawed_r):
        """Materially incorrect correlation coefficients (0.90, 0.81, etc.) are FLAGGED."""
        tc = make_tc_record(
            "analyze_correlations",
            {"status": "success", "matrix": {"salary": {"experience_years": 0.8525}}},
        )
        answer = f"The correlation between salary and experience is {flawed_r}."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.FLAGGED
        assert res.is_flagged is True
        assert res.is_verified is False
        assert any("Numerical contradiction" in r for r in res.reasons)

    # ── 2. Negation and Causal-Claim Detection ─────────────────────────────────

    @pytest.mark.parametrize("disclaimer", [
        "Correlation does not prove causation.",
        "Correlation does not necessarily prove causation.",
        "Correlation does not directly prove causation.",
        "Correlation does not automatically prove causation.",
        "Correlation cannot prove causation.",
    ])
    def test_negation_disclaimer_with_adverbs_not_flagged(self, disclaimer):
        """Disclaimers with intervening adverbs are recognized as negated and NOT flagged."""
        tc = make_tc_record(
            "analyze_correlations",
            {"status": "success", "matrix": {"salary": {"experience_years": 0.8525}}},
        )
        answer = f"The correlation is 0.85. Note that {disclaimer}"
        res = validate_grounding(answer, tool_call_records=[tc])

        # Must not be flagged as an unsupported causal claim
        assert res.status != GroundingStatus.FLAGGED
        assert res.is_flagged is False

    def test_negation_disclaimer_followed_by_explicit_causal_claim(self):
        """A disclaimer followed by an explicit unsupported causal claim in the next sentence is FLAGGED."""
        tc = make_tc_record(
            "analyze_correlations",
            {"status": "success", "matrix": {"salary": {"experience_years": 0.8525}}},
        )
        answer = "Correlation does not prove causation. However, higher experience causes higher salary."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.FLAGGED
        assert res.is_flagged is True
        assert any("causal relationship" in r for r in res.reasons)

    def test_negation_contrastive_clause_with_causal_claim(self):
        """A contrasting clause asserting causation despite an initial disclaimer is FLAGGED."""
        tc = make_tc_record(
            "analyze_correlations",
            {"status": "success", "matrix": {"salary": {"experience_years": 0.8525}}},
        )
        answer = "While correlation does not prove causation, experience causes higher salary."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.FLAGGED
        assert res.is_flagged is True
        assert any("causal relationship" in r for r in res.reasons)

    def test_negation_multi_sentence_isolation(self):
        """Negation token in sentence 1 does not suppress flagging causal claim in sentence 2."""
        tc = make_tc_record(
            "analyze_correlations",
            {"status": "success", "matrix": {"salary": {"experience_years": 0.8525}}},
        )
        answer = "There were no missing values in the dataset. Gaining experience causes higher salary."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.FLAGGED
        assert res.is_flagged is True

    def test_negation_token_not_treated_as_safe_harbor(self):
        """Phrasing like 'no doubts that experience causes' is NOT treated as a negated disclaimer."""
        tc = make_tc_record(
            "analyze_correlations",
            {"status": "success", "matrix": {"salary": {"experience_years": 0.8525}}},
        )
        answer = "There are no doubts that experience causes higher salary."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.FLAGGED
        assert res.is_flagged is True

    # ── 3. Mixed Correct and Incorrect Claims ──────────────────────────────────

    def test_mixed_claims_verified_plus_contradictory_metric(self):
        """An answer with one verified claim and one contradictory claim must be FLAGGED."""
        tc = make_tc_record(
            "group_and_aggregate",
            {
                "status": "success",
                "results": [
                    {"department": "Engineering", "salary_mean": 2273750.0},
                    {"department": "HR", "salary_mean": 942000.0},
                ],
            },
        )
        answer = (
            "The average salary in Engineering is $2,273,750. "
            "In HR, the average salary is $50,000."
        )
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.FLAGGED
        assert res.is_flagged is True
        assert res.is_verified is False
        assert any("HR" in r and "50,000" in r for r in res.reasons)

    def test_mixed_claims_same_group_alternative_contradiction(self):
        """Citing a correct number alongside an alternative contradictory number for the same group is FLAGGED."""
        tc = make_tc_record(
            "group_and_aggregate",
            {
                "status": "success",
                "results": [{"department": "Engineering", "salary_mean": 2273750.0}],
            },
        )
        answer = "The Engineering average salary is $2,273,750, or alternatively $50,000."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.FLAGGED
        assert res.is_flagged is True
        assert res.is_verified is False

    def test_conflicting_claims_highest_average_salary_wrong_department(self):
        """Claiming a non-maximum department has the highest average salary is FLAGGED."""
        tc = make_tc_record(
            "group_and_aggregate",
            {
                "status": "success",
                "results": [
                    {"department": "Engineering", "salary_mean": 2273750.0},
                    {"department": "HR", "salary_mean": 942000.0},
                    {"department": "Sales", "salary_mean": 780000.0},
                ],
            },
        )
        answer = "Sales has the highest average salary."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.FLAGGED
        assert res.is_flagged is True
        assert any("Sales" in r and "Engineering" in r for r in res.reasons)

    def test_conflicting_claims_highest_average_salary_multiple_departments(self):
        """Conflicting claims asserting multiple departments as highest is FLAGGED."""
        tc = make_tc_record(
            "group_and_aggregate",
            {
                "status": "success",
                "results": [
                    {"department": "Engineering", "salary_mean": 2273750.0},
                    {"department": "HR", "salary_mean": 942000.0},
                ],
            },
        )
        answer = "Engineering has the highest average salary, but HR also has the highest average salary."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.FLAGGED
        assert res.is_flagged is True
        assert any("Multiple departments" in r for r in res.reasons)

    def test_verified_highest_average_salary_department(self):
        """Correctly identifying the highest average salary department is VERIFIED."""
        tc = make_tc_record(
            "group_and_aggregate",
            {
                "status": "success",
                "results": [
                    {"department": "Engineering", "salary_mean": 2273750.0},
                    {"department": "HR", "salary_mean": 942000.0},
                ],
            },
        )
        answer = "Engineering has the highest average salary with $2,273,750."
        res = validate_grounding(answer, tool_call_records=[tc])

        assert res.status == GroundingStatus.VERIFIED
        assert res.is_verified is True
        assert res.is_flagged is False
        assert any("Engineering" in c and "highest" in c for c in res.claims_checked)

    # ── 4. Multiple Tool Results ───────────────────────────────────────────────

    def test_multiple_tools_number_in_unrelated_tool_cannot_validate(self):
        """A number from an overview result (e.g. 50 rows) cannot validate an unrelated salary claim."""
        tc_ov = make_tc_record("get_dataset_overview", {"rows": 50, "columns": 5})
        tc_agg = make_tc_record(
            "group_and_aggregate",
            {"status": "success", "results": [{"department": "Engineering", "salary_mean": 2273750.0}]},
        )
        answer = "The average salary in Engineering is $50."
        res = validate_grounding(answer, tool_call_records=[tc_ov, tc_agg])

        assert res.status == GroundingStatus.FLAGGED
        assert res.is_flagged is True

    def test_multiple_tools_unrelated_failure_preserves_supported_partial_answer(self):
        """Unrelated failure in filter_data does not invalidate supported overview claim."""
        tc_ov = make_tc_record("get_dataset_overview", {"rows": 120, "columns": 5})
        tc_fail = make_tc_record("filter_data", {"status": "error", "error": "Invalid column"})

        answer = "The dataset has 120 rows."
        res = validate_grounding(answer, tool_call_records=[tc_ov, tc_fail])

        assert res.status == GroundingStatus.VERIFIED
        assert res.is_verified is True
        assert res.is_flagged is False

    def test_multiple_tools_failure_followed_by_retry_validates_against_retry(self):
        """Failed tool call followed by successful retry validates against the retry result."""
        tc_fail = make_tc_record("filter_data", {"status": "error", "error": "Syntax error"})
        tc_retry = make_tc_record("filter_data", {"status": "success", "matched_rows": 42})

        answer = "Filtering succeeded and found 42 matching rows."
        res = validate_grounding(answer, tool_call_records=[tc_fail, tc_retry])

        assert res.status == GroundingStatus.VERIFIED
        assert res.is_verified is True
        assert res.is_flagged is False

    def test_dataset_overview_formatted_rows_with_commas(self):
        """Dataset overview row count formatted with commas (e.g. '1,200 rows') matches correctly."""
        tc_ov = make_tc_record("get_dataset_overview", {"rows": 1200, "columns": 8})
        answer_valid = "The dataset has 1,200 rows."
        res_valid = validate_grounding(answer_valid, tool_call_records=[tc_ov])

        assert res_valid.status == GroundingStatus.VERIFIED
        assert res_valid.is_verified is True

        answer_invalid = "The dataset has 9,999 rows."
        res_invalid = validate_grounding(answer_invalid, tool_call_records=[tc_ov])

        assert res_invalid.status == GroundingStatus.FLAGGED
        assert res_invalid.is_flagged is True

    # ── 5. Grounding-Status Semantics ──────────────────────────────────────────

    def test_semantics_flagged_makes_is_success_false(self):
        """FLAGGED answers strictly result in is_success=False on AgentResult."""
        gr = GroundingResult(status=GroundingStatus.FLAGGED, reasons=["Contradiction"])
        result = AgentResult(answer="Contradictory claim", grounding=gr)

        assert result.is_success is False
        assert result.is_flagged is True
        assert result.is_verified is False
        assert result.execution_success is True
        assert result.error is None

    def test_semantics_execution_error_distinct_from_grounding_failure(self):
        """Execution errors remain distinct from grounding failures."""
        result = AgentResult(answer="", error="API Timeout Error")

        assert result.is_success is False
        assert result.execution_success is False
        assert result.is_flagged is False
        assert result.is_verified is False
        assert result.error == "API Timeout Error"
        assert result.grounding is None

    def test_semantics_unverifiable_not_treated_as_verified(self):
        """UNVERIFIABLE is NOT treated as VERIFIED, but does not falsely flag is_success."""
        gr = GroundingResult(status=GroundingStatus.UNVERIFIABLE, reasons=["Qualitative claim"])
        result = AgentResult(answer="Qualitative statement", grounding=gr)

        assert result.is_success is True
        assert result.is_verified is False
        assert result.is_flagged is False
        assert result.grounding.is_unverifiable is True

    def test_semantics_serialized_output_consistency(self):
        """AgentResult.to_dict() and GroundingResult.to_dict() maintain full consistency."""
        gr = GroundingResult(
            status=GroundingStatus.VERIFIED,
            reasons=["Matches"],
            claims_checked=["Claim 1"],
            details={"k": "v"},
        )
        res = AgentResult(answer="Good answer", grounding=gr)
        d = res.to_dict()

        assert d["is_success"] is True
        assert d["is_verified"] is True
        assert d["is_flagged"] is False
        assert d["execution_success"] is True
        assert d["grounding"]["status"] == "VERIFIED"
        assert d["grounding"]["is_verified"] is True
        assert d["grounding"]["is_flagged"] is False
        assert d["grounding"]["is_unverifiable"] is False

    # ── 6. Malformed Data Robustness ───────────────────────────────────────────

    @pytest.mark.parametrize("bad_answer", [123, None, {}, [], True])
    def test_malformed_non_string_answer_safe_handling(self, bad_answer):
        """Non-string answer types fail safely as UNVERIFIABLE without crashing."""
        res = validate_grounding(bad_answer)

        assert res.status == GroundingStatus.UNVERIFIABLE
        assert res.is_unverifiable is True
        assert res.is_flagged is False

    def test_malformed_unexpected_tool_record_types(self):
        """Unexpected tool record objects (None, int, dict without keys) do not crash."""
        bad_records = [None, 123, "string_record", {}, {"tool_name": "unknown"}]
        res = validate_grounding("Some general answer.", tool_call_records=bad_records)

        assert res.status == GroundingStatus.UNVERIFIABLE
        assert res.is_flagged is False

    def test_malformed_tool_output_nan_and_inf_metrics(self):
        """NaN and Inf values in tool outputs are handled defensively without false matches."""
        tc_nan = make_tc_record(
            "analyze_correlations",
            {"status": "success", "matrix": {"salary": {"exp": float("nan")}}},
        )
        tc_inf = make_tc_record(
            "group_and_aggregate",
            {"status": "success", "results": [{"department": "Eng", "salary_mean": float("inf")}]},
        )
        answer = "The correlation is 0.85 and Eng salary is $100,000."
        res = validate_grounding(answer, tool_call_records=[tc_nan, tc_inf])

        # Should not crash and should not falsely verify against NaN/Inf
        assert res.status == GroundingStatus.UNVERIFIABLE
        assert res.is_flagged is False
        assert res.is_verified is False

    def test_malformed_empty_tool_records_and_missing_output_fields(self):
        """Missing output fields in successful tool outputs fail safely."""
        tc_empty = make_tc_record("group_and_aggregate", {"status": "success"})
        tc_empty_corr = make_tc_record("analyze_correlations", {"status": "success"})
        answer = "Engineering has the highest average salary."
        res = validate_grounding(answer, tool_call_records=[tc_empty, tc_empty_corr])

        assert res.status == GroundingStatus.UNVERIFIABLE
        assert res.is_flagged is False

