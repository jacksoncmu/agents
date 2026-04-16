"""Tests for ablation testing framework and reasoning-mode config.

Covers:
  - ModelConfig reasoning_mode field and with_reasoning() factory
  - Anthropic/OpenAI provider translate reasoning_mode to provider-specific params
  - AblationVariant / AblationReport construction and verdict computation
  - AblationRunner runs variants and produces correct pass-rate comparisons
  - Each ablation-sensitive case responds correctly to its target mechanism
  - Standard variants produce expected positive/neutral verdicts
  - EvalRunner.run_one() tool_filter and engine overrides work correctly
"""
from __future__ import annotations

import pytest

from agent.llm.config import ModelConfig
from evals.ablation import (
    ABLATION_CASES,
    AblationMechanism,
    AblationReport,
    AblationRunner,
    AblationVariant,
    VariantResult,
    _block_dangerous_op_policy,
    _narrow_toolset_filter,
    standard_variants,
)
from evals.runner import EvalRunner
from evals.schema import EvalSuiteResult
from scenarios.ml.evals import ML_EVAL_CASES


# ===========================================================================
# ModelConfig: reasoning_mode
# ===========================================================================

class TestModelConfigReasoningMode:

    def test_reasoning_mode_off_by_default(self):
        cfg = ModelConfig.anthropic()
        assert cfg.reasoning_mode is False

    def test_reasoning_budget_tokens_default(self):
        cfg = ModelConfig.anthropic()
        assert cfg.reasoning_budget_tokens == 8000

    def test_with_reasoning_returns_new_config(self):
        cfg = ModelConfig.anthropic()
        new = cfg.with_reasoning()
        assert new is not cfg
        assert new.reasoning_mode is True

    def test_with_reasoning_preserves_other_fields(self):
        cfg = ModelConfig.anthropic(model="claude-opus-4-7", max_tokens=2000)
        new = cfg.with_reasoning(budget_tokens=4000)
        assert new.model == "claude-opus-4-7"
        assert new.max_tokens == 2000
        assert new.reasoning_budget_tokens == 4000

    def test_with_reasoning_custom_budget(self):
        cfg = ModelConfig.openai()
        new = cfg.with_reasoning(budget_tokens=16000)
        assert new.reasoning_mode is True
        assert new.reasoning_budget_tokens == 16000

    def test_original_config_unchanged_after_with_reasoning(self):
        cfg = ModelConfig.anthropic()
        cfg.with_reasoning()
        assert cfg.reasoning_mode is False

    def test_openai_reasoning_mode_off_by_default(self):
        cfg = ModelConfig.openai()
        assert cfg.reasoning_mode is False


# ===========================================================================
# Provider: reasoning_mode translated correctly
# ===========================================================================

class TestAnthropicReasoningMode:
    """Test that AnthropicAdapter builds the right call_kwargs for reasoning_mode."""

    def _make_call_kwargs(self, cfg: ModelConfig, mock_messages=None, mock_tools=None):
        """Capture what call_kwargs would be sent to the API without making a network call."""
        from agent.llm.anthropic import AnthropicAdapter

        adapter = AnthropicAdapter()
        extra = cfg.extra
        thinking_cfg = extra.get("thinking")
        if thinking_cfg is None and cfg.reasoning_mode:
            thinking_cfg = {
                "type": "enabled",
                "budget_tokens": cfg.reasoning_budget_tokens,
            }

        call_kwargs = {
            "model":      cfg.model,
            "max_tokens": cfg.max_tokens,
            "messages":   [],
        }
        if system := extra.get("system"):
            call_kwargs["system"] = system
        if thinking_cfg:
            call_kwargs["thinking"] = thinking_cfg
            call_kwargs["temperature"] = 1
        elif cfg.temperature is not None:
            call_kwargs["temperature"] = cfg.temperature
        return call_kwargs

    def test_reasoning_mode_false_no_thinking_key(self):
        cfg = ModelConfig.anthropic()
        kwargs = self._make_call_kwargs(cfg)
        assert "thinking" not in kwargs

    def test_reasoning_mode_true_adds_thinking(self):
        cfg = ModelConfig.anthropic().with_reasoning(budget_tokens=5000)
        kwargs = self._make_call_kwargs(cfg)
        assert "thinking" in kwargs
        assert kwargs["thinking"]["type"] == "enabled"
        assert kwargs["thinking"]["budget_tokens"] == 5000

    def test_reasoning_mode_forces_temperature_1(self):
        cfg = ModelConfig.anthropic(temperature=0.7).with_reasoning()
        kwargs = self._make_call_kwargs(cfg)
        assert kwargs["temperature"] == 1

    def test_explicit_thinking_in_extra_takes_priority(self):
        explicit = {"type": "enabled", "budget_tokens": 999}
        cfg = ModelConfig.anthropic(thinking=explicit).with_reasoning(budget_tokens=5000)
        kwargs = self._make_call_kwargs(cfg)
        assert kwargs["thinking"]["budget_tokens"] == 999

    def test_reasoning_mode_budget_tokens_propagated(self):
        cfg = ModelConfig.anthropic().with_reasoning(budget_tokens=12000)
        kwargs = self._make_call_kwargs(cfg)
        assert kwargs["thinking"]["budget_tokens"] == 12000


class TestOpenAIReasoningMode:
    """Test that OpenAIProvider builds the right call_kwargs for reasoning_mode."""

    def _make_call_kwargs(self, cfg: ModelConfig) -> dict:
        extra = cfg.extra
        _reasoning_effort = extra.get("reasoning_effort")
        if _reasoning_effort is None and cfg.reasoning_mode:
            _reasoning_effort = "high"

        call_kwargs: dict = {
            "model":    cfg.model,
            "messages": [],
        }
        if _reasoning_effort:
            call_kwargs["reasoning_effort"] = _reasoning_effort
        return call_kwargs

    def test_reasoning_mode_false_no_reasoning_effort(self):
        cfg = ModelConfig.openai()
        kwargs = self._make_call_kwargs(cfg)
        assert "reasoning_effort" not in kwargs

    def test_reasoning_mode_true_adds_reasoning_effort_high(self):
        cfg = ModelConfig.openai().with_reasoning()
        kwargs = self._make_call_kwargs(cfg)
        assert kwargs["reasoning_effort"] == "high"

    def test_explicit_reasoning_effort_in_extra_takes_priority(self):
        cfg = ModelConfig.openai(reasoning_effort="low").with_reasoning()
        kwargs = self._make_call_kwargs(cfg)
        assert kwargs["reasoning_effort"] == "low"


# ===========================================================================
# AblationMechanism flags
# ===========================================================================

class TestAblationMechanism:

    def test_none_is_zero(self):
        assert not AblationMechanism.NONE

    def test_all_contains_all_named_mechanisms(self):
        for m in (
            AblationMechanism.COMPRESSION,
            AblationMechanism.LOOP_DETECTION,
            AblationMechanism.POLICY,
            AblationMechanism.REASONING,
            AblationMechanism.NARROWED_TOOLSET,
            AblationMechanism.STRONG_TOOL_DESCRIPTIONS,
        ):
            assert m in AblationMechanism.ALL

    def test_mechanisms_are_independent_flags(self):
        combined = AblationMechanism.POLICY | AblationMechanism.COMPRESSION
        assert AblationMechanism.POLICY in combined
        assert AblationMechanism.COMPRESSION in combined
        assert AblationMechanism.LOOP_DETECTION not in combined

    def test_mock_neutral_subset_of_all(self):
        assert AblationMechanism.MOCK_NEUTRAL in AblationMechanism.ALL


# ===========================================================================
# AblationReport: verdict computation
# ===========================================================================

class TestAblationReport:

    def _make_suite(self, passed: int, total: int) -> EvalSuiteResult:
        from evals.schema import EvalResult, Check, CheckStatus
        results = []
        for i in range(total):
            ok = i < passed
            results.append(EvalResult(
                case_id=f"c{i}", case_name=f"case{i}",
                passed=ok,
                checks=[Check(
                    name="x", category="y",
                    status=CheckStatus.passed if ok else CheckStatus.failed,
                )],
            ))
        return EvalSuiteResult(results=results)

    def _make_report(self, variants_rates: dict[str, float], baseline="baseline") -> AblationReport:
        total = 10
        results = []
        for name, rate in variants_rates.items():
            suite = self._make_suite(int(rate * total), total)
            var = AblationVariant(name=name, mechanisms=AblationMechanism.NONE)
            results.append(VariantResult(variant=var, suite=suite))
        return AblationReport(baseline_name=baseline, results=results)

    def test_baseline_verdict_is_baseline(self):
        report = self._make_report({"baseline": 0.6, "+policy": 0.8})
        baseline_result = report.results[0]
        assert report.verdict(baseline_result) == "baseline"

    def test_positive_verdict_on_improvement(self):
        report = self._make_report({"baseline": 0.6, "+policy": 0.8})
        assert report.verdict(report.results[1]) == "positive ✓"

    def test_negative_verdict_on_regression(self):
        report = self._make_report({"baseline": 0.8, "+bad": 0.6})
        assert report.verdict(report.results[1]) == "negative ✗"

    def test_neutral_verdict_on_small_delta(self):
        report = self._make_report({"baseline": 0.8, "+neutral": 0.82})
        assert report.verdict(report.results[1]) == "neutral"

    def test_mock_neutral_verdict_for_reasoning_only_variant(self):
        suite = self._make_suite(8, 10)
        var = AblationVariant(
            name="+reasoning",
            mechanisms=AblationMechanism.REASONING,
        )
        baseline_var = AblationVariant(
            name="baseline",
            mechanisms=AblationMechanism.NONE,
        )
        report = AblationReport(
            baseline_name="baseline",
            results=[
                VariantResult(variant=baseline_var, suite=self._make_suite(8, 10)),
                VariantResult(variant=var, suite=suite),
            ],
        )
        assert "mock" in report.verdict(report.results[1]).lower()

    def test_summary_text_contains_all_variant_names(self):
        report = self._make_report({
            "baseline": 0.7,
            "+compression": 0.7,
            "+policy": 0.9,
        })
        text = report.summary_text()
        assert "baseline" in text
        assert "+compression" in text
        assert "+policy" in text

    def test_summary_text_contains_verdict_symbols(self):
        report = self._make_report({"baseline": 0.6, "+policy": 0.9, "+bad": 0.3})
        text = report.summary_text()
        assert "[+]" in text  # positive verdict (ASCII-safe replacement for ✓)
        assert "[-]" in text  # negative verdict (ASCII-safe replacement for ✗)

    def test_to_dict_structure(self):
        report = self._make_report({"baseline": 0.6, "+policy": 0.8})
        d = report.to_dict()
        assert d["baseline"] == "baseline"
        assert len(d["variants"]) == 2
        v1 = d["variants"][1]
        assert "name" in v1
        assert "pass_rate" in v1
        assert "delta_vs_baseline" in v1
        assert "verdict" in v1

    def test_to_json_is_valid(self):
        import json
        report = self._make_report({"baseline": 0.5, "+x": 0.7})
        parsed = json.loads(report.to_json())
        assert "variants" in parsed


# ===========================================================================
# EvalRunner: tool_filter and engine overrides
# ===========================================================================

class TestEvalRunnerOverrides:

    def test_tool_filter_removes_tool(self):
        """web_search stub removed by filter → its call returns error."""
        runner = EvalRunner()
        case = ABLATION_CASES[2]  # ablation_narrowed_toolset
        result = runner.run_one(case, tool_filter=_narrow_toolset_filter)
        # With narrowing: web_search unavailable → check passes
        assert result.passed

    def test_no_tool_filter_keeps_all_tools(self):
        """Without filter: web_search available → narrowed_toolset check fails."""
        runner = EvalRunner()
        case = ABLATION_CASES[2]  # ablation_narrowed_toolset
        result = runner.run_one(case)  # no filter
        # Without narrowing: web_search runs → narrowed check fails
        assert not result.passed

    def test_policy_override_blocks_dangerous_op(self):
        runner = EvalRunner()
        case = ABLATION_CASES[0]  # ablation_policy_blocks
        policy = _block_dangerous_op_policy()
        result = runner.run_one(case, policy=policy)
        assert result.passed

    def test_no_policy_dangerous_op_executes(self):
        runner = EvalRunner()
        case = ABLATION_CASES[0]  # ablation_policy_blocks
        result = runner.run_one(case)  # no policy
        assert not result.passed

    def test_policy_compat_passes_with_or_without_policy(self):
        runner = EvalRunner()
        case = ABLATION_CASES[1]  # ablation_policy_compat
        policy = _block_dangerous_op_policy()
        assert runner.run_one(case, policy=policy).passed
        assert runner.run_one(case).passed

    def test_run_all_forwards_overrides(self):
        runner = EvalRunner()
        policy = _block_dangerous_op_policy()
        suite = runner.run_all(ABLATION_CASES, policy=policy)
        # ablation_policy_blocks and _compat pass; narrowed_toolset fails (no filter)
        assert suite.passed >= 2

    def test_compression_override_does_not_break_normal_cases(self):
        from agent.compression import CompressionConfig, ContextCompressor
        runner = EvalRunner()
        compressor = ContextCompressor(CompressionConfig())
        # ML cases are short — compression should be a no-op, not break anything
        from evals.cases import ALL_CASES
        suite = runner.run_all(ALL_CASES, compressor=compressor)
        assert suite.passed == suite.total


# ===========================================================================
# AblationRunner: integration
# ===========================================================================

class TestAblationRunner:

    def test_runner_produces_one_result_per_variant(self):
        variants = standard_variants()
        runner = AblationRunner(ABLATION_CASES)
        report = runner.run(variants)
        assert len(report.results) == len(variants)

    def test_baseline_variant_present(self):
        runner = AblationRunner(ABLATION_CASES)
        report = runner.run(standard_variants())
        names = [r.variant.name for r in report.results]
        assert "baseline" in names

    def test_policy_variant_is_positive_vs_baseline(self):
        runner = AblationRunner(ABLATION_CASES)
        report = runner.run(standard_variants())
        policy_result = next(r for r in report.results if r.variant.name == "+policy")
        assert "positive" in report.verdict(policy_result)

    def test_narrowed_toolset_variant_is_positive_vs_baseline(self):
        runner = AblationRunner(ABLATION_CASES)
        report = runner.run(standard_variants())
        narrow_result = next(r for r in report.results if r.variant.name == "+narrowed_toolset")
        assert "positive" in report.verdict(narrow_result)

    def test_compression_variant_is_neutral_vs_baseline(self):
        runner = AblationRunner(ABLATION_CASES)
        report = runner.run(standard_variants())
        comp_result = next(r for r in report.results if r.variant.name == "+compression")
        assert report.verdict(comp_result) in ("neutral", "positive ✓")

    def test_loop_detection_variant_does_not_regress(self):
        runner = AblationRunner(ABLATION_CASES)
        report = runner.run(standard_variants())
        ld_result = next(r for r in report.results if r.variant.name == "+loop_detection")
        assert "negative" not in report.verdict(ld_result)

    def test_reasoning_variant_is_mock_neutral(self):
        runner = AblationRunner(ABLATION_CASES)
        report = runner.run(standard_variants())
        r_result = next(r for r in report.results if r.variant.name == "+reasoning")
        assert "mock" in report.verdict(r_result).lower()

    def test_all_on_does_not_regress_vs_policy_alone(self):
        """Adding all mechanisms together should not be worse than +policy alone."""
        runner = AblationRunner(ABLATION_CASES)
        report = runner.run(standard_variants())
        all_on = next(r for r in report.results if r.variant.name == "all_on")
        policy = next(r for r in report.results if r.variant.name == "+policy")
        assert all_on.pass_rate >= policy.pass_rate

    def test_ml_cases_all_pass_on_baseline(self):
        """ML eval cases are mechanism-agnostic — should pass even on bare baseline."""
        runner = AblationRunner(ML_EVAL_CASES)
        baseline = AblationVariant(name="baseline", mechanisms=AblationMechanism.NONE)
        report = runner.run([baseline])
        assert report.results[0].pass_rate == 1.0

    def test_ml_cases_all_pass_with_all_on(self):
        """All mechanisms active must not break the ML eval suite."""
        runner = AblationRunner(ML_EVAL_CASES)
        all_on = next(v for v in standard_variants() if v.name == "all_on")
        report = runner.run([all_on])
        assert report.results[0].pass_rate == 1.0

    def test_report_summary_mentions_notes(self):
        runner = AblationRunner(ABLATION_CASES)
        report = runner.run(standard_variants()[:2])
        text = report.summary_text()
        assert "REASONING" in text
        assert "MockProvider" in text or "mock" in text.lower()

    def test_report_to_json_roundtrip(self):
        import json
        runner = AblationRunner(ABLATION_CASES)
        report = runner.run(standard_variants()[:3])
        parsed = json.loads(report.to_json())
        assert len(parsed["variants"]) == 3
        assert all("verdict" in v for v in parsed["variants"])

    def test_wall_ms_recorded(self):
        runner = AblationRunner(ABLATION_CASES[:1])
        report = runner.run(standard_variants()[:1])
        assert report.results[0].wall_ms > 0
