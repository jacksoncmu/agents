"""ML/data-science scenario — tabular EDA and supervised learning.

Public API:
  MLScenarioConfig  — all scenario-layer configuration knobs
  MLDataStore       — in-memory state for one agent session
  make_ml_tools     — builds all 20 ML tools bound to a data store
  SYSTEM_PROMPT     — system-layer prompt (role, honesty, tool discipline)
  SCENARIO_PROMPT   — scenario-layer prompt (ML workflow rules, leakage, metrics)
  EXECUTION_PROMPT  — execution-layer prompt (output format, concrete numbers)
  ML_EVAL_CASES     — list of 5 machine-checkable eval cases

Quick start:
    from scenarios.ml import MLScenarioConfig, MLDataStore, make_ml_tools, ML_EVAL_CASES
    from evals.runner import EvalRunner

    runner = EvalRunner()
    suite  = runner.run_all(ML_EVAL_CASES)
    print(suite.summary_text())
"""
from scenarios.ml.config import MLScenarioConfig
from scenarios.ml.tools import MLDataStore, make_ml_tools
from scenarios.ml.prompts import EXECUTION_PROMPT, SCENARIO_PROMPT, SYSTEM_PROMPT
from scenarios.ml.evals import ML_EVAL_CASES

__all__ = [
    "MLScenarioConfig",
    "MLDataStore",
    "make_ml_tools",
    "SYSTEM_PROMPT",
    "SCENARIO_PROMPT",
    "EXECUTION_PROMPT",
    "ML_EVAL_CASES",
]
