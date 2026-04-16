"""MLScenarioConfig — all tuneable knobs for the tabular ML scenario.

System-layer, scenario-layer, and execution-layer concerns are separated:
this module owns only scenario-layer configuration (workflow rules, allowed
models, metric discipline, artifact policy).
"""
from __future__ import annotations

from dataclasses import dataclass, field


# Tools present in a typical general runtime that are DISABLED in this scenario.
# Capturing them here makes the exclusion explicit and auditable.
DISABLED_FROM_GENERAL_RUNTIME: list[str] = [
    # Web / network
    "web_browse", "web_search", "http_request", "fetch_url",
    # Shell / code execution
    "shell_exec", "run_command", "execute_code", "run_python",
    # File-system exploration beyond scoped project files
    "read_file", "write_file", "list_directory", "glob_files", "find_files",
    # Messaging / calendar
    "send_email", "send_message", "slack_post", "calendar_event",
    # Generic database / API
    "database_query", "sql_query", "api_call", "graphql_query",
    # Unrelated utilities
    "calculator", "get_current_time", "search_notes",
]

# Exact set of tools visible to the model in this scenario.
ML_ALLOWED_TOOLS: list[str] = [
    "load_dataset",
    "inspect_schema",
    "preview_rows",
    "summarize_columns",
    "compute_missingness",
    "filter_rows",
    "transform_columns",
    "groupby_aggregate",
    "join_tables",
    "plot_histogram",
    "plot_boxplot",
    "plot_scatter",
    "plot_correlation_matrix",
    "detect_outliers",
    "split_dataset",
    "train_model",
    "evaluate_model",
    "compare_models",
    "save_artifact",
    "generate_report",
]


@dataclass
class MLScenarioConfig:
    """
    All scenario-layer configuration for the tabular ML workflow.

    Three concerns kept separate:
    - Tool access control (allowed/disabled)
    - ML workflow discipline (leakage, splits, metrics)
    - Default values and output limits
    """

    # --- Tool access control ---
    allowed_tools: list[str] = field(
        default_factory=lambda: list(ML_ALLOWED_TOOLS)
    )
    disabled_tools: list[str] = field(
        default_factory=lambda: list(DISABLED_FROM_GENERAL_RUNTIME)
    )

    # --- Preview / output caps ---
    max_rows_preview: int = 20
    max_columns_preview: int = 50
    max_plots_per_run: int = 8

    # --- Allowed ML primitives ---
    allowed_model_families: list[str] = field(default_factory=lambda: [
        "linear_regression",
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
    ])
    allowed_task_types: list[str] = field(default_factory=lambda: [
        "classification",
        "regression",
    ])
    allowed_metrics_by_task: dict[str, list[str]] = field(default_factory=lambda: {
        "classification": ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"],
        "regression":     ["mae", "rmse", "r2", "mape"],
    })

    # --- Workflow discipline flags ---
    # Require train / validation / test partitions — no single-split allowed.
    require_train_validation_test_discipline: bool = True
    # Require a leakage check before train_model executes.
    require_leakage_check: bool = True
    # Block split_dataset / train_model until target_column is declared.
    require_target_declaration_before_training: bool = True
    # Pause and confirm with user when classification vs regression is ambiguous.
    require_problem_type_inference_confirmation_when_ambiguous: bool = True

    # --- Defaults injected when user omits them ---
    default_split_ratios: dict[str, float] = field(default_factory=lambda: {
        "train":      0.70,
        "validation": 0.15,
        "test":       0.15,
    })
    default_random_seed: int = 42
    max_auto_models_per_run: int = 5

    # --- Artifact policy ---
    default_save_artifacts: bool = True  # cleaned datasets, plots, reports

    @classmethod
    def default(cls) -> "MLScenarioConfig":
        """Return the default scenario config for v1 (narrow + reliable)."""
        return cls()
