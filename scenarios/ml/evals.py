"""ML scenario eval cases.

Five machine-checkable eval cases covering the core tabular ML workflow:
  ml_eda_basic          Basic EDA on a clean dataset
  ml_classification     Classification baseline with model comparison
  ml_regression         Regression baseline with feature discussion
  ml_data_quality       Data cleaning before modeling
  ml_leakage_trap       Leakage column detection and exclusion

Each case has:
- scripted LLM responses (mock provider)
- exact expected tool call sequence
- custom checks for process-level assertions (EDA before modeling, etc.)
"""
from __future__ import annotations

from agent.llm.base import LLMResponse
from agent.types import SessionState, ToolCall

from evals.schema import (
    Check,
    CheckStatus,
    EvalCase,
    ExpectedState,
    ExpectedToolCall,
)
from scenarios.ml.tools import MLDataStore, make_ml_tools


# ---------------------------------------------------------------------------
# Helper: extract ordered tool call names from a session
# ---------------------------------------------------------------------------

def _tool_calls(session) -> list[str]:
    return [tc.name for m in session.messages for tc in m.tool_calls]


def _tool_results_content(session) -> list[str]:
    return [tr.content for m in session.messages for tr in m.tool_results]


def _has_tool_error(session) -> bool:
    return any(tr.error for m in session.messages for tr in m.tool_results)


# ---------------------------------------------------------------------------
# Eval 1: Basic EDA on a clean tabular dataset
# ---------------------------------------------------------------------------
#
# User: "Give me a quick EDA summary of this customer churn dataset."
# Expected flow: load → inspect → preview → summarize → missingness → histogram
# No training must occur.

def _build_eda_case() -> EvalCase:
    store = MLDataStore()
    tools = make_ml_tools(store)

    _load  = ToolCall.new("load_dataset",       {"source": "churn.csv",     "dataset_id": "churn_ds"})
    _insp  = ToolCall.new("inspect_schema",      {"dataset_id": "churn_ds"})
    _prev  = ToolCall.new("preview_rows",        {"dataset_id": "churn_ds", "n_rows": 5, "offset": 0})
    _summ  = ToolCall.new("summarize_columns",   {"dataset_id": "churn_ds", "columns": [], "include_quantiles": True})
    _miss  = ToolCall.new("compute_missingness", {"dataset_id": "churn_ds", "columns": []})
    _hist  = ToolCall.new("plot_histogram",      {"dataset_id": "churn_ds", "column": "tenure", "bins": 20})

    llm_responses = [
        LLMResponse(content="", tool_calls=[_load]),
        LLMResponse(content="", tool_calls=[_insp]),
        LLMResponse(content="", tool_calls=[_prev]),
        LLMResponse(content="", tool_calls=[_summ]),
        LLMResponse(content="", tool_calls=[_miss]),
        LLMResponse(content="", tool_calls=[_hist]),
        LLMResponse(content=(
            "EDA summary for the churn dataset (1,000 rows, 8 columns):\n\n"
            "**Schema**: customerID (categorical), gender (categorical), tenure (numeric, range 0–72, mean 32.4), "
            "Contract (categorical), MonthlyCharges (numeric), TotalCharges (numeric, nullable), "
            "PhoneService (boolean), Churn (categorical).\n\n"
            "**Missingness**: TotalCharges has 30 missing values (3.0%). All other columns are complete.\n\n"
            "**Distributions**: tenure is right-skewed with median 29. MonthlyCharges ranges widely (18–120).\n\n"
            "**Churn rate**: approximately 26% of customers churned.\n\n"
            "A histogram of tenure has been saved as a plot artifact."
        )),
    ]

    def check_no_training(session, text, _):
        calls = _tool_calls(session)
        forbidden = {"split_dataset", "train_model", "evaluate_model"}
        used = forbidden & set(calls)
        return [Check(
            name="no_model_training",
            category="process",
            status=CheckStatus.failed if used else CheckStatus.passed,
            detail=f"Training tools used: {used}" if used else "No training tools called — correct.",
        )]

    def check_missingness_reported(session, text, _):
        mentioned = "3.0%" in text or "30 missing" in text or "TotalCharges" in text
        return [Check(
            name="missingness_in_response",
            category="content",
            status=CheckStatus.passed if mentioned else CheckStatus.failed,
            detail="Response must mention TotalCharges missingness rate.",
        )]

    def check_numeric_insight(session, text, _):
        has_number = any(c.isdigit() for c in text)
        return [Check(
            name="numeric_insight_present",
            category="content",
            status=CheckStatus.passed if has_number else CheckStatus.failed,
            detail="Response must include at least one concrete numeric value.",
        )]

    return EvalCase(
        id="ml_eda_basic",
        name="Basic EDA on clean churn dataset",
        description=(
            "User requests a quick EDA summary. Agent must inspect schema, preview rows, "
            "summarize columns, compute missingness, and generate at least one plot. "
            "No model training should occur."
        ),
        user_message="Give me a quick EDA summary of this customer churn dataset.",
        llm_responses=llm_responses,
        tools=tools,
        expected_tool_sequence=[
            ExpectedToolCall(name="load_dataset",       args_subset={"source": "churn.csv", "dataset_id": "churn_ds"}),
            ExpectedToolCall(name="inspect_schema",      args_subset={"dataset_id": "churn_ds"}),
            ExpectedToolCall(name="preview_rows",        args_subset={"dataset_id": "churn_ds"}),
            ExpectedToolCall(name="summarize_columns",   args_subset={"dataset_id": "churn_ds"}),
            ExpectedToolCall(name="compute_missingness", args_subset={"dataset_id": "churn_ds"}),
            ExpectedToolCall(name="plot_histogram",      args_subset={"dataset_id": "churn_ds"}),
        ],
        expected_state=ExpectedState(
            session_state=SessionState.waiting_for_user,
            min_messages=14,
        ),
        required_response_substrings=["TotalCharges", "tenure", "churn"],
        forbidden_response_substrings=[],
        custom_checks=[check_no_training, check_missingness_reported, check_numeric_insight],
        tags=["ml", "eda", "no_training"],
    )


# ---------------------------------------------------------------------------
# Eval 2: Classification baseline with model comparison
# ---------------------------------------------------------------------------
#
# User: "Predict customer churn and compare a simple baseline with a tree-based model."
# Expected: inspect → split → 2x train → 2x evaluate (validation) → compare
# No regression metrics; no test-set selection; target not in features.

def _build_classification_case() -> EvalCase:
    store = MLDataStore()
    tools = make_ml_tools(store)

    _features = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "PhoneService"]

    _load  = ToolCall.new("load_dataset",   {"source": "churn.csv", "dataset_id": "churn_ds"})
    _insp  = ToolCall.new("inspect_schema", {"dataset_id": "churn_ds"})
    _split = ToolCall.new("split_dataset",  {
        "dataset_id": "churn_ds", "target_column": "Churn",
        "task_type": "classification",
        "train_fraction": 0.70, "validation_fraction": 0.15, "test_fraction": 0.15,
        "random_seed": 42, "stratify": True,
    })
    _lr = ToolCall.new("train_model", {
        "split_id": "split_001", "model_family": "logistic_regression",
        "feature_columns": _features, "target_column": "Churn", "hyperparameters": {},
    })
    _rf = ToolCall.new("train_model", {
        "split_id": "split_001", "model_family": "random_forest",
        "feature_columns": _features, "target_column": "Churn", "hyperparameters": {},
    })
    _eval_lr = ToolCall.new("evaluate_model", {"model_id": "model_001", "evaluation_split": "validation", "metrics": ["accuracy", "f1", "roc_auc"]})
    _eval_rf = ToolCall.new("evaluate_model", {"model_id": "model_002", "evaluation_split": "validation", "metrics": ["accuracy", "f1", "roc_auc"]})
    _cmp     = ToolCall.new("compare_models", {"model_ids": ["model_001", "model_002"], "evaluation_split": "validation", "primary_metric": "roc_auc"})

    llm_responses = [
        LLMResponse(content="", tool_calls=[_load]),
        LLMResponse(content="", tool_calls=[_insp]),
        LLMResponse(content="", tool_calls=[_split]),
        LLMResponse(content="", tool_calls=[_lr]),
        LLMResponse(content="", tool_calls=[_rf]),
        LLMResponse(content="", tool_calls=[_eval_lr]),
        LLMResponse(content="", tool_calls=[_eval_rf]),
        LLMResponse(content="", tool_calls=[_cmp]),
        LLMResponse(content=(
            "Classification results (target: Churn, validation split):\n\n"
            "| Model               | Accuracy | F1   | ROC-AUC |\n"
            "|---------------------|----------|------|---------|\n"
            "| Logistic Regression | 0.79     | 0.71 | 0.82    |\n"
            "| Random Forest       | 0.83     | 0.78 | 0.87    |\n\n"
            "**Best model**: Random Forest (ROC-AUC 0.87 on validation).\n\n"
            "The Random Forest outperforms logistic regression on all three metrics. "
            "I recommend it as the baseline for this task. "
            "Note: test-set evaluation has not been run; use the test split only for the final estimate."
        )),
    ]

    def check_no_regression_metrics(session, text, _):
        regression_metrics = {"mae", "rmse", "r2", "mape"}
        found = {m for m in regression_metrics if m in text.lower()}
        return [Check(
            name="no_regression_metrics",
            category="process",
            status=CheckStatus.failed if found else CheckStatus.passed,
            detail=f"Regression metrics found in response: {found}" if found else "No regression metrics — correct.",
        )]

    def check_validation_not_test_for_selection(session, text, _):
        calls = _tool_calls(session)
        eval_indices = [i for i, c in enumerate(calls) if c == "evaluate_model"]
        cmp_indices  = [i for i, c in enumerate(calls) if c == "compare_models"]
        # All evaluate calls before compare must use validation (checked via args)
        msgs_with_tc = [m for m in session.messages if m.tool_calls]
        eval_msgs = [m for m in msgs_with_tc if any(tc.name == "evaluate_model" for tc in m.tool_calls)]
        used_test = any(
            tc.arguments.get("evaluation_split") == "test"
            for m in eval_msgs
            for tc in m.tool_calls
            if tc.name == "evaluate_model"
        )
        return [Check(
            name="validation_used_for_selection",
            category="process",
            status=CheckStatus.failed if used_test else CheckStatus.passed,
            detail="evaluate_model used 'test' split during model selection." if used_test else "Validation split used — correct.",
        )]

    def check_target_not_in_features(session, text, _):
        leakage = any(
            "Churn" in tc.arguments.get("feature_columns", [])
            for m in session.messages
            for tc in m.tool_calls
            if tc.name == "train_model"
        )
        return [Check(
            name="target_not_in_features",
            category="process",
            status=CheckStatus.failed if leakage else CheckStatus.passed,
            detail="Target column 'Churn' appeared in feature_columns." if leakage else "Target excluded from features — correct.",
        )]

    def check_best_model_named(session, text, _):
        mentioned = "random forest" in text.lower() or "best" in text.lower()
        return [Check(
            name="best_model_named_in_response",
            category="content",
            status=CheckStatus.passed if mentioned else CheckStatus.failed,
            detail="Response must name the best validation model.",
        )]

    return EvalCase(
        id="ml_classification",
        name="Classification baseline with model comparison",
        description=(
            "User wants to predict churn. Agent must identify target, split with train/val/test "
            "discipline, train logistic regression and random forest, evaluate both on validation, "
            "and compare with classification metrics only."
        ),
        user_message="Predict customer churn and compare a simple baseline with a tree-based model.",
        llm_responses=llm_responses,
        tools=tools,
        expected_tool_sequence=[
            ExpectedToolCall(name="load_dataset",   args_subset={"dataset_id": "churn_ds"}),
            ExpectedToolCall(name="inspect_schema",  args_subset={"dataset_id": "churn_ds"}),
            ExpectedToolCall(name="split_dataset",   args_subset={"target_column": "Churn", "task_type": "classification"}),
            ExpectedToolCall(name="train_model",     args_subset={"model_family": "logistic_regression"}),
            ExpectedToolCall(name="train_model",     args_subset={"model_family": "random_forest"}),
            ExpectedToolCall(name="evaluate_model",  args_subset={"evaluation_split": "validation"}),
            ExpectedToolCall(name="evaluate_model",  args_subset={"evaluation_split": "validation"}),
            ExpectedToolCall(name="compare_models",  args_subset={"evaluation_split": "validation", "primary_metric": "roc_auc"}),
        ],
        expected_state=ExpectedState(
            session_state=SessionState.waiting_for_user,
            min_messages=18,
        ),
        required_response_substrings=["0.87", "validation", "Random Forest"],
        forbidden_response_substrings=[],
        custom_checks=[
            check_no_regression_metrics,
            check_validation_not_test_for_selection,
            check_target_not_in_features,
            check_best_model_named,
        ],
        tags=["ml", "classification", "model_comparison"],
    )


# ---------------------------------------------------------------------------
# Eval 3: Regression baseline with feature discussion
# ---------------------------------------------------------------------------
#
# User: "Predict house prices and tell me which features seem most useful."
# Expected: load → inspect → summarize → corr_matrix → split → train (linear + rf) → evaluate → response

def _build_regression_case() -> EvalCase:
    store = MLDataStore()
    tools = make_ml_tools(store)

    _features = ["area_sqft", "bedrooms", "bathrooms", "year_built"]

    _load  = ToolCall.new("load_dataset",          {"source": "housing.csv", "dataset_id": "housing_ds"})
    _insp  = ToolCall.new("inspect_schema",         {"dataset_id": "housing_ds"})
    _summ  = ToolCall.new("summarize_columns",      {"dataset_id": "housing_ds", "columns": [], "include_quantiles": True})
    _corr  = ToolCall.new("plot_correlation_matrix",{"dataset_id": "housing_ds", "columns": []})
    _split = ToolCall.new("split_dataset", {
        "dataset_id": "housing_ds", "target_column": "price",
        "task_type": "regression",
        "train_fraction": 0.70, "validation_fraction": 0.15, "test_fraction": 0.15,
        "random_seed": 42, "stratify": False,
    })
    _lr = ToolCall.new("train_model", {
        "split_id": "split_001", "model_family": "linear_regression",
        "feature_columns": _features, "target_column": "price", "hyperparameters": {},
    })
    _rf = ToolCall.new("train_model", {
        "split_id": "split_001", "model_family": "random_forest",
        "feature_columns": _features, "target_column": "price", "hyperparameters": {},
    })
    _eval_lr = ToolCall.new("evaluate_model", {"model_id": "model_001", "evaluation_split": "validation", "metrics": ["mae", "rmse", "r2"]})
    _eval_rf = ToolCall.new("evaluate_model", {"model_id": "model_002", "evaluation_split": "validation", "metrics": ["mae", "rmse", "r2"]})

    llm_responses = [
        LLMResponse(content="", tool_calls=[_load]),
        LLMResponse(content="", tool_calls=[_insp]),
        LLMResponse(content="", tool_calls=[_summ]),
        LLMResponse(content="", tool_calls=[_corr]),
        LLMResponse(content="", tool_calls=[_split]),
        LLMResponse(content="", tool_calls=[_lr]),
        LLMResponse(content="", tool_calls=[_rf]),
        LLMResponse(content="", tool_calls=[_eval_lr]),
        LLMResponse(content="", tool_calls=[_eval_rf]),
        LLMResponse(content=(
            "Regression results (target: price, validation split):\n\n"
            "| Model             | MAE       | RMSE      | R²   |\n"
            "|-------------------|-----------|-----------|------|\n"
            "| Linear Regression | $45,200   | $68,100   | 0.71 |\n"
            "| Random Forest     | $28,400   | $41,200   | 0.86 |\n\n"
            "**Best model**: Random Forest (R²=0.86, MAE=$28,400 on validation).\n\n"
            "**Most useful features**: area_sqft shows the strongest correlation with price. "
            "bedrooms and bathrooms are moderately correlated. year_built has weaker predictive power.\n\n"
            "The linear model provides a useful baseline (R²=0.71) but the Random Forest "
            "captures non-linear interactions and reduces error by 37%."
        )),
    ]

    def check_no_classification_metrics(session, text, _):
        clf_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"}
        found = {m for m in clf_metrics if m in text.lower()}
        return [Check(
            name="no_classification_metrics",
            category="process",
            status=CheckStatus.failed if found else CheckStatus.passed,
            detail=f"Classification metrics found: {found}" if found else "No classification metrics — correct.",
        )]

    def check_regression_metrics_reported(session, text, _):
        has_reg = any(m in text.lower() for m in ("mae", "rmse", "r2", "r²"))
        return [Check(
            name="regression_metrics_in_response",
            category="content",
            status=CheckStatus.passed if has_reg else CheckStatus.failed,
            detail="Response must report at least one regression metric (mae/rmse/r2).",
        )]

    def check_feature_discussion(session, text, _):
        mentioned = any(f in text.lower() for f in ("area_sqft", "feature", "important", "useful", "correl"))
        return [Check(
            name="feature_discussion_present",
            category="content",
            status=CheckStatus.passed if mentioned else CheckStatus.failed,
            detail="Response must include a discussion of which features are most useful.",
        )]

    return EvalCase(
        id="ml_regression",
        name="Regression baseline with feature discussion",
        description=(
            "User wants to predict house prices. Agent must inspect schema, run EDA, "
            "split for regression, train a linear and a tree baseline, evaluate with "
            "regression metrics only, and discuss feature importance."
        ),
        user_message="Predict house prices and tell me which features seem most useful.",
        llm_responses=llm_responses,
        tools=tools,
        expected_tool_sequence=[
            ExpectedToolCall(name="load_dataset",           args_subset={"dataset_id": "housing_ds"}),
            ExpectedToolCall(name="inspect_schema",          args_subset={"dataset_id": "housing_ds"}),
            ExpectedToolCall(name="summarize_columns",       args_subset={"dataset_id": "housing_ds"}),
            ExpectedToolCall(name="plot_correlation_matrix", args_subset={"dataset_id": "housing_ds"}),
            ExpectedToolCall(name="split_dataset",           args_subset={"target_column": "price", "task_type": "regression"}),
            ExpectedToolCall(name="train_model",             args_subset={"model_family": "linear_regression"}),
            ExpectedToolCall(name="train_model",             args_subset={"model_family": "random_forest"}),
            ExpectedToolCall(name="evaluate_model",          args_subset={"evaluation_split": "validation"}),
            ExpectedToolCall(name="evaluate_model",          args_subset={"evaluation_split": "validation"}),
        ],
        expected_state=ExpectedState(
            session_state=SessionState.waiting_for_user,
            min_messages=20,
        ),
        required_response_substrings=["0.86", "validation", "Random Forest", "area_sqft"],
        forbidden_response_substrings=[],
        custom_checks=[check_no_classification_metrics, check_regression_metrics_reported, check_feature_discussion],
        tags=["ml", "regression", "feature_importance"],
    )


# ---------------------------------------------------------------------------
# Eval 4: Data quality problem — clean before modeling
# ---------------------------------------------------------------------------
#
# User: "Can you clean this dataset and tell me if it's usable for modeling?"
# Dataset has missing values, duplicate rows, inconsistent labels.
# Expected: load → inspect → missingness → preview → transform → save_artifact → response

def _build_data_quality_case() -> EvalCase:
    store = MLDataStore()
    tools = make_ml_tools(store)

    _load  = ToolCall.new("load_dataset",       {"source": "dirty_data.csv", "dataset_id": "raw_ds"})
    _insp  = ToolCall.new("inspect_schema",      {"dataset_id": "raw_ds"})
    _miss  = ToolCall.new("compute_missingness", {"dataset_id": "raw_ds", "columns": []})
    _prev  = ToolCall.new("preview_rows",        {"dataset_id": "raw_ds", "n_rows": 20, "offset": 0})
    _txfm  = ToolCall.new("transform_columns", {
        "dataset_id": "raw_ds",
        "operations": [
            {"column": "age",      "operation": "fill_missing",      "params": {"strategy": "median"}},
            {"column": "income",   "operation": "fill_missing",      "params": {"strategy": "median"}},
            {"column": "category", "operation": "normalize_labels",  "params": {}},
            {"column": "__rows__", "operation": "drop_duplicates",   "params": {}},
        ],
        "output_dataset_id": "clean_ds",
    })
    _save  = ToolCall.new("save_artifact", {
        "artifact_type": "dataset", "source_id": "clean_ds",
        "filename": "clean_data.csv", "metadata": {},
    })

    llm_responses = [
        LLMResponse(content="", tool_calls=[_load]),
        LLMResponse(content="", tool_calls=[_insp]),
        LLMResponse(content="", tool_calls=[_miss]),
        LLMResponse(content="", tool_calls=[_prev]),
        LLMResponse(content="", tool_calls=[_txfm]),
        LLMResponse(content="", tool_calls=[_save]),
        LLMResponse(content=(
            "Data quality assessment for dirty_data.csv (300 rows, 6 columns):\n\n"
            "**Issues found and fixed:**\n"
            "- Missing values: age (12.0%), income (8.0%), category (5.0%) → imputed with median/mode.\n"
            "- Duplicate rows: 23 removed (7.7% of original data).\n"
            "- Inconsistent category labels: normalized.\n\n"
            "**Cleaned dataset saved** as clean_data.csv (277 rows remaining).\n\n"
            "**Remaining risks:**\n"
            "- customerID has 2.0% missing values — consider whether it's needed for modeling.\n"
            "- After cleaning, class balance in 'target' should be checked before training.\n\n"
            "The dataset is usable for modeling with the caveats above."
        )),
    ]

    def check_cleaning_before_training(session, text, _):
        calls = _tool_calls(session)
        has_clean  = "transform_columns" in calls
        has_train  = "train_model" in calls
        clean_idx  = calls.index("transform_columns") if has_clean else None
        train_idx  = calls.index("train_model")        if has_train  else None
        if has_train and has_clean and clean_idx > train_idx:
            status = CheckStatus.failed
            detail = "train_model called before transform_columns."
        elif has_train and not has_clean:
            status = CheckStatus.failed
            detail = "train_model called without any cleaning step."
        else:
            status = CheckStatus.passed
            detail = "No training before cleaning — correct." if not has_train else "Cleaning occurred before training."
        return [Check(name="cleaning_before_training", category="process", status=status, detail=detail)]

    def check_artifact_saved(session, text, _):
        calls = _tool_calls(session)
        saved = "save_artifact" in calls
        return [Check(
            name="cleaned_artifact_saved",
            category="process",
            status=CheckStatus.passed if saved else CheckStatus.failed,
            detail="save_artifact must be called to save the cleaned dataset." if not saved else "Cleaned dataset artifact saved.",
        )]

    def check_issues_distinguished(session, text, _):
        mentions_fixed    = any(w in text.lower() for w in ("fixed", "removed", "imputed", "cleaned"))
        mentions_remaining = any(w in text.lower() for w in ("remaining", "risk", "caveat", "consider", "still"))
        ok = mentions_fixed and mentions_remaining
        return [Check(
            name="fixed_vs_unresolved_distinguished",
            category="content",
            status=CheckStatus.passed if ok else CheckStatus.failed,
            detail="Response must distinguish fixed issues from unresolved risks.",
        )]

    return EvalCase(
        id="ml_data_quality",
        name="Data cleaning before modeling",
        description=(
            "Dataset contains missing values, duplicate rows, and inconsistent labels. "
            "Agent must detect issues, clean the data, save a cleaned artifact, "
            "and distinguish fixed issues from remaining risks. No training should occur."
        ),
        user_message="Can you clean this dataset and tell me if it's usable for modeling?",
        llm_responses=llm_responses,
        tools=tools,
        expected_tool_sequence=[
            ExpectedToolCall(name="load_dataset",       args_subset={"source": "dirty_data.csv"}),
            ExpectedToolCall(name="inspect_schema",      args_subset={"dataset_id": "raw_ds"}),
            ExpectedToolCall(name="compute_missingness", args_subset={"dataset_id": "raw_ds"}),
            ExpectedToolCall(name="preview_rows",        args_subset={"dataset_id": "raw_ds"}),
            ExpectedToolCall(name="transform_columns",   args_subset={"output_dataset_id": "clean_ds"}),
            ExpectedToolCall(name="save_artifact",       args_subset={"artifact_type": "dataset", "filename": "clean_data.csv"}),
        ],
        expected_state=ExpectedState(
            session_state=SessionState.waiting_for_user,
            min_messages=14,
        ),
        required_response_substrings=["12.0%", "23", "clean_data.csv"],
        forbidden_response_substrings=[],
        custom_checks=[check_cleaning_before_training, check_artifact_saved, check_issues_distinguished],
        tags=["ml", "data_quality", "cleaning"],
    )


# ---------------------------------------------------------------------------
# Eval 5: Leakage trap — exclude post-outcome column
# ---------------------------------------------------------------------------
#
# User: "Train the best churn model you can."
# Dataset: churn_leaky.csv contains "days_since_churn" — a post-outcome leakage column.
# Expected: inspect → detect leakage → split → train WITHOUT leakage column → evaluate → response

def _build_leakage_case() -> EvalCase:
    store = MLDataStore()
    tools = make_ml_tools(store)

    # Safe features — days_since_churn intentionally excluded
    _safe_features = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "PhoneService"]

    _load  = ToolCall.new("load_dataset",   {"source": "churn_leaky.csv", "dataset_id": "churn_leaky_ds"})
    _insp  = ToolCall.new("inspect_schema", {"dataset_id": "churn_leaky_ds"})
    _split = ToolCall.new("split_dataset",  {
        "dataset_id": "churn_leaky_ds", "target_column": "Churn",
        "task_type": "classification",
        "train_fraction": 0.70, "validation_fraction": 0.15, "test_fraction": 0.15,
        "random_seed": 42, "stratify": True,
    })
    _train = ToolCall.new("train_model", {
        "split_id": "split_001", "model_family": "logistic_regression",
        "feature_columns": _safe_features,
        "target_column": "Churn", "hyperparameters": {},
    })
    _eval  = ToolCall.new("evaluate_model", {"model_id": "model_001", "evaluation_split": "validation", "metrics": ["accuracy", "f1", "roc_auc"]})

    llm_responses = [
        LLMResponse(content="", tool_calls=[_load]),
        LLMResponse(content="", tool_calls=[_insp]),
        LLMResponse(content="", tool_calls=[_split]),
        LLMResponse(content="", tool_calls=[_train]),
        LLMResponse(content="", tool_calls=[_eval]),
        LLMResponse(content=(
            "Leakage handling: 'days_since_churn' was identified as a post-outcome feature "
            "(populated only when Churn='Yes') and was excluded from training.\n\n"
            "Model: Logistic Regression (safe features only)\n"
            "Validation: accuracy=0.79, F1=0.71, ROC-AUC=0.82\n\n"
            "These metrics reflect realistic generalization performance without leakage. "
            "A model trained with 'days_since_churn' would show artificially inflated metrics."
        )),
    ]

    def check_leakage_excluded_from_features(session, text, _):
        leaked = any(
            "days_since_churn" in tc.arguments.get("feature_columns", [])
            for m in session.messages
            for tc in m.tool_calls
            if tc.name == "train_model"
        )
        return [Check(
            name="leakage_column_excluded",
            category="process",
            status=CheckStatus.failed if leaked else CheckStatus.passed,
            detail="'days_since_churn' appeared in feature_columns — leakage not handled." if leaked else "Leakage column excluded from features.",
        )]

    def check_leakage_mentioned_in_response(session, text, _):
        mentioned = "leakage" in text.lower() or "days_since_churn" in text.lower() or "post-outcome" in text.lower()
        return [Check(
            name="leakage_mentioned_in_response",
            category="content",
            status=CheckStatus.passed if mentioned else CheckStatus.failed,
            detail="Response must explicitly mention leakage handling.",
        )]

    def check_inspect_before_train(session, text, _):
        calls = _tool_calls(session)
        insp_idx  = next((i for i, c in enumerate(calls) if c == "inspect_schema"), None)
        train_idx = next((i for i, c in enumerate(calls) if c == "train_model"),    None)
        if insp_idx is None:
            return [Check(name="inspect_before_train", category="process", status=CheckStatus.failed, detail="inspect_schema was never called.")]
        if train_idx is None:
            return [Check(name="inspect_before_train", category="process", status=CheckStatus.skipped, detail="No train_model call found.")]
        ok = insp_idx < train_idx
        return [Check(
            name="inspect_before_train",
            category="process",
            status=CheckStatus.passed if ok else CheckStatus.failed,
            detail=f"inspect_schema at step {insp_idx}, train_model at step {train_idx}.",
        )]

    return EvalCase(
        id="ml_leakage_trap",
        name="Leakage column detection and exclusion",
        description=(
            "Dataset contains a post-outcome leakage column ('days_since_churn'). "
            "Agent must inspect the schema, identify the leakage column, exclude it from "
            "feature_columns, and explicitly mention leakage handling in the response."
        ),
        user_message="Train the best churn model you can.",
        llm_responses=llm_responses,
        tools=tools,
        expected_tool_sequence=[
            ExpectedToolCall(name="load_dataset",   args_subset={"source": "churn_leaky.csv"}),
            ExpectedToolCall(name="inspect_schema",  args_subset={"dataset_id": "churn_leaky_ds"}),
            ExpectedToolCall(name="split_dataset",   args_subset={"target_column": "Churn", "task_type": "classification"}),
            ExpectedToolCall(name="train_model",     args_subset={"model_family": "logistic_regression"}),
            ExpectedToolCall(name="evaluate_model",  args_subset={"evaluation_split": "validation"}),
        ],
        expected_state=ExpectedState(
            session_state=SessionState.waiting_for_user,
            min_messages=12,
        ),
        required_response_substrings=["leakage", "days_since_churn", "0.82"],
        forbidden_response_substrings=[],
        custom_checks=[
            check_leakage_excluded_from_features,
            check_leakage_mentioned_in_response,
            check_inspect_before_train,
        ],
        tags=["ml", "leakage", "feature_safety"],
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_ml_eval_cases() -> list[EvalCase]:
    """Return a freshly built list of ML eval cases with new MLDataStore instances.

    Call this instead of referencing ML_EVAL_CASES directly whenever you need
    to run the cases more than once (e.g. across multiple ablation variants) —
    each call creates independent stores so state does not bleed between runs.
    """
    return [
        _build_eda_case(),
        _build_classification_case(),
        _build_regression_case(),
        _build_data_quality_case(),
        _build_leakage_case(),
    ]


ML_EVAL_CASES: list[EvalCase] = build_ml_eval_cases()
