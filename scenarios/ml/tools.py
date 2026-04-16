"""ML scenario tool definitions with precise schemas and mock implementations.

Every tool description is exact: required fields, types, enums, error conditions,
and output schema are fully specified so the model cannot guess or hallucinate them.

Tools are created via make_ml_tools(store) which binds all 20 tools to a shared
MLDataStore instance. One store per agent session keeps state isolated.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from agent.tools import ToolContext, ToolDefinition, ToolParam


# ---------------------------------------------------------------------------
# Shared in-memory data store
# ---------------------------------------------------------------------------

@dataclass
class MLDataStore:
    """Registry for datasets, splits, models, and artifacts within one session."""
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)
    splits:   dict[str, dict[str, Any]] = field(default_factory=dict)
    models:   dict[str, dict[str, Any]] = field(default_factory=dict)
    artifacts: list[dict[str, Any]]     = field(default_factory=list)
    _split_ctr:    int = field(default=0, repr=False)
    _model_ctr:    int = field(default=0, repr=False)
    _artifact_ctr: int = field(default=0, repr=False)

    def next_split_id(self) -> str:
        self._split_ctr += 1
        return f"split_{self._split_ctr:03d}"

    def next_model_id(self) -> str:
        self._model_ctr += 1
        return f"model_{self._model_ctr:03d}"

    def next_artifact_id(self) -> str:
        self._artifact_ctr += 1
        return f"artifact_{self._artifact_ctr:03d}"


# ---------------------------------------------------------------------------
# Built-in dataset templates
# ---------------------------------------------------------------------------

_SCHEMAS: dict[str, dict[str, Any]] = {
    "churn": {
        "row_count": 1000,
        "columns": [
            {"name": "customerID",     "dtype": "categorical", "nullable": False},
            {"name": "gender",         "dtype": "categorical", "nullable": False},
            {"name": "tenure",         "dtype": "numeric",     "nullable": False},
            {"name": "Contract",       "dtype": "categorical", "nullable": False},
            {"name": "MonthlyCharges", "dtype": "numeric",     "nullable": False},
            {"name": "TotalCharges",   "dtype": "numeric",     "nullable": True},
            {"name": "PhoneService",   "dtype": "boolean",     "nullable": False},
            {"name": "Churn",          "dtype": "categorical", "nullable": False},
        ],
        "missingness": {"TotalCharges": 0.03},
        "duplicate_rows": 0,
        "quality_issues": [],
    },
    "churn_leaky": {
        "row_count": 1000,
        "columns": [
            {"name": "customerID",       "dtype": "categorical", "nullable": False},
            {"name": "gender",           "dtype": "categorical", "nullable": False},
            {"name": "tenure",           "dtype": "numeric",     "nullable": False},
            {"name": "Contract",         "dtype": "categorical", "nullable": False},
            {"name": "MonthlyCharges",   "dtype": "numeric",     "nullable": False},
            {"name": "TotalCharges",     "dtype": "numeric",     "nullable": True},
            {"name": "PhoneService",     "dtype": "boolean",     "nullable": False},
            {"name": "days_since_churn", "dtype": "numeric",     "nullable": True},
            {"name": "Churn",            "dtype": "categorical", "nullable": False},
        ],
        "missingness": {"TotalCharges": 0.03, "days_since_churn": 0.73},
        "duplicate_rows": 0,
        "quality_issues": ["possible_leakage"],
        "leakage_columns": ["days_since_churn"],
    },
    "housing": {
        "row_count": 500,
        "columns": [
            {"name": "id",         "dtype": "numeric",     "nullable": False},
            {"name": "area_sqft",  "dtype": "numeric",     "nullable": False},
            {"name": "bedrooms",   "dtype": "numeric",     "nullable": False},
            {"name": "bathrooms",  "dtype": "numeric",     "nullable": True},
            {"name": "location",   "dtype": "categorical", "nullable": False},
            {"name": "year_built", "dtype": "numeric",     "nullable": True},
            {"name": "price",      "dtype": "numeric",     "nullable": False},
        ],
        "missingness": {"bathrooms": 0.04, "year_built": 0.06},
        "duplicate_rows": 0,
        "quality_issues": [],
    },
    "dirty_data": {
        "row_count": 300,
        "columns": [
            {"name": "customerID", "dtype": "categorical", "nullable": True},
            {"name": "age",        "dtype": "numeric",     "nullable": True},
            {"name": "income",     "dtype": "numeric",     "nullable": True},
            {"name": "category",   "dtype": "categorical", "nullable": True},
            {"name": "score",      "dtype": "numeric",     "nullable": False},
            {"name": "target",     "dtype": "categorical", "nullable": False},
        ],
        "missingness": {"age": 0.12, "income": 0.08, "category": 0.05, "customerID": 0.02},
        "duplicate_rows": 23,
        "quality_issues": ["missing_values", "duplicate_rows", "inconsistent_labels"],
    },
}


def _builtin_key(source: str) -> str | None:
    s = source.lower().removesuffix(".csv").strip()
    for key in _SCHEMAS:
        if s == key or s.endswith("/" + key) or s.endswith("\\" + key):
            return key
    return None


def _mock_rows(schema_key: str, n: int, offset: int) -> list[dict[str, Any]]:
    rng = random.Random(42 + offset)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        idx = offset + i
        base = schema_key.replace("_leaky", "").replace("dirty_", "")
        if base in ("churn",):
            row: dict[str, Any] = {
                "customerID":     f"CUST{1000 + idx:04d}",
                "gender":         rng.choice(["Male", "Female"]),
                "tenure":         rng.randint(0, 72),
                "Contract":       rng.choice(["Month-to-month", "One year", "Two year"]),
                "MonthlyCharges": round(rng.uniform(18.0, 120.0), 2),
                "TotalCharges":   round(rng.uniform(18.0, 8000.0), 2) if rng.random() > 0.03 else None,
                "PhoneService":   rng.choice([True, False]),
                "Churn":          rng.choice(["Yes", "No"]),
            }
            if schema_key == "churn_leaky":
                row["days_since_churn"] = rng.randint(1, 365) if row["Churn"] == "Yes" else None
        elif schema_key == "housing":
            row = {
                "id":         1000 + idx,
                "area_sqft":  rng.randint(600, 5000),
                "bedrooms":   rng.randint(1, 6),
                "bathrooms":  rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, None]),
                "location":   rng.choice(["downtown", "suburb", "rural", "beachfront"]),
                "year_built": rng.choice([rng.randint(1950, 2023), None]),
                "price":      float(round(rng.uniform(120000.0, 1500000.0))),
            }
        else:
            cols = _SCHEMAS.get(schema_key, {}).get("columns", [])
            row = {c["name"]: None for c in cols}
        rows.append(row)
    return rows


# Mock metric tables keyed by task_type → model_family → metric → value.
_MOCK_METRICS: dict[str, dict[str, dict[str, float]]] = {
    "classification": {
        "logistic_regression": {"accuracy": 0.79, "precision": 0.74, "recall": 0.69, "f1": 0.71, "roc_auc": 0.82, "log_loss": 0.46},
        "random_forest":       {"accuracy": 0.83, "precision": 0.80, "recall": 0.76, "f1": 0.78, "roc_auc": 0.87, "log_loss": 0.38},
        "xgboost":             {"accuracy": 0.84, "precision": 0.81, "recall": 0.77, "f1": 0.79, "roc_auc": 0.88, "log_loss": 0.36},
        "lightgbm":            {"accuracy": 0.85, "precision": 0.82, "recall": 0.78, "f1": 0.80, "roc_auc": 0.89, "log_loss": 0.35},
    },
    "regression": {
        "linear_regression": {"mae": 45200.0, "rmse": 68100.0, "r2": 0.71, "mape": 0.142},
        "random_forest":     {"mae": 28400.0, "rmse": 41200.0, "r2": 0.86, "mape": 0.089},
        "xgboost":           {"mae": 26100.0, "rmse": 38700.0, "r2": 0.88, "mape": 0.081},
        "lightgbm":          {"mae": 25800.0, "rmse": 38200.0, "r2": 0.88, "mape": 0.080},
    },
}

_ALLOWED_METRICS = {
    "classification": {"accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"},
    "regression":     {"mae", "rmse", "r2", "mape"},
}

_ALLOWED_MODEL_FAMILIES = {
    "linear_regression", "logistic_regression", "random_forest", "xgboost", "lightgbm",
}

_REGRESSION_FAMILIES = {"linear_regression"}
_CLASSIFICATION_FAMILIES = {"logistic_regression"}


# ---------------------------------------------------------------------------
# Tool factory: all 20 tools
# ---------------------------------------------------------------------------

def make_ml_tools(store: MLDataStore) -> list[ToolDefinition]:
    """Return all 20 ML scenario tools bound to *store*."""
    return [
        _make_load_dataset(store),
        _make_inspect_schema(store),
        _make_preview_rows(store),
        _make_summarize_columns(store),
        _make_compute_missingness(store),
        _make_filter_rows(store),
        _make_transform_columns(store),
        _make_groupby_aggregate(store),
        _make_join_tables(store),
        _make_plot_histogram(store),
        _make_plot_boxplot(store),
        _make_plot_scatter(store),
        _make_plot_correlation_matrix(store),
        _make_detect_outliers(store),
        _make_split_dataset(store),
        _make_train_model(store),
        _make_evaluate_model(store),
        _make_compare_models(store),
        _make_save_artifact(store),
        _make_generate_report(store),
    ]


# --- load_dataset -----------------------------------------------------------

def _make_load_dataset(store: MLDataStore) -> ToolDefinition:
    def handler(source: str, dataset_id: str, *, ctx: ToolContext) -> str:
        if dataset_id in store.datasets:
            raise ValueError(f"dataset_id already in use: {dataset_id!r}. Choose a different id.")
        key = _builtin_key(source)
        if key is None:
            raise ValueError(
                f"Unknown dataset source: {source!r}. "
                "Built-in sources: churn.csv, churn_leaky.csv, housing.csv, dirty_data.csv. "
                "Pass the exact filename."
            )
        tmpl = _SCHEMAS[key]
        store.datasets[dataset_id] = {
            "source": source,
            "builtin_key": key,
            "schema": tmpl,
        }
        ctx.log(f"Loaded {key!r} → dataset_id={dataset_id!r}, rows={tmpl['row_count']}")
        return json.dumps({
            "dataset_id":   dataset_id,
            "source":       source,
            "row_count":    tmpl["row_count"],
            "column_count": len(tmpl["columns"]),
            "status":       "loaded",
        })

    return ToolDefinition(
        name="load_dataset",
        description="""Load a tabular dataset into the session registry.

Required input object:
{
  "source":     string  — file path or built-in dataset name (e.g. "churn.csv", "housing.csv")
  "dataset_id": string  — identifier to assign; used in all subsequent tool calls
}

Rules:
- source must be a filename or path string; do not pass raw data or JSON.
- dataset_id must be unique within the session; re-using an id is an error.
- dataset_id must not contain spaces or special characters.
- Built-in sources: "churn.csv", "churn_leaky.csv", "housing.csv", "dirty_data.csv".

Returns:
{
  "dataset_id":   string,
  "source":       string,
  "row_count":    integer,
  "column_count": integer,
  "status":       "loaded"
}

Errors:
- "dataset_id already in use: <id>" if dataset_id is already registered.
- "Unknown dataset source: <source>" if source does not match a built-in dataset.""",
        params=[
            ToolParam("source",     "string", "File path or built-in dataset name.", required=True),
            ToolParam("dataset_id", "string", "Unique identifier to assign to this dataset.", required=True),
        ],
        handler=handler,
    )


# --- inspect_schema ---------------------------------------------------------

def _make_inspect_schema(store: MLDataStore) -> ToolDefinition:
    def handler(dataset_id: str, *, ctx: ToolContext) -> str:
        if dataset_id not in store.datasets:
            raise ValueError(f"dataset_id not found: {dataset_id!r}. Call load_dataset first.")
        ds = store.datasets[dataset_id]
        tmpl = ds["schema"]
        ctx.log(f"Inspecting schema for {dataset_id!r}")
        return json.dumps({
            "dataset_id":    dataset_id,
            "row_count":     tmpl["row_count"],
            "column_count":  len(tmpl["columns"]),
            "columns":       tmpl["columns"],
            "quality_issues": tmpl.get("quality_issues", []),
        })

    return ToolDefinition(
        name="inspect_schema",
        description="""Inspect the schema of one loaded dataset.

Required input object:
{
  "dataset_id": string  — must refer to a dataset loaded via load_dataset
}

Rules:
- dataset_id must be a string, not an array or file path.
- Only inspects one dataset at a time.

Returns:
{
  "dataset_id":    string,
  "row_count":     integer,
  "column_count":  integer,
  "columns": [
    {
      "name":     string,
      "dtype":    "numeric" | "categorical" | "boolean" | "datetime" | "text" | "unknown",
      "nullable": boolean
    }
  ],
  "quality_issues": string[]  — e.g. ["possible_leakage", "missing_values", "duplicate_rows"]
}

Errors:
- "dataset_id not found: <id>" if the dataset has not been loaded.""",
        params=[
            ToolParam("dataset_id", "string", "ID of the dataset to inspect.", required=True),
        ],
        handler=handler,
    )


# --- preview_rows -----------------------------------------------------------

def _make_preview_rows(store: MLDataStore) -> ToolDefinition:
    def handler(dataset_id: str, n_rows: int, offset: int = 0, *, ctx: ToolContext) -> str:
        if dataset_id not in store.datasets:
            raise ValueError(f"dataset_id not found: {dataset_id!r}.")
        if not (1 <= n_rows <= 100):
            raise ValueError(f"n_rows must be 1–100, got {n_rows}.")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}.")
        key = store.datasets[dataset_id]["builtin_key"]
        total = store.datasets[dataset_id]["schema"]["row_count"]
        rows = _mock_rows(key, min(n_rows, total - offset), offset)
        ctx.log(f"Previewing {len(rows)} rows from offset {offset}")
        return json.dumps({
            "dataset_id": dataset_id,
            "total_rows": total,
            "offset":     offset,
            "n_rows":     len(rows),
            "rows":       rows,
        })

    return ToolDefinition(
        name="preview_rows",
        description="""Return a sample of rows from one loaded dataset.

Required input object:
{
  "dataset_id": string   — must refer to a loaded dataset
  "n_rows":     integer  — number of rows to return; must be 1–100
  "offset":     integer  — row offset to start from; must be >= 0; defaults to 0 if omitted
}

Rules:
- n_rows must be an integer between 1 and 100 inclusive.
- offset must be a non-negative integer.
- Do not pass column filters here; use filter_rows for that.

Returns:
{
  "dataset_id": string,
  "total_rows": integer,
  "offset":     integer,
  "n_rows":     integer,
  "rows":       array of objects  — each object is one row with column-name keys
}

Errors:
- "dataset_id not found: <id>"
- "n_rows must be 1–100, got <n>"
- "offset must be >= 0, got <n>" """,
        params=[
            ToolParam("dataset_id", "string",  "ID of the dataset.", required=True),
            ToolParam("n_rows",     "integer", "Rows to return (1–100).", required=True),
            ToolParam("offset",     "integer", "Row index to start from (default 0).", required=False),
        ],
        handler=handler,
    )


# --- summarize_columns ------------------------------------------------------

def _make_summarize_columns(store: MLDataStore) -> ToolDefinition:
    def handler(dataset_id: str, columns: list[str], include_quantiles: bool, *, ctx: ToolContext) -> str:
        if dataset_id not in store.datasets:
            raise ValueError(f"dataset_id not found: {dataset_id!r}.")
        tmpl = store.datasets[dataset_id]["schema"]
        all_cols = {c["name"]: c for c in tmpl["columns"]}
        target_cols = list(all_cols.keys()) if not columns else columns
        for col in target_cols:
            if col not in all_cols:
                raise ValueError(f"Column not found in dataset: {col!r}.")

        rng = random.Random(42)
        summaries = []
        for col in target_cols:
            meta = all_cols[col]
            dtype = meta["dtype"]
            miss_rate = tmpl.get("missingness", {}).get(col, 0.0)
            miss_count = int(miss_rate * tmpl["row_count"])
            summary: dict[str, Any] = {
                "column":       col,
                "dtype":        dtype,
                "missing_count": miss_count,
                "missing_rate": round(miss_rate, 4),
            }
            if dtype == "numeric":
                mean = round(rng.uniform(10, 500), 2)
                std  = round(mean * rng.uniform(0.1, 0.5), 2)
                mn   = round(mean - 2 * std, 2)
                mx   = round(mean + 2 * std, 2)
                ns: dict[str, Any] = {
                    "mean":   mean,
                    "std":    std,
                    "min":    mn,
                    "max":    mx,
                }
                if include_quantiles:
                    ns["p25"]    = round(mean - 0.67 * std, 2)
                    ns["median"] = round(mean - 0.1 * std, 2)
                    ns["p75"]    = round(mean + 0.67 * std, 2)
                summary["numeric_summary"] = ns
            elif dtype == "categorical":
                summary["categorical_summary"] = {
                    "unique_count": rng.randint(2, 20),
                    "top_values": [
                        {"value": f"val_{i}", "count": rng.randint(10, 200)}
                        for i in range(3)
                    ],
                }
            summaries.append(summary)

        ctx.log(f"Summarized {len(summaries)} columns for {dataset_id!r}")
        return json.dumps({"dataset_id": dataset_id, "summaries": summaries})

    return ToolDefinition(
        name="summarize_columns",
        description="""Compute per-column summary statistics for selected columns in one dataset.

Required input object:
{
  "dataset_id":        string    — must refer to a loaded dataset
  "columns":           string[]  — array of exact column names to summarize; pass [] to summarize all columns
  "include_quantiles": boolean   — if true, include p25/median/p75; must be true or false, never omitted
}

Rules:
- columns must be an array of strings (column names), not column indexes.
- Use [] only when the intent is to summarize all columns.
- include_quantiles must be explicitly true or false.
- Column names must exactly match names returned by inspect_schema.

Returns:
{
  "dataset_id": string,
  "summaries": [
    {
      "column":        string,
      "dtype":         "numeric" | "categorical" | "boolean" | "datetime" | "text" | "unknown",
      "missing_count": integer,
      "missing_rate":  number   (0.0–1.0),
      "numeric_summary"?: {
        "mean":   number | null,
        "std":    number | null,
        "min":    number | null,
        "p25"?:   number | null,
        "median"?: number | null,
        "p75"?:   number | null,
        "max":    number | null
      },
      "categorical_summary"?: {
        "unique_count": integer,
        "top_values": [{"value": string, "count": integer}]
      }
    }
  ]
}

Errors:
- "dataset_id not found: <id>"
- "Column not found in dataset: <col>" if any element of columns does not exist.""",
        params=[
            ToolParam("dataset_id",        "string",  "ID of the dataset.", required=True),
            ToolParam("columns",           "array",   "Column names to summarize; [] = all.", required=True),
            ToolParam("include_quantiles", "boolean", "Whether to include p25/median/p75.", required=True),
        ],
        handler=handler,
    )


# --- compute_missingness ----------------------------------------------------

def _make_compute_missingness(store: MLDataStore) -> ToolDefinition:
    def handler(dataset_id: str, columns: list[str], *, ctx: ToolContext) -> str:
        if dataset_id not in store.datasets:
            raise ValueError(f"dataset_id not found: {dataset_id!r}.")
        tmpl = store.datasets[dataset_id]["schema"]
        all_cols = {c["name"]: c for c in tmpl["columns"]}
        target_cols = list(all_cols.keys()) if not columns else columns
        for col in target_cols:
            if col not in all_cols:
                raise ValueError(f"Column not found: {col!r}.")
        miss_map = tmpl.get("missingness", {})
        total = tmpl["row_count"]
        report = []
        for col in target_cols:
            rate  = miss_map.get(col, 0.0)
            count = int(rate * total)
            report.append({"column": col, "missing_count": count, "missing_rate": round(rate, 4), "nullable": all_cols[col]["nullable"]})
        ctx.log(f"Missingness computed for {len(report)} columns")
        return json.dumps({"dataset_id": dataset_id, "total_rows": total, "columns": report})

    return ToolDefinition(
        name="compute_missingness",
        description="""Compute missing-value counts and rates for selected columns in one dataset.

Required input object:
{
  "dataset_id": string    — must refer to a loaded dataset
  "columns":    string[]  — column names to check; pass [] to check all columns
}

Rules:
- columns must be an array of strings; do not pass column indexes.
- Pass [] to check every column in the dataset.

Returns:
{
  "dataset_id": string,
  "total_rows": integer,
  "columns": [
    {
      "column":        string,
      "missing_count": integer,
      "missing_rate":  number  (0.0–1.0),
      "nullable":      boolean
    }
  ]
}

Errors:
- "dataset_id not found: <id>"
- "Column not found: <col>" """,
        params=[
            ToolParam("dataset_id", "string", "ID of the dataset.", required=True),
            ToolParam("columns",    "array",  "Column names to check; [] = all.", required=True),
        ],
        handler=handler,
    )


# --- filter_rows ------------------------------------------------------------

def _make_filter_rows(store: MLDataStore) -> ToolDefinition:
    def handler(dataset_id: str, condition: str, output_dataset_id: str, *, ctx: ToolContext) -> str:
        if dataset_id not in store.datasets:
            raise ValueError(f"dataset_id not found: {dataset_id!r}.")
        if output_dataset_id in store.datasets:
            raise ValueError(f"output_dataset_id already in use: {output_dataset_id!r}.")
        if not condition.strip():
            raise ValueError("condition must be a non-empty string.")
        src = store.datasets[dataset_id]
        total = src["schema"]["row_count"]
        kept  = int(total * 0.65)  # mock: 65% pass filter
        new_schema = dict(src["schema"])
        new_schema = {**src["schema"], "row_count": kept}
        store.datasets[output_dataset_id] = {"source": f"filter({dataset_id})", "builtin_key": src["builtin_key"], "schema": new_schema}
        ctx.log(f"filter_rows: {total} → {kept} rows")
        return json.dumps({"input_dataset_id": dataset_id, "output_dataset_id": output_dataset_id, "condition": condition, "input_rows": total, "output_rows": kept, "rows_removed": total - kept})

    return ToolDefinition(
        name="filter_rows",
        description="""Filter rows in a dataset by a condition and write the result to a new dataset id.

Required input object:
{
  "dataset_id":        string — source dataset
  "condition":         string — filter expression (e.g. "tenure > 12", "Churn == 'Yes'")
  "output_dataset_id": string — id to assign to the filtered dataset; must not already exist
}

Rules:
- condition must be a non-empty string using column names exactly as they appear in the schema.
- output_dataset_id must be a new unique id, not an existing dataset.
- The original dataset is not modified.

Returns:
{
  "input_dataset_id":  string,
  "output_dataset_id": string,
  "condition":         string,
  "input_rows":        integer,
  "output_rows":       integer,
  "rows_removed":      integer
}

Errors:
- "dataset_id not found: <id>"
- "output_dataset_id already in use: <id>"
- "condition must be a non-empty string" """,
        params=[
            ToolParam("dataset_id",        "string", "Source dataset id.", required=True),
            ToolParam("condition",         "string", "Row filter expression.", required=True),
            ToolParam("output_dataset_id", "string", "Id for the output dataset.", required=True),
        ],
        handler=handler,
    )


# --- transform_columns ------------------------------------------------------

def _make_transform_columns(store: MLDataStore) -> ToolDefinition:
    _VALID_OPS = {
        "fill_missing", "drop_missing", "normalize_labels",
        "drop_duplicates", "cast_numeric", "cast_categorical",
        "log_transform", "clip_outliers",
    }

    def handler(dataset_id: str, operations: list[dict], output_dataset_id: str, *, ctx: ToolContext) -> str:
        if dataset_id not in store.datasets:
            raise ValueError(f"dataset_id not found: {dataset_id!r}.")
        if output_dataset_id in store.datasets:
            raise ValueError(f"output_dataset_id already in use: {output_dataset_id!r}.")
        if not isinstance(operations, list) or len(operations) == 0:
            raise ValueError("operations must be a non-empty array.")
        for op in operations:
            if not isinstance(op, dict):
                raise ValueError("Each operation must be an object with 'column' and 'operation' keys.")
            if "operation" not in op:
                raise ValueError("Each operation object must have an 'operation' key.")
            if op["operation"] not in _VALID_OPS:
                raise ValueError(f"Unknown operation: {op['operation']!r}. Allowed: {sorted(_VALID_OPS)}.")

        src_schema = store.datasets[dataset_id]["schema"]
        new_schema = {**src_schema, "row_count": src_schema["row_count"] - src_schema.get("duplicate_rows", 0), "duplicate_rows": 0, "quality_issues": [], "missingness": {}}
        store.datasets[output_dataset_id] = {"source": f"transform({dataset_id})", "builtin_key": store.datasets[dataset_id]["builtin_key"], "schema": new_schema}
        applied = [op.get("operation") for op in operations]
        ctx.log(f"transform_columns: applied {applied}")
        return json.dumps({"input_dataset_id": dataset_id, "output_dataset_id": output_dataset_id, "operations_applied": len(operations), "applied": applied, "status": "ok"})

    return ToolDefinition(
        name="transform_columns",
        description="""Apply one or more transformation operations to columns and write the result to a new dataset id.

Required input object:
{
  "dataset_id":        string  — source dataset
  "operations":        array   — list of operation objects; must be non-empty
  "output_dataset_id": string  — id for the output dataset; must not already exist
}

Each operation object:
{
  "column":    string  — column to apply the operation to; use "__rows__" for row-level ops
  "operation": string  — one of: "fill_missing" | "drop_missing" | "normalize_labels" |
                         "drop_duplicates" | "cast_numeric" | "cast_categorical" |
                         "log_transform" | "clip_outliers"
  "params":    object  — operation-specific parameters; use {} if none required
}

fill_missing params:  {"strategy": "mean" | "median" | "mode" | "constant", "value"?: any}
clip_outliers params: {"lower_quantile": number, "upper_quantile": number}

Rules:
- operations must be a non-empty array.
- Each element must be an object with at least "column" and "operation" keys.
- operation must be one of the exact enum values above.
- Use column="__rows__" for drop_duplicates (applies to the entire row).
- The original dataset is not modified.

Returns:
{
  "input_dataset_id":  string,
  "output_dataset_id": string,
  "operations_applied": integer,
  "applied":           string[],
  "status":            "ok"
}

Errors:
- "dataset_id not found: <id>"
- "output_dataset_id already in use: <id>"
- "operations must be a non-empty array"
- "Unknown operation: <op>" """,
        params=[
            ToolParam("dataset_id",        "string", "Source dataset id.", required=True),
            ToolParam("operations",        "array",  "List of operation objects.", required=True),
            ToolParam("output_dataset_id", "string", "Id for the output dataset.", required=True),
        ],
        handler=handler,
    )


# --- groupby_aggregate ------------------------------------------------------

def _make_groupby_aggregate(store: MLDataStore) -> ToolDefinition:
    _VALID_FUNCS = {"mean", "sum", "count", "min", "max", "std", "median"}

    def handler(dataset_id: str, group_by: list[str], aggregations: list[dict], output_dataset_id: str, *, ctx: ToolContext) -> str:
        if dataset_id not in store.datasets:
            raise ValueError(f"dataset_id not found: {dataset_id!r}.")
        if output_dataset_id in store.datasets:
            raise ValueError(f"output_dataset_id already in use: {output_dataset_id!r}.")
        if not group_by:
            raise ValueError("group_by must be a non-empty array.")
        if not aggregations:
            raise ValueError("aggregations must be a non-empty array.")
        for agg in aggregations:
            if agg.get("function") not in _VALID_FUNCS:
                raise ValueError(f"Invalid aggregation function: {agg.get('function')!r}. Allowed: {sorted(_VALID_FUNCS)}.")
        rng = random.Random(42)
        groups = rng.randint(3, 30)
        schema = store.datasets[dataset_id]["schema"]
        new_schema = {**schema, "row_count": groups}
        store.datasets[output_dataset_id] = {"source": f"groupby({dataset_id})", "builtin_key": "grouped", "schema": new_schema}
        ctx.log(f"groupby_aggregate: {groups} groups, {len(aggregations)} agg(s)")
        return json.dumps({"input_dataset_id": dataset_id, "output_dataset_id": output_dataset_id, "group_count": groups, "columns_computed": len(aggregations)})

    return ToolDefinition(
        name="groupby_aggregate",
        description="""Group rows by one or more columns and compute aggregations, writing results to a new dataset id.

Required input object:
{
  "dataset_id":        string   — source dataset
  "group_by":          string[] — non-empty array of column names to group by
  "aggregations":      array    — non-empty array of aggregation objects
  "output_dataset_id": string   — id for the output dataset; must not already exist
}

Each aggregation object:
{
  "column":   string — column to aggregate
  "function": "mean" | "sum" | "count" | "min" | "max" | "std" | "median"
}

Rules:
- group_by must be a non-empty array of exact column names.
- aggregations must be a non-empty array.
- function must be one of the exact enum values above.

Returns:
{
  "input_dataset_id":  string,
  "output_dataset_id": string,
  "group_count":       integer,
  "columns_computed":  integer
}

Errors:
- "dataset_id not found: <id>"
- "output_dataset_id already in use: <id>"
- "group_by must be a non-empty array"
- "Invalid aggregation function: <fn>" """,
        params=[
            ToolParam("dataset_id",        "string", "Source dataset id.", required=True),
            ToolParam("group_by",          "array",  "Column names to group by.", required=True),
            ToolParam("aggregations",      "array",  "Aggregation objects.", required=True),
            ToolParam("output_dataset_id", "string", "Id for the output dataset.", required=True),
        ],
        handler=handler,
    )


# --- join_tables ------------------------------------------------------------

def _make_join_tables(store: MLDataStore) -> ToolDefinition:
    _VALID_HOW = {"inner", "left", "right", "outer"}

    def handler(left_dataset_id: str, right_dataset_id: str, on: Any, how: str, output_dataset_id: str, *, ctx: ToolContext) -> str:
        for did in (left_dataset_id, right_dataset_id):
            if did not in store.datasets:
                raise ValueError(f"dataset_id not found: {did!r}.")
        if output_dataset_id in store.datasets:
            raise ValueError(f"output_dataset_id already in use: {output_dataset_id!r}.")
        if how not in _VALID_HOW:
            raise ValueError(f"how must be one of {sorted(_VALID_HOW)}, got {how!r}.")
        left_rows  = store.datasets[left_dataset_id]["schema"]["row_count"]
        right_rows = store.datasets[right_dataset_id]["schema"]["row_count"]
        joined_rows = int(min(left_rows, right_rows) * 0.9)
        left_cols  = store.datasets[left_dataset_id]["schema"]["columns"]
        right_cols = [c for c in store.datasets[right_dataset_id]["schema"]["columns"] if c["name"] != (on if isinstance(on, str) else on[0])]
        merged_cols = left_cols + right_cols
        new_schema = {"row_count": joined_rows, "columns": merged_cols, "missingness": {}, "quality_issues": [], "duplicate_rows": 0}
        store.datasets[output_dataset_id] = {"source": f"join({left_dataset_id},{right_dataset_id})", "builtin_key": "joined", "schema": new_schema}
        ctx.log(f"join_tables: {how} join → {joined_rows} rows")
        return json.dumps({"left_dataset_id": left_dataset_id, "right_dataset_id": right_dataset_id, "output_dataset_id": output_dataset_id, "how": how, "joined_rows": joined_rows})

    return ToolDefinition(
        name="join_tables",
        description="""Join two loaded datasets on a shared key column and write the result to a new dataset id.

Required input object:
{
  "left_dataset_id":   string          — id of the left dataset
  "right_dataset_id":  string          — id of the right dataset
  "on":                string | string[] — shared join key(s); string for single key, array for composite
  "how":               "inner" | "left" | "right" | "outer"
  "output_dataset_id": string          — id for the joined dataset; must not already exist
}

Rules:
- Both dataset ids must refer to loaded datasets.
- on must be a column name (string) or array of column names that exist in both datasets.
- how must be exactly one of the four enum values.
- output_dataset_id must be a new unique id.

Returns:
{
  "left_dataset_id":   string,
  "right_dataset_id":  string,
  "output_dataset_id": string,
  "how":               string,
  "joined_rows":       integer
}

Errors:
- "dataset_id not found: <id>"
- "output_dataset_id already in use: <id>"
- "how must be one of ['inner','left','outer','right'], got <how>" """,
        params=[
            ToolParam("left_dataset_id",   "string", "Id of the left dataset.", required=True),
            ToolParam("right_dataset_id",  "string", "Id of the right dataset.", required=True),
            ToolParam("on",                "string", "Join key column name (or array for composite keys).", required=True),
            ToolParam("how",               "string", "Join type.", required=True, enum=["inner", "left", "right", "outer"]),
            ToolParam("output_dataset_id", "string", "Id for the joined dataset.", required=True),
        ],
        handler=handler,
    )


# --- plot_histogram ----------------------------------------------------------

def _make_plot_histogram(store: MLDataStore) -> ToolDefinition:
    def handler(dataset_id: str, column: str, bins: int, artifact_id: str = "", *, ctx: ToolContext) -> str:
        if dataset_id not in store.datasets:
            raise ValueError(f"dataset_id not found: {dataset_id!r}.")
        if not (5 <= bins <= 100):
            raise ValueError(f"bins must be 5–100, got {bins}.")
        schema = store.datasets[dataset_id]["schema"]
        col_meta = next((c for c in schema["columns"] if c["name"] == column), None)
        if col_meta is None:
            raise ValueError(f"Column not found: {column!r}.")
        aid = artifact_id or f"plot_hist_{column}_{dataset_id}"
        store.artifacts.append({"artifact_id": aid, "type": "plot", "subtype": "histogram", "column": column, "dataset_id": dataset_id})
        ctx.log(f"plot_histogram: {column!r} with {bins} bins → {aid!r}")
        return json.dumps({"artifact_id": aid, "plot_type": "histogram", "dataset_id": dataset_id, "column": column, "bins": bins, "status": "rendered"})

    return ToolDefinition(
        name="plot_histogram",
        description="""Render a histogram for one numeric or categorical column and save it as a plot artifact.

Required input object:
{
  "dataset_id":  string  — must refer to a loaded dataset
  "column":      string  — exact column name to plot
  "bins":        integer — number of bins; must be 5–100
  "artifact_id": string  — optional; id to assign the plot artifact; auto-generated if omitted
}

Rules:
- column must exist in the dataset schema.
- bins must be an integer between 5 and 100 inclusive.
- artifact_id is optional; if omitted, an id is auto-generated.
- Plots count toward the session plot limit (max 8 per run).

Returns:
{
  "artifact_id": string,
  "plot_type":   "histogram",
  "dataset_id":  string,
  "column":      string,
  "bins":        integer,
  "status":      "rendered"
}

Errors:
- "dataset_id not found: <id>"
- "Column not found: <col>"
- "bins must be 5–100, got <n>" """,
        params=[
            ToolParam("dataset_id",  "string",  "Id of the dataset.", required=True),
            ToolParam("column",      "string",  "Column name to plot.", required=True),
            ToolParam("bins",        "integer", "Number of histogram bins (5–100).", required=True),
            ToolParam("artifact_id", "string",  "Optional plot artifact id.", required=False),
        ],
        handler=handler,
    )


# --- plot_boxplot ------------------------------------------------------------

def _make_plot_boxplot(store: MLDataStore) -> ToolDefinition:
    def handler(dataset_id: str, column: str, group_by: str = "", artifact_id: str = "", *, ctx: ToolContext) -> str:
        if dataset_id not in store.datasets:
            raise ValueError(f"dataset_id not found: {dataset_id!r}.")
        schema = store.datasets[dataset_id]["schema"]
        names = {c["name"] for c in schema["columns"]}
        if column not in names:
            raise ValueError(f"Column not found: {column!r}.")
        if group_by and group_by not in names:
            raise ValueError(f"group_by column not found: {group_by!r}.")
        aid = artifact_id or f"plot_box_{column}_{dataset_id}"
        store.artifacts.append({"artifact_id": aid, "type": "plot", "subtype": "boxplot", "column": column, "dataset_id": dataset_id})
        grp_label = group_by or "none"
        ctx.log(f"plot_boxplot: {column!r} grouped by {grp_label!r}")
        return json.dumps({"artifact_id": aid, "plot_type": "boxplot", "dataset_id": dataset_id, "column": column, "group_by": group_by or None, "status": "rendered"})

    return ToolDefinition(
        name="plot_boxplot",
        description="""Render a box plot for one numeric column, optionally grouped by a categorical column.

Required input object:
{
  "dataset_id":  string — must refer to a loaded dataset
  "column":      string — numeric column to plot; must exist in schema
  "group_by":    string — optional categorical column to group boxes by; omit or pass "" for no grouping
  "artifact_id": string — optional plot artifact id; auto-generated if omitted
}

Rules:
- column must exist in the dataset schema.
- group_by, if provided, must exist in the dataset schema and be a categorical column.
- Plots count toward the session plot limit (max 8 per run).

Returns:
{
  "artifact_id": string,
  "plot_type":   "boxplot",
  "dataset_id":  string,
  "column":      string,
  "group_by":    string | null,
  "status":      "rendered"
}

Errors:
- "dataset_id not found: <id>"
- "Column not found: <col>"
- "group_by column not found: <col>" """,
        params=[
            ToolParam("dataset_id",  "string", "Id of the dataset.", required=True),
            ToolParam("column",      "string", "Numeric column to plot.", required=True),
            ToolParam("group_by",    "string", "Optional grouping column.", required=False),
            ToolParam("artifact_id", "string", "Optional plot artifact id.", required=False),
        ],
        handler=handler,
    )


# --- plot_scatter ------------------------------------------------------------

def _make_plot_scatter(store: MLDataStore) -> ToolDefinition:
    def handler(dataset_id: str, x_column: str, y_column: str, color_column: str = "", artifact_id: str = "", *, ctx: ToolContext) -> str:
        if dataset_id not in store.datasets:
            raise ValueError(f"dataset_id not found: {dataset_id!r}.")
        names = {c["name"] for c in store.datasets[dataset_id]["schema"]["columns"]}
        for col in (x_column, y_column):
            if col not in names:
                raise ValueError(f"Column not found: {col!r}.")
        if color_column and color_column not in names:
            raise ValueError(f"color_column not found: {color_column!r}.")
        aid = artifact_id or f"plot_scatter_{x_column}_{y_column}"
        store.artifacts.append({"artifact_id": aid, "type": "plot", "subtype": "scatter"})
        ctx.log(f"plot_scatter: x={x_column!r} y={y_column!r}")
        return json.dumps({"artifact_id": aid, "plot_type": "scatter", "dataset_id": dataset_id, "x_column": x_column, "y_column": y_column, "color_column": color_column or None, "status": "rendered"})

    return ToolDefinition(
        name="plot_scatter",
        description="""Render a scatter plot of two numeric columns, optionally coloured by a third column.

Required input object:
{
  "dataset_id":    string — must refer to a loaded dataset
  "x_column":      string — column for the x-axis; must exist in schema
  "y_column":      string — column for the y-axis; must exist in schema
  "color_column":  string — optional column for point colour; omit or pass "" for no colouring
  "artifact_id":   string — optional plot artifact id; auto-generated if omitted
}

Rules:
- x_column and y_column must exist in the dataset schema.
- color_column, if provided, must exist in the schema.
- x_column, y_column, and color_column must all be different columns.
- Plots count toward the session plot limit (max 8 per run).

Returns:
{
  "artifact_id":  string,
  "plot_type":    "scatter",
  "dataset_id":   string,
  "x_column":     string,
  "y_column":     string,
  "color_column": string | null,
  "status":       "rendered"
}

Errors:
- "dataset_id not found: <id>"
- "Column not found: <col>"
- "color_column not found: <col>" """,
        params=[
            ToolParam("dataset_id",   "string", "Id of the dataset.", required=True),
            ToolParam("x_column",     "string", "X-axis column.", required=True),
            ToolParam("y_column",     "string", "Y-axis column.", required=True),
            ToolParam("color_column", "string", "Optional colour column.", required=False),
            ToolParam("artifact_id",  "string", "Optional plot artifact id.", required=False),
        ],
        handler=handler,
    )


# --- plot_correlation_matrix ------------------------------------------------

def _make_plot_correlation_matrix(store: MLDataStore) -> ToolDefinition:
    def handler(dataset_id: str, columns: list[str], artifact_id: str = "", *, ctx: ToolContext) -> str:
        if dataset_id not in store.datasets:
            raise ValueError(f"dataset_id not found: {dataset_id!r}.")
        schema = store.datasets[dataset_id]["schema"]
        numeric_cols = [c["name"] for c in schema["columns"] if c["dtype"] == "numeric"]
        target_cols  = numeric_cols if not columns else columns
        if len(target_cols) < 2:
            raise ValueError("At least 2 numeric columns are required for a correlation matrix.")
        rng = random.Random(42)
        corr: dict[str, dict[str, float]] = {}
        for i, c1 in enumerate(target_cols):
            corr[c1] = {}
            for j, c2 in enumerate(target_cols):
                corr[c1][c2] = 1.0 if i == j else round(rng.uniform(-0.9, 0.9), 3)
        aid = artifact_id or f"plot_corr_{dataset_id}"
        store.artifacts.append({"artifact_id": aid, "type": "plot", "subtype": "correlation_matrix"})
        ctx.log(f"plot_correlation_matrix: {len(target_cols)} columns")
        return json.dumps({"artifact_id": aid, "plot_type": "correlation_matrix", "dataset_id": dataset_id, "columns": target_cols, "correlation_matrix": corr, "status": "rendered"})

    return ToolDefinition(
        name="plot_correlation_matrix",
        description="""Compute and render a Pearson correlation matrix for selected numeric columns.

Required input object:
{
  "dataset_id":  string    — must refer to a loaded dataset
  "columns":     string[]  — numeric column names to include; pass [] to use all numeric columns
  "artifact_id": string    — optional plot artifact id; auto-generated if omitted
}

Rules:
- columns must be an array of strings (column names); pass [] to auto-select all numeric columns.
- At least 2 numeric columns must be present/selected.
- Non-numeric columns are silently excluded when columns=[].
- Plots count toward the session plot limit (max 8 per run).

Returns:
{
  "artifact_id":        string,
  "plot_type":          "correlation_matrix",
  "dataset_id":         string,
  "columns":            string[],
  "correlation_matrix": object  — keys are column names; values are objects of {column: correlation_coefficient}
  "status":             "rendered"
}

Errors:
- "dataset_id not found: <id>"
- "At least 2 numeric columns are required for a correlation matrix." """,
        params=[
            ToolParam("dataset_id",  "string", "Id of the dataset.", required=True),
            ToolParam("columns",     "array",  "Numeric column names; [] = all numeric.", required=True),
            ToolParam("artifact_id", "string", "Optional plot artifact id.", required=False),
        ],
        handler=handler,
    )


# --- detect_outliers --------------------------------------------------------

def _make_detect_outliers(store: MLDataStore) -> ToolDefinition:
    _VALID_METHODS = {"iqr", "zscore"}

    def handler(dataset_id: str, columns: list[str], method: str, threshold: float, *, ctx: ToolContext) -> str:
        if dataset_id not in store.datasets:
            raise ValueError(f"dataset_id not found: {dataset_id!r}.")
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be 'iqr' or 'zscore', got {method!r}.")
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}.")
        schema = store.datasets[dataset_id]["schema"]
        names  = {c["name"] for c in schema["columns"]}
        for col in columns:
            if col not in names:
                raise ValueError(f"Column not found: {col!r}.")
        rng = random.Random(42)
        results = [{"column": col, "outlier_count": rng.randint(0, 40), "outlier_rate": round(rng.uniform(0, 0.05), 4)} for col in columns]
        ctx.log(f"detect_outliers: method={method!r}, {len(columns)} columns")
        return json.dumps({"dataset_id": dataset_id, "method": method, "threshold": threshold, "columns": results})

    return ToolDefinition(
        name="detect_outliers",
        description="""Detect outliers in selected numeric columns using IQR or Z-score method.

Required input object:
{
  "dataset_id": string    — must refer to a loaded dataset
  "columns":    string[]  — non-empty array of numeric column names to check
  "method":     "iqr" | "zscore"
  "threshold":  number    — IQR multiplier (e.g. 1.5) for iqr; standard deviations for zscore; must be > 0
}

Rules:
- columns must be a non-empty array of exact column names.
- method must be exactly "iqr" or "zscore".
- threshold must be a positive number (> 0).
- Typical values: threshold=1.5 for iqr, threshold=3.0 for zscore.

Returns:
{
  "dataset_id": string,
  "method":     string,
  "threshold":  number,
  "columns": [
    {
      "column":        string,
      "outlier_count": integer,
      "outlier_rate":  number  (0.0–1.0)
    }
  ]
}

Errors:
- "dataset_id not found: <id>"
- "Column not found: <col>"
- "method must be 'iqr' or 'zscore', got <method>"
- "threshold must be > 0, got <n>" """,
        params=[
            ToolParam("dataset_id", "string", "Id of the dataset.", required=True),
            ToolParam("columns",    "array",  "Numeric column names to check.", required=True),
            ToolParam("method",     "string", "Detection method.", required=True, enum=["iqr", "zscore"]),
            ToolParam("threshold",  "number", "Outlier threshold (e.g. 1.5 for iqr, 3.0 for zscore).", required=True),
        ],
        handler=handler,
    )


# --- split_dataset ----------------------------------------------------------

def _make_split_dataset(store: MLDataStore) -> ToolDefinition:
    def handler(
        dataset_id: str,
        target_column: str,
        task_type: str,
        train_fraction: float,
        validation_fraction: float,
        test_fraction: float,
        random_seed: int,
        stratify: bool,
        *,
        ctx: ToolContext,
    ) -> str:
        if dataset_id not in store.datasets:
            raise ValueError(f"dataset_id not found: {dataset_id!r}.")
        if task_type not in ("classification", "regression"):
            raise ValueError(f"task_type must be 'classification' or 'regression', got {task_type!r}.")
        total_frac = round(train_fraction + validation_fraction + test_fraction, 6)
        if abs(total_frac - 1.0) > 1e-5:
            raise ValueError(f"train_fraction + validation_fraction + test_fraction must equal 1.0, got {total_frac}.")
        if stratify and task_type != "classification":
            raise ValueError("stratify=true is only valid for classification tasks.")
        schema = store.datasets[dataset_id]["schema"]
        col_names = {c["name"] for c in schema["columns"]}
        if target_column not in col_names:
            raise ValueError(f"target_column {target_column!r} not found in dataset schema.")
        total = schema["row_count"]
        split_id = store.next_split_id()
        partitions = {
            "train_rows":      int(total * train_fraction),
            "validation_rows": int(total * validation_fraction),
            "test_rows":       total - int(total * train_fraction) - int(total * validation_fraction),
        }
        store.splits[split_id] = {
            "dataset_id":    dataset_id,
            "task_type":     task_type,
            "target_column": target_column,
            "partitions":    partitions,
            "random_seed":   random_seed,
        }
        ctx.log(f"split_dataset: {split_id!r}, target={target_column!r}, task={task_type!r}")
        return json.dumps({
            "split_id":      split_id,
            "dataset_id":    dataset_id,
            "task_type":     task_type,
            "target_column": target_column,
            "partitions":    partitions,
            "random_seed":   random_seed,
            "stratify":      stratify,
        })

    return ToolDefinition(
        name="split_dataset",
        description="""Split one loaded dataset into train / validation / test partitions for supervised learning.

Required input object:
{
  "dataset_id":          string  — must refer to a loaded dataset
  "target_column":       string  — exact name of the prediction target column; must exist in schema
  "task_type":           "classification" | "regression"
  "train_fraction":      number  — fraction of rows for training (e.g. 0.70)
  "validation_fraction": number  — fraction of rows for validation (e.g. 0.15)
  "test_fraction":       number  — fraction of rows for testing (e.g. 0.15)
  "random_seed":         integer — seed for reproducibility (e.g. 42)
  "stratify":            boolean — whether to stratify the split; only valid for classification
}

Rules:
- train_fraction + validation_fraction + test_fraction must equal exactly 1.0 (within 1e-5).
- stratify may only be true when task_type is "classification".
- target_column must be a single string, not an array.
- Do not call this tool until the target column is explicitly identified.
- target_column must exist in the dataset schema; error if missing.

Returns:
{
  "split_id":      string,
  "dataset_id":    string,
  "task_type":     "classification" | "regression",
  "target_column": string,
  "partitions": {
    "train_rows":      integer,
    "validation_rows": integer,
    "test_rows":       integer
  },
  "random_seed":   integer,
  "stratify":      boolean
}

Errors:
- "dataset_id not found: <id>"
- "task_type must be 'classification' or 'regression', got <type>"
- "train_fraction + validation_fraction + test_fraction must equal 1.0, got <n>"
- "stratify=true is only valid for classification tasks"
- "target_column '<col>' not found in dataset schema" """,
        params=[
            ToolParam("dataset_id",          "string",  "Id of the dataset.", required=True),
            ToolParam("target_column",        "string",  "Target column name.", required=True),
            ToolParam("task_type",            "string",  "Task type.", required=True, enum=["classification", "regression"]),
            ToolParam("train_fraction",       "number",  "Fraction for training (e.g. 0.70).", required=True),
            ToolParam("validation_fraction",  "number",  "Fraction for validation (e.g. 0.15).", required=True),
            ToolParam("test_fraction",        "number",  "Fraction for testing (e.g. 0.15).", required=True),
            ToolParam("random_seed",          "integer", "Random seed for reproducibility.", required=True),
            ToolParam("stratify",             "boolean", "Stratify on target (classification only).", required=True),
        ],
        handler=handler,
    )


# --- train_model ------------------------------------------------------------

def _make_train_model(store: MLDataStore) -> ToolDefinition:
    def handler(
        split_id: str,
        model_family: str,
        feature_columns: list[str],
        target_column: str,
        hyperparameters: dict,
        *,
        ctx: ToolContext,
    ) -> str:
        if split_id not in store.splits:
            raise ValueError(f"split_id not found: {split_id!r}. Call split_dataset first.")
        if model_family not in _ALLOWED_MODEL_FAMILIES:
            raise ValueError(f"model_family must be one of {sorted(_ALLOWED_MODEL_FAMILIES)}, got {model_family!r}.")
        split = store.splits[split_id]
        if target_column != split["target_column"]:
            raise ValueError(f"target_column {target_column!r} does not match split target {split['target_column']!r}.")
        if target_column in feature_columns:
            raise ValueError(f"target_column {target_column!r} must not appear in feature_columns.")
        if not feature_columns:
            raise ValueError("feature_columns must be a non-empty array.")
        task_type = split["task_type"]
        if task_type == "regression" and model_family == "logistic_regression":
            raise ValueError("logistic_regression is only valid for classification tasks.")
        if task_type == "classification" and model_family == "linear_regression":
            raise ValueError("linear_regression is only valid for regression tasks.")
        model_id = store.next_model_id()
        store.models[model_id] = {
            "split_id":       split_id,
            "model_family":   model_family,
            "task_type":      task_type,
            "feature_columns": feature_columns,
            "target_column":  target_column,
            "hyperparameters": hyperparameters,
        }
        ctx.log(f"train_model: {model_id!r}, family={model_family!r}, task={task_type!r}")
        return json.dumps({
            "model_id":     model_id,
            "model_family": model_family,
            "task_type":    task_type,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "training_summary": {
                "train_rows":      split["partitions"]["train_rows"],
                "validation_rows": split["partitions"]["validation_rows"],
                "fit_status":      "success",
            },
        })

    return ToolDefinition(
        name="train_model",
        description="""Train one baseline model on an existing supervised split.

Required input object:
{
  "split_id":        string    — must come from a prior split_dataset call
  "model_family":    "linear_regression" | "logistic_regression" | "random_forest" | "xgboost" | "lightgbm"
  "feature_columns": string[]  — non-empty array of exact column names; must not include target_column
  "target_column":   string    — must exactly match the target declared in split_dataset
  "hyperparameters": object    — model-specific params; use {} to request defaults
}

Rules:
- split_id must refer to a split created by split_dataset.
- model_family must be exactly one of the five enum values.
- feature_columns must be a non-empty array; never include target_column in this array.
- target_column must exactly match the split's target_column; error if it differs.
- logistic_regression is only valid for classification tasks.
- linear_regression is only valid for regression tasks.
- hyperparameters must be an object; use {} to request model defaults.
- Do not train on the test partition; training uses the train partition only.

Returns:
{
  "model_id":        string,
  "model_family":    string,
  "task_type":       "classification" | "regression",
  "feature_columns": string[],
  "target_column":   string,
  "training_summary": {
    "train_rows":      integer,
    "validation_rows": integer,
    "fit_status":      "success" | "failed"
  }
}

Errors:
- "split_id not found: <id>"
- "model_family must be one of [...], got <family>"
- "target_column '<col>' does not match split target '<split_target>'"
- "target_column '<col>' must not appear in feature_columns"
- "feature_columns must be a non-empty array"
- "logistic_regression is only valid for classification tasks"
- "linear_regression is only valid for regression tasks" """,
        params=[
            ToolParam("split_id",         "string", "Id from split_dataset.", required=True),
            ToolParam("model_family",     "string", "Model algorithm.", required=True,
                      enum=["linear_regression", "logistic_regression", "random_forest", "xgboost", "lightgbm"]),
            ToolParam("feature_columns",  "array",  "Feature column names (must not include target).", required=True),
            ToolParam("target_column",    "string", "Target column name (must match split).", required=True),
            ToolParam("hyperparameters",  "object", "Model hyperparameters; {} for defaults.", required=True),
        ],
        handler=handler,
    )


# --- evaluate_model ---------------------------------------------------------

def _make_evaluate_model(store: MLDataStore) -> ToolDefinition:
    def handler(model_id: str, evaluation_split: str, metrics: list[str], *, ctx: ToolContext) -> str:
        if model_id not in store.models:
            raise ValueError(f"model_id not found: {model_id!r}. Call train_model first.")
        if evaluation_split not in ("validation", "test"):
            raise ValueError(f"evaluation_split must be 'validation' or 'test', got {evaluation_split!r}.")
        if not metrics:
            raise ValueError("metrics must be a non-empty array.")
        model = store.models[model_id]
        task_type = model["task_type"]
        allowed = _ALLOWED_METRICS[task_type]
        for m in metrics:
            if m not in allowed:
                raise ValueError(f"Metric {m!r} is not allowed for {task_type} tasks. Allowed: {sorted(allowed)}.")
        mock_all = _MOCK_METRICS[task_type].get(model["model_family"], {})
        result_metrics = {m: mock_all.get(m, 0.0) for m in metrics}
        ctx.log(f"evaluate_model: {model_id!r} on {evaluation_split!r} split")
        return json.dumps({
            "model_id":         model_id,
            "model_family":     model["model_family"],
            "task_type":        task_type,
            "evaluation_split": evaluation_split,
            "metrics":          result_metrics,
        })

    return ToolDefinition(
        name="evaluate_model",
        description="""Evaluate one trained model on its validation or test partition.

Required input object:
{
  "model_id":         string   — must refer to a model from train_model
  "evaluation_split": "validation" | "test"
  "metrics":          string[] — non-empty array of metric names
}

Allowed metrics by task type:
- classification: ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"]
- regression:     ["mae", "rmse", "r2", "mape"]

Rules:
- model_id must refer to a trained model.
- evaluation_split must be exactly "validation" or "test".
- metrics must be a non-empty array.
- All metrics must belong to the correct task type; mixing classification and regression metrics is an error.
- During model selection, always evaluate on "validation"; never use "test" for selection.
- Use "test" at most once for the final performance estimate.

Returns:
{
  "model_id":         string,
  "model_family":     string,
  "task_type":        "classification" | "regression",
  "evaluation_split": "validation" | "test",
  "metrics":          { "<metric_name>": number }
}

Errors:
- "model_id not found: <id>"
- "evaluation_split must be 'validation' or 'test', got <val>"
- "metrics must be a non-empty array"
- "Metric '<m>' is not allowed for <task_type> tasks" """,
        params=[
            ToolParam("model_id",         "string", "Id from train_model.", required=True),
            ToolParam("evaluation_split", "string", "Which partition to evaluate on.", required=True, enum=["validation", "test"]),
            ToolParam("metrics",          "array",  "Metric names to compute.", required=True),
        ],
        handler=handler,
    )


# --- compare_models ---------------------------------------------------------

def _make_compare_models(store: MLDataStore) -> ToolDefinition:
    def handler(model_ids: list[str], evaluation_split: str, primary_metric: str, *, ctx: ToolContext) -> str:
        if not model_ids or len(model_ids) < 2:
            raise ValueError("model_ids must contain at least 2 model ids.")
        if evaluation_split not in ("validation", "test"):
            raise ValueError(f"evaluation_split must be 'validation' or 'test', got {evaluation_split!r}.")
        for mid in model_ids:
            if mid not in store.models:
                raise ValueError(f"model_id not found: {mid!r}.")
        task_types = {store.models[mid]["task_type"] for mid in model_ids}
        if len(task_types) > 1:
            raise ValueError("All models must have the same task_type.")
        task_type = task_types.pop()
        allowed = _ALLOWED_METRICS[task_type]
        if primary_metric not in allowed:
            raise ValueError(f"primary_metric {primary_metric!r} not valid for {task_type}. Allowed: {sorted(allowed)}.")
        rows = []
        for mid in model_ids:
            fam  = store.models[mid]["model_family"]
            vals = _MOCK_METRICS[task_type].get(fam, {})
            rows.append({"model_id": mid, "model_family": fam, primary_metric: vals.get(primary_metric, 0.0)})
        higher_is_better = primary_metric not in {"mae", "rmse", "log_loss", "mape"}
        best = max(rows, key=lambda r: r[primary_metric]) if higher_is_better else min(rows, key=lambda r: r[primary_metric])
        ctx.log(f"compare_models: best={best['model_id']!r} by {primary_metric!r}")
        return json.dumps({
            "evaluation_split": evaluation_split,
            "primary_metric":   primary_metric,
            "higher_is_better": higher_is_better,
            "comparison":       rows,
            "best_model_id":    best["model_id"],
            "best_model_family": best["model_family"],
        })

    return ToolDefinition(
        name="compare_models",
        description="""Compare two or more trained models on a shared evaluation split using a primary metric.

Required input object:
{
  "model_ids":        string[] — array of at least 2 model ids from train_model
  "evaluation_split": "validation" | "test"
  "primary_metric":   string  — metric to rank models by; must be valid for the shared task type
}

Rules:
- model_ids must contain at least 2 ids.
- All models must share the same task_type; mixing classification and regression models is an error.
- evaluation_split must be "validation" or "test".
- primary_metric must be valid for the task type (see evaluate_model for allowed lists).
- Use "validation" for model selection; use "test" only for the final comparison.

Returns:
{
  "evaluation_split":  "validation" | "test",
  "primary_metric":    string,
  "higher_is_better":  boolean,
  "comparison": [
    {
      "model_id":     string,
      "model_family": string,
      "<primary_metric>": number
    }
  ],
  "best_model_id":     string,
  "best_model_family": string
}

Errors:
- "model_ids must contain at least 2 model ids"
- "model_id not found: <id>"
- "All models must have the same task_type"
- "evaluation_split must be 'validation' or 'test', got <val>"
- "primary_metric '<m>' not valid for <task_type>" """,
        params=[
            ToolParam("model_ids",        "array",  "At least 2 model ids.", required=True),
            ToolParam("evaluation_split", "string", "Partition to evaluate on.", required=True, enum=["validation", "test"]),
            ToolParam("primary_metric",   "string", "Metric to rank models by.", required=True),
        ],
        handler=handler,
    )


# --- save_artifact ----------------------------------------------------------

def _make_save_artifact(store: MLDataStore) -> ToolDefinition:
    _VALID_TYPES = {"dataset", "model", "plot", "report"}

    def handler(artifact_type: str, source_id: str, filename: str, metadata: dict = None, *, ctx: ToolContext) -> str:
        if artifact_type not in _VALID_TYPES:
            raise ValueError(f"artifact_type must be one of {sorted(_VALID_TYPES)}, got {artifact_type!r}.")
        if not filename.strip():
            raise ValueError("filename must be a non-empty string.")
        if artifact_type == "dataset" and source_id not in store.datasets:
            raise ValueError(f"source_id (dataset) not found: {source_id!r}.")
        if artifact_type == "model" and source_id not in store.models:
            raise ValueError(f"source_id (model) not found: {source_id!r}.")
        meta = metadata or {}
        aid = store.next_artifact_id()
        store.artifacts.append({"artifact_id": aid, "artifact_type": artifact_type, "source_id": source_id, "filename": filename, "metadata": meta})
        ctx.log(f"save_artifact: {aid!r} → {filename!r}")
        return json.dumps({"artifact_id": aid, "artifact_type": artifact_type, "source_id": source_id, "filename": filename, "status": "saved"})

    return ToolDefinition(
        name="save_artifact",
        description="""Save a dataset, model, plot, or report as a named artifact for later retrieval or sharing.

Required input object:
{
  "artifact_type": "dataset" | "model" | "plot" | "report"
  "source_id":     string  — id of the dataset/model/plot/report to save
  "filename":      string  — desired output filename (e.g. "clean_data.csv", "rf_model.pkl")
  "metadata":      object  — optional key/value metadata; use {} if none
}

Rules:
- artifact_type must be exactly one of the four enum values.
- source_id must refer to an existing entity of the matching type.
- filename must be a non-empty string; include the appropriate extension.
- metadata must be an object; use {} if no metadata is needed.

Returns:
{
  "artifact_id":   string,
  "artifact_type": string,
  "source_id":     string,
  "filename":      string,
  "status":        "saved"
}

Errors:
- "artifact_type must be one of ['dataset','model','plot','report'], got <type>"
- "filename must be a non-empty string"
- "source_id (dataset) not found: <id>"
- "source_id (model) not found: <id>" """,
        params=[
            ToolParam("artifact_type", "string", "Type of artifact to save.", required=True, enum=["dataset", "model", "plot", "report"]),
            ToolParam("source_id",     "string", "Id of the entity to save.", required=True),
            ToolParam("filename",      "string", "Output filename.", required=True),
            ToolParam("metadata",      "object", "Optional metadata key/value pairs.", required=False),
        ],
        handler=handler,
    )


# --- generate_report --------------------------------------------------------

def _make_generate_report(store: MLDataStore) -> ToolDefinition:
    _VALID_CONTENT_TYPES = {"text", "metric_table", "plot_ref"}

    def handler(title: str, sections: list[dict], artifact_id: str = "", *, ctx: ToolContext) -> str:
        if not title.strip():
            raise ValueError("title must be a non-empty string.")
        if not sections:
            raise ValueError("sections must be a non-empty array.")
        for i, sec in enumerate(sections):
            if not isinstance(sec, dict):
                raise ValueError(f"sections[{i}] must be an object.")
            if sec.get("content_type") not in _VALID_CONTENT_TYPES:
                raise ValueError(f"sections[{i}].content_type must be one of {sorted(_VALID_CONTENT_TYPES)}.")
        aid = artifact_id or store.next_artifact_id()
        store.artifacts.append({"artifact_id": aid, "type": "report", "title": title, "section_count": len(sections)})
        ctx.log(f"generate_report: {aid!r}, {len(sections)} sections")
        return json.dumps({"artifact_id": aid, "title": title, "section_count": len(sections), "status": "generated"})

    return ToolDefinition(
        name="generate_report",
        description="""Generate a structured analysis report and save it as an artifact.

Required input object:
{
  "title":       string  — report title; must be non-empty
  "sections":    array   — non-empty array of section objects
  "artifact_id": string  — optional report artifact id; auto-generated if omitted
}

Each section object:
{
  "heading":      string  — section heading text
  "content_type": "text" | "metric_table" | "plot_ref"
  "content":      string  — for "text": narrative text;
                            for "metric_table": JSON-encoded table rows;
                            for "plot_ref": artifact_id of a plot
}

Rules:
- title must be a non-empty string.
- sections must be a non-empty array.
- Each section must have heading, content_type, and content.
- content_type must be exactly one of the three enum values.

Returns:
{
  "artifact_id":   string,
  "title":         string,
  "section_count": integer,
  "status":        "generated"
}

Errors:
- "title must be a non-empty string"
- "sections must be a non-empty array"
- "sections[<i>] must be an object"
- "sections[<i>].content_type must be one of ['metric_table','plot_ref','text']" """,
        params=[
            ToolParam("title",       "string", "Report title.", required=True),
            ToolParam("sections",    "array",  "Report sections.", required=True),
            ToolParam("artifact_id", "string", "Optional artifact id.", required=False),
        ],
        handler=handler,
    )
