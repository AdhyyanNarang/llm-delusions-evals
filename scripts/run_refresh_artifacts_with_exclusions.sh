#!/usr/bin/env bash
set -euo pipefail

# Regenerate report + analysis artifacts with participant exclusions.
#
# This script does NOT rerun evals. It only rebuilds report/analysis outputs
# from existing logs.
#
# Usage:
#   bash scripts/run_refresh_artifacts_with_exclusions.sh \
#     [logs_dir] [context_logs_dir] [report_dir] [uniform_sample_upper_limit]
#
# Examples:
#   bash scripts/run_refresh_artifacts_with_exclusions.sh
#   bash scripts/run_refresh_artifacts_with_exclusions.sh logs logs-context report 350
#   bash scripts/run_refresh_artifacts_with_exclusions.sh logs logs-context report ""

LOGS_DIR="${1:-logs}"
CONTEXT_LOGS_DIR="${2:-logs-context}"
REPORT_DIR="${3:-report}"
UNIFORM_SAMPLE_UPPER_LIMIT="${4:-350}"

# Top-level exports for this refresh run.
export LLM_DELUSIONS_EXCLUDED_PARTICIPANTS="${LLM_DELUSIONS_EXCLUDED_PARTICIPANTS:-103,107,115}"
export LLM_DELUSIONS_WINDOWS_PATH="${LLM_DELUSIONS_WINDOWS_PATH:-../llm-delusions/subsets/items_sanitized.parquet}"
export LLM_DELUSIONS_TRANSCRIPTS_PATH="${LLM_DELUSIONS_TRANSCRIPTS_PATH:-../llm-delusions/transcripts_data/transcripts.parquet}"
# Optional override for methods stats dataset:
#   LLM_DELUSIONS_ITEMS_SANITIZED_PATH=../llm-delusions/subsets/items_sanitized_no_103_107_115.parquet
# If unset, compute_methods_stats uses its internal default path.
export LLM_DELUSIONS_ITEMS_SANITIZED_PATH="${LLM_DELUSIONS_ITEMS_SANITIZED_PATH:-}"
EXPORT_OVERLEAF_ASSETS="${EXPORT_OVERLEAF_ASSETS:-0}"

CONTEXT_CACHE_ARGS=()
if [[ "${NO_CACHE_CONTEXT:-1}" == "1" ]]; then
  CONTEXT_CACHE_ARGS=(--no-cache)
fi

SUMMARY_PATH="${REPORT_DIR}/summary.json"

echo "== Refresh Config =="
echo "Logs dir: ${LOGS_DIR}"
echo "Context logs dir: ${CONTEXT_LOGS_DIR}"
echo "Report dir: ${REPORT_DIR}"
echo "Summary path: ${SUMMARY_PATH}"
echo "Excluded participants: ${LLM_DELUSIONS_EXCLUDED_PARTICIPANTS:-<none>}"
echo "Windows path: ${LLM_DELUSIONS_WINDOWS_PATH}"
echo "Transcripts path: ${LLM_DELUSIONS_TRANSCRIPTS_PATH}"
if [[ -n "${LLM_DELUSIONS_ITEMS_SANITIZED_PATH}" ]]; then
  echo "Methods sanitized path override: ${LLM_DELUSIONS_ITEMS_SANITIZED_PATH}"
else
  echo "Methods sanitized path override: <default>"
fi
echo "Uniform sample upper limit: ${UNIFORM_SAMPLE_UPPER_LIMIT:-<disabled>}"
echo "Context cache mode: ${NO_CACHE_CONTEXT:-1} (1 means --no-cache)"
echo "Export overleaf assets: ${EXPORT_OVERLEAF_ASSETS} (1 means enabled)"

echo
echo "== 1) Generate report snapshot =="
uv run python -m llm_delusions_evals.scripts.generate_report \
  --logs-dir "${LOGS_DIR}" \
  --output-dir "${REPORT_DIR}"

echo
echo "== 2) Regenerate figures/tables from summary =="
uv run python -m analysis.generate_figures \
  --summary "${SUMMARY_PATH}"

echo
echo "== 3) Regenerate context effects (non-uniform outputs) =="
uv run python -m analysis.compute_context_effects \
  --logs-dir "${CONTEXT_LOGS_DIR}" \
  --baseline-logs-dir "${LOGS_DIR}" \
  "${CONTEXT_CACHE_ARGS[@]}"

echo
echo "== 4) Regenerate context code-control outputs =="
uv run python -m analysis.compute_context_code_controls \
  --logs-dir "${CONTEXT_LOGS_DIR}" \
  --baseline-logs-dir "${LOGS_DIR}" \
  "${CONTEXT_CACHE_ARGS[@]}"

if [[ -n "${UNIFORM_SAMPLE_UPPER_LIMIT}" ]]; then
  echo
  echo "== 5) Regenerate uniform context outputs =="
  uv run python -m analysis.compute_context_effects \
    --logs-dir "${CONTEXT_LOGS_DIR}" \
    --baseline-logs-dir "${LOGS_DIR}" \
    --uniform-sample-upper-limit "${UNIFORM_SAMPLE_UPPER_LIMIT}" \
    "${CONTEXT_CACHE_ARGS[@]}"
fi

echo
echo "== 6) Export methods high-level stats =="
uv run python -m analysis.compute_methods_stats

echo
echo "== 7) Regenerate salient qualitative examples =="
uv run python -m analysis.extract_salient_examples \
  --group-by category \
  --rows-path "${REPORT_DIR}/eval_rows.parquet" \
  --summary-path "${SUMMARY_PATH}" \
  --csv-out analysis/data/salient_examples.csv \
  --tex-out analysis/tables/salient_examples.tex

uv run python -m analysis.extract_salient_examples \
  --group-by code \
  --rows-path "${REPORT_DIR}/eval_rows.parquet" \
  --summary-path "${SUMMARY_PATH}" \
  --csv-out analysis/data/salient_code_examples.csv \
  --tex-out analysis/tables/salient_code_examples.tex

if [[ "${EXPORT_OVERLEAF_ASSETS}" == "1" ]]; then
  echo
  echo "== 8) Export selected artifacts to overleaf repo =="
  uv run python -m analysis.export_overleaf_assets
fi

echo
echo "Done. Refreshed report + context artifacts with participant exclusions."
