"""Unified delusions eval task.

Supports two context modes controlled by the ``context_mode`` parameter:

- ``context_mode=0`` (default): window-only -- the model sees only the
  conversation window with no prior context.
- ``context_mode=1``: context + window -- the full conversation history
  before the window is prepended.

Usage::

    # Window-only (default)
    uv run inspect eval src/llm_delusions_evals/tasks/delusions_eval.py \
      --model openai/gpt-4o-mini -T context_mode=0

    # Context + window
    uv run inspect eval src/llm_delusions_evals/tasks/delusions_eval.py \
      --model openai/gpt-4o-mini -T context_mode=1
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
from inspect_ai import Task, task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from llm_delusions_annotations.annotation_metadata import (
    filter_analysis_metadata,
    load_annotation_metadata_with_role_splits,
)
from llm_delusions_annotations.annotation_prompts import ANNOTATIONS_FILE
from llm_delusions_subsets.eval_dataset import (
    load_context_window_samples,
    load_window_samples,
)

from llm_delusions_evals.constants import get_source_id, normalize_id
from llm_delusions_evals.paths import (
    DEFAULT_TRANSCRIPTS_PATH,
    DEFAULT_WINDOWS_PATH,
    resolve_path,
)
from llm_delusions_evals.scorers.annotation_scorer import metadata_annotation_scorer

_metadata_raw = filter_analysis_metadata(
    load_annotation_metadata_with_role_splits(ANNOTATIONS_FILE),
)

# Apply ID rename mapping to metadata
_metadata = {normalize_id(aid): metadata for aid, metadata in _metadata_raw.items()}

_WINDOW_ANNOTATION_IDS = sorted(aid for aid in _metadata if aid.startswith("bot-"))

_GRADER_ALIASES = {
    "gemini-3": "google/vertex/gemini-3-flash-preview",
    "gemini-2.5": "google/vertex/gemini-2.5-flash",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-5": "openai/gpt-5.1-2025-11-13",
    "mock": "mockllm/model",
}


def _filter_samples_by_min_context_length(
    raw_samples: list[dict], min_context_messages: int
) -> list[dict]:
    """Filter context-mode samples by effective context length.

    Parameters
    ----------
    raw_samples:
        Raw samples returned by ``load_context_window_samples``.
    min_context_messages:
        Minimum effective context length to keep.

    Returns
    -------
    list[dict]
        Samples with ``metadata.context_length >= min_context_messages``.
    """
    kept_samples: list[dict] = []
    for sample in raw_samples:
        metadata = sample.get("metadata", {}) if isinstance(sample, dict) else {}
        raw_context_length = metadata.get("context_length")
        try:
            context_length = int(raw_context_length)
        except (TypeError, ValueError):
            continue
        if context_length >= min_context_messages:
            kept_samples.append(sample)
    return kept_samples


def _resolve_min_context_messages(max_context_messages: int) -> int:
    """Resolve min context filtering from environment configuration.

    Parameters
    ----------
    max_context_messages:
        Requested maximum context length for the run.

    Returns
    -------
    int
        Minimum effective context length threshold.
    """
    raw_value = os.getenv("LLM_DELUSIONS_MIN_CONTEXT_MESSAGES", "0")
    try:
        min_context_messages = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "LLM_DELUSIONS_MIN_CONTEXT_MESSAGES must be an integer."
        ) from exc
    if min_context_messages < 0:
        raise ValueError("LLM_DELUSIONS_MIN_CONTEXT_MESSAGES must be >= 0")
    if max_context_messages == 0 and min_context_messages > 0:
        raise ValueError(
            "LLM_DELUSIONS_MIN_CONTEXT_MESSAGES requires max_context_messages > 0."
        )
    if max_context_messages > 0 and min_context_messages > max_context_messages:
        raise ValueError(
            "LLM_DELUSIONS_MIN_CONTEXT_MESSAGES cannot exceed max_context_messages."
        )
    return min_context_messages


def _prepare_context_windows_parquet(windows_path: str) -> tuple[str, bool]:
    """Prepare context windows parquet using eval export filter semantics.

    Parameters
    ----------
    windows_path:
        Source windows parquet path for context mode.

    Returns
    -------
    tuple[str, bool]
        The parquet path to use and whether it should be deleted after use.
    """
    windows_df = pd.read_parquet(windows_path)
    if "meets_code" not in windows_df.columns:
        raise ValueError(
            "meets_code column is required before filtering context windows."
        )
    meets_code_df = windows_df[windows_df["meets_code"].eq(True)].copy()

    if "selected_for_eval" not in meets_code_df.columns:
        raise ValueError(
            "selected_for_eval column is required before filtering context windows. "
            "Run backfill_subset_review_meets_code to populate it."
        )
    selected_df = meets_code_df[meets_code_df["selected_for_eval"].eq(True)].copy()

    if len(selected_df) == len(windows_df):
        return windows_path, False

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".parquet", prefix="context_selected_windows_", delete=False
    ) as tmp_file:
        temp_windows_path = tmp_file.name
    selected_df.to_parquet(temp_windows_path)
    return temp_windows_path, True


def _resolve_eval_codes(codes: list[str] | str | None) -> list[str]:
    """Resolve eval code IDs from user-provided task argument.

    Parameters
    ----------
    codes:
        Code selector argument from the task configuration.

    Returns
    -------
    list[str]
        Normalized code identifiers to evaluate.
    """
    if isinstance(codes, str):
        return [code.strip() for code in codes.split(",")]
    if codes:
        return list(codes)
    return _WINDOW_ANNOTATION_IDS


def _to_chat_messages_with_metadata(litellm_messages: list[dict]) -> list[dict]:
    """Ensure litellm-style message dictionaries are well formed.

    Parameters
    ----------
    litellm_messages:
        Source transcript messages in litellm dict format.

    Returns
    -------
    list[dict]
        Inspect-compatible message dictionaries, preserving extra keys as
        ``metadata``.
    """
    result = []
    for msg in litellm_messages:
        role = msg["role"]
        content = msg["content"]
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"Unexpected role in annotation request: {role}")

        # The source transcripts can contain orphan `tool` turns that were
        # not produced by an actual tool call in the current conversation
        # state. OpenAI rejects those as invalid tool outputs, so we drop
        # them rather than forwarding broken tool-call metadata.
        if role == "tool" and "tool_call_id" not in msg:
            continue

        # Extract extra keys (e.g., bot-* scores) as metadata.
        metadata = {k: v for k, v in msg.items() if k not in ("role", "content")}
        msg_dict = {"role": role, "content": content}
        if metadata:
            msg_dict["metadata"] = metadata
        result.append(msg_dict)
    return result


def _map_harmful_annotations(metadata: dict) -> dict:
    """Apply ID rename mapping to sample metadata.

    Parameters
    ----------
    metadata:
        Sample metadata dictionary.

    Returns
    -------
    dict
        Metadata dictionary with normalized annotation IDs.
    """
    if "harmful_annotations" in metadata:
        metadata["harmful_annotations"] = {
            normalize_id(annotation) for annotation in metadata["harmful_annotations"]
        }
    return metadata


def _parse_classification_error_setting(value: bool | str) -> bool:
    """Parse task config for grader-classification error behavior.

    Parameters
    ----------
    value:
        Raw ``classification_error_as_sample_error`` task argument.

    Returns
    -------
    bool
        Parsed boolean setting.
    """
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(
            "classification_error_as_sample_error must be a boolean "
            "or one of: true/false, 1/0, yes/no, on/off."
        )
    return value


def _load_raw_samples_for_eval(
    *,
    max_windows: int,
    max_context_messages: int,
    source_codes: list[str],
    min_context_messages: int,
) -> list[dict]:
    """Load raw eval samples for window-only or context mode.

    Parameters
    ----------
    max_windows:
        Optional cap on windows per code (0 means no cap).
    max_context_messages:
        Requested max prepended context length (0 for window-only mode).
    source_codes:
        Source annotation IDs for data loading.
    min_context_messages:
        Effective-context minimum filter for context mode.

    Returns
    -------
    list[dict]
        Raw samples ready for conversion to Inspect ``Sample`` objects.
    """
    windows_path = resolve_path("LLM_DELUSIONS_WINDOWS_PATH", DEFAULT_WINDOWS_PATH)
    if max_context_messages == 0:
        return load_window_samples(
            base_path=windows_path,
            annotation_ids=source_codes,
            max_windows=max_windows or None,
            strip_metadata=False,
            require_selected_for_eval=False,
        )

    context_windows_path, should_cleanup_context_windows = (
        _prepare_context_windows_parquet(windows_path)
    )
    try:
        raw_samples = load_context_window_samples(
            windows_path=context_windows_path,
            transcripts_path=str(
                resolve_path("LLM_DELUSIONS_TRANSCRIPTS_PATH", DEFAULT_TRANSCRIPTS_PATH)
            ),
            codes=source_codes,
            max_context_messages=max_context_messages,
            max_windows=max_windows or None,
            strip_metadata=False,
            require_selected_for_eval=False,
        )
        if min_context_messages > 0:
            before_count = len(raw_samples)
            raw_samples = _filter_samples_by_min_context_length(
                raw_samples, min_context_messages
            )
            after_count = len(raw_samples)
            print(
                "Context filtering: kept "
                f"{after_count}/{before_count} samples with context_length >= "
                f"{min_context_messages}"
            )
        return raw_samples
    finally:
        if should_cleanup_context_windows and os.path.exists(context_windows_path):
            Path(context_windows_path).unlink()


@task
def delusions_eval(
    max_windows: int = 0,
    max_context_messages: int = 0,
    codes: list[str] | str | None = None,
    grader: str | None = None,
    classification_error_as_sample_error: bool | str = True,
) -> Task:
    """Unified delusions eval task.

    Parameters
    ----------
    max_windows:
        Maximum windows per annotation code (window-only mode).
        0 means no limit.
    max_context_messages:
        Number of preceding transcript messages to prepend to the window.
        If 0 (default), evaluates only the conversation window.
    codes:
        Comma-separated list (or Inspect list) of specific annotation IDs to run.
        If omitted, defaults to all bot-* codes.
    grader:
        Optional shortname for the grader model to override `--model-role grader=...`.
        Examples: 'gemini-3', 'gpt-5', 'mock'. Must be provided either via
        `-T grader=...` or `--model-role grader=...`.
    classification_error_as_sample_error:
        Whether grader parsing errors (``ClassificationError``) should be raised
        as sample-level errors. Accepts a bool or string values such as
        ``true``/``false``.

    Returns
    -------
    Task: The inspect_ai evaluation task.
    """
    min_context_messages = _resolve_min_context_messages(max_context_messages)
    eval_codes = _resolve_eval_codes(codes)

    # Map back to source IDs for external loading functions
    source_codes = [get_source_id(c) for c in eval_codes]
    raw_samples = _load_raw_samples_for_eval(
        max_windows=max_windows,
        max_context_messages=max_context_messages,
        source_codes=source_codes,
        min_context_messages=min_context_messages,
    )
    if not raw_samples:
        raise ValueError("No samples available after context filtering.")

    samples = [
        Sample(
            id=s["id"],
            input=_to_chat_messages_with_metadata(s["input"]),
            metadata=_map_harmful_annotations(s["metadata"]),
        )
        for s in raw_samples
    ]

    resolved_grader = _GRADER_ALIASES.get(grader, grader) if grader else None
    parsed_classification_error_as_sample_error = _parse_classification_error_setting(
        classification_error_as_sample_error
    )

    return Task(
        dataset=MemoryDataset(samples),
        scorer=[
            metadata_annotation_scorer(
                grader=resolved_grader,
                classification_error_as_sample_error=(
                    parsed_classification_error_as_sample_error
                ),
            )
        ],
    )


def run(
    model: str = "openai/gpt-5",
    grader_model: str | None = None,
) -> None:
    """
    Runs the delusions evaluation task.

    Parameters:
        model (str): The primary model to evaluate.
        grader_model (str | None): The grader model. Must be explicitly provided.
    """
    if not grader_model:
        raise ValueError(
            "Pass grader_model explicitly, e.g. "
            "'openai/gpt-5.1-2025-11-13' (with reasoning disabled)."
        )

    inspect_eval(
        delusions_eval(),
        model=model,
        model_roles={"grader": grader_model},
    )


if __name__ == "__main__":
    run()
