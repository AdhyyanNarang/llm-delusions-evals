"""Helpers for the interactive user-simulator spot-checking notebook.

The notebook is intentionally stateful: choose a real window prefix, generate
one synthetic message at a time, inspect it, and optionally fast-forward a few
turns. This file keeps the notebook compact while exposing the requested helper
functions in the notebook namespace.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
from IPython.display import HTML, display

DATA_PATH = Path("items-sanitized.parquet")
DEFAULT_OUTPUT_PATH = Path("sim_outputs/spotchecks.jsonl")

QWEN_MODEL_ID = "Qwen/Qwen3-8B"
ASSISTANT_MODEL_ID = "gpt-4o-mini"
OPENAI_API_KEY_ENV = "OPENAI_EXPT_API_KEY"

USE_4BIT = False
QWEN_GENERATION_PARAMS: dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_new_tokens": 256,
}
ASSISTANT_MAX_OUTPUT_TOKENS = 512

USER_SIMULATOR_SYSTEM_PROMPT = """You are simulating the next user message in a real conversation.

You will receive the conversation so far. Continue as the user only.
Output only the next user message text.
Do not include a role label, analysis, explanation, or multiple turns.
Preserve the local style, level of specificity, and conversational momentum.
"""

ASSISTANT_SYSTEM_PROMPT = """Continue as the assistant in this conversation.

Output only the next assistant message text.
Do not include a role label, analysis, or multiple turns.
"""

WINDOWS_DF: pd.DataFrame | None = None
QWEN_TOKENIZER = None
QWEN_MODEL = None
OPENAI_CLIENT = None


def utc_now_iso() -> str:
    """Return a UTC timestamp suitable for JSONL records."""

    return datetime.now(timezone.utc).isoformat()


@dataclass
class SpotcheckState:
    """Current transcript state for one spot-check."""

    eval_subset_id: str | None = None
    source_label: str | None = None
    k: int | None = None
    real_prefix: list[dict[str, Any]] = field(default_factory=list)
    next_original_message: dict[str, Any] | None = None
    generated_messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: utc_now_iso())
    updated_at: str = field(default_factory=lambda: utc_now_iso())


STATE = SpotcheckState()


def normalize_message(message: Any) -> dict[str, Any]:
    """Normalize one parquet message object into a plain dictionary."""

    if isinstance(message, dict):
        result = dict(message)
    else:
        try:
            result = dict(message)
        except Exception:
            result = {"role": "unknown", "content": repr(message)}

    result["role"] = str(result.get("role", "")).strip()
    result["content"] = "" if result.get("content") is None else str(result["content"])
    return result


def compact_message(message: dict[str, Any]) -> dict[str, Any]:
    """Keep role/content plus non-null score metadata for saved records."""

    compact = {
        "role": message.get("role"),
        "content": message.get("content", ""),
    }
    for key, value in message.items():
        if key in compact or value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        compact[key] = value
    return compact


def load_windows(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the selected window parquet and normalize nested messages."""

    global WINDOWS_DF

    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet file: {parquet_path.resolve()}")

    df = pd.read_parquet(parquet_path).copy()
    required = {"eval_subset_id", "label", "messages"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    df["eval_subset_id"] = df["eval_subset_id"].astype(str)
    df["messages"] = df["messages"].map(lambda messages: [normalize_message(m) for m in list(messages)])
    df["message_count"] = df["messages"].map(len)
    df["first_role"] = df["messages"].map(lambda messages: messages[0].get("role") if messages else None)
    WINDOWS_DF = df
    return df


def _windows(windows_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return provided windows or the module-level loaded dataframe."""

    global WINDOWS_DF
    if windows_df is not None:
        return windows_df
    if WINDOWS_DF is None:
        WINDOWS_DF = load_windows()
    return WINDOWS_DF


def message_preview(message: dict[str, Any] | None, limit: int = 180) -> str:
    """Return a one-line preview for a message."""

    if not message:
        return ""
    text = re.sub(r"\s+", " ", str(message.get("content", ""))).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def eligible_prefixes(k: int, windows_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return windows where ``messages[:k]`` is followed by a real user turn."""

    if k < 0:
        raise ValueError("k must be >= 0")

    df = _windows(windows_df).copy()

    def is_eligible(messages: list[dict[str, Any]]) -> bool:
        return len(messages) > k and messages[k].get("role") == "user"

    out = df[df["messages"].map(is_eligible)].copy()
    out["k"] = k
    out["next_original_role"] = out["messages"].map(lambda messages: messages[k].get("role"))
    out["next_original_preview"] = out["messages"].map(lambda messages: message_preview(messages[k]))
    return out[
        [
            "eval_subset_id",
            "label",
            "meets_code",
            "message_count",
            "k",
            "next_original_role",
            "next_original_preview",
        ]
    ].reset_index(drop=True)


def _get_window(eval_subset_id: str, windows_df: pd.DataFrame | None = None) -> pd.Series:
    """Look up one source window by eval subset id."""

    df = _windows(windows_df)
    matches = df[df["eval_subset_id"].astype(str).eq(str(eval_subset_id))]
    if matches.empty:
        raise KeyError(f"Unknown eval_subset_id: {eval_subset_id}")
    return matches.iloc[0]


def reset_state(eval_subset_id: str, k: int) -> SpotcheckState:
    """Reset global state to the first ``k`` real messages of one window."""

    row = _get_window(eval_subset_id)
    messages = list(row["messages"])
    if k < 0:
        raise ValueError("k must be >= 0")
    if len(messages) <= k:
        raise ValueError(
            f"Window {eval_subset_id} has only {len(messages)} messages; k={k} is too large."
        )
    if messages[k].get("role") != "user":
        raise ValueError(
            "This spot-check requires messages[k] to be a user turn. "
            f"For k={k}, found role={messages[k].get('role')!r}."
        )

    now = utc_now_iso()
    STATE.eval_subset_id = str(row["eval_subset_id"])
    STATE.source_label = str(row["label"])
    STATE.k = int(k)
    STATE.real_prefix = [compact_message(m) for m in messages[:k]]
    STATE.next_original_message = compact_message(messages[k])
    STATE.generated_messages = []
    STATE.created_at = now
    STATE.updated_at = now
    return STATE


def current_transcript() -> list[dict[str, Any]]:
    """Return the current real-prefix plus synthetic continuation."""

    return [
        {"role": m["role"], "content": m.get("content", "")}
        for m in STATE.real_prefix + STATE.generated_messages
    ]


def expected_next_role() -> str:
    """Return the role that should be generated next."""

    if not STATE.eval_subset_id:
        raise RuntimeError("Call reset_state(eval_subset_id, k) first.")
    if not STATE.generated_messages:
        return "user"
    last_role = STATE.generated_messages[-1]["role"]
    return "assistant" if last_role == "user" else "user"


def check_gpu() -> dict[str, Any]:
    """Return CUDA/GPU status without requiring model loading."""

    try:
        import torch
    except ImportError:
        return {"torch_available": False, "cuda_available": False}

    status: dict[str, Any] = {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        status.update(
            {
                "current_device": device_index,
                "device_name": torch.cuda.get_device_name(device_index),
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
            }
        )
    return status


def load_qwen_model(
    model_id: str = QWEN_MODEL_ID,
    *,
    use_4bit: bool = USE_4BIT,
    force_reload: bool = False,
) -> tuple[Any, Any]:
    """Load Qwen3-8B for user-turn generation."""

    global QWEN_TOKENIZER, QWEN_MODEL

    if QWEN_TOKENIZER is not None and QWEN_MODEL is not None and not force_reload:
        return QWEN_TOKENIZER, QWEN_MODEL

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_kwargs: dict[str, Any] = {"device_map": "auto"}

    if use_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    QWEN_TOKENIZER = tokenizer
    QWEN_MODEL = model
    return tokenizer, model


def _transcript_for_prompt(messages: list[dict[str, Any]]) -> str:
    """Format transcript messages for the simulator instruction."""

    parts = []
    for index, message in enumerate(messages):
        role = str(message.get("role", "unknown")).upper()
        content = str(message.get("content", "")).strip()
        parts.append(f"{index:02d} {role}:\n{content}")
    return "\n\n".join(parts)


def _apply_qwen_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Apply Qwen chat template with non-thinking mode when supported."""

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def clean_generated_text(text: str, *, target_role: str) -> str:
    """Remove role labels, thinking blocks, and accidental extra turns."""

    cleaned = str(text or "")
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()
    cleaned = re.sub(r"^```(?:text|markdown)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(
        rf"^\s*(?:{target_role}|user|assistant)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    boundary_patterns = [
        r"\n\s*User\s*:",
        r"\n\s*Assistant\s*:",
        r"\n\s*System\s*:",
        r"\n\s*###",
    ]
    for pattern in boundary_patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            cleaned = cleaned[: match.start()].strip()
    return cleaned


def generate_qwen_user(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Generate the next synthetic user message from the current transcript."""

    if QWEN_TOKENIZER is None or QWEN_MODEL is None:
        raise RuntimeError("Call load_qwen_model() before generating user messages.")

    import torch

    prompt_messages = [
        {"role": "system", "content": USER_SIMULATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Conversation so far:\n\n"
                f"{_transcript_for_prompt(messages)}\n\n"
                "Write the next USER message only."
            ),
        },
    ]
    prompt = _apply_qwen_chat_template(QWEN_TOKENIZER, prompt_messages)
    inputs = QWEN_TOKENIZER(prompt, return_tensors="pt")
    device = getattr(QWEN_MODEL, "device", None)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    gen_kwargs = dict(QWEN_GENERATION_PARAMS)
    gen_kwargs["do_sample"] = True
    gen_kwargs["pad_token_id"] = QWEN_TOKENIZER.eos_token_id

    with torch.no_grad():
        output_ids = QWEN_MODEL.generate(**inputs, **gen_kwargs)

    input_length = inputs["input_ids"].shape[-1]
    raw = QWEN_TOKENIZER.decode(output_ids[0][input_length:], skip_special_tokens=True)
    cleaned = clean_generated_text(raw, target_role="user")
    return {"raw_output": raw, "content": cleaned}


def get_openai_client() -> Any:
    """Return an OpenAI client using OPENAI_EXPT_API_KEY."""

    global OPENAI_CLIENT
    if OPENAI_CLIENT is not None:
        return OPENAI_CLIENT

    api_key = os.environ.get(OPENAI_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set {OPENAI_API_KEY_ENV} before generating assistant turns.")

    from openai import OpenAI

    OPENAI_CLIENT = OpenAI(api_key=api_key)
    return OPENAI_CLIENT


def generate_openai_assistant(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Generate the next assistant message using gpt-4o-mini."""

    client = get_openai_client()
    input_messages = [
        {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
        for m in messages
        if str(m.get("role", "")) in {"user", "assistant", "system"}
    ]
    response = client.responses.create(
        model=ASSISTANT_MODEL_ID,
        instructions=ASSISTANT_SYSTEM_PROMPT,
        input=input_messages,
        max_output_tokens=ASSISTANT_MAX_OUTPUT_TOKENS,
    )
    raw = getattr(response, "output_text", "") or ""
    cleaned = clean_generated_text(raw, target_role="assistant")
    return {"raw_output": raw, "content": cleaned}


def generate_next() -> dict[str, Any]:
    """Generate and append exactly one synthetic message."""

    role = expected_next_role()
    transcript = current_transcript()

    if role == "user":
        generated = generate_qwen_user(transcript)
        model_id = QWEN_MODEL_ID
    elif role == "assistant":
        generated = generate_openai_assistant(transcript)
        model_id = ASSISTANT_MODEL_ID
    else:
        raise RuntimeError(f"Unsupported next role: {role}")

    record = {
        "role": role,
        "content": generated["content"],
        "raw_output": generated["raw_output"],
        "cleaned_output": generated["content"],
        "model": model_id,
        "source": "synthetic",
        "generated_at": utc_now_iso(),
    }
    STATE.generated_messages.append(record)
    STATE.updated_at = utc_now_iso()
    return record


def fast_forward(n: int) -> list[dict[str, Any]]:
    """Generate and append ``n`` messages using the same one-step path."""

    if n < 0:
        raise ValueError("n must be >= 0")
    return [generate_next() for _ in range(int(n))]


def state_to_record() -> dict[str, Any]:
    """Convert current state to a JSON-serializable rollout record."""

    if not STATE.eval_subset_id:
        raise RuntimeError("No active state. Call reset_state(...) first.")

    return {
        "timestamp": utc_now_iso(),
        "eval_subset_id": STATE.eval_subset_id,
        "source_label": STATE.source_label,
        "k": STATE.k,
        "models": {
            "user": QWEN_MODEL_ID,
            "assistant": ASSISTANT_MODEL_ID,
        },
        "generation_params": {
            "qwen": QWEN_GENERATION_PARAMS,
            "assistant_max_output_tokens": ASSISTANT_MAX_OUTPUT_TOKENS,
            "qwen_use_4bit_default": USE_4BIT,
        },
        "real_prefix": STATE.real_prefix,
        "next_original_message": STATE.next_original_message,
        "generated_messages": STATE.generated_messages,
        "created_at": STATE.created_at,
        "updated_at": STATE.updated_at,
    }


def save_state_jsonl(path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Append the current rollout state to JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(state_to_record(), ensure_ascii=False) + "\n")
    return output_path


def load_saved_rollouts(path: str | Path = DEFAULT_OUTPUT_PATH) -> list[dict[str, Any]]:
    """Load saved rollout records from JSONL."""

    input_path = Path(path)
    if not input_path.exists():
        return []
    records = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def render_messages(
    real_prefix: list[dict[str, Any]],
    generated_messages: list[dict[str, Any]],
    *,
    title: str = "Spot-check transcript",
    next_original_message: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Render real and synthetic messages as HTML."""

    meta = metadata or {
        "eval_subset_id": STATE.eval_subset_id,
        "source_label": STATE.source_label,
        "k": STATE.k,
    }
    parts = [
        "<style>",
        ".sim-wrap{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.45}",
        ".sim-meta{padding:10px 12px;border:1px solid #d1d5db;background:#f9fafb;margin:8px 0 12px 0;font-size:13px}",
        ".sim-msg{border:1px solid #d1d5db;margin:10px 0;padding:10px 12px;border-radius:6px}",
        ".sim-role{font-weight:700;text-transform:uppercase;letter-spacing:.04em;font-size:12px;margin-bottom:6px}",
        ".sim-content{white-space:pre-wrap;font-size:14px;color:#111827}",
        ".real{background:#f8fafc;border-left:5px solid #64748b}",
        ".synthetic.user{background:#f2f7ff;border-left:5px solid #2456a6}",
        ".synthetic.assistant{background:#fff8e8;border-left:5px solid #8a5a00}",
        "</style>",
        "<div class='sim-wrap'>",
        f"<div class='sim-meta'><b>{escape(title)}</b><br>",
        f"<b>eval_subset_id:</b> {escape(str(meta.get('eval_subset_id')))} &nbsp; ",
        f"<b>label:</b> {escape(str(meta.get('source_label')))} &nbsp; ",
        f"<b>k:</b> {escape(str(meta.get('k')))} &nbsp; ",
        f"<b>generated:</b> {len(generated_messages)}",
        "</div>",
    ]

    if next_original_message:
        preview = message_preview(next_original_message, limit=260)
        parts.append(
            "<div class='sim-meta'>"
            "<b>Next original user message preview:</b><br>"
            f"{escape(preview)}"
            "</div>"
        )

    for idx, message in enumerate(real_prefix):
        role = str(message.get("role", "unknown"))
        content = str(message.get("content", ""))
        parts.append(
            "<div class='sim-msg real'>"
            f"<div class='sim-role'>real {escape(role)} turn {idx}</div>"
            f"<div class='sim-content'>{escape(content)}</div>"
            "</div>"
        )

    for idx, message in enumerate(generated_messages):
        role = str(message.get("role", "unknown"))
        content = str(message.get("content", ""))
        parts.append(
            f"<div class='sim-msg synthetic {escape(role)}'>"
            f"<div class='sim-role'>synthetic {escape(role)} gen {idx}</div>"
            f"<div class='sim-content'>{escape(content)}</div>"
            "</div>"
        )

    parts.append("</div>")
    display(HTML("\n".join(parts)))


def render_state() -> None:
    """Render the current global spot-check state."""

    render_messages(
        STATE.real_prefix,
        STATE.generated_messages,
        next_original_message=STATE.next_original_message,
    )


def _dropdown_options_for_k(k: int, windows_df: pd.DataFrame) -> list[tuple[str, str]]:
    """Build dropdown labels for eligible windows."""

    options = []
    for row in eligible_prefixes(k, windows_df).itertuples(index=False):
        preview = str(row.next_original_preview).replace("\n", " ")
        label = f"{row.eval_subset_id} | {row.label} | next: {preview[:80]}"
        options.append((label, row.eval_subset_id))
    return options


def build_spotcheck_ui(windows_df: pd.DataFrame | None = None) -> Any:
    """Build the ipywidgets UI for one-message-at-a-time spot-checking."""

    import ipywidgets as widgets

    df = _windows(windows_df)
    max_k = max(int(df["message_count"].max()) - 1, 0)

    k_widget = widgets.BoundedIntText(value=2, min=0, max=max_k, description="k")
    window_widget = widgets.Dropdown(description="window", options=[])
    use_4bit_widget = widgets.Checkbox(value=USE_4BIT, description="load Qwen 4-bit")
    fast_n_widget = widgets.BoundedIntText(value=3, min=0, max=50, description="N")

    refresh_button = widgets.Button(description="Refresh windows")
    reset_button = widgets.Button(description="Reset")
    load_qwen_button = widgets.Button(description="Load Qwen")
    generate_button = widgets.Button(description="Generate next")
    fast_button = widgets.Button(description="Fast forward N")
    save_button = widgets.Button(description="Save JSONL")

    status_out = widgets.Output()
    transcript_out = widgets.Output()

    def set_status(message: str) -> None:
        with status_out:
            status_out.clear_output(wait=True)
            print(message)

    def refresh_windows(*_args: Any) -> None:
        options = _dropdown_options_for_k(int(k_widget.value), df)
        window_widget.options = options
        if options:
            window_widget.value = options[0][1]
            set_status(f"{len(options)} eligible windows for k={k_widget.value}.")
        else:
            set_status(f"No eligible windows for k={k_widget.value}.")

    def redraw() -> None:
        with transcript_out:
            transcript_out.clear_output(wait=True)
            if STATE.eval_subset_id:
                render_state()

    def do_reset(*_args: Any) -> None:
        if not window_widget.value:
            set_status("No eligible window selected.")
            return
        try:
            reset_state(str(window_widget.value), int(k_widget.value))
            set_status("State reset. First generated turn will be user.")
            redraw()
        except Exception as exc:
            set_status(f"Reset failed: {exc}")

    def do_load_qwen(*_args: Any) -> None:
        try:
            load_qwen_model(use_4bit=bool(use_4bit_widget.value))
            set_status(f"Loaded {QWEN_MODEL_ID}.")
        except Exception as exc:
            set_status(f"Qwen load failed: {exc}")

    def do_generate(*_args: Any) -> None:
        try:
            record = generate_next()
            set_status(f"Generated {record['role']} message with {record['model']}.")
            redraw()
        except Exception as exc:
            set_status(f"Generation failed: {exc}")

    def do_fast_forward(*_args: Any) -> None:
        try:
            records = fast_forward(int(fast_n_widget.value))
            roles = " -> ".join(record["role"] for record in records)
            set_status(f"Fast-forwarded {len(records)} message(s): {roles}")
            redraw()
        except Exception as exc:
            set_status(f"Fast forward failed: {exc}")

    def do_save(*_args: Any) -> None:
        try:
            output_path = save_state_jsonl()
            set_status(f"Saved rollout to {output_path}.")
        except Exception as exc:
            set_status(f"Save failed: {exc}")

    refresh_button.on_click(refresh_windows)
    reset_button.on_click(do_reset)
    load_qwen_button.on_click(do_load_qwen)
    generate_button.on_click(do_generate)
    fast_button.on_click(do_fast_forward)
    save_button.on_click(do_save)
    k_widget.observe(lambda _change: refresh_windows(), names="value")

    controls = widgets.VBox(
        [
            widgets.HBox([k_widget, window_widget, refresh_button]),
            widgets.HBox([reset_button, load_qwen_button, use_4bit_widget]),
            widgets.HBox([generate_button, fast_n_widget, fast_button, save_button]),
            status_out,
            transcript_out,
        ]
    )

    display(controls)
    refresh_windows()
    do_reset()
    return controls
