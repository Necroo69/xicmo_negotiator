"""Format validated conversations into training-ready `messages` JSONL.

Produces BOTH framings from the same clean data so you can compare them:

  single-turn : one example per negotiator turn. The user message carries the
                budget + prior deal_state summary + the creator's latest line.
                Cleaner signal for JSON/contract discipline.
  multi-turn  : one example per conversation. The full DM thread as alternating
                user/assistant messages — a faithful reconstruction of what the
                teacher negotiator actually saw during generation (system prompt
                holds the budget; the running state lives in its own prior JSON).
                Train with loss on the assistant turns only.

Every example is in portable `messages` format:
    {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}

The split is 90/10 BY CONVERSATION (never by turn), stratified per persona, so
turns from one conversation never straddle train/val and both splits stay
balanced across personas. Deterministic: sorted by conversation_id.

Reuses validate.py's filters, so rejected/duplicate conversations never leak in.

Usage:
    python generator/format_for_training.py
    python generator/format_for_training.py --val-frac 0.1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

# --- make imports work whether run as a script or as a module -----------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_PROJECT_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from negotiation_contract import build_system_prompt, parse_model_reply, ContractError
from validate import (
    RAW_DIR,
    CLEAN_DIR,
    BRANDS_BY_NAME,
    rejection_reason,
    conversation_to_examples,
)

DEFAULT_VAL_FRAC = 0.10


# ---------------------------------------------------------------------------
# Loading + filtering (mirrors validate.py's accept logic)
# ---------------------------------------------------------------------------

def load_passing_conversations() -> list[dict]:
    """Return every conversation that passes validate.py's checks + opening dedup."""
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "conv_*.json")))
    passing: list[dict] = []
    seen_openings: set[str] = set()

    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                conv = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if rejection_reason(conv) is not None:
            continue

        opening = (conv.get("turns") or [{}])[0].get("dm_text")
        if opening:
            key = " ".join(opening.split())
            if key in seen_openings:
                continue
            seen_openings.add(key)

        passing.append(conv)

    return passing


# ---------------------------------------------------------------------------
# Example construction
# ---------------------------------------------------------------------------

def _system_prompt_for(conv: dict) -> str:
    """Rebuild the exact system prompt the negotiator saw for this conversation."""
    budget = conv["budget"]
    meta = conv.get("meta", {})
    brand_cfg = BRANDS_BY_NAME.get(meta.get("brand"))
    return build_system_prompt(
        budget_max=budget,
        payout_method=meta.get("payout_method") or "PayPal",
        brand_facts=brand_cfg["brand_facts"] if brand_cfg else "",
        brand=brand_cfg["brand"] if brand_cfg else "the brand",
        brand_website=brand_cfg["brand_website"] if brand_cfg else "",
    )


def single_turn_examples(conv: dict) -> list[dict]:
    """One {messages:[system,user,assistant]} per turn (via validate's flattener)."""
    out = []
    for ex in conversation_to_examples(conv):
        out.append({"messages": [
            {"role": "system", "content": ex["system"]},
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["assistant"]},
        ]})
    return out


def multi_turn_example(conv: dict) -> dict | None:
    """One {messages:[system, (user,assistant)*]} per conversation.

    Reconstructs the teacher's actual view: system holds the budget, each user
    turn is the creator's raw message, each assistant turn is the canonical JSON.
    """
    messages = [{"role": "system", "content": _system_prompt_for(conv)}]
    for t in conv.get("turns", []):
        if t.get("parse_error"):
            continue
        try:
            assistant = parse_model_reply(t["negotiator_raw"]).to_json()
        except ContractError:
            continue
        messages.append({"role": "user", "content": t["creator_message"]})
        messages.append({"role": "assistant", "content": assistant})

    # Need at least one full user/assistant exchange to be usable.
    if len(messages) < 3:
        return None
    return {"messages": messages}


# ---------------------------------------------------------------------------
# Split (by conversation, stratified per persona)
# ---------------------------------------------------------------------------

def split_by_persona(convs: list[dict], val_frac: float) -> tuple[list[dict], list[dict]]:
    """Deterministic 90/10-ish split, stratified per persona, split by conversation."""
    by_persona: dict[str, list[dict]] = defaultdict(list)
    for c in convs:
        by_persona[c.get("persona_name", "unknown")].append(c)

    train: list[dict] = []
    val: list[dict] = []
    for persona, group in by_persona.items():
        group = sorted(group, key=lambda c: c["conversation_id"])
        n_val = round(len(group) * val_frac)
        # take val from the front deterministically; keep >=1 in train
        n_val = min(n_val, len(group) - 1) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _persona_counts(convs: list[dict]) -> str:
    c = Counter(x.get("persona_name", "unknown") for x in convs)
    return ", ".join(f"{k}={v}" for k, v in sorted(c.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Format clean conversations into messages JSONL.")
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC,
                        help="fraction of conversations held out for validation (default 0.10)")
    args = parser.parse_args()

    os.makedirs(CLEAN_DIR, exist_ok=True)
    convs = load_passing_conversations()
    if not convs:
        raise SystemExit(f"No passing conversations found in {RAW_DIR}. Run selfplay first.")

    train_convs, val_convs = split_by_persona(convs, args.val_frac)

    # Build all four files.
    outputs = {
        "train_singleturn.jsonl": [ex for c in train_convs for ex in single_turn_examples(c)],
        "val_singleturn.jsonl":   [ex for c in val_convs for ex in single_turn_examples(c)],
        "train_multiturn.jsonl":  [ex for c in train_convs if (ex := multi_turn_example(c))],
        "val_multiturn.jsonl":    [ex for c in val_convs if (ex := multi_turn_example(c))],
    }
    for name, rows in outputs.items():
        _write_jsonl(os.path.join(CLEAN_DIR, name), rows)

    # --- summary ---
    print("=== Formatting summary ===")
    print(f"Passing conversations : {len(convs)}")
    print(f"  train conversations : {len(train_convs)}  ({_persona_counts(train_convs)})")
    print(f"  val   conversations : {len(val_convs)}  ({_persona_counts(val_convs)})")
    print()
    print("Written to", CLEAN_DIR)
    for name, rows in outputs.items():
        print(f"  {name:26s} {len(rows):5d} examples")


if __name__ == "__main__":
    main()
