"""Validate / filter self-play transcripts into training-ready data.

Reads every conversation from data/raw/, applies business-rule checks, and
writes the turns of every passing conversation to data/clean/train.jsonl — one
JSON line per negotiator turn, in {system, user, assistant} form.

Reject rules (a conversation is dropped if ANY apply):
  - any turn's price exceeds the HARD ceiling (budget * 1.10)
  - a turn is marked terms_locked but the 3 close fields (price + payout + email)
    are not all filled
  - a turn is terms_locked AND escalate_to_human at the same time (contradiction)
  - any turn has an empty dm_text
  - fewer than 2 turns total (too short to be useful)
  - the conversation ended in a ContractError (parse failed mid-conversation)

Usage:
    python generator/validate.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

# --- make imports work whether run as a script or as a module -----------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_PROJECT_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from negotiation_contract import (
    build_system_prompt,
    parse_model_reply,
    ContractError,
    DealState,
)
from personas import PERSONAS
from selfplay import BRANDS

RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
CLEAN_DIR = os.path.join(_PROJECT_ROOT, "data", "clean")
CLEAN_FILE = os.path.join(CLEAN_DIR, "train.jsonl")

# The hard ceiling is 110% of the target budget — the negotiator is allowed to
# close anywhere at or under this. Mirrors negotiation_contract's ceiling.
CEILING_MULTIPLIER = 1.10

# Look up a brand's facts/website by display name so we can rebuild the exact
# system prompt the negotiator saw during generation.
BRANDS_BY_NAME = {b["brand"]: b for b in BRANDS}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _required_fields_filled(ds: dict) -> bool:
    """Close gate per the contract: price + payout_method + email (NOT deliverables)."""
    return DealState(
        price=ds.get("price"),
        payout_method=ds.get("payout_method"),
        email=ds.get("email"),
    ).required_fields_filled()


def rejection_reason(conv: dict) -> str | None:
    """Return the first failing check's name, or None if the conversation passes."""
    turns = conv.get("turns", [])
    budget = conv.get("budget")
    ceiling = budget * CEILING_MULTIPLIER if budget is not None else None

    # 1. price above the HARD ceiling (budget * 1.10) on any turn
    for t in turns:
        ds = t.get("deal_state")
        price = ds.get("price") if ds else None
        if price is not None and ceiling is not None and price > ceiling:
            return "price_over_budget"

    # 2. terms_locked but the 3 close fields aren't all filled
    for t in turns:
        ds = t.get("deal_state")
        if ds and ds.get("status") == "terms_locked" and not _required_fields_filled(ds):
            return "terms_locked_incomplete"

    # 2b. terms_locked AND escalate_to_human on the same turn is a contradiction
    for t in turns:
        ds = t.get("deal_state")
        if ds and ds.get("status") == "terms_locked" and ds.get("escalate_to_human"):
            return "escalate_on_lock"

    # 3. empty dm_text on any turn
    for t in turns:
        dm = t.get("dm_text")
        if not dm or not str(dm).strip():
            return "empty_dm_text"

    # 4. too short
    if conv.get("total_turns", len(turns)) < 2:
        return "too_short"

    # 5. ended in a parse failure
    if conv.get("final_status") == "contract_error":
        return "contract_error"

    return None


# ---------------------------------------------------------------------------
# Training-example construction
# ---------------------------------------------------------------------------

def conversation_to_examples(conv: dict) -> list[dict]:
    """Turn a passing conversation into {system, user, assistant} training lines."""
    budget = conv["budget"]
    meta = conv.get("meta", {})
    brand_cfg = BRANDS_BY_NAME.get(meta.get("brand"))
    # Rebuild the exact system prompt the negotiator was given for this run.
    system = build_system_prompt(
        budget_max=budget,
        payout_method=meta.get("payout_method") or "PayPal",
        brand_facts=brand_cfg["brand_facts"] if brand_cfg else "",
        brand=brand_cfg["brand"] if brand_cfg else "the brand",
        brand_website=brand_cfg["brand_website"] if brand_cfg else "",
    )
    examples: list[dict] = []

    for t in conv.get("turns", []):
        if t.get("parse_error"):
            continue  # shouldn't happen in a passing conversation, but be safe

        # Normalize the assistant target to canonical JSON via the contract.
        try:
            assistant = parse_model_reply(t["negotiator_raw"]).to_json()
        except ContractError:
            continue

        prior = t.get("prior_deal_state", {})
        user = (
            f"Budget ceiling: {budget}\n"
            f"Current deal state: {json.dumps(prior, ensure_ascii=False)}\n"
            f"Creator: {t['creator_message']}"
        )
        examples.append({"system": system, "user": user, "assistant": assistant})

    return examples


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(CLEAN_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "conv_*.json")))

    total = len(paths)
    passed = 0
    rejected = 0
    reasons: Counter[str] = Counter()
    example_count = 0
    persona_coverage: Counter[str] = Counter()
    persona_brands: dict[str, Counter] = {}  # persona -> Counter of brand names
    seen_openings: set[str] = set()  # turn-1 negotiator dm_text, for dedup

    with open(CLEAN_FILE, "w", encoding="utf-8") as out:
        for path in paths:
            try:
                with open(path, encoding="utf-8") as f:
                    conv = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                rejected += 1
                reasons["unreadable_file"] += 1
                print(f"  skip {os.path.basename(path)}: {e}")
                continue

            reason = rejection_reason(conv)
            if reason is not None:
                rejected += 1
                reasons[reason] += 1
                continue

            # Dedup: drop a conversation whose opening DM matches one we've kept.
            opening = (conv.get("turns") or [{}])[0].get("dm_text")
            if opening:
                key = " ".join(opening.split())  # normalize whitespace
                if key in seen_openings:
                    rejected += 1
                    reasons["duplicate_opening"] += 1
                    continue
                seen_openings.add(key)

            examples = conversation_to_examples(conv)
            if not examples:
                rejected += 1
                reasons["no_usable_turns"] += 1
                continue

            for ex in examples:
                out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            example_count += len(examples)
            passed += 1
            persona_name = conv.get("persona_name", "unknown")
            persona_coverage[persona_name] += 1
            brand_name = conv.get("meta", {}).get("brand", "unknown")
            persona_brands.setdefault(persona_name, Counter())[brand_name] += 1

    # --- summary ---
    print("\n=== Validation summary ===")
    print(f"Total raw conversations : {total}")
    print(f"Passed                  : {passed}")
    print(f"Rejected                : {rejected}")
    if reasons:
        print("Rejection reasons:")
        for reason, n in reasons.most_common():
            print(f"  {reason:24s}: {n}")

    # --- per-persona coverage over the passing (clean) conversations ---
    print("\nPersona coverage:")
    ordered = sorted(
        (p.name for p in PERSONAS),
        key=lambda name: persona_coverage.get(name, 0),
        reverse=True,
    )
    for name in ordered:
        n = persona_coverage.get(name, 0)
        pct = (n / passed * 100) if passed else 0.0
        label = name.replace("_", "-")
        print(f"  {label:16s} → {n:3d} conversations ({pct:.0f}%)")
        brands = persona_brands.get(name)
        if brands:
            brand_str = ", ".join(f"{b}×{c}" for b, c in brands.most_common())
            print(f"  {'':16s}   brands: {brand_str}")

    print(f"\nWrote {example_count} training examples to {CLEAN_FILE}")


if __name__ == "__main__":
    main()
