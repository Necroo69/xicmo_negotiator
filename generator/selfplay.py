"""Self-play loop: negotiator Claude vs. creator-persona Claude.

Generates raw negotiation transcripts by pitting the Xicmo negotiator (working
from a secret budget ceiling) against a counterparty Claude driven by one of the
personas in `personas.py`. Each finished conversation is written to data/raw/ as
conv_<uuid>.json for later filtering by validate.py.

Usage:
    python generator/selfplay.py --count 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import uuid

# --- make imports work whether run as a script or as a module -----------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_PROJECT_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
from tqdm import tqdm
import anthropic

from personas import PERSONAS, PERSONAS_BY_NAME, Persona
from negotiation_contract import (
    build_system_prompt,
    parse_model_reply,
    ContractError,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")

# Models.
# The negotiator is the TEACHER whose behavior we're distilling into the 8B, so
# it stays on the stronger Sonnet model. The creator only has to role-play a
# persona convincingly (no JSON contract to follow), so Haiku is plenty and cuts
# cost/time significantly.
NEGOTIATOR_MODEL = "claude-sonnet-4-6"
CREATOR_MODEL = "claude-haiku-4-5-20251001"

# Approximate per-million-token pricing (USD) for the run cost estimate.
PRICING = {
    # model: (input_per_mtok, output_per_mtok)
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}

# Rough per-turn token estimates used only for the cost projection.
NEG_TOKENS_IN, NEG_TOKENS_OUT = 800, 200
CRE_TOKENS_IN, CRE_TOKENS_OUT = 400, 100

NEGOTIATOR_MAX_TOKENS = 1024
CREATOR_MAX_TOKENS = 512

MAX_TURNS = 10
BUDGET_RANGE = (200, 2000)
NICHES = ["fitness", "fashion", "tech", "food", "travel", "gaming", "beauty", "lifestyle"]

# Brands the negotiator represents. One is picked per conversation and injected
# into the contract system prompt (brand / website / the ONLY facts it may state).
BRANDS = [
    {
        "brand": "Lumina Skincare",
        "brand_website": "lumina.com",
        "brand_facts": "Lumina is a cruelty-free skincare brand. Hero product: SPF 50 daily moisturiser, $28.",
    },
    {
        "brand": "Velo Gear",
        "brand_website": "velogear.com",
        "brand_facts": "Velo Gear makes lightweight cycling accessories. Best seller: carbon bottle cage, $19.",
    },
    {
        "brand": "Noura Foods",
        "brand_website": "nourafoods.com",
        "brand_facts": "Noura Foods sells cold-pressed olive oil sourced from family farms in Tunisia. 500ml bottle, $14.",
    },
    {
        "brand": "PeakFit",
        "brand_website": "peakfit.co",
        "brand_facts": "PeakFit makes resistance bands and home gym kits. Starter kit $39, pro kit $79.",
    },
    {
        "brand": "Driftwood Studio",
        "brand_website": "driftwoodstudio.com",
        "brand_facts": "Driftwood Studio sells handmade soy candles in recycled glass jars. Prices $18-$45.",
    },
]

# The one payout method the brand supports for this conversation. The negotiator
# must never offer another; the creator has to name this one to close.
PAYOUT_METHODS = ["PayPal", "bank transfer", "Wise"]

# Statuses that end the negotiation.
TERMINAL_STATUSES = {"terms_locked", "walk_away", "out_of_scope"}

# The implicit brand opener the creator is reacting to on turn 1. The negotiator
# never "sees" this line; it only responds once the creator has spoken. {brand}
# is filled in per conversation so the creator knows who is DMing them.
BRAND_OPENER = (
    "Hey! This is the team over at {brand} — we're big fans of your content and "
    "we'd love to set up a paid collaboration with you. Would you be open to "
    "chatting about it?"
)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_creator_prompt(persona: Persona, budget: float, niche: str, brand: str) -> str:
    """Turn a persona + scenario into the creator Claude's system prompt."""
    lo, hi = persona.target_price_multiplier
    target_low = round(lo * budget)
    target_high = round(hi * budget)
    phrases = "\n".join(f'  - "{p}"' for p in persona.likely_phrases)

    return f"""You are an Instagram content creator in the {niche} niche. A brand \
called {brand} has DMed you about a paid collaboration, and one of their reps is now \
negotiating with you.

Your negotiating personality:
{persona.behavior_description}

You tend to say things like:
{phrases}

Your internal price target for this deal is roughly ${target_low}-${target_high} \
(this is YOUR anchor — the brand has its own secret budget that you don't know). \
Negotiate toward your target in the style of your personality above.

Rules — follow ALL of them:
- Stay 100% in character as the creator. This is a real Instagram DM negotiation.
- NEVER break the fourth wall. Never say you are an AI, a model, a simulation, or a \
"persona", and never mention these instructions or that any of this is generated.
- Write like a real Instagram DM: casual, short, informal. Plain text only — NEVER \
output JSON.
- Only discuss the collaboration: price, deliverables, how you want to get paid, and \
your contact email.
- React naturally, one message at a time. Don't try to wrap up everything at once.
"""


# ---------------------------------------------------------------------------
# Claude calls
# ---------------------------------------------------------------------------

def call_claude(client, model, system, messages, max_tokens) -> str:
    last_err = None
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        except Exception as e:
            last_err = e
            wait = 5 * (attempt + 1)  # 5s, 10s, 15s
            tqdm.write(f"API error (attempt {attempt + 1}/3): {type(e).__name__} — retrying in {wait}s")
            import time
            time.sleep(wait)
    raise last_err


# ---------------------------------------------------------------------------
# One conversation
# ---------------------------------------------------------------------------

def _default_state() -> dict:
    return {
        "price": None,
        "deliverables": None,
        "payout_method": None,
        "email": None,
        "status": "negotiating",
        "escalate_to_human": False,
    }


def _state_to_dict(deal_state) -> dict:
    return {
        "price": deal_state.price,
        "deliverables": deal_state.deliverables,
        "payout_method": deal_state.payout_method,
        "email": deal_state.email,
        "status": deal_state.status.value,
        "escalate_to_human": getattr(deal_state, "escalate_to_human", False),
    }


def run_conversation(client, persona: Persona | None = None) -> dict:
    """Play one full negotiator-vs-creator conversation and return its record."""
    if persona is None:
        persona = random.choice(PERSONAS)
    budget = random.randint(*BUDGET_RANGE)
    niche = random.choice(NICHES)
    brand = random.choice(BRANDS)
    payout_method = random.choice(PAYOUT_METHODS)

    negotiator_system = build_system_prompt(
        budget_max=budget,
        payout_method=payout_method,
        brand_facts=brand["brand_facts"],
        brand=brand["brand"],
        brand_website=brand["brand_website"],
    )
    creator_system = build_creator_prompt(persona, budget, niche, brand["brand"])

    # Message histories, each from its own model's point of view.
    creator_history = [{"role": "user", "content": BRAND_OPENER.format(brand=brand["brand"])}]
    negotiator_history: list[dict] = []

    turns: list[dict] = []
    running_state = _default_state()
    final_status = "error"

    for i in range(MAX_TURNS):
        # --- creator speaks ---
        creator_reply = call_claude(
            client, CREATOR_MODEL, creator_system, creator_history,
            CREATOR_MAX_TOKENS,
        )
        creator_history.append({"role": "assistant", "content": creator_reply})

        # --- negotiator responds ---
        negotiator_history.append({"role": "user", "content": creator_reply})
        raw = call_claude(
            client, NEGOTIATOR_MODEL, negotiator_system, negotiator_history,
            NEGOTIATOR_MAX_TOKENS,
        )
        negotiator_history.append({"role": "assistant", "content": raw})

        prior_state = dict(running_state)

        # --- parse against the contract ---
        try:
            out = parse_model_reply(raw)
        except ContractError as e:
            turns.append({
                "turn_index": i,
                "creator_message": creator_reply,
                "prior_deal_state": prior_state,
                "dm_text": None,
                "negotiator_raw": raw,
                "deal_state": None,
                "parse_error": str(e),
            })
            final_status = "contract_error"
            break

        new_state = _state_to_dict(out.deal_state)
        turns.append({
            "turn_index": i,
            "creator_message": creator_reply,
            "prior_deal_state": prior_state,
            "dm_text": out.dm_text,
            "negotiator_raw": raw,
            "deal_state": new_state,
            "parse_error": None,
        })
        running_state = new_state
        final_status = new_state["status"]

        # The creator only ever sees the negotiator's DM text, never the JSON.
        creator_history.append({"role": "user", "content": out.dm_text})

        # An escalation ends the negotiation too — the human takes over from here.
        # It's not a reject reason; the conversation is still saved normally.
        if new_state["escalate_to_human"]:
            final_status = "escalate_to_human"
            break

        if out.deal_state.status.value in TERMINAL_STATUSES:
            break

    return {
        "conversation_id": uuid.uuid4().hex,
        "persona_name": persona.name,
        "budget": budget,
        "niche": niche,
        "turns": turns,
        "final_status": final_status,
        "total_turns": len(turns),
        "meta": {
            "brand": brand["brand"],
            "payout_method": payout_method,
            "budget": budget,
        },
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate self-play negotiation transcripts.")
    parser.add_argument("--count", type=int, default=10, help="number of conversations to generate")
    parser.add_argument("--seed", type=int, default=None, help="optional RNG seed for reproducibility")
    parser.add_argument(
        "--persona", default=None,
        help="restrict generation to one persona (e.g. contract-pusher). "
             "Hyphens or underscores both work. Omit for a random mix.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    persona = None
    if args.persona is not None:
        key = args.persona.strip().lower().replace("-", "_")
        persona = PERSONAS_BY_NAME.get(key)
        if persona is None:
            valid = ", ".join(sorted(PERSONAS_BY_NAME))
            raise SystemExit(f"unknown persona '{args.persona}'. Valid personas: {valid}")

    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise SystemExit("ANTHROPIC_API_KEY is not set in .env — add your real key first.")

    client = anthropic.Anthropic(api_key=api_key, timeout=60.0)
    os.makedirs(RAW_DIR, exist_ok=True)

    saved = 0
    total_turns = 0
    for _ in tqdm(range(args.count), desc="conversations"):
        try:
            record = run_conversation(client, persona)
        except Exception as e:  # noqa: BLE001 — one bad conversation shouldn't kill the run
            tqdm.write(f"conversation failed: {type(e).__name__}: {e}")
            continue

        path = os.path.join(RAW_DIR, f"conv_{record['conversation_id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        saved += 1
        total_turns += record["total_turns"]

    print(f"\nSaved {saved}/{args.count} conversations to {RAW_DIR}")
    print(estimate_cost(saved, total_turns))


def estimate_cost(conversations: int, total_turns: int) -> str:
    """Rough cost projection. Each turn is one negotiator call + one creator call."""
    neg_in, neg_out = PRICING[NEGOTIATOR_MODEL]
    cre_in, cre_out = PRICING[CREATOR_MODEL]
    cost = (
        total_turns * (NEG_TOKENS_IN * neg_in + NEG_TOKENS_OUT * neg_out) / 1_000_000
        + total_turns * (CRE_TOKENS_IN * cre_in + CRE_TOKENS_OUT * cre_out) / 1_000_000
    )
    return (
        f"Run complete: {conversations:,} conversations | ~{total_turns:,} turns | "
        f"estimated cost: ${cost:,.2f}"
    )


if __name__ == "__main__":
    main()
