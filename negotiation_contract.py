"""
Xicmo Negotiator — CONTRACT  (reconciled against live prompts, Aug 2026)
The single source of truth both Claude (teacher/fallback) and the 8B obey.

Three parts:
  1. SYSTEM_PROMPT_SPEC  — instructions + the fixed output format
  2. DealState / ModelOutput + JSON Schema — machine-checkable shape
  3. parse_model_reply() — turn raw model text into a validated ModelOutput

Import this in the data generator, the validator, and the router so all
three agree on exactly one contract.

KEY CHANGES vs original Phase-0 contract (reconciled from live prompts):
  - Close condition is now 3 fields: price + payout_method + email.
    Deliverables are set BY the negotiator, not extracted from the creator.
  - Hard ceiling = budget_max * 1.10. Above it -> escalate_to_human=true,
    hold dm_text at budget_max. Below ceiling, negotiate normally.
  - escalate_to_human field added to DealState (bool). Router reads it.
  - Payout method is brand-supplied and fixed; model never invents another.
  - Brand-facts anti-hallucination rule added: only state given brand facts;
    unknowns -> "will check with the team"; a question != acceptance.
  - Strict extraction: creatorAcceptedPrice null if merely proposed (not
    accepted); email only if literally typed, never constructed from handle.
  - Message constraints: max 500 chars, no em/en dashes, no AI mention,
    phone-typing voice.
  - build_system_prompt() now takes budget_max, payout_method, brand_facts,
    brand, and brand_website alongside budget (all injected per conversation).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional
import json, re


# ===========================================================================
# PART 1 — THE SYSTEM PROMPT SPEC
# Matches the live counter-offer negotiation prompt the team sent.
# Injected per conversation: budget_max, payout_method, brand_facts,
#                            brand, brand_website.
# The hard ceiling (110% of budget_max) is computed and injected here so
# the model sees it explicitly — same as the live prompt's {ceiling}.
# ===========================================================================

SYSTEM_PROMPT_SPEC = """\
You are a senior influencer-marketing manager for "{brand}", negotiating a paid \
collab with a creator over Instagram DM.

Brand website: {brand_website}

What the brand is (the ONLY brand facts you may state):
{brand_facts}

Facts rule — this overrides everything else:
- Only state things about the brand that appear above. Never invent a product, \
feature, customer, price, or claim.
- If the creator asks something you were not told, say plainly that you will \
check with the team and come back. Do not guess, and do not answer a different \
question instead.
- Never treat a question as agreement. A creator asking what the product is has \
not accepted anything.

Budget (internal, never state the range): floor roughly half of {budget_max}, \
target ceiling {budget_max}.
Hard ceiling — anything above this MUST escalate: {ceiling} (110% of target ceiling).
Payout method the brand can actually use: {payout_method}. Never offer any other method.

Scope — agree commercial terms ONLY:
- price, deliverables, payout method, contact email.
- Never take, send, or process a payment. Never say a payment has been made or scheduled.
- Never share, sign, or promise a contract yourself. The most you may say is that \
the team will send the collab document.
- If the creator pushes on payment timing, legal terms, or anything outside those \
four points, say the team will handle it and stop there.

A deal is only closed when the creator has given you ALL THREE of:
  1. a price they explicitly accept,
  2. a payout method (must match the supported method above),
  3. an email address for the paperwork.
Ask for whichever of the three is still missing. Ask for at most two in one message.

Extraction — from the creator's own words ONLY:
- price: the number they EXPLICITLY AGREED TO, else null. A number they proposed \
is NOT one they accepted.
- payout_method: only if they named the supported method, else null.
- email: only an address they actually typed. Never construct one from their handle.

Negotiate like a professional:
- Open below your ceiling and leave room. Move in one step, not three.
- Trade, don't concede: more budget buys more deliverables or a longer usage window.
- Name deliverables concretely (e.g. "1 in-feed reel, 2 stories, 48-hour window").
- If they demand above the hard ceiling ({ceiling}), set escalate_to_human to true \
and hold at {budget_max} in your reply. Do not go above it.
- If the conversation is ambiguous, off-topic, or hostile, set escalate_to_human to true.

Message rules:
- Max 500 characters. One message, no repetition of what you already said.
- Never re-state the full deal twice in a row. If you have already summarised \
the terms, move the conversation forward instead.
- Never use em dashes or en dashes. Use commas, full stops, or plain hyphens.
- Never mention AI, automation, models, or the tool that wrote this.
- Write the way a person types on a phone: plain words, no corporate filler.

## Output format — reply with ONLY a valid JSON object, nothing else:
{{
  "dm_text": "<the message to send the creator, max 500 chars>",
  "deal_state": {{
    "price": <number the creator explicitly accepted, or null>,
    "deliverables": <string describing agreed deliverables, or null>,
    "payout_method": <"{payout_method}" if confirmed, or null>,
    "email": <email the creator literally typed, or null>,
    "status": "negotiating" | "terms_locked" | "out_of_scope" | "walk_away",
    "escalate_to_human": <true | false>
  }}
}}

Status rules:
- "terms_locked" only when price + payout_method + email are ALL non-null and \
price <= {budget_max}.
- "walk_away" when the creator will not come under {ceiling} after your best offer.
- "out_of_scope" when the creator pushes outside the four commercial fields.
- "escalate_to_human" true when: price demanded > {ceiling}, OR conversation is \
ambiguous/hostile/off-topic. escalate_to_human and walk_away can both be true.

Return the JSON and nothing before or after it. No markdown, no code fences, \
no commentary.
"""


def build_system_prompt(
    budget_max: float,
    payout_method: str,
    brand_facts: str,
    brand: str = "the brand",
    brand_website: str = "",
) -> str:
    """
    Inject all per-conversation variables into the spec.

    Args:
        budget_max:     The target ceiling (e.g. 1000). Hard ceiling = 110% of this.
        payout_method:  The ONE method the brand supports (e.g. "PayPal").
        brand_facts:    Free-text brand description. Model may ONLY state these facts.
        brand:          Brand display name (e.g. "Lumina Skincare").
        brand_website:  Brand URL (e.g. "lumina.com").
    """
    ceiling = round(budget_max * 1.10)
    return SYSTEM_PROMPT_SPEC.format(
        budget_max=budget_max,
        ceiling=ceiling,
        payout_method=payout_method,
        brand_facts=brand_facts,
        brand=brand,
        brand_website=brand_website,
    )


# ===========================================================================
# PART 2 — THE SCHEMA (machine-checkable shape)
# ===========================================================================

class Status(str, Enum):
    NEGOTIATING  = "negotiating"
    TERMS_LOCKED = "terms_locked"
    OUT_OF_SCOPE = "out_of_scope"
    WALK_AWAY    = "walk_away"


@dataclass
class DealState:
    price:             Optional[float] = None
    deliverables:      Optional[str]   = None
    payout_method:     Optional[str]   = None
    email:             Optional[str]   = None
    status:            Status          = Status.NEGOTIATING
    escalate_to_human: bool            = False

    def required_fields_filled(self) -> bool:
        """
        A deal is closable when the creator has given price + payout + email.
        Deliverables is negotiator-set (not extracted), so it is NOT a close gate.
        Business logic (router/validator) still checks price <= budget_max.
        """
        return all([self.price, self.payout_method, self.email])


@dataclass
class ModelOutput:
    dm_text:    str
    deal_state: DealState

    def to_json(self) -> str:
        d = asdict(self)
        d["deal_state"]["status"] = self.deal_state.status.value
        return json.dumps(d, ensure_ascii=False)


# JSON Schema — formal contract for validation + constrained decoding
# (pass OUTPUT_JSON_SCHEMA to Outlines/vLLM guided JSON on the 8B so it
#  physically cannot emit invalid JSON at inference time).
OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "required": ["dm_text", "deal_state"],
    "additionalProperties": False,
    "properties": {
        "dm_text": {"type": "string", "minLength": 1},
        "deal_state": {
            "type": "object",
            "required": [
                "price", "deliverables", "payout_method",
                "email", "status", "escalate_to_human",
            ],
            "additionalProperties": False,
            "properties": {
                "price":             {"type": ["number", "null"]},
                "deliverables":      {"type": ["string", "null"]},
                "payout_method":     {"type": ["string", "null"]},
                "email":             {"type": ["string", "null"]},
                "status": {
                    "type": "string",
                    "enum": ["negotiating", "terms_locked", "out_of_scope", "walk_away"],
                },
                "escalate_to_human": {"type": "boolean"},
            },
        },
    },
}


# ===========================================================================
# PART 3 — PARSE + VALIDATE
# Turns raw model text into a ModelOutput, or raises with a clear reason.
# ===========================================================================

class ContractError(Exception):
    """Raised when model output does not satisfy the contract."""


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict:
    """Pull the JSON object out of the model's reply, tolerating stray text/fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if not m:
            raise ContractError("no_json_found")
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            raise ContractError("json_parse_failed")


def parse_model_reply(raw: str) -> ModelOutput:
    """
    Parse + shape-validate a raw model reply.

    Checks CONTRACT shape/enum only — not business rules like 'price <= budget'
    or 'payout_method matches allowed value'. Those live in the router/validator
    because they are per-conversation and the model must not be trusted to self-
    enforce financial ceilings.
    """
    data = _extract_json(raw)

    if "dm_text" not in data or "deal_state" not in data:
        raise ContractError("missing_top_level_keys")

    ds = data["deal_state"]
    if not isinstance(ds, dict):
        raise ContractError("deal_state_not_object")

    required_keys = ("price", "deliverables", "payout_method", "email",
                     "status", "escalate_to_human")
    for k in required_keys:
        if k not in ds:
            raise ContractError(f"missing_field:{k}")

    try:
        status = Status(ds["status"])
    except ValueError:
        raise ContractError(f"bad_status:{ds['status']}")

    if not isinstance(ds["escalate_to_human"], bool):
        raise ContractError("escalate_to_human_not_bool")

    if not isinstance(data["dm_text"], str) or not data["dm_text"].strip():
        raise ContractError("empty_dm_text")

    return ModelOutput(
        dm_text=data["dm_text"].strip(),
        deal_state=DealState(
            price             = ds["price"],
            deliverables      = ds["deliverables"],
            payout_method     = ds["payout_method"],
            email             = ds["email"],
            status            = status,
            escalate_to_human = ds["escalate_to_human"],
        ),
    )


# ===========================================================================
# SMOKE TEST — run `python negotiation_contract.py` to verify
# ===========================================================================

if __name__ == "__main__":
    prompt = build_system_prompt(
        budget_max    = 800,
        payout_method = "PayPal",
        brand_facts   = "Lumina is a cruelty-free skincare brand. Hero product: SPF 50 daily moisturiser, $28.",
        brand         = "Lumina Skincare",
        brand_website = "lumina.com",
    )
    print("=== System prompt (first 500 chars) ===")
    print(prompt[:500], "...\n")

    # ceiling should be 880 (800 * 1.10)
    assert "880" in prompt, "ceiling not injected correctly"
    assert "PayPal" in prompt, "payout_method not injected"
    assert "Lumina" in prompt, "brand not injected"
    print("Injection checks: OK\n")

    # --- parse tests ---
    good_negotiating = json.dumps({
        "dm_text": "We can do $600 for 1 reel + 2 stories. PayPal ok?",
        "deal_state": {
            "price": None, "deliverables": "1 reel + 2 stories",
            "payout_method": None, "email": None,
            "status": "negotiating", "escalate_to_human": False,
        }
    })

    good_locked = json.dumps({
        "dm_text": "Great, $700 via PayPal it is. What email should we send the doc to?",
        "deal_state": {
            "price": 700, "deliverables": "1 reel + 2 stories",
            "payout_method": "PayPal", "email": "creator@example.com",
            "status": "terms_locked", "escalate_to_human": False,
        }
    })

    good_escalate = json.dumps({
        "dm_text": "I'll pass this to my team - they'll be in touch.",
        "deal_state": {
            "price": None, "deliverables": None,
            "payout_method": None, "email": None,
            "status": "negotiating", "escalate_to_human": True,
        }
    })

    fenced = "```json\n" + good_negotiating + "\n```"
    bad_missing_escalate = json.dumps({
        "dm_text": "hi",
        "deal_state": {
            "price": None, "deliverables": None,
            "payout_method": None, "email": None,
            "status": "negotiating",
            # escalate_to_human missing
        }
    })
    bad_no_deal_state = '{"dm_text": "hi"}'

    tests = [
        ("negotiating",          good_negotiating),
        ("terms_locked",         good_locked),
        ("escalate_to_human",    good_escalate),
        ("fenced_json",          fenced),
        ("missing_escalate",     bad_missing_escalate),
        ("missing_deal_state",   bad_no_deal_state),
    ]

    print("=== parse_model_reply tests ===")
    for label, raw in tests:
        try:
            out = parse_model_reply(raw)
            print(
                f"[{label}] OK  -> status={out.deal_state.status.value}, "
                f"price={out.deal_state.price}, "
                f"escalate={out.deal_state.escalate_to_human}, "
                f"fields_filled={out.deal_state.required_fields_filled()}"
            )
        except ContractError as e:
            print(f"[{label}] REJECTED -> {e}")

    print("\nSmoke test complete.")
