# Xicmo Instagram Negotiator — Data Generator

A self-play pipeline that generates supervised fine-tuning (SFT) data for teaching
a small (8B) LLM to negotiate paid Instagram brand collaborations. A strong
"teacher" model (Claude Sonnet) plays the **negotiator**; a cheaper model
(Claude Haiku) plays the **creator** across a set of adversarial personas. Every
negotiator reply must obey a strict JSON **contract**, and only conversations
that pass business-rule validation become training data.

## Layout

```
negotiation_contract.py     # single source of truth: system prompt spec, schema, parser
generator/
  personas.py               # 8 creator counterparty personas
  selfplay.py               # two-model self-play loop -> data/raw/
  validate.py               # business-rule filtering -> data/clean/train.jsonl
  format_for_training.py    # -> messages-format JSONL, single- and multi-turn, train/val split
data/
  raw/                      # unfiltered generated conversations (gitignored)
  clean/                    # validated, training-ready JSONL (gitignored)
requirements.txt
.env                        # ANTHROPIC_API_KEY=... (gitignored, never commit)
```

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
echo ANTHROPIC_API_KEY=sk-ant-... > .env
```

## Pipeline

1. **Generate** — self-play conversations, one persona per run for balance:
   ```bash
   python generator/selfplay.py --count 100 --persona reasonable
   ```
   Each conversation randomizes budget ($200–$2000), niche, brand, and payout
   method. Runs stop per-conversation on parse errors and retry transient API
   failures. Output: `data/raw/conv_<uuid>.json`.

2. **Validate** — drop bad conversations, emit flat training JSONL:
   ```bash
   python generator/validate.py
   ```

3. **Format** — produce portable `messages` JSONL with a train/val split:
   ```bash
   python generator/format_for_training.py
   ```

Steps 2 and 3 are **local-only** (no API calls / credits).

## The contract

Every negotiator turn must emit exactly one JSON object:

```json
{
  "dm_text": "<message to the creator, max 500 chars>",
  "deal_state": {
    "price": <number the creator explicitly accepted, or null>,
    "deliverables": <string or null>,
    "payout_method": <supported method if confirmed, or null>,
    "email": <email the creator literally typed, or null>,
    "status": "negotiating" | "terms_locked" | "out_of_scope" | "walk_away",
    "escalate_to_human": <true | false>
  }
}
```

Key rules enforced by `validate.py`: a deal closes (`terms_locked`) only when
price + payout_method + email are all present and price ≤ budget × 1.10 (the hard
ceiling); `terms_locked` + `escalate_to_human` together is a contradiction and is
rejected; empty `dm_text`, <2 turns, and mid-conversation parse failures are
rejected; duplicate opening DMs are deduped.

## Personas

`over_asker`, `contract_pusher`, `ghost`, `reasonable`, `lowballer`,
`scope_creeper`, `vague`, `aggressive` — each stress-tests a specific failure
mode (anchoring, out-of-scope contract demands, non-response, bait-and-switch,
scope creep, vagueness, hostility). See `generator/personas.py`.

## Training data format

Both framings are produced from the same clean conversations, in `data/clean/`:

| File | Unit | Use |
|------|------|-----|
| `train_singleturn.jsonl` / `val_singleturn.jsonl` | one example per negotiator turn | prior context is compressed into a `deal_state` summary in the user message; cleaner signal for JSON/contract discipline |
| `train_multiturn.jsonl` / `val_multiturn.jsonl` | one example per conversation | full DM thread as alternating messages; faithful to how the teacher was prompted |

Every line is portable `messages` format:

```json
{"messages": [
  {"role": "system", "content": "<negotiator system prompt with budget/brand injected>"},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "{\"dm_text\": ...}"}
]}
```

The split is 90/10 **by conversation** (never by turn), stratified per persona,
so turns from one conversation never straddle train/val.

### Training notes

- **Match training to serving.** If your router feeds the model a compact
  `deal_state` + latest creator message at inference, train on **single-turn**.
  If it feeds the whole DM thread, train on **multi-turn**.
- **Multi-turn requires assistant-only loss masking** — compute loss only on the
  assistant turns (mask system + user), or the model learns to imitate the
  creator too. Most stacks do this for `messages` data: HF `SFTTrainer` with a
  chat template, Axolotl `type: chat_template`, Unsloth `train_on_responses_only`.
- **Constrained decoding**: `OUTPUT_JSON_SCHEMA` in `negotiation_contract.py` can
  be passed to a guided-JSON decoder (Outlines / vLLM) so the 8B physically
  cannot emit invalid JSON at inference time.

## Notes

- The data is **fully synthetic** (Claude-generated). Creator emails / payout
  handles that appear in transcripts are invented placeholders, not real PII.
- Generation cost is roughly $0.02–$0.05 per conversation depending on persona
  (short-closing personas like `aggressive` are cheapest; `ghost` runs the full
  turn cap and is priciest).
