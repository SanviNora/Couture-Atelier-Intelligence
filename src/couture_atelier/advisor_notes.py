"""Advisor notes: a short reason to reach out, grounded only in the client's row.

A note may be drafted by a language model, but it is published only if it
passes ``check_note``: every number and category it mentions must come from the
client's facts, it must not use discount language, and it must stay short. A
note that fails falls back to the template, so the advisor always sees
something correct. The public demo never calls a model; it reads checked notes
from a cache built offline by ``scripts/build_advisor_notes.py``.
"""

from __future__ import annotations

import hashlib
import json
import re

FACT_FIELDS = ["tier", "weeks_since_last_visit", "spend_last_12m_eur", "favourite_category", "champion_action",
               "challenger_action"]
BANNED = re.compile(r"discount|sale\b|% off|promo|deal|bargain|cheap|clearance|guarantee", re.IGNORECASE)
MAX_CHARS = 280
TIER_WORDS = {"VIC", "core", "aspirational"}

SYSTEM_PROMPT = (
    "You write one-line reasons for a luxury client advisor to contact a client this quarter. Use only the facts "
    "given. Never invent purchases, products, events, prices or personal details. Never mention discounts, sales "
    "or offers. Quiet, warm, specific; at most two sentences and 280 characters. If the action is 'no action', "
    "say briefly why waiting is better."
)


def facts_of(row: dict) -> dict:
    """Plain Python types, so a row read back from CSV hashes like the row it was written from."""
    facts = {field: row[field] for field in FACT_FIELDS}
    facts["weeks_since_last_visit"] = int(facts["weeks_since_last_visit"])
    facts["spend_last_12m_eur"] = float(facts["spend_last_12m_eur"])
    return {k: v if isinstance(v, int | float) else str(v) for k, v in facts.items()}


def facts_key(facts: dict) -> str:
    return hashlib.sha256(json.dumps(facts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def template_note(facts: dict) -> str:
    weeks = int(facts["weeks_since_last_visit"])
    when = "this week" if weeks == 0 else "last week" if weeks == 1 else f"{weeks} weeks ago"
    category = str(facts["favourite_category"])
    if facts["champion_action"] == "contact" or facts["challenger_action"] == "contact":
        return (f"Visited {when}; {category} is a favourite. Share the new-season {category} pieces "
                "before they reach the floor.")
    return f"Visited {when}. No outreach this quarter; let the relationship breathe."


def _allowed_numbers(facts: dict) -> set[str]:
    allowed = set()
    for value in (facts["weeks_since_last_visit"], facts["spend_last_12m_eur"]):
        number = float(value)
        allowed |= {f"{number:.0f}", f"{number:,.0f}", f"{number / 1000:.0f}", f"{number / 1000:.1f}"}
    return allowed


def check_note(note: str, facts: dict, categories: tuple[str, ...] = ()) -> list[str]:
    """Problems with a note; an empty list means it can be shown to an advisor."""
    problems = []
    for category in categories:
        if category != facts["favourite_category"] and re.search(rf"\b{re.escape(category)}\b", note, re.IGNORECASE):
            problems.append(f"mentions {category}, which is not the client's favourite category")
    if len(note) > MAX_CHARS:
        problems.append(f"longer than {MAX_CHARS} characters")
    if BANNED.search(note):
        problems.append("discount or promotional language")
    allowed = _allowed_numbers(facts)
    for number in re.findall(r"\d[\d,]*(?:\.\d+)?", note):
        if number.rstrip(".") not in allowed:
            problems.append(f"number {number} is not in the client's facts")
    mentioned_tiers = {word for word in TIER_WORDS if re.search(rf"\b{word}\b", note, re.IGNORECASE)}
    if mentioned_tiers - {str(facts["tier"])}:
        problems.append(f"mentions tier {sorted(mentioned_tiers)} but the client is {facts['tier']}")
    return problems


def published_note(facts: dict, cache: dict[str, str], categories: tuple[str, ...] = ()) -> tuple[str, str]:
    """The note to show and where it came from: a checked model draft, or the template."""
    draft = cache.get(facts_key(facts))
    if draft and not check_note(draft, facts, categories):
        return draft, "model draft, checked"
    return template_note(facts), "template"
