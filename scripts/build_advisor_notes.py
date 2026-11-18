"""Draft advisor notes for the illustrative synthetic book with a language model (optional, offline).

Needs ``pip install anthropic`` and ANTHROPIC_API_KEY. Each draft is checked by
``couture_atelier.advisor_notes.check_note``; failures are counted and fall
back to the template when the bundle is built. Drafts are cached by a hash of
the client's facts, so the public bundle rebuilds without calling the model.

Writes ``reports/advisor_notes_cache.json`` and ``reports/advisor_notes_eval.json``.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from couture_atelier.advisor_notes import SYSTEM_PROMPT, check_note, facts_key, facts_of
from couture_atelier.paths import PUBLIC_BUNDLE, REPORTS

MODEL = os.environ.get("ADVISOR_NOTES_MODEL", "claude-haiku-5-5")
CACHE = REPORTS / "advisor_notes_cache.json"


def draft(client, facts: dict) -> str:
    message = client.messages.create(
        model=MODEL, max_tokens=200, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Client facts: {json.dumps(facts, default=str)}\nWrite the note."}],
    )
    return message.content[0].text.strip()


def build_advisor_notes() -> dict:
    import anthropic  # optional dependency, only needed here

    client = anthropic.Anthropic()
    book = pd.read_csv(PUBLIC_BUNDLE / "demo_advisor_book.csv")
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    categories = tuple(book["favourite_category"].unique())
    failures = []
    for row in book.to_dict("records"):
        facts = facts_of(row)
        note = draft(client, facts)
        problems = check_note(note, facts, categories)
        if problems:
            failures.append({"client": row["client"], "note": note, "problems": problems})
        else:
            cache[facts_key(facts)] = note
    CACHE.write_text(json.dumps(cache, indent=2) + "\n")
    summary = {"model": MODEL, "drafted": len(book), "passed_check": len(book) - len(failures),
               "failed_check": failures}
    (REPORTS / "advisor_notes_eval.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(build_advisor_notes(), indent=2))
