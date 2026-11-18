"""Render a personal interview guide from the project's reports and gaps log.

Every number is read from ``reports/``; every question comes from
``docs/gaps_and_decisions.md``. Personal pitch notes are read from
``prep/pitch.md`` if present. Output goes to ``prep/`` (git-ignored):
Markdown, HTML, and a PDF when Playwright is installed.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime

import markdown

from couture_atelier.paths import REPORTS, ROOT

PREP = ROOT / "prep"
STYLES = """
@page { size: A4; margin: 16mm 16mm 18mm; }
body { color: #16151a; font: 10pt/1.5 -apple-system, "Helvetica Neue", sans-serif; }
h1, h2, h3 { font-family: Georgia, serif; font-weight: 400; line-height: 1.15; page-break-after: avoid; }
h1 { font-size: 26pt; margin: 0 0 4px; } h1 em, h2 em { color: #5b4fe0; }
h2 { font-size: 16pt; margin-top: 26px; border-bottom: 1px solid #e8e2d7; padding-bottom: 4px; }
h3 { font-size: 13pt; margin: 22px 0 6px; color: #4436c9; }
h4 { font-size: 10.5pt; margin: 14px 0 3px; font-family: -apple-system, sans-serif; font-weight: 600; page-break-after: avoid; }
table { width: 100%; border-collapse: collapse; margin: 8px 0 14px; font-size: 9pt; page-break-inside: avoid; }
th { background: #f3efe8; text-align: left; } th, td { padding: 5px 7px; border: 1px solid #e8e2d7; vertical-align: top; }
blockquote { border-left: 3px solid #5b4fe0; background: #f6f4fe; margin: 8px 0; padding: 6px 12px; }
.q { page-break-inside: avoid; margin-bottom: 10px; } .meta { color: #75717a; font-size: 8.5pt; }
code { background: #f3efe8; padding: 0 3px; border-radius: 3px; }
"""


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text())


def headline_numbers() -> list[tuple[str, str, str]]:
    cv, cl = load("client_value.json")["evaluation"]["test"], load("clienteling.json")
    demand, alloc = load("demand.json"), load("allocation.json")
    ca, ps = load("client_allocation.json"), load("price_strategy.json")
    ci = cv["paired_bootstrap_95ci"]["return_auc_ml_minus_recency"]
    robust = cl["robustness_across_pilot_draws"]
    lf = demand["launch_forecast"]
    chosen = lf["models"][lf["chosen_on_validation"]]["test"]
    base = lf["models"]["store_category_average"]["test"]
    store = {r["policy"]: r["revenue_eur_m"] for r in alloc["store_allocation"]}
    hero = ca["policies"]
    ten = next(r for r in ps["cut"]["cuts"] if round(r["cut"], 2) == 0.10)
    gap = ps["cohort"]["entry_minus_higher_within_boutique_month"]["returned_2y"]
    diag = cv["spend_bias_diagnosis"]
    return [
        ("Return-next-year AUC", f"{cv['lapse']['return_auc_ml']:.3f}",
         f"+{ci['difference']:.3f} over recency alone, 95% CI [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"),
        ("Spend bias, 2019", (f"{cv['spend']['selected_ml']['total_bias_pct']:+.0f}% → "
                              f"{diag['selected_ml_trend_level']['total_bias_pct']:+.0f}%"),
         "falling return rate; trend correction found after the test, reported as a diagnosis"),
        ("Clienteling, share of best achievable", f"{robust['status_quo_share']:.1%} champion",
         (f"uplift with trust check {robust['mean_with_trust_check']:.1%}; shortlist hybrid "
          f"{robust.get('mean_shortlist_hybrid', float('nan')):.1%} (mean of 3 pilot draws)")),
        ("Launch forecast WAPE", f"{chosen['wape']:.3f}", f"vs {base['wape']:.3f} store × category average"),
        ("80% interval coverage", f"{lf['quantile_interval_test']['coverage_q10_q90']:.1%}", "calibrated"),
        ("Store allocation", f"+€{store['optimal_with_uncertainty'] - store['retailer_opening_share']:.1f}M",
         "over the retailer's own split at 60% of demand"),
        ("Hero pieces to clients",
         (f"€{hero['relationship_value_with_pipeline_reserve']['expected_total_value_eur_m']:.0f}M"
          f" vs €{hero['vic_first']['expected_total_value_eur_m']:.0f}M"), "relationship-value MILP vs VICs first"),
        ("10% price cut", f"needs +{ten['breakeven_volume_gain']:.0%}, gets +{ten['volume_gain_sustained']:.0%}",
         f"sustained elasticity {ps['cut']['elasticity']['sustained']:.1f} from real markdowns"),
        ("Entry-first clients", f"{gap['difference'] * 100:+.1f} pts return rate",
         f"same boutique and month, CI [{gap['ci_low'] * 100:.1f}, {gap['ci_high'] * 100:.1f}]"),
    ]


def questions() -> str:
    pattern = re.compile(r"^\| (G\d+) \| (.+?) \| (.+?) \| (Fixed(?: \(partly\))?|Flagged|Accepted) \| (.+?) \| (.+) \|$")
    by_area: dict[str, list[str]] = {}
    for line in (ROOT / "docs" / "gaps_and_decisions.md").read_text().splitlines():
        match = pattern.match(line)
        if not match:
            continue
        gid, area, finding, status, handled, question = match.groups()
        by_area.setdefault(area, []).append(
            f"#### {question.strip(chr(34))}\n\n**What happened.** {finding}\n\n"
            f"**What I did.** {handled} <span class='meta'>{gid} · {status}</span>")
    return "\n\n".join(f"### {area}\n\n" + "\n\n".join(items) for area, items in sorted(by_area.items()))


def render() -> None:
    PREP.mkdir(exist_ok=True)
    pitch = (PREP / "pitch.md").read_text() if (PREP / "pitch.md").exists() else ""
    rows = "\n".join(re.sub(r"-(?=\d)", "−", f"| {a} | **{b}** | {c} |") for a, b, c in headline_numbers())
    source = f"""# Couture Atelier *Intelligence*

Interview guide · {datetime.now(UTC):%d %B %Y} · every number read from `reports/`

## Numbers to know cold

| Result | Number | Context |
|---|---|---|
{rows}

{pitch}

## Questions the project invites

Each comes from the gaps log: what went wrong or was judged, and what was done about it.

{questions()}
"""
    (PREP / "interview_guide.md").write_text(source)
    body = markdown.markdown(source, extensions=["tables"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{STYLES}</style></head><body>{body}</body></html>"
    (PREP / "interview_guide.html").write_text(html)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("wrote prep/interview_guide.html (install playwright for the PDF)", file=sys.stderr)
        return
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto((PREP / "interview_guide.html").as_uri())
        page.pdf(path=str(PREP / "interview_guide.pdf"), format="A4", print_background=True,
                 margin={"top": "16mm", "bottom": "18mm", "left": "16mm", "right": "16mm"})
        browser.close()
    print("wrote prep/interview_guide.pdf")


if __name__ == "__main__":
    render()
