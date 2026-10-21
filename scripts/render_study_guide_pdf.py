"""Render the Markdown study guide as a styled PDF."""

from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "COUTURE_ATELIER_STUDY_GUIDE.md"
OUTPUT = ROOT / "docs" / "COUTURE_ATELIER_STUDY_GUIDE.html"

STYLES = """
@page { size: A4; margin: 18mm 17mm 20mm; @bottom-right { content: counter(page); color: #66716D; font: 9px sans-serif; } }
body { color: #18211F; font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; font-size: 10.5pt; line-height: 1.5; }
h1, h2, h3 { color: #1F5B52; font-family: Georgia, serif; line-height: 1.2; page-break-after: avoid; }
h1 { font-size: 25pt; border-bottom: 2px solid #B9975B; padding-bottom: 8px; margin-top: 0; }
h2 { font-size: 17pt; margin-top: 25px; border-bottom: 1px solid #DCDDD5; padding-bottom: 4px; }
h3 { font-size: 12.5pt; margin-top: 18px; color: #315F57; }
p, ul, ol, table, pre { page-break-inside: avoid; }
code { background: #F1EFE8; color: #315F57; padding: 1px 3px; border-radius: 2px; }
pre { background: #18211F; color: #F5F3EE; padding: 10px; border-radius: 3px; white-space: pre-wrap; font-size: 8.5pt; }
table { width: 100%; border-collapse: collapse; margin: 10px 0 16px; font-size: 9pt; }
th { background: #1F5B52; color: #F5F3EE; text-align: left; }
th, td { padding: 6px 8px; border: 1px solid #DCDDD5; }
tr:nth-child(even) { background: #F5F3EE; }
blockquote { border-left: 3px solid #B9975B; background: #EFE9DC; margin: 12px 0; padding: 8px 14px; }
hr { border: 0; border-top: 1px solid #DCDDD5; margin: 22px 0; }
"""


def render() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    body = markdown.markdown(source, extensions=["tables", "fenced_code", "toc"])
    document = f"<!doctype html><html><head><meta charset='utf-8'><style>{STYLES}</style></head><body>{body}</body></html>"
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    render()
