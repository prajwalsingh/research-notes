#!/usr/bin/env python3
"""
Scaffold a new note: creates notes/<slug>.md and adds one entry to
notes/manifest.json. Paste your raw Gemini markdown/LaTeX straight into
the generated .md file afterwards — no reformatting needed.

Usage:
    python3 scripts/new_note.py "Title of the note" --tags eeg,math

Run from the repo root (the folder containing index.html).
"""
import argparse
import datetime
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "notes"
MANIFEST = NOTES_DIR / "manifest.json"


def slugify(title: str) -> str:
    s = title.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def main():
    parser = argparse.ArgumentParser(description="Create a new note.")
    parser.add_argument("title", help="Note title, e.g. \"Why RoPE extrapolates\"")
    parser.add_argument("--tags", default="", help="Comma-separated tags, e.g. eeg,math")
    parser.add_argument("--excerpt", default="", help="One-line summary shown on the index page")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD, defaults to today")
    args = parser.parse_args()

    slug = slugify(args.title)
    date = args.date or datetime.date.today().isoformat()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    md_path = NOTES_DIR / f"{slug}.md"

    if md_path.exists():
        sys.exit(f"notes/{slug}.md already exists — pick a different title or delete it first.")

    md_path.write_text(
        f"<!-- paste raw Gemini markdown/LaTeX below this line -->\n\n", encoding="utf-8"
    )

    manifest = {"notes": []}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    manifest["notes"].append({
        "slug": slug,
        "title": args.title,
        "date": date,
        "tags": tags,
        "excerpt": args.excerpt,
        "file": f"{slug}.md",
    })
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"created notes/{slug}.md")
    print(f"registered in notes/manifest.json")
    print(f"next: paste your Gemini text into notes/{slug}.md, then preview locally.")


if __name__ == "__main__":
    main()
