# notebook

A minimal, no-build-step site for turning raw Gemini markdown/LaTeX output
into a continuously-scrolling notes page, hostable on GitHub Pages.

Paste a Gemini response into a `.md` file, add one line to a manifest, push.
There is no conversion script and nothing to compile — `marked.js`, `KaTeX`,
and `highlight.js` are loaded from a CDN and do the rendering in the browser.

## How it's laid out

```
index.html            the notebook listing (search + tag filters)
note.html             renders a single note, given ?slug=...
assets/css/style.css  all styling, both themes
assets/js/common.js   theme toggle + the EEG-trace scroll indicator
assets/js/index.js    listing page logic
assets/js/note.js     markdown -> HTML -> math/code rendering, TOC, scrollspy
notes/manifest.json   one entry per note: title, date, tags, excerpt, filename
notes/*.md            note content, exactly as pasted from Gemini
notes/_template.md    a starting skeleton — not published, kept out of manifest
scripts/new_note.py   scaffolds a new note + manifest entry from the terminal
```

A note's `.md` file holds *only* body content — title, date, and tags all
live in `manifest.json`, so you can paste Gemini's raw output in without
stripping or adding anything.

## Adding a note

**Option A — by hand**

1. Drop a new file in `notes/`, e.g. `notes/rope-extrapolation.md`, and paste
   your Gemini text into it as-is.
2. Add one entry to `notes/manifest.json`:

   ```json
   {
     "slug": "rope-extrapolation",
     "title": "Why RoPE extrapolates past training length",
     "date": "2026-08-05",
     "tags": ["nerf", "positional-encoding"],
     "excerpt": "One line for the listing page.",
     "file": "rope-extrapolation.md"
   }
   ```
3. Refresh the browser.

**Option B — from the terminal**

```bash
python3 scripts/new_note.py "Why RoPE extrapolates past training length" \
  --tags nerf,positional-encoding \
  --excerpt "One line for the listing page."
```

This creates `notes/why-rope-extrapolates-past-training-length.md` and
appends the manifest entry for you. Paste the Gemini text into the new file
and you're done.

## Previewing locally on Ubuntu 20.04

The pages `fetch()` `manifest.json` and the `.md` files, which browsers
block over a bare `file://` URL. Serve the folder instead — Python 3 ships
with Ubuntu 20.04, so no install is needed:

```bash
cd lab-notes
python3 -m http.server 8000
```

Then open `http://localhost:8000` in a browser. Leave the server running
and just refresh the tab as you edit notes.

## Publishing to GitHub Pages

1. Push this folder to a repo (as the repo root, or under `/docs`).
2. In the repo's **Settings → Pages**, set the source to the branch/folder
   you pushed to.
3. Your notebook is live at `https://<username>.github.io/<repo>/`.

No build step runs on GitHub's side — it's serving static files exactly as
they sit in the repo, so what you see locally is what ships.

## Writing notes that render cleanly

- Inline math: `$...$`. Display math: `$$...$$` or `\[...\]`.
- Underscores and asterisks inside math (`\theta_I`, `S_{ij}`, `x_i`) are
  protected from markdown's own emphasis parser before rendering, so LaTeX
  subscripts and multiplication won't get eaten or turn into stray italics —
  including when a `$$...$$` block spans a blank line.
- Fenced code blocks (` ```python `) get syntax highlighting automatically.
- Headings (`##`, `###`) automatically populate the "on this page" bar on
  the note page.
- Tables, blockquotes, and images use plain GitHub-flavored markdown.
- If you paste text copied from a rendered chat UI (rather than raw
  markdown source), section titles and paragraph breaks often collapse
  into one run-on block of text. You'll need to re-add `##` headings and
  blank lines between paragraphs by hand — the renderer can't recover
  structure that isn't in the text.

## Design notes

Palette, type, and the EEG-trace scroll indicator are all defined as CSS
custom properties at the top of `assets/css/style.css` — change `--bg`,
`--ink`, `--accent`, `--signal`, or the font stacks there and the whole site
follows. The trace indicator in the header is generated per-page from a
seeded pseudo-random walk in `common.js`, so it's stable across reloads of
the same page and traces itself in as you scroll, rather than a plain
progress bar.
