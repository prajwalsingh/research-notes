<!--
  This file is a starting point, not a published note — it is deliberately
  left out of manifest.json. Copy it to a new file (or just paste raw
  Gemini output into a new .md file) and add one entry to manifest.json.

  You do not need to touch this comment block or add any front-matter
  inside the .md file itself — title, date, and tags all live in
  manifest.json, so the note body can be exactly what Gemini gave you.
-->

## A section heading

Ordinary prose goes here. Inline math like $E = mc^2$ and display math like

$$
\nabla_\theta \mathcal{L} = \mathbb{E}\left[ \nabla_\theta \log p_\theta(x) \right]
$$

both render automatically — no conversion step. Keep display-math blocks
free of blank lines inside the `$$ ... $$` fence so the markdown parser
doesn't split them across paragraphs.

- bullet points work as usual
- so do **bold**, *italics*, and `inline code`

```python
# fenced code blocks get syntax highlighting
def f(x):
    return x ** 2
```

> blockquotes render as callouts, useful for "rule of thumb" style asides.
