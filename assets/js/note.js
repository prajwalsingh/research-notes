(function(){
  wireThemeToggle(document.getElementById('themeToggle'));
  mountTrace(document.getElementById('trace'));

  const params = new URLSearchParams(location.search);
  const slug = params.get('slug');
  const bodyEl = document.getElementById('noteBody');
  const titleEl = document.getElementById('noteTitle');
  const metaEl = document.getElementById('noteMeta');
  const tocEl = document.getElementById('tocList');
  const footerSlugEl = document.getElementById('footerSlug');

  const fmtDate = (iso) => {
    const d = new Date(iso + 'T00:00:00');
    if (isNaN(d)) return iso;
    return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: '2-digit' });
  };

  function slugify(text){
    return text.toLowerCase().trim()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-');
  }

  function buildToc(){
    const heads = bodyEl.querySelectorAll('h2, h3');
    const used = {};
    const items = [];
    heads.forEach(h => {
      let id = slugify(h.textContent);
      if (used[id] != null){ used[id]++; id = `${id}-${used[id]}`; } else { used[id] = 0; }
      h.id = id;
      const anchor = document.createElement('a');
      anchor.href = `#${id}`;
      anchor.className = 'heading-anchor';
      anchor.textContent = '#';
      h.appendChild(anchor);
      items.push({ id, level: h.tagName === 'H2' ? 2 : 3, text: h.textContent.replace(/#$/, '') });
    });

    if (items.length === 0){
      document.getElementById('noteToc').style.display = 'none';
      return;
    }
    tocEl.innerHTML = items.map(it =>
      `<li><a href="#${it.id}" data-level="${it.level}">${it.text}</a></li>`
    ).join('');

    const links = Array.from(tocEl.querySelectorAll('a'));
    const onScroll = () => {
      let current = null;
      heads.forEach(h => { if (h.getBoundingClientRect().top < 120) current = h.id; });
      links.forEach(l => l.classList.toggle('active', l.getAttribute('href') === `#${current}`));
    };
    document.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  function renderMath(){
    if (window.renderMathInElement){
      renderMathInElement(bodyEl, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '\\[', right: '\\]', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false }
        ],
        throwOnError: false
      });
    }
  }

  function highlightCode(){
    if (window.hljs){
      bodyEl.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
    }
  }

  function readingTime(markdown){
    const words = markdown.trim().split(/\s+/).length;
    return Math.max(1, Math.round(words / 220));
  }

  function loadNote(){
    if (!slug){
      bodyEl.innerHTML = `<p class="error">no note specified — go back to the <a href="index.html">notebook</a>.</p>`;
      return;
    }

    fetch('notes/manifest.json')
      .then(r => r.json())
      .then(data => {
        const entry = (data.notes || []).find(n => n.slug === slug);
        if (!entry) throw new Error(`slug "${slug}" not found in notes/manifest.json`);

        document.title = `${entry.title} — notebook`;
        titleEl.textContent = entry.title;
        footerSlugEl.textContent = entry.file;
        metaEl.innerHTML = `
          <span>${fmtDate(entry.date)}</span>
          ${entry.tags && entry.tags.length ? `<span>${entry.tags.join(' · ')}</span>` : ''}
          <span id="readTime"></span>
        `;

        return fetch(`notes/${entry.file}`).then(r => {
          if (!r.ok) throw new Error(`notes/${entry.file} not found (HTTP ${r.status})`);
          return r.text();
        }).then(md => {
          const readEl = document.getElementById('readTime');
          if (readEl) readEl.textContent = `${readingTime(md)} min read`;
          marked.setOptions({ gfm: true, breaks: false });
          bodyEl.innerHTML = marked.parse(md);
          buildToc();
          highlightCode();
          renderMath();
        });
      })
      .catch(err => {
        bodyEl.innerHTML = `<p class="error">${err.message}</p>`;
      });
  }

  loadNote();
})();
