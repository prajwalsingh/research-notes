(function(){
  wireThemeToggle(document.getElementById('themeToggle'));
  mountTrace(document.getElementById('trace'));

  const listEl = document.getElementById('noteList');
  const searchEl = document.getElementById('search');
  const tagsEl = document.getElementById('tagFilters');

  const fmtDate = (iso) => {
    const d = new Date(iso + 'T00:00:00');
    if (isNaN(d)) return iso;
    return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: '2-digit' });
  };

  let notes = [];
  let activeTag = null;

  function render(){
    const q = searchEl.value.trim().toLowerCase();
    const filtered = notes.filter(n => {
      const matchesTag = !activeTag || (n.tags || []).includes(activeTag);
      const hay = [n.title, n.excerpt, ...(n.tags || [])].join(' ').toLowerCase();
      const matchesQuery = !q || hay.includes(q);
      return matchesTag && matchesQuery;
    });

    if (filtered.length === 0){
      listEl.innerHTML = `<li class="empty-state">nothing matches — try a different search or tag.</li>`;
      return;
    }

    listEl.innerHTML = filtered.map(n => `
      <li class="note-item">
        <a href="note.html?slug=${encodeURIComponent(n.slug)}">
          <div class="note-meta">
            <span>${fmtDate(n.date)}</span>
            <span>${n.readMins ? n.readMins + ' min read' : ''}</span>
          </div>
          <h2 class="note-title">${n.title}</h2>
          ${n.excerpt ? `<p class="note-excerpt">${n.excerpt}</p>` : ''}
          ${n.tags && n.tags.length ? `<div class="note-tags">${n.tags.map(t => `<span>${t}</span>`).join('')}</div>` : ''}
        </a>
      </li>
    `).join('');
  }

  function renderTags(){
    const allTags = Array.from(new Set(notes.flatMap(n => n.tags || []))).sort();
    tagsEl.innerHTML = allTags.map(t =>
      `<button class="tag-chip" data-tag="${t}" aria-pressed="false" type="button">${t}</button>`
    ).join('');
    tagsEl.querySelectorAll('.tag-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const tag = chip.dataset.tag;
        activeTag = (activeTag === tag) ? null : tag;
        tagsEl.querySelectorAll('.tag-chip').forEach(c =>
          c.setAttribute('aria-pressed', c.dataset.tag === activeTag ? 'true' : 'false'));
        render();
      });
    });
  }

  fetch('notes/manifest.json', { cache: 'no-store' })
    .then(r => {
      if (!r.ok) throw new Error('manifest.json not found (HTTP ' + r.status + ')');
      return r.json();
    })
    .then(data => {
      notes = (data.notes || []).slice().sort((a, b) => (a.date < b.date ? 1 : -1));
      renderTags();
      render();
    })
    .catch(err => {
      listEl.innerHTML = `<li class="error">could not load notes/manifest.json — ${err.message}.
        If you opened this file directly (file://), serve it instead: run
        <code>python3 -m http.server</code> in this folder and open
        <code>http://localhost:8000</code>.</li>`;
    });

  searchEl.addEventListener('input', render);
})();
