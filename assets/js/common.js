/* Shared across index.html and note.html:
   - light/dark theme toggle, persisted in localStorage
   - the EEG-trace scroll indicator (the site's one signature element)
*/

(function initTheme(){
  const saved = localStorage.getItem('ln-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const theme = saved || (prefersDark ? 'dark' : 'light');
  if (theme === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
})();

function wireThemeToggle(btn){
  if (!btn) return;
  const setLabel = () => {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    btn.textContent = isDark ? 'light' : 'dark';
    btn.setAttribute('aria-label', isDark ? 'Switch to light theme' : 'Switch to dark theme');
  };
  setLabel();
  btn.addEventListener('click', () => {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    if (isDark){
      document.documentElement.removeAttribute('data-theme');
      localStorage.setItem('ln-theme', 'light');
    } else {
      document.documentElement.setAttribute('data-theme', 'dark');
      localStorage.setItem('ln-theme', 'dark');
    }
    setLabel();
  });
}

/* Deterministic pseudo-random EEG-like trace, same on every load for a given
   page (seeded by pathname) so it doesn't jitter on re-render. */
function seededRandom(seed){
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return function(){
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function buildTracePoints(width, height, seed){
  const rand = seededRandom(seed);
  const points = [];
  const step = 6;
  let y = height / 2;
  for (let x = 0; x <= width; x += step){
    // mostly-flat baseline with occasional spikes, like a real EEG channel
    const spike = rand() < 0.12 ? (rand() - 0.5) * height * 1.6 : (rand() - 0.5) * height * 0.25;
    y = Math.max(2, Math.min(height - 2, height / 2 + spike));
    points.push(`${x},${y.toFixed(1)}`);
  }
  return points.join(' ');
}

function mountTrace(container){
  if (!container) return;
  const width = 1000, height = 26;
  const seed = Array.from(location.pathname + location.search).reduce((a, c) => a + c.charCodeAt(0), 7);
  const pts = buildTracePoints(width, height, seed || 7);
  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true">
      <polyline class="trace-bg" points="${pts}" />
      <polyline class="trace-fg" points="${pts}" style="clip-path: inset(0 100% 0 0);" />
    </svg>`;
  const fg = container.querySelector('.trace-fg');
  const onScroll = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    const pct = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
    fg.style.clipPath = `inset(0 ${(100 - pct * 100).toFixed(2)}% 0 0)`;
  };
  document.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  onScroll();
}
