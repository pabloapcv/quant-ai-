// Ramon web app: a thin client over the Ramon API (same origin).
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const store = {
    get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };

  // ---------------------------------------------------------------- formatting
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const pct = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(d)}%`);
  const spct = (v, d = 1) => (v === null || v === undefined ? '—' : `${v > 0 ? '+' : ''}${(v * 100).toFixed(d)}%`);
  const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d));
  const sgn = (v) => (v > 0 ? 'pos' : v < 0 ? 'neg' : '');
  const pill = (stance) => {
    const s = String(stance || '');
    const cls = s.includes('overweight') ? 'ow' : s.includes('underweight') ? 'uw' : '';
    return `<span class="pill ${cls}">${esc(s.replace('_', ' '))}</span>`;
  };
  const tile = (v, l) => `<div class="tile"><div class="v">${v}</div><div class="l">${esc(l)}</div></div>`;
  const table = (cols, rows) => `<thead><tr>${cols.map((c) => `<th class="${c.num ? 'num' : ''}">${esc(c.h)}</th>`).join('')}</tr></thead>` +
    `<tbody>${rows.map((r) => `<tr>${cols.map((c) => `<td class="${c.num ? 'num' : ''} ${c.cls ? c.cls(r) : ''}">${c.f(r)}</td>`).join('')}</tr>`).join('')}</tbody>`;

  // Minimal, safe markdown: escape first, then headings, bold, code, lists, tables, paragraphs.
  function md(src) {
    const lines = esc(src).split('\n');
    const out = [];
    let list = false;
    let tbl = [];
    const inline = (t) => t.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`([^`]+)`/g, '<code>$1</code>');
    const flushTable = () => {
      if (!tbl.length) return;
      const rows = tbl.filter((r) => !/^\s*\|?\s*:?-{2,}/.test(r)).map((r) => r.trim().replace(/^\||\|$/g, '').split('|').map((c) => inline(c.trim())));
      const [head, ...body] = rows;
      out.push(`<div class="wrap"><table><thead><tr>${head.map((c) => `<th>${c}</th>`).join('')}</tr></thead><tbody>${body.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`);
      tbl = [];
    };
    for (const line of lines) {
      if (/^\s*\|.*\|\s*$/.test(line)) { if (list) { out.push('</ul>'); list = false; } tbl.push(line); continue; }
      flushTable();
      const m = line.match(/^\s*[-*]\s+(.*)/);
      if (m) { if (!list) { out.push('<ul>'); list = true; } out.push(`<li>${inline(m[1])}</li>`); continue; }
      if (list) { out.push('</ul>'); list = false; }
      const h = line.match(/^#{1,4}\s+(.*)/);
      if (h) { out.push(`<p><strong>${inline(h[1])}</strong></p>`); continue; }
      if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    }
    flushTable();
    if (list) out.push('</ul>');
    return out.join('');
  }

  // ---------------------------------------------------------------- API
  const headers = () => {
    const t = store.get('ramon_token', '');
    return t ? { Authorization: `Bearer ${t}` } : {};
  };
  async function api(path, opts = {}) {
    const res = await fetch(path, { ...opts, headers: { ...(opts.headers || {}), ...headers() } });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
    return body;
  }
  const failBox = (el, e) => { el.innerHTML = `<p class="err">Couldn't load: ${esc(e.message)}</p>`; };

  // ---------------------------------------------------------------- tabs
  const loaded = new Set();
  function show(tab) {
    $$('#tabs button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === tab)));
    $$('.tab').forEach((s) => s.classList.toggle('active', s.id === `tab-${tab}`));
    store.set('ramon_tab', tab);
    if (!loaded.has(tab) && loaders[tab]) { loaded.add(tab); loaders[tab](); }
  }
  $$('#tabs button').forEach((b) => b.addEventListener('click', () => show(b.dataset.tab)));

  // ---------------------------------------------------------------- universes
  let universes = [];
  async function fillUniverses() {
    if (!universes.length) universes = (await api('/v1/universes')).universes;
    $$('select[id$="-universe"]').forEach((sel) => {
      if (sel.options.length) return;
      sel.innerHTML = universes.map((u) => `<option value="${esc(u.id)}">${esc(u.description)}</option>`).join('');
      sel.value = store.get(`ramon_${sel.id}`, 'ind12');
      if (!sel.value) sel.selectedIndex = 0;
    });
  }

  // ---------------------------------------------------------------- Ask
  let history = [];
  const chat = $('#chat');
  function addMsg(cls, html) {
    const d = document.createElement('div');
    d.className = `msg ${cls}`;
    d.innerHTML = html;
    chat.append(d);
    d.scrollIntoView({ behavior: 'smooth', block: 'end' });
    return d;
  }
  async function ask(question) {
    $('.hint')?.remove();
    addMsg('user', esc(question));
    const bot = addMsg('bot', '<div class="tools"></div><div class="answer"><span class="spinner"></span> Thinking…</div>');
    const tools = $('.tools', bot);
    const answerEl = $('.answer', bot);
    $('#send').disabled = true;
    try {
      const res = await fetch('/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...headers() },
        body: JSON.stringify({ question, history }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf('\n\n')) >= 0) {
          const block = buf.slice(0, i);
          buf = buf.slice(i + 2);
          const ev = (block.match(/^event: (.*)$/m) || [])[1];
          const data = JSON.parse((block.match(/^data: (.*)$/m) || [])[1] || '{}');
          if (ev === 'tool') {
            const s = document.createElement('span');
            s.textContent = `${data.name}${data.error ? ' (error)' : ''}`;
            tools.append(s);
          } else if (ev === 'answer') {
            answerEl.innerHTML = md(data.text);
            const ok = data.grounded;
            const line = document.createElement('div');
            line.className = `check-line ${ok ? '' : 'bad'}`;
            line.textContent = ok ? '✓ Every number checked against the engine data'
              : `⚠ Not traceable to engine data: ${data.unsupported.join(', ')}`;
            bot.append(line);
            history.push({ question, answer: data.text });
          } else if (ev === 'error') {
            answerEl.innerHTML = `<p class="err">${esc(data.kind === 'budget' ? 'Daily token budget reached: ' : '')}${esc(data.message)}</p>`;
          }
        }
      }
    } catch (e) {
      answerEl.innerHTML = `<p class="err">${esc(e.message)}</p>`;
    } finally {
      $('#send').disabled = false;
      loadUsage();
    }
  }
  $('#ask-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const q = $('#question').value.trim();
    if (!q) return;
    $('#question').value = '';
    ask(q);
  });
  $('#question').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('#ask-form').requestSubmit(); }
  });
  $$('.ex').forEach((b) => b.addEventListener('click', () => ask(b.textContent)));
  $('#clear').addEventListener('click', () => { history = []; chat.innerHTML = ''; });
  async function loadUsage() {
    try {
      const u = await api('/v1/usage');
      const b = u.daily_budget;
      $('#usage-line').textContent = `Claude today: ${Math.round(u.today.weighted).toLocaleString()} of ${b ? b.toLocaleString() : 'unlimited'} weighted tokens`;
    } catch { $('#usage-line').textContent = ''; }
  }

  // ---------------------------------------------------------------- Macro
  function cbCard(el, d, name) {
    const body = $('.body', el);
    if (d.error) { body.innerHTML = `<p class="muted">${esc(d.error)}</p>`; return; }
    const r = d.extracted_reading || {};
    const ch = d.changes_since_previous || { added: [], removed: [] };
    body.innerHTML = `
      <p><strong>${esc(d.meeting_date)}</strong> — ${r.action ? `${esc(r.action)} ${r.change_bp > 0 ? '+' : ''}${esc(r.change_bp)}bp, stance ${r.stance > 0 ? '+' : ''}${esc(r.stance)} (−2 dovish … +2 hawkish)` : 'no reading yet'}</p>
      ${r.summary ? `<p class="muted">${esc(r.summary)}</p>` : ''}
      <p class="small muted">Changes since ${esc(d.previous_meeting_date || '—')}:</p>
      <ul class="changes">${ch.added.slice(0, 6).map((s) => `<li class="added">${esc(s)}</li>`).join('')}${ch.removed.slice(0, 4).map((s) => `<li class="removed">${esc(s)}</li>`).join('')}</ul>
      <p class="small"><a href="${esc(d.url)}" target="_blank" rel="noopener">${esc(name)} statement ↗</a></p>`;
  }
  async function loadMacro() {
    const wait = '<span class="spinner"></span> Loading…';
    $('#macro-tiles').innerHTML = `<div class="tile">${wait}</div>`;
    ['#quadrants', '#pillars', '#fomc-card .body', '#ecb-card .body'].forEach((s) => { $(s).innerHTML = wait; });
    try {
      const r = await api('/v1/regime');
      const g = r.gdp_nowcast;
      $('#macro-tiles').innerHTML = [
        tile(esc(String(r.regime).replace('_', ' ')), `Regime, ${r.data_month}`),
        tile(pct(r.recession?.prob_12m), 'Recession probability, 12 months'),
        tile(g ? `${num(g.gdp_saar_pct, 1)}%` : '—', g ? `GDPNow ${g.quarter} (updated ${g.updated})` : 'GDPNow'),
        tile(num(r.financial_conditions.score), 'Financial conditions (above 0 = tighter)'),
      ].join('');
      const order = ['goldilocks', 'overheating', 'stagflation', 'deflationary_slowdown'];
      $('#quadrants').innerHTML = order.map((k) => `<div class="qrow"><span>${esc(k.replace('_', ' '))}</span><div class="bar"><div style="width:${(r.quadrant_probs[k] || 0) * 100}%"></div></div><span class="num">${pct(r.quadrant_probs[k], 0)}</span></div>`).join('');
      $('#pillars').innerHTML = `<table>${table([
        { h: 'Pillar', f: (p) => esc(p.k) },
        { h: 'Score', num: true, f: (p) => num(p.v.score), cls: (p) => sgn(p.v.score) },
        { h: '3m change', num: true, f: (p) => num(p.v.change_3m) },
        { h: 'Above trend', num: true, f: (p) => pct(p.v.prob_above_trend, 0) },
      ], [['growth', r.growth], ['inflation', r.inflation], ['financial conditions', r.financial_conditions]].map(([k, v]) => ({ k, v })))}</table>`;
    } catch (e) { failBox($('#macro-tiles'), e); }
    try {
      const cb = await api('/v1/central-banks');
      cbCard($('#fomc-card'), cb.fomc, 'FOMC');
      cbCard($('#ecb-card'), cb.ecb, 'ECB');
    } catch (e) { failBox($('#fomc-card .body'), e); }
  }

  // ---------------------------------------------------------------- Views
  async function loadViews() {
    await fillUniverses();
    const u = $('#views-universe').value;
    const h = $('#views-horizon').value;
    store.set('ramon_views-universe', u);
    $('#views-table').innerHTML = '<tbody><tr><td><span class="spinner"></span> Computing views…</td></tr></tbody>';
    try {
      const v = await api(`/v1/views?universe=${encodeURIComponent(u)}&horizon=${h}`);
      $('#views-head').textContent = `As of ${v.as_of_month}, ${v.horizon} horizon, vs ${v.benchmark}`;
      $('#views-table').innerHTML = table([
        { h: 'Asset', f: (r) => esc(r.asset) },
        { h: 'Stance', f: (r) => pill(r.stance) },
        { h: 'P(beat median)', num: true, f: (r) => pct(r.prob_beat_median, 0) },
        { h: 'Expected vs avg', num: true, f: (r) => spct(r.expected_excess_return), cls: (r) => sgn(r.expected_excess_return) },
        { h: '80% range', num: true, f: (r) => `${spct(r.range_p10)} … ${spct(r.range_p90)}` },
        { h: 'Confidence', f: (r) => esc(r.confidence) },
        { h: 'Top driver', f: (r) => esc(r.top_drivers?.[0]?.signal || '—') },
      ], v.views);
    } catch (e) { failBox($('#views-table'), e); }
  }
  $('#views-universe').addEventListener('change', loadViews);
  $('#views-horizon').addEventListener('change', loadViews);

  // ---------------------------------------------------------------- Scenario
  $('#scenario-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const shocks = {};
    $$('.shocks input').forEach((i) => { if (i.value !== '') shocks[i.name] = Number(i.value); });
    const out = $('#scenario-out');
    if (!Object.keys(shocks).length) { out.innerHTML = '<p class="err">Enter at least one shock.</p>'; return; }
    out.innerHTML = '<div class="card"><span class="spinner"></span> Running…</div>';
    try {
      const r = await api('/v1/scenario', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ universe: $('#scenario-universe').value, shocks, same_regime: $('#same-regime').checked }),
      });
      const avg = r.equal_weight_average_return;
      const implied = Object.entries(r.implied_moves_of_other_factors).map(([k, m]) => `${esc(k)} ${m.value > 0 ? '+' : ''}${num(m.value, 3)} ${esc(m.unit)}`).join(' · ');
      out.innerHTML = `<div class="card">
        <p>Fit on ${r.fit.months} months (${esc(r.fit.from)} … ${esc(r.fit.to)})${r.fit.regime_filter ? `, ${esc(r.fit.regime_filter)} months only` : ''}.
        Equal-weight average: <strong class="${sgn(avg.expected)}">${spct(avg.expected)}</strong> (80%: ${spct(avg.range_p10)} … ${spct(avg.range_p90)}).</p>
        <p class="small muted">Other factors usually moved: ${implied || '—'}</p>
        <div class="wrap"><table>${table([
          { h: 'Asset', f: (x) => esc(x.asset) },
          { h: 'Same-month move vs avg', num: true, f: (x) => spct(x.expected), cls: (x) => sgn(x.expected) },
          { h: '80% range', num: true, f: (x) => `${spct(x.range_p10)} … ${spct(x.range_p90)}` },
        ], r.excess_vs_average)}</table></div>
        <p class="small muted">${esc(r.note)}</p></div>`;
    } catch (err) { failBox(out, err); }
  });

  // ---------------------------------------------------------------- Analogs
  async function loadAnalogs() {
    await fillUniverses();
    const u = $('#analogs-universe').value;
    store.set('ramon_analogs-universe', u);
    $('#analogs-table').innerHTML = '<tbody><tr><td><span class="spinner"></span> Searching history…</td></tr></tbody>';
    try {
      const a = await api(`/v1/analogs?universe=${encodeURIComponent(u)}&k=10`);
      $('#analogs-table').innerHTML = table([
        { h: 'Month', f: (x) => esc(x.month.slice(0, 7)) },
        { h: 'Distance', num: true, f: (x) => num(x.distance) },
        { h: 'Best next 12m', f: (x) => `${esc(x.next_12m.best || '—')} <span class="pos">${spct(x.next_12m.best_excess, 0)}</span>` },
        { h: 'Worst next 12m', f: (x) => `${esc(x.next_12m.worst || '—')} <span class="neg">${spct(x.next_12m.worst_excess, 0)}</span>` },
      ], a.analogs);
      $('#analogs-fwd').innerHTML = table([
        { h: 'Asset', f: (x) => esc(x.asset) },
        { h: 'Median', num: true, f: (x) => spct(x.median_excess), cls: (x) => sgn(x.median_excess) },
        { h: 'Share positive', num: true, f: (x) => pct(x.share_of_analogs_positive, 0) },
      ], [...a.forward_excess_by_asset['12m']].sort((p, q) => q.median_excess - p.median_excess));
    } catch (e) { failBox($('#analogs-table'), e); }
  }
  $('#analogs-universe').addEventListener('change', loadAnalogs);

  // ---------------------------------------------------------------- System
  async function loadSystem() {
    $('#system-tiles').innerHTML = '<div class="tile"><span class="spinner"></span> Loading…</div>';
    $('#track-table').innerHTML = '<tbody><tr><td><span class="spinner"></span> Scoring track records…</td></tr></tbody>';
    try {
      const [h, u, al] = await Promise.all([api('/health'), api('/v1/usage'), api('/v1/alerts?days=14')]);
      $('#system-tiles').innerHTML = [
        tile(h.last_pipeline_ok ? 'OK' : 'Failed', `Last pipeline run ${String(h.last_pipeline_run || '—').slice(0, 16)}`),
        tile(Math.round(u.today.weighted).toLocaleString(), 'Claude weighted tokens today'),
        tile(String(u.today.calls), 'Claude calls today'),
        tile(String(al.alerts.length), 'Alerts in the last 14 days'),
      ].join('');
      $('#alerts').innerHTML = al.alerts.length
        ? `<ul class="changes">${al.alerts.slice(0, 12).map((x) => `<li><span class="small muted">${esc(String(x.at).slice(0, 10))}</span> ${esc(x.message)}</li>`).join('')}</ul>`
        : '<p class="muted">No alerts.</p>';
      await fillUniverses();
      const rows = [];
      for (const un of universes) {
        try {
          const t = await api(`/v1/track-record?universe=${encodeURIComponent(un.id)}&horizon=3`);
          rows.push({ u: un.id, t });
        } catch { /* skip */ }
      }
      $('#track-table').innerHTML = table([
        { h: 'Universe', f: (r) => esc(r.u) },
        { h: 'Long-run IC', num: true, f: (r) => num(r.t.ic_mean, 3) },
        { h: 't', num: true, f: (r) => num(r.t.ic_tstat) },
        { h: '2021–25 IC', num: true, f: (r) => num(r.t.holdout_2021_2025?.ic_mean, 3), cls: (r) => sgn(r.t.holdout_2021_2025?.ic_mean) },
        { h: 't', num: true, f: (r) => num(r.t.holdout_2021_2025?.ic_tstat) },
      ], rows);
    } catch (e) { failBox($('#system-tiles'), e); }
  }
  $('#token').value = store.get('ramon_token', '');
  $('#token').addEventListener('change', (e) => store.set('ramon_token', e.target.value));

  const loaders = { ask: loadUsage, macro: loadMacro, views: loadViews, analogs: loadAnalogs, scenario: fillUniverses, system: loadSystem };
  show(store.get('ramon_tab', 'ask'));
})();
