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

  // ---------------------------------------------------------------- analyst mode
  const analyst = () => document.body.classList.contains('analyst');
  function setAnalyst(on) {
    document.body.classList.toggle('analyst', on);
    $('#analyst-mode').checked = on;
    store.set('ramon_analyst', on ? '1' : '');
    const cur = store.get('ramon_tab', 'brief');
    if (!on && $(`#tabs button[data-tab="${cur}"]`)?.classList.contains('analyst-only')) show('brief');
  }
  const TOOL_WORDS = {
    list_universes: 'Checked coverage', get_views: 'Read market leans', get_track_record: 'Checked track record',
    get_signal_gates: 'Checked model tests', get_portfolio: 'Read model portfolio', search_catalog: 'Searched data',
    get_series: 'Read economic data', get_regime: 'Read the economy', find_analogs: 'Compared with history',
    run_scenario: 'Ran a what-if', search_documents: 'Searched central-bank texts',
    get_ecb_statement: 'Read ECB statement', get_fomc_statement: 'Read Fed statement',
    get_business_lens: 'Looked at your business', get_key_prices: 'Checked key prices', get_credit_crypto_views: 'Read credit & crypto views',
  };

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
        body: JSON.stringify({ question, history, style: analyst() ? 'analyst' : 'executive' }),
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
            s.textContent = `${analyst() ? data.name : (TOOL_WORDS[data.name] || 'Checked data')}${data.error ? ' (error)' : ''}`;
            tools.append(s);
          } else if (ev === 'answer') {
            answerEl.innerHTML = md(data.text);
            const ok = data.grounded;
            const line = document.createElement('div');
            line.className = `check-line ${ok ? '' : 'bad'}`;
            line.textContent = ok ? '✓ Every number checked against Ramon\'s data'
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

  // ---------------------------------------------------------------- Briefing
  const light = (l, text) => `<span class="light ${esc(l)}"><i></i>${esc(text)}</span>`;
  const cap = (t) => (t ? t[0].toUpperCase() + t.slice(1) : t);
  const monthName = (d) => (d ? new Date(`${d.slice(0, 10)}T00:00:00Z`).toLocaleDateString(undefined, { month: 'long', year: 'numeric', timeZone: 'UTC' }) : '');
  const dayName = (d) => (d ? new Date(`${d.slice(0, 10)}T00:00:00Z`).toLocaleDateString(undefined, { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' }) : '');
  function leanList(items) {
    return `<ul class="leans">${items.map((x) => `<li><span>${esc(x.asset)}</span><span class="muted small">${pct(x.chance_beat_peers, 0)} chance of beating peers</span></li>`).join('')}</ul>`;
  }
  async function loadBrief() {
    const el = $('#brief');
    try {
      const b = await api('/v1/briefing');
      const e = b.economy;
      const parts = [];
      if (e) {
        const g = e.gdp_nowcast;
        parts.push(`<div class="card hero">
          <div class="eyebrow">The economy in one sentence · data through ${esc(monthName(e.data_month))}</div>
          <p class="headline">${esc(e.sentence)}</p>
          <div class="kpis">
            <div class="kpi"><div class="k">Overall picture</div><div class="v">${light(e.regime_light, e.regime)}</div>
              <div class="small muted">${e.regime_confidence === 'clear' ? 'Signals agree' : 'Signals are mixed: treat as a tentative read'}</div></div>
            <div class="kpi"><div class="k">Recession risk, next 12 months</div><div class="v">${light(e.recession.light, `${cap(e.recession.word)} · ${pct(e.recession.probability, 0)}`)}</div></div>
            <div class="kpi"><div class="k">Economic growth right now</div><div class="v">${g ? `${num(g.gdp_saar_pct, 1)}% <span class="small muted">annual pace</span>` : '—'}</div>
              <div class="small muted">${g ? `Atlanta Fed estimate for ${esc(String(g.quarter).replace(/(\d{4})Q(\d)/, 'Q$2 $1'))}, updated ${esc(dayName(g.updated))}` : ''}</div></div>
            <div class="kpi"><div class="k">Borrowing &amp; market conditions</div><div class="v">${esc(cap(e.financial_conditions.split(' (')[0]))}</div>
              <div class="small muted">${esc((e.financial_conditions.split(' (')[1] || '').replace(')', ''))}</div></div>
          </div></div>`);
      }
      const wc = b.what_changed || { items: [] };
      parts.push(`<div class="card"><h3>What changed since ${esc(dayName(wc.since))}</h3>${wc.items.length
        ? `<ul class="changes">${wc.items.map((t) => `<li>${esc(t)}</li>`).join('')}</ul>`
        : '<p class="muted">No significant changes in the past month.</p>'}</div>`);
      parts.push(`<div id="watch-wrap"></div>`);
      if ((b.prices || []).length) {
        const mv = (x, kind) => (x === null || x === undefined ? '—' : kind === 'pct' ? spct(x, 0) : `${x > 0 ? '+' : ''}${num(x, 2)} pts`);
        const rows = b.prices.map((pr) => `<tr${pr.big ? ' class="big"' : ''}><td>${esc(pr.name)}${pr.level !== null ? ` <span class="muted small">now ${num(pr.level, 2)}%</span>` : ''}${pr.big ? ' <span class="tag">big move</span>' : ''}</td>
          <td class="num">${mv(pr.change_3m, pr.kind)}</td><td class="num">${mv(pr.change_12m, pr.kind)}</td></tr>`).join('');
        parts.push(`<h2>Prices that matter</h2><div class="card"><div class="wrap"><table class="prices"><thead><tr><th></th><th class="num">Past 3 months</th><th class="num">Past year</th></tr></thead><tbody>${rows}</tbody></table></div>
          <p class="small muted">Commodities are world benchmark prices in US dollars, as monthly averages; rates are changes in percentage points. Rises are not colored as good or bad: that depends on whether you buy or sell. "Big move" means 15% or more in 3 months or 25% in a year (half a point or a full point for rates).</p></div>`);
      }
      parts.push(`<h2>Your business</h2><div class="card" id="lens">
        <div class="seg no-print" id="lens-markets" role="tablist"></div>
        <div class="controls"><label><span id="lens-prompt">My business is closest to</span> <select id="lens-asset"><option>Loading…</option></select></label></div>
        <div id="lens-body" class="muted small">Pick what matters to your business to see what Ramon's data says about it.</div></div>`);
      parts.push(`<h2>Central banks</h2><div class="grid2">${b.central_banks.map((c) => `<div class="card">
          <div class="eyebrow">${esc(c.bank)}${c.meeting_date ? ` · ${esc(dayName(c.meeting_date))}` : ''}</div>
          <p class="sub">${esc(c.headline)}</p>
          ${c.tone ? `<p><span class="tone ${c.facts.stance > 0 ? 'hawk' : c.facts.stance < 0 ? 'dove' : ''}">${esc(cap(c.tone.split(':')[0]))}</span> <span class="muted">${esc(c.tone.split(':').slice(1).join(':'))}</span></p>` : ''}
          ${c.summary ? `<p class="small">${esc(c.summary)}</p>` : ''}
          ${c.url ? `<p class="small"><a href="${esc(c.url)}" target="_blank" rel="noopener">Read the statement ↗</a></p>` : ''}
        </div>`).join('')}</div>
        <p class="small muted">"Hawkish" means focused on fighting inflation, which usually means higher borrowing costs; "dovish" means focused on supporting growth.</p>`);
      parts.push(`<h2>Where the models lean</h2><div class="grid2">${b.leans.map((l) => (l.as_of ? `<div class="card">
          <div class="eyebrow">${esc(l.label)} · next 3 months, as of ${esc(monthName(l.as_of))}</div>
          ${l.all_neutral ? '<p class="small muted">No strong leans: every one is close to a coin flip.</p>' : ''}
          <div class="lean-cols"><div><div class="k pos">Slightly favored</div>${leanList(l.favored)}</div>
          <div><div class="k neg">Least favored</div>${leanList(l.least)}</div></div></div>`
        : `<div class="card"><div class="eyebrow">${esc(l.label)}</div><p class="muted">Not available.</p></div>`)).join('')}</div>
        <div class="callout">${esc(b.trust_note)} <a href="#" class="how-link">How Ramon works</a></div>`);
      parts.push(`<h2>Recent warnings</h2><div class="card">${b.alerts.length
        ? `<ul class="changes">${b.alerts.map((a) => `<li><span class="small muted">${esc(dayName(a.date))}</span> ${esc(a.message)}</li>`).join('')}</ul>`
        : '<p class="muted">Nothing notable in the last two weeks.</p>'}</div>`);
      parts.push(`<div class="card no-print"><h3>Go deeper</h3><div class="examples">${[
        'What does the Fed decision mean for borrowing costs?',
        'Which industries tend to do well in an economy like this one?',
        'What would change your view on recession risk?',
      ].map((q) => `<button class="ex go">${esc(q)}</button>`).join('')}</div></div>`);
      el.innerHTML = parts.join('');
      initLens();
      renderWatch();
      $$('.how-link', el).forEach((a) => a.addEventListener('click', (ev) => { ev.preventDefault(); show('how'); }));
      $$('.go', el).forEach((btn) => btn.addEventListener('click', () => { show('ask'); ask(btn.textContent); }));
    } catch (err) { failBox(el, err); }
  }

  $('#print').addEventListener('click', () => window.print());

  // ---------------------------------------------------------------- Your business
  let lensMarkets = [];
  async function initLens() {
    try {
      lensMarkets = (await api('/v1/lens')).markets;
      if (!store.get('ramon_lens_ind12', '') && store.get('ramon_industry', '')) store.set('ramon_lens_ind12', store.get('ramon_industry', ''));
      $('#lens-markets').innerHTML = lensMarkets.map((m) => `<button type="button" data-m="${esc(m.id)}">${esc(m.label)}</button>`).join('');
      $$('#lens-markets button').forEach((b) => b.addEventListener('click', () => pickMarket(b.dataset.m)));
      $('#lens-asset').addEventListener('change', (e) => {
        const m = store.get('ramon_lens_market', 'ind12');
        store.set(`ramon_lens_${m}`, e.target.value);
        if (e.target.value) loadLens(m, e.target.value);
      });
      pickMarket(store.get('ramon_lens_market', 'ind12'));
    } catch (e) { failBox($('#lens-body'), e); }
  }
  async function pickMarket(id) {
    const m = lensMarkets.find((x) => x.id === (id === 'ctry_usd' ? 'ctry_world' : id)) || lensMarkets[0];
    store.set('ramon_lens_market', m.id);
    $$('#lens-markets button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.m === m.id)));
    $('#lens-prompt').textContent = m.prompt;
    const sel = $('#lens-asset');
    sel.innerHTML = '<option>Loading…</option>';
    sel.disabled = true;
    const body = $('#lens-body');
    body.className = 'muted small';
    body.innerHTML = `<span class="spinner"></span> Getting the list (the first look after a new data run can take a few minutes)…`;
    try {
      const { assets } = await api(`/v1/lens?market=${encodeURIComponent(m.id)}`);
      if (store.get('ramon_lens_market', 'ind12') !== m.id) return; // switched again meanwhile
      sel.innerHTML = `<option value="">Choose a ${esc(m.noun)}…</option>` + assets.map((a) => `<option>${esc(a)}</option>`).join('');
      sel.disabled = false;
      const saved = store.get(`ramon_lens_${m.id}`, '');
      if (saved && assets.includes(saved)) { sel.value = saved; loadLens(m.id, saved); }
      else body.textContent = `Pick a ${m.noun} to see what Ramon's data says about it.`;
    } catch (e) { failBox(body, e); }
  }
  async function loadLens(market, asset) {
    const body = $('#lens-body');
    body.className = '';
    body.innerHTML = `<span class="spinner"></span> Looking at ${esc(asset)}…`;
    try {
      const L = await api(`/v1/lens?market=${encodeURIComponent(market)}&asset=${encodeURIComponent(asset)}`);
      const max = Math.max(...L.events.map((e) => Math.abs(e.vs_average)), 1e-9);
      const bars = [...L.events].sort((a, b) => b.vs_average - a.vs_average).map((e) => {
        const w = (Math.abs(e.vs_average) / max) * 50;
        return `<div class="brow"><span class="name">${esc(e.label)}</span><div class="track"><div class="fill ${e.vs_average >= 0 ? 'up' : 'down'}" style="${e.vs_average >= 0 ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`}"></div></div><span class="val ${sgn(e.vs_average)}">${spct(e.vs_average)}</span></div>`;
      }).join('');
      const drv = esc(L.lean.drivers_sentence || '');
      body.innerHTML = `<ul class="changes">${L.summary.map((t) => `<li>${esc(t)}</li>`).join('')}</ul>
        ${drv ? `<p class="small muted">What drives the lean. ${drv}</p>` : ''}
        <div class="k small muted" style="margin-top:12px">How ${esc(asset)} moved vs ${esc(L.avg)} in the same month, by event</div>
        <div class="bars">${bars}</div>
        <div class="callout">${esc(L.trust_note)}</div>
        <div class="examples no-print"><button class="ex pin" type="button"></button><button class="ex go2">What are the biggest risks for ${esc(asset)} right now?</button></div>`;
      $$('.go2', body).forEach((btn) => btn.addEventListener('click', () => { show('ask'); ask(btn.textContent); }));
      const pin = $('.pin', body);
      const paint = () => { pin.textContent = isWatched(market, asset) ? '✓ On your watchlist (remove)' : '+ Add to your watchlist'; };
      paint();
      pin.addEventListener('click', () => { toggleWatch(market, asset); paint(); renderWatch(); });
    } catch (e) { failBox(body, e); }
  }

  // ---------------------------------------------------------------- Watchlist
  // Per-browser convenience: the pinned items live in this browser only.
  // ctry_usd (14 markets) became ctry_world (33) on 2026-10-26: older pins move over
  const watchList = () => { try { return (JSON.parse(store.get('ramon_watchlist', '[]')) || []).map((w) => (w.market === 'ctry_usd' ? { ...w, market: 'ctry_world' } : w)); } catch { return []; } };
  const isWatched = (m, a) => watchList().some((w) => w.market === m && w.asset === a);
  function toggleWatch(m, a) {
    const list = watchList();
    const next = isWatched(m, a) ? list.filter((w) => !(w.market === m && w.asset === a)) : [...list, { market: m, asset: a }].slice(-12);
    store.set('ramon_watchlist', JSON.stringify(next));
  }
  async function renderWatch() {
    const wrap = $('#watch-wrap');
    if (!wrap) return;
    const list = watchList();
    if (!list.length) { wrap.innerHTML = ''; return; }
    if (!lensMarkets.length) { try { lensMarkets = (await api('/v1/lens')).markets; } catch { /* labels fall back to ids */ } }
    const label = (m) => (lensMarkets.find((x) => x.id === m) || {}).label || m;
    wrap.innerHTML = `<h2>Your watchlist</h2><div class="card"><ul class="watch">${list.map((w, i) => `<li data-i="${i}"><button type="button" class="watch-open"><strong>${esc(w.asset)}</strong> <span class="muted small">${esc(label(w.market))}</span><span class="watch-line muted"><span class="spinner"></span></span></button><button type="button" class="watch-x no-print" title="Remove" aria-label="Remove ${esc(w.asset)}">×</button></li>`).join('')}</ul></div>`;
    $$('.watch li', wrap).forEach((li) => {
      const w = list[Number(li.dataset.i)];
      $('.watch-open', li).addEventListener('click', async () => {
        store.set(`ramon_lens_${w.market}`, w.asset);
        await pickMarket(w.market);
        $('#lens').scrollIntoView({ behavior: 'smooth' });
      });
      $('.watch-x', li).addEventListener('click', () => { toggleWatch(w.market, w.asset); renderWatch(); });
      api(`/v1/lens?market=${encodeURIComponent(w.market)}&asset=${encodeURIComponent(w.asset)}`)
        .then((L) => { $('.watch-line', li).textContent = L.headline; })
        .catch((e) => { $('.watch-line', li).textContent = `Not available: ${e.message}`; });
    });
  }

  // ---------------------------------------------------------------- What if
  let PRESETS = [];
  const UNIT_WORD = { ind12: 'the average US sector', ctry_usd: 'the average market', ctry_world: 'the average market', fx_g10: 'the average currency', fx_em: 'the average emerging-market currency', rates_dev: 'the average bond market', rates_em: 'the average bond market', cmdty: 'the average commodity' };
  let lastPreset = null;
  async function loadPresets() {
    if (PRESETS.length) return;
    try {
      PRESETS = (await api('/v1/presets')).presets;
      $('#presets').innerHTML = PRESETS.map((p, i) => `<button type="button" class="preset" data-i="${i}">${esc(p.label)}</button>`).join('');
      $$('.preset').forEach((b) => b.addEventListener('click', () => runPreset(Number(b.dataset.i))));
    } catch (e) { failBox($('#presets'), e); }
  }
  $('#whatif-universe').addEventListener('change', () => { if (lastPreset !== null) runPreset(lastPreset); });
  async function runPreset(i) {
    lastPreset = i;
    const p = PRESETS[i];
    $$('.preset').forEach((b) => b.classList.toggle('on', Number(b.dataset.i) === i));
    const u = $('#whatif-universe').value;
    const out = $('#whatif-out');
    out.innerHTML = '<div class="card"><span class="spinner"></span> Looking through history…</div>';
    try {
      const r = await api('/v1/scenario', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ universe: u, shocks: p.shocks }) });
      const rows = [...r.excess_vs_average].sort((a, b) => b.expected - a.expected);
      const max = Math.max(...rows.map((x) => Math.abs(x.expected)), 1e-9);
      const best = rows[0]; const worst = rows[rows.length - 1];
      const avg = r.equal_weight_average_return.expected;
      const bars = rows.map((x) => {
        const w = (Math.abs(x.expected) / max) * 50;
        return `<div class="brow"><span class="name">${esc(x.asset)}</span><div class="track"><div class="fill ${x.expected >= 0 ? 'up' : 'down'}" style="${x.expected >= 0 ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`}"></div></div><span class="val ${sgn(x.expected)}">${spct(x.expected)}</span></div>`;
      }).join('');
      out.innerHTML = `<div class="card">
        <div class="eyebrow">${esc(p.label)}${p.note ? ` (${esc(p.note)})` : ''}</div>
        <p class="sub">Historically, <strong>${esc(best.asset)}</strong> did best (${spct(best.expected)} vs ${esc(UNIT_WORD[u])}) and <strong>${esc(worst.asset)}</strong> fared worst (${spct(worst.expected)}).</p>
        <p class="small muted">${esc(cap(UNIT_WORD[u]))} itself typically ${avg >= 0 ? 'rose' : 'fell'} ${pct(Math.abs(avg))} in such a month. The bars show each one's move compared with that average.</p>
        <div class="bars">${bars}</div>
        <p class="small muted">Based on ${esc(String(r.fit.months))} months of history (${esc(monthName(r.fit.from))} to ${esc(monthName(r.fit.to))}). This shows how things usually moved together in the same month, not a forecast; real events rarely match history exactly.</p>
      </div>`;
    } catch (err) { failBox(out, err); }
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
  async function loadSleeves() {
    try {
      const sl = await api('/v1/sleeves');
      $('#sleeves-table').innerHTML = table([
        { h: 'Asset', f: (r) => esc(r.asset_name) },
        { h: 'Month', f: (r) => esc(String(r.as_of).slice(0, 7)) },
        { h: 'P(beat)', num: true, f: (r) => pct(r.prob_positive_excess, 0) },
        { h: 'Stance', f: (r) => pill(r.stance) },
        { h: '12m trend', num: true, f: (r) => spct(r.drivers.trend_12m) },
        { h: 'Spread vs 5y avg', num: true, f: (r) => (r.drivers.spread_vs_5y_avg === null ? '—' : `${num(Math.abs(r.drivers.spread_vs_5y_avg) < 0.005 ? 0 : r.drivers.spread_vs_5y_avg, 2)} pts`) },
      ], sl.views);
      $('#sleeves-note').textContent = sl.trust_note;
    } catch (e) { failBox($('#sleeves-table'), e); }
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

  const loaders = { brief: loadBrief, whatif: loadPresets, ask: loadUsage, macro: loadMacro, views: () => { loadViews(); loadSleeves(); }, analogs: loadAnalogs, scenario: fillUniverses, system: loadSystem };
  $('#analyst-mode').addEventListener('change', (e) => setAnalyst(e.target.checked));
  setAnalyst(store.get('ramon_analyst', '') === '1');
  show(store.get('ramon_tab', 'brief'));
})();
