/* FraudGuard AI - front-end logic. Every number shown comes from the /api endpoints. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const money = n => Number(n).toLocaleString(undefined, {maximumFractionDigits: 2});
  const pct = p => (p * 100).toFixed(1) + '%';
  const when = iso => new Date(iso).toLocaleString();

  async function api(url, opts = {}) {
    const o = {credentials: 'same-origin', headers: {}, ...opts};
    if (o.body && typeof o.body !== 'string') { o.body = JSON.stringify(o.body); o.headers['Content-Type'] = 'application/json'; }
    const res = await fetch(url, o);
    let data = null; try { data = await res.json(); } catch (e) {}
    if (res.status === 401 && !url.startsWith('/api/auth/login')) { location.href = '/login'; throw new Error('Authentication required'); }
    if (!res.ok) { const err = new Error((data && data.error) || `Request failed (${res.status})`); err.status = res.status; throw err; }
    return data;
  }
  function toast(msg, kind = 'danger') {
    const el = document.createElement('div'); el.className = `alert alert-${kind} shadow`; el.textContent = msg;
    $('#toasts').appendChild(el); setTimeout(() => el.remove(), 5000);
  }
  const pill = v => `<span class="pill ${esc(v)}">${esc(v)}</span>`;
  function counter(el, to, decimals = 0, suffix = '') {
    if (!el) return;
    if (reduceMotion || !to) { el.textContent = to.toFixed(decimals) + suffix; return; }
    const t0 = performance.now(), dur = 900;
    const tick = t => { const k = Math.min((t - t0) / dur, 1), e = 1 - Math.pow(1 - k, 3);
      el.textContent = (to * e).toFixed(decimals) + suffix; if (k < 1) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }
  // chart theme
  Chart.defaults.color = '#8fa0c4'; Chart.defaults.borderColor = 'rgba(120,160,255,.12)';
  Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
  if (reduceMotion) Chart.defaults.animation = false; else Chart.defaults.animation.duration = 900;
  const C = {blue: '#3b82f6', cyan: '#22d3ee', green: '#22c55e', amber: '#f59e0b', red: '#ef4444'};
  const mk = (id, cfg) => { const el = document.getElementById(id); return el ? new Chart(el, cfg) : null; };

  async function modelBanner() {
    const box = $('#modelBanner'); if (!box) return;
    try {
      const s = await api('/api/model/status');
      if (!s.available) box.innerHTML = `<div class="alert alert-warning"><b>Model not available.</b> ${esc(s.reason)}</div>`;
    } catch (e) {}
  }
  const rowLink = id => `<a class="btn-ghost btn-sm py-1" href="/transactions/${id}">View</a>`;

  // ---------- pages ----------
  const pages = {
    async auth() {
      const f = $('#authForm'); const err = $('#formError');
      f.addEventListener('submit', async e => {
        e.preventDefault(); err.classList.add('d-none');
        const body = Object.fromEntries(new FormData(f)); const btn = f.querySelector('button'); btn.disabled = true;
        try { await api(`/api/auth/${f.dataset.mode}`, {method: 'POST', body}); location.href = '/dashboard'; }
        catch (ex) { err.textContent = ex.message; err.classList.remove('d-none'); btn.disabled = false; }
      });
    },
    async dashboard() {
      modelBanner();
      const s = await api('/api/dashboard/stats');
      counter($('#k-total'), s.total); counter($('#k-fraud'), s.fraudulent); counter($('#k-legit'), s.legitimate);
      counter($('#k-rate'), s.fraud_rate, 2, '%');
      mk('trendChart', {type: 'line', data: {labels: s.trend.dates, datasets: [
        {label: 'Analyzed', data: s.trend.total, borderColor: C.blue, backgroundColor: 'rgba(59,130,246,.15)', fill: true, tension: .35},
        {label: 'Flagged fraudulent', data: s.trend.fraudulent, borderColor: C.red, tension: .35}]},
        options: {maintainAspectRatio: false, scales: {y: {beginAtZero: true, ticks: {precision: 0}}}}});
      mk('riskChart', {type: 'doughnut', data: {labels: ['Low', 'Medium', 'High'], datasets: [{data: Object.values(s.risk_distribution),
        backgroundColor: [C.green, C.amber, C.red], borderWidth: 0}]}, options: {maintainAspectRatio: false, cutout: '68%'}});
      fillTable('#recentTable', '#recentEmpty', s.recent_suspicious, false);
    },
    async detect() {
      modelBanner();
      const f = $('#detectForm'), err = $('#formError'), ov = $('#overlay');
      const examples = {
        routine: {type: 'CASH_OUT', hour_of_day: 14, amount: 450, oldbalanceOrg: 25000, oldbalanceDest: 3000},
        drain: {type: 'TRANSFER', hour_of_day: 3, amount: 181000, oldbalanceOrg: 181000, oldbalanceDest: 0}};
      $$('[data-example]').forEach(b => b.addEventListener('click', () => {
        const ex = examples[b.dataset.example]; Object.entries(ex).forEach(([k, v]) => { f.elements[k].value = v; }); }));
      f.addEventListener('submit', async e => {
        e.preventDefault(); err.classList.add('d-none');
        const body = Object.fromEntries(new FormData(f));
        if (!f.checkValidity()) { f.reportValidity(); return; }
        ov.classList.add('show'); $('#submitBtn').disabled = true;
        try { const r = await api('/api/transactions/predict', {method: 'POST', body}); location.href = `/transactions/${r.id}`; }
        catch (ex) { ov.classList.remove('show'); $('#submitBtn').disabled = false; err.textContent = ex.message; err.classList.remove('d-none'); }
      });
    },
    async result() {
      const root = $('#resultRoot'); const id = document.body.dataset.tx;
      let t; try { t = await api(`/api/transactions/${id}`); } catch (ex) { root.innerHTML = `<div class="fg-card empty">${esc(ex.message)}</div>`; return; }
      const color = {Low: C.green, Medium: C.amber, High: C.red}[t.risk_level]; const L = 251.3;
      const ex = t.explanation;
      root.innerHTML = `
      <div class="row g-3">
        <div class="col-lg-5 reveal"><div class="fg-card text-center h-100"><h2>Fraud risk</h2>
          <svg class="gauge" viewBox="0 0 200 120" role="img" aria-label="Fraud probability ${pct(t.fraud_probability)}">
            <path class="track" d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke-width="16" stroke-linecap="round"/>
            <path class="fill" id="gaugeFill" d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="${color}" stroke-width="16" stroke-linecap="round" stroke-dasharray="${L}" stroke-dashoffset="${L}"/>
            <text x="100" y="92" text-anchor="middle" fill="#fff" font-size="30" font-weight="800">${pct(t.fraud_probability)}</text></svg>
          <div class="mt-2">${pill(t.label)} ${pill(t.risk_level)}</div>
          <p class="muted small mt-3 mb-0">Decision threshold used: ${t.threshold.toFixed(3)} &middot; Model ${esc(t.model_version)}</p></div></div>
        <div class="col-lg-7 reveal d1"><div class="fg-card h-100"><h2>Transaction #${t.id}</h2>
          <div class="row g-2 small">${[['Type', t.type], ['Hour of day', String(t.hour_of_day).padStart(2, '0') + ':00'], ['Amount', money(t.amount)], ['Analyzed', when(t.created_at)],
            ['Sender balance before', money(t.oldbalanceOrg)], ['Receiver balance before', money(t.oldbalanceDest)]]
            .map(([k, v]) => `<div class="col-6"><div class="muted">${k}</div><div class="fw-semibold">${esc(v)}</div></div>`).join('')}</div></div></div>
        <div class="col-12 reveal d2"><div class="fg-card"><h2><i class="bi bi-eye me-1"></i>Why this score? (SHAP)</h2>
          ${ex ? `<div class="shap-note mb-3">${esc(ex.summary)}</div>
            <div class="row g-3"><div class="col-lg-7"><div style="height:${Math.max(220, ex.contributions.slice(0, 8).length * 38)}px"><canvas id="shapChart"></canvas></div></div>
            <div class="col-lg-5"><div class="table-responsive"><table class="table fg-table table-sm"><thead><tr><th>Feature</th><th>Value</th><th>Effect</th></tr></thead><tbody>
              ${ex.contributions.slice(0, 8).map(c => `<tr><td>${esc(c.label)}</td><td>${esc(c.display)}</td>
              <td class="${c.shap > 0 ? 'text-danger' : 'text-success'}">${c.shap > 0 ? '&uarr; raises risk' : '&darr; lowers risk'}</td></tr>`).join('')}</tbody></table></div></div></div>`
          : `<div class="empty">No SHAP explanation was stored for this transaction (the SHAP library was unavailable when it was scored).</div>`}
          <p class="disclaimer mt-3 mb-0">Positive SHAP values push the score toward fraud, negative values toward legitimate. The prediction is an estimate of fraud risk, not confirmed fraud.</p></div></div>
      </div>`;
      requestAnimationFrame(() => setTimeout(() => { $('#gaugeFill').style.strokeDashoffset = L * (1 - Math.min(t.fraud_probability, 1)); }, 80));
      if (ex) { const top = ex.contributions.slice(0, 8);
        mk('shapChart', {type: 'bar', data: {labels: top.map(c => c.label), datasets: [{data: top.map(c => c.shap),
          backgroundColor: top.map(c => c.shap > 0 ? C.red : C.green), borderRadius: 6}]},
          options: {indexAxis: 'y', maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {x: {title: {display: true, text: 'SHAP value (impact on fraud score)'}}}}}); }
    },
    async history() {
      const f = $('#filterForm'); let page = 1;
      const load = async () => {
        const p = new URLSearchParams({page, per_page: 10});
        ['q', 'label', 'risk', 'date_from', 'date_to'].forEach(k => { const v = $('#' + k).value.trim(); if (v) p.set(k, v); });
        try {
          const r = await api('/api/transactions?' + p);
          fillTable('#histTable', '#histEmpty', r.items, true);
          $('#histInfo').textContent = `${r.total} transaction(s) - page ${r.page} of ${Math.max(r.pages, 1)}`;
          const pg = $('#pager'); pg.innerHTML = '';
          const add = (txt, to, dis, act) => { const li = document.createElement('li'); li.className = `page-item ${dis ? 'disabled' : ''} ${act ? 'active' : ''}`;
            li.innerHTML = `<a class="page-link" href="#">${txt}</a>`; li.onclick = e => { e.preventDefault(); if (!dis) { page = to; load(); } }; pg.appendChild(li); };
          add('Prev', page - 1, page <= 1); add(page, page, false, true); add('Next', page + 1, page >= r.pages);
        } catch (ex) { toast(ex.message); }
      };
      f.addEventListener('submit', e => { e.preventDefault(); page = 1; load(); }); load();
    },
    async suspicious() {
      const load = async risk => { const r = await api('/api/suspicious' + (risk ? `?risk=${risk}` : '')); fillTable('#suspTable', '#suspEmpty', r.items, true); };
      $$('[data-risk]').forEach(b => b.addEventListener('click', () => { $$('[data-risk]').forEach(x => x.classList.remove('active')); b.classList.add('active'); load(b.dataset.risk); }));
      load('');
    },
    async profile() {
      const p = await api('/api/auth/profile');
      $('#p-name').textContent = p.full_name; $('#p-email').textContent = p.email;
      $('#p-since').textContent = new Date(p.member_since).toLocaleDateString(); $('#p-count').textContent = p.transaction_count;
      $('#logoutBtn2').addEventListener('click', logout);
    },
    async analytics() {
      const [s, a, m] = await Promise.all([api('/api/dashboard/stats'), api('/api/analytics'), api('/api/model/status')]);
      const empty = s.total === 0;
      mk('c-split', {type: 'doughnut', data: {labels: ['Legitimate', 'Fraudulent'], datasets: [{data: [s.legitimate, s.fraudulent], backgroundColor: [C.green, C.red], borderWidth: 0}]}, options: {maintainAspectRatio: false, cutout: '65%'}});
      mk('c-trend', {type: 'line', data: {labels: s.trend.dates, datasets: [{label: 'Analyzed', data: s.trend.total, borderColor: C.blue, tension: .35}, {label: 'Fraudulent', data: s.trend.fraudulent, borderColor: C.red, tension: .35}]}, options: {maintainAspectRatio: false, scales: {y: {beginAtZero: true, ticks: {precision: 0}}}}});
      mk('c-risk', {type: 'bar', data: {labels: ['Low', 'Medium', 'High'], datasets: [{data: Object.values(s.risk_distribution), backgroundColor: [C.green, C.amber, C.red], borderRadius: 8}]}, options: {maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {y: {beginAtZero: true, ticks: {precision: 0}}}}});
      const stack = obj => ({labels: Object.keys(obj), datasets: [{label: 'Legitimate', data: Object.values(obj).map(v => v.legitimate), backgroundColor: C.green}, {label: 'Fraudulent', data: Object.values(obj).map(v => v.fraudulent), backgroundColor: C.red}]});
      const so = {maintainAspectRatio: false, scales: {x: {stacked: true}, y: {stacked: true, beginAtZero: true, ticks: {precision: 0}}}};
      mk('c-amount', {type: 'bar', data: stack(a.by_amount), options: so}); mk('c-type', {type: 'bar', data: stack(a.by_type), options: so});
      mk('c-hist', {type: 'bar', data: {labels: a.probability_histogram.map((_, i) => `${i * 10}-${i * 10 + 10}%`), datasets: [{data: a.probability_histogram, backgroundColor: C.cyan, borderRadius: 4}]}, options: {maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {y: {beginAtZero: true, ticks: {precision: 0}}}}});
      if (!empty) $('#avgNote').textContent = `Average amount - flagged: ${a.avg_amount_fraud != null ? money(a.avg_amount_fraud) : 'n/a'} | legitimate: ${a.avg_amount_legit != null ? money(a.avg_amount_legit) : 'n/a'}`;
      else $('#avgNote').textContent = 'No transactions yet. Analyze some to populate these charts.';
      const mp = $('#modelPanel');
      if (!m.available) { mp.innerHTML = `<div class="alert alert-warning mb-0">${esc(m.reason)}</div>`; return; }
      const t = m.test_metrics || {};
      const f = v => v == null ? 'n/a' : Number(v).toFixed(4);
      mp.innerHTML = `<p>Model: <b class="text-white">${esc(m.model_name)}</b> &middot; version ${esc(m.model_version)} &middot; threshold ${m.threshold.toFixed(3)} &middot; dataset ${esc(m.dataset)}</p>
        <div class="table-responsive"><table class="table fg-table table-sm"><thead><tr><th>Model</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>PR-AUC</th><th>Accuracy</th></tr></thead><tbody>
        ${Object.entries(m.comparison).map(([k, v]) => `<tr><td>${esc(k)}${k === m.model_name ? ' (selected)' : ''}</td><td>${f(v.precision)}</td><td>${f(v.recall)}</td><td>${f(v.f1)}</td><td>${f(v.roc_auc)}</td><td>${f(v.pr_auc)}</td><td>${f(v.accuracy)}</td></tr>`).join('')}</tbody></table></div>
        <p class="disclaimer">Held-out test-set metrics from your training run (ml/train.py). Accuracy alone is misleading on imbalanced fraud data; prioritise recall, precision and PR-AUC.</p>`;
      const imgs = [['shap_summary.png', 'SHAP summary'], ['shap_importance.png', 'Global feature importance'], ['confusion_matrix.png', 'Confusion matrix'], ['pr_curves.png', 'Precision-recall curves']];
      $('#shapImgs').innerHTML = imgs.map(([f, n]) => `<div class="col-md-6 col-lg-3"><img class="img-fluid rounded bg-white" alt="${n}" src="/static/img/ml/${f}" onerror="this.parentElement.remove()"><div class="small muted mt-1">${n}</div></div>`).join('');
    }
  };

  function fillTable(tableSel, emptySel, items, withPred) {
    const tb = $(tableSel + ' tbody'); tb.innerHTML = '';
    $(emptySel).classList.toggle('d-none', items.length > 0);
    tb.innerHTML = items.map(t => withPred
      ? `<tr><td>#${t.id}</td><td>${when(t.created_at)}</td><td>${esc(t.type)}</td><td>${money(t.amount)}</td><td>${pill(t.label)}</td><td>${pct(t.fraud_probability)}</td><td>${pill(t.risk_level)}</td><td>${rowLink(t.id)}</td></tr>`
      : `<tr><td>#${t.id}</td><td>${when(t.created_at)}</td><td>${esc(t.type)}</td><td>${money(t.amount)}</td><td>${pct(t.fraud_probability)}</td><td>${pill(t.risk_level)}</td><td>${rowLink(t.id)}</td></tr>`).join('');
  }
  async function logout(e) { e && e.preventDefault(); try { await api('/api/auth/logout', {method: 'POST'}); } catch (x) {} location.href = '/'; }

  document.addEventListener('DOMContentLoaded', () => {
    $$('#logoutBtn').forEach(b => b.addEventListener('click', logout));
    // smooth page transitions
    document.addEventListener('click', e => { const a = e.target.closest('a[href^="/"]');
      if (a && !reduceMotion && !e.ctrlKey && !e.metaKey && a.target !== '_blank' && !a.getAttribute('href').startsWith('#')) {
        e.preventDefault(); document.body.classList.add('leaving'); setTimeout(() => location.href = a.href, 140); } });
    const page = document.body.dataset.page; const key = $('#authForm') ? 'auth' : page;
    if (pages[key]) pages[key]().catch(ex => { if (ex.message !== 'Authentication required') toast(ex.message); });
  });
  window.addEventListener('pageshow', e => { if (e.persisted) document.body.classList.remove('leaving'); });
})();
