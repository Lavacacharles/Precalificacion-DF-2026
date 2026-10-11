// Conversion panel explorer — plain d3 v7, no build step.
// Data: data/clients.csv (one row per client) + data/eda.json (precomputed by ../eda.py)

const NUM = ["edad", "ingresos", "ratio_deuda_ingresos", "antiguedad_cuenta_meses", "numero_productos",
  "saldo_promedio", "dias_ultima_transaccion", "antiguedad_direccion_meses", "visitas_web_ultimos_90_dias",
  "distancia_sucursal_km", "dia_preferido_pago"];
const CAT = ["ocupacion", "region", "canal_adquisicion", "banda_riesgo", "dispositivo_principal"];
const BOOL = ["tiene_tarjeta_credito", "activo_movil", "es_nuevo_cliente", "tiene_prestamo", "tiene_seguro"];
const SERIES = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)"];
const LEVEL_ORDER = { banda_riesgo: ["low", "medium", "high"] };
const pct = d3.format(".1%"), num = d3.format(","), f2 = d3.format(".2f"), f3 = d3.format("+.3f");

// Per-variable reading of what the variable is "describing", shown in the audit header.
const READING = {
  edad: "Uniform 22–71, never ages over the year. A label attached to the client, not a life-cycle driver.",
  ingresos: "Smooth bell with a hard floor at 18,000 (1.6% of clients). Unrelated to occupation, debt, risk, or conversion.",
  ratio_deuda_ingresos: "Flat until ~0.55, then conversion drops for low and medium risk only (high risk is already low). A tail rule; independent of having a loan and of the risk band itself.",
  antiguedad_cuenta_meses: "Uniform 1–179 and frozen across months. Contradicts es_nuevo_cliente (18% of 10-year accounts are 'new').",
  numero_productos: "Strongest structural driver, with a step at 3 products, but only for low-risk clients. Medium/high risk barely respond.",
  saldo_promedio: "Floor at 300 (6.7% of clients) and cap at 90,000. No relation to conversion.",
  dias_ultima_transaccion: "Uniform 1–364 and frozen: it does NOT grow while the client stays, so it is a fixed 'activity level' trait, not a recency. Shape differs by risk band: a HARD STEP at 180 days for high risk (14% → 5%), a gentle linear slope for low/medium. Try it in the Segment explorer split by banda_riesgo.",
  antiguedad_direccion_meses: "Uniform, pure noise.",
  visitas_web_ultimos_90_dias: "Poisson-like (mean 8) for everyone, independent of device or app. Digital intent is not encoded here.",
  distancia_sucursal_km: "Log-normal with an 80 km cap. Independent of region and conversion.",
  dia_preferido_pago: "Uniform 1–28. Placebo variable, flat as it should be.",
  ocupacion: "Same income, age and balance across occupations. A label without economic content.",
  region: "Same branch distance in every region. No geography encoded.",
  canal_adquisicion: "Unrelated to device or app use, and flat for low/medium risk. BUT one hidden rule: high-risk clients acquired at a branch convert ~4 pts less (5.6% vs 9.2%). See tab 7.",
  banda_riesgo: "The dominant variable and a GATE: low-risk clients respond to products, app and recency; high-risk clients convert rarely unless very recently active. Not derived from debt/income.",
  dispositivo_principal: "Unrelated to app activity or web visits. Noise.",
  tiene_tarjeta_credito: "Lift exists ONLY in low risk (16.0% → 19.6%); exactly zero in medium and high. The overall +1.5 pts is a diluted average.",
  activo_movil: "Works only for low-risk clients who already hold 2+ products (e.g. 2 products: 13.7% → 21.6%); no effect for 1-product or high-risk clients.",
  es_nuevo_cliente: "Independent of account tenure and of conversion. The 'new client' flag carries no information.",
  tiene_prestamo: "Independent of debt ratio and conversion.",
  tiene_seguro: "Noise.",
  dias_ultima_interaccion: "The only column that changes month to month, but by being overwritten with random values (share of untouched copies falls from 100% in Jan to 9% in Nov). Its apparent signal is borrowed from dias_ultima_transaccion and vanishes over time.",
};

const tip = d3.select("#tip");
function showTip(ev, html) {
  tip.html(html).style("opacity", 1);
  const w = tip.node().offsetWidth, x = ev.clientX + 14;
  tip.style("left", (x + w > window.innerWidth ? ev.clientX - w - 14 : x) + "px").style("top", (ev.clientY + 12) + "px");
}
const hideTip = () => tip.style("opacity", 0);
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function wilson(k, n) {
  if (!n) return [0, 0];
  const z = 1.96, p = k / n, d = 1 + z * z / n;
  const c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  return [Math.max(0, c - h), c + h];
}

function frame(sel, { w = 600, h = 300, m = { t: 12, r: 16, b: 40, l: 52 }, fixed = false } = {}) {
  const el = d3.select(sel);
  el.selectAll("*").remove();
  const cw = el.node().clientWidth;
  if (!fixed && cw > 200) { h = Math.round(h * Math.min(1, Math.max(0.75, cw / w))); w = cw; }
  const svg = el.append("svg").attr("viewBox", `0 0 ${w} ${h}`);
  const g = svg.append("g").attr("transform", `translate(${m.l},${m.t})`);
  return { el, svg, g, iw: w - m.l - m.r, ih: h - m.t - m.b, m };
}
function yAxis(g, y, iw, fmt = pct, label) {
  g.append("g").attr("class", "gridline").call(d3.axisLeft(y).ticks(5).tickSize(-iw).tickFormat(""));
  g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickFormat(fmt));
  if (label) g.append("text").attr("x", -40).attr("y", -2).attr("text-anchor", "start").text(label).attr("transform", null);
}
function xAxis(g, x, ih, label, iw) {
  const ax = g.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`)
    .call(x.bandwidth ? d3.axisBottom(x) : d3.axisBottom(x).ticks(8));
  if (label) g.append("text").attr("x", iw / 2).attr("y", ih + 34).attr("text-anchor", "middle").text(label);
  return ax;
}
function baseline(g, y, iw, rate) {
  g.append("line").attr("x1", 0).attr("x2", iw).attr("y1", y(rate)).attr("y2", y(rate))
    .attr("stroke", "var(--muted)").attr("stroke-dasharray", "4 4");
  g.append("text").attr("x", 4).attr("y", y(rate) + 13).attr("text-anchor", "start").style("fill", "var(--muted)").text(`base ${pct(rate)}`);
}
function legend(sel, items) {
  d3.select(sel).insert("div", ":first-child").attr("class", "legend")
    .selectAll("span").data(items).join("span").style("--c", d => d.color).text(d => d.label);
}

// ---------- binning & hazard ----------
function binner(data, v, mode = "rate") {
  if (CAT.includes(v) || BOOL.includes(v)) {
    const lv = LEVEL_ORDER[v] || [...new Set(data.map(d => d[v]))].sort();
    return { type: "cat", levels: lv.map(String), key: d => String(d[v]) };
  }
  const vals = data.map(d => d[v]).sort(d3.ascending);
  const uniq = [...new Set(vals)];
  if (uniq.length <= 12) return { type: "cat", levels: uniq.map(String), key: d => String(d[v]) };
  if (mode === "hist") {
    const [a, b] = d3.extent(vals), th = d3.ticks(a, b, 30);
    return { type: "num", thresholds: th, hist: true };
  }
  const th = [...new Set(d3.range(1, 10).map(i => d3.quantileSorted(vals, i / 10)))];
  const edges = [vals[0], ...th, vals[vals.length - 1]];
  const levels = edges.slice(0, -1).map((e, i) => `${fmtV(e)}–${fmtV(edges[i + 1])}`);
  const key = d => { const x = d[v]; let i = d3.bisectRight(th, x); return levels[i]; };
  return { type: "num", levels, key, th };
}
const fmtV = x => Math.abs(x) >= 1000 ? d3.format(".3s")(x) : (Number.isInteger(x) ? x : d3.format(".2f")(x));

function hazardBy(data, keyFn, levels) {
  const roll = d3.rollup(data, a => ({ conv: d3.sum(a, d => d.converted), expo: d3.sum(a, d => d.n_months), clients: a.length }), keyFn);
  return levels.map(l => {
    const r = roll.get(l) || { conv: 0, expo: 0, clients: 0 };
    const [lo, hi] = wilson(r.conv, r.expo);
    return { level: l, ...r, rate: r.expo ? r.conv / r.expo : NaN, lo, hi };
  });
}

// ---------- load ----------
Promise.all([d3.csv("data/clients.csv", d3.autoType), d3.json("data/eda.json")]).then(([clients, eda]) => {
  window._c = clients;
  const base = eda.panel.base_rate;
  const tabs = {
    panel: () => drawPanel(eda),
    audit: () => drawAudit(clients, eda, base),
    integrity: () => drawIntegrity(eda),
    explore: () => drawExplore(clients, base),
    personas: () => drawPersonas(eda, base),
    interaccion: () => drawInteraccion(eda),
    groups: () => drawGroups(),
    guide: () => {},
  };
  const drawn = {};
  function open(t) {
    d3.selectAll("#tabs button").classed("active", function () { return this.dataset.tab === t; });
    d3.selectAll(".tab").classed("active", function () { return this.id === t; });
    if (!drawn[t]) { tabs[t](); drawn[t] = true; }
    try { localStorage.setItem("tab", t); } catch (e) {}
    history.replaceState(null, "", "#" + t);
  }
  d3.selectAll("#tabs button").on("click", function () { open(this.dataset.tab); });
  let start = "guide";
  try { start = localStorage.getItem("tab") || "guide"; } catch (e) {}
  if (location.hash.slice(1) in tabs) start = location.hash.slice(1);
  open(tabs[start] ? start : "guide");
});

// ---------- 1. panel ----------
function drawPanel(eda) {
  const p = eda.panel;
  const frozenShare = Object.values(eda.frozen).filter(v => v === 0).length;
  const tiles = [
    [num(p.n_clients), "unique clients"], [num(p.n_rows), "client-months"],
    [pct(p.base_rate), "monthly conversion rate"],
    [num(p.never_converted), "never convert, all still present in Nov"],
    [`${frozenShare} / ${Object.keys(eda.frozen).length}`, "variables frozen per client across months"],
    [eda.consistency.at(-1).observed, "max |Spearman| between any two features"],
  ];
  d3.select("#tiles").selectAll(".tile").data(tiles).join("div").attr("class", "tile")
    .html(d => `<div class="v">${d[0]}</div><div class="l">${d[1]}</div>`);

  // stacked bars: returning vs new entrants
  const rows = p.by_month.map(d => ({ ...d, returning: d.rows - d.new_entrants }));
  const F = frame("#ch-months");
  const x = d3.scaleBand(rows.map(d => d.t), [0, F.iw]).padding(0.25);
  const y = d3.scaleLinear([0, d3.max(rows, d => d.rows) * 1.05], [F.ih, 0]);
  yAxis(F.g, y, F.iw, d3.format("~s"));
  xAxis(F.g, x, F.ih, "calendar month (2026)", F.iw);
  const stack = d3.stack().keys(["returning", "new_entrants"])(rows);
  const colors = { returning: "var(--s1)", new_entrants: "var(--s2)" };
  F.g.selectAll(".layer").data(stack).join("g").style("fill", d => colors[d.key])
    .selectAll("rect").data(d => d).join("rect")
    .attr("x", d => x(d.data.t)).attr("width", x.bandwidth())
    .attr("y", d => y(d[1])).attr("height", d => Math.max(0, y(d[0]) - y(d[1]) - 2)).attr("rx", 2)
    .on("mousemove", (ev, d) => showTip(ev, `<b>Month ${d.data.t}</b><br>rows ${num(d.data.rows)}<br>already in panel ${num(d.data.returning)}<br>first appearance ${num(d.data.new_entrants)}<br>conversions ${num(d.data.conversions)} (${pct(d.data.rate)})`))
    .on("mouseleave", hideTip);
  legend("#ch-months", [{ label: "already in panel", color: colors.returning }, { label: "first appearance this month", color: colors.new_entrants }]);

  // hazard by k
  const K = frame("#ch-k");
  const kx = d3.scaleLinear([0, 10], [0, K.iw]);
  const ky = d3.scaleLinear([0, 0.22], [K.ih, 0]);
  yAxis(K.g, ky, K.iw);
  xAxis(K.g, kx, K.ih, "months the client has already been in the panel", K.iw);
  baseline(K.g, ky, K.iw, p.base_rate);
  const kd = p.hazard_by_k.map(d => ({ ...d, ci: wilson(d.rate * d.n, d.n) }));
  K.g.append("path").datum(kd).attr("fill", "var(--s1)").attr("opacity", 0.15)
    .attr("d", d3.area().x(d => kx(d.k)).y0(d => ky(d.ci[0])).y1(d => ky(d.ci[1])));
  K.g.append("path").datum(kd).attr("fill", "none").attr("stroke", "var(--s1)").attr("stroke-width", 2)
    .attr("d", d3.line().x(d => kx(d.k)).y(d => ky(d.rate)));
  K.g.selectAll("circle").data(kd).join("circle").attr("cx", d => kx(d.k)).attr("cy", d => ky(d.rate)).attr("r", 4)
    .attr("fill", "var(--s1)").attr("stroke", "var(--surface)").attr("stroke-width", 2)
    .on("mousemove", (ev, d) => showTip(ev, `<b>${d.k} months already in panel</b><br>conversion rate ${pct(d.rate)}<br>95% CI ${pct(d.ci[0])}–${pct(d.ci[1])}<br>${num(d.n)} rows`))
    .on("mouseleave", hideTip);

  // cohort heatmap
  const C = frame("#ch-cohort", { w: 900, h: 330, m: { t: 10, r: 80, b: 40, l: 70 } });
  const cx = d3.scaleBand(d3.range(11), [0, C.iw]).padding(0.04);
  const cy = d3.scaleBand(d3.range(1, 12), [0, C.ih]).padding(0.04);
  const col = d3.scaleSequential(d3.interpolateRgb(cssVar("--seq-lo"), cssVar("--seq-hi"))).domain([0.10, 0.22]).clamp(true);
  C.g.append("g").attr("class", "axis").attr("transform", `translate(0,${C.ih})`).call(d3.axisBottom(cx));
  C.g.append("g").attr("class", "axis").call(d3.axisLeft(cy).tickFormat(t => t === 1 ? "Jan (stock)" : "first seen " + t));
  C.g.append("text").attr("x", C.iw / 2).attr("y", C.ih + 34).attr("text-anchor", "middle").text("months already in the panel");
  C.g.selectAll("rect").data(p.cohort_k).join("rect")
    .attr("x", d => cx(d.k)).attr("y", d => cy(d.t0)).attr("width", cx.bandwidth()).attr("height", cy.bandwidth()).attr("rx", 2)
    .attr("fill", d => col(d.rate))
    .on("mousemove", (ev, d) => showTip(ev, `<b>first seen month ${d.t0}, ${d.k} months in panel</b><br>conversion rate ${pct(d.rate)}<br>${num(d.n)} rows`))
    .on("mouseleave", hideTip);
  C.g.selectAll("text.v").data(p.cohort_k.filter(d => d.n >= 300)).join("text").attr("class", "v")
    .attr("x", d => cx(d.k) + cx.bandwidth() / 2).attr("y", d => cy(d.t0) + cy.bandwidth() / 2 + 4)
    .attr("text-anchor", "middle").style("font-size", "10px")
    .style("fill", d => d.rate > 0.17 ? "var(--surface)" : "var(--ink2)").text(d => d3.format(".0%")(d.rate));
  // legend ramp
  const lg = C.g.append("g").attr("transform", `translate(${C.iw + 16},0)`);
  const ly = d3.scaleLinear([0.10, 0.22], [C.ih, 0]);
  lg.selectAll("rect").data(d3.range(0.10, 0.22, 0.005)).join("rect").attr("x", 0).attr("width", 12)
    .attr("y", d => ly(d + 0.005)).attr("height", d => ly(d) - ly(d + 0.005) + 0.5).attr("fill", d => col(d));
  lg.append("g").attr("class", "axis").attr("transform", "translate(12,0)").call(d3.axisRight(ly).ticks(5).tickFormat(d3.format(".0%")));
}

// ---------- 2. audit ----------
function drawAudit(clients, eda, base) {
  const list = [...eda.audit].sort((a, b) => Math.abs(b.gini) - Math.abs(a.gini));
  const items = d3.select("#var-list").selectAll(".var-item").data(list).join("div").attr("class", "var-item")
    .html(d => `<span>${d.var}<br><span class="badge ${d.verdict}">${d.verdict}</span></span><span class="g">Gini ${f3(d.gini)}</span>`)
    .on("click", (ev, d) => select(d));
  function select(a) {
    items.classed("active", d => d === a);
    const clip = eda.clips.filter(c => c.var === a.var)
      .map(c => `${c.side} = ${fmtV(c.value)} (${pct(c.share_clients)} of clients)`).join("; ");
    d3.select("#audit-head").html(`
      <h3>${a.var} <span class="badge ${a.verdict}">${a.verdict}</span></h3>
      <div class="audit-grid">
        <div><b>A banker would expect</b>${a.expected}</div>
        <div><b>What the data shows</b>${READING[a.var] || ""}</div>
        <div><b>Gini of this variable alone</b>${f3(a.gini)}${a.type === "cat" ? " (levels ordered by their rate, in-sample)" : ""}</div>
        <div><b>Changes over months?</b>${a.time_varying ? "yes, overwritten with noise" : `no (${pct(eda.frozen[a.var] || 0)} of clients vary)`}${clip ? `<br><b style="margin-top:6px">Clipping</b>${clip}` : ""}</div>
      </div>`);
    if (a.var === "dias_ultima_interaccion") { interShare("#ch-dist", eda); interGini("#ch-haz", eda); return; }
    drawDist("#ch-dist", clients, a.var);
    drawHazard("#ch-haz", clients, a.var, base);
  }
  select(list[0]);
}

function drawDist(sel, data, v) {
  const b = binner(data, v, "hist");
  const F = frame(sel, { h: 260 });
  let bars;
  if (b.type === "cat") {
    const cnt = d3.rollup(data, a => a.length, b.key);
    bars = b.levels.map(l => ({ label: l, n: cnt.get(l) || 0 }));
    const x = d3.scaleBand(b.levels, [0, F.iw]).padding(0.25);
    const y = d3.scaleLinear([0, d3.max(bars, d => d.n) * 1.08], [F.ih, 0]);
    yAxis(F.g, y, F.iw, d3.format("~s"));
    xAxis(F.g, x, F.ih, "", F.iw);
    F.g.selectAll("rect").data(bars).join("rect").attr("x", d => x(d.label)).attr("width", x.bandwidth())
      .attr("y", d => y(d.n)).attr("height", d => F.ih - y(d.n)).attr("rx", 2).attr("fill", "var(--s1)")
      .on("mousemove", (ev, d) => showTip(ev, `<b>${d.label}</b><br>${num(d.n)} clients (${pct(d.n / data.length)})`)).on("mouseleave", hideTip);
  } else {
    const vals = data.map(d => d[v]);
    const x = d3.scaleLinear(d3.extent(vals), [0, F.iw]).nice();
    const bins = d3.bin().domain(x.domain()).thresholds(x.ticks(40))(vals);
    const y = d3.scaleLinear([0, d3.max(bins, d => d.length) * 1.08], [F.ih, 0]);
    yAxis(F.g, y, F.iw, d3.format("~s"));
    xAxis(F.g, x, F.ih, v, F.iw);
    F.g.selectAll("rect").data(bins).join("rect").attr("x", d => x(d.x0) + 1).attr("width", d => Math.max(0, x(d.x1) - x(d.x0) - 2))
      .attr("y", d => y(d.length)).attr("height", d => F.ih - y(d.length)).attr("rx", 1).attr("fill", "var(--s1)")
      .on("mousemove", (ev, d) => showTip(ev, `<b>${fmtV(d.x0)}–${fmtV(d.x1)}</b><br>${num(d.length)} clients`)).on("mouseleave", hideTip);
  }
}

function drawHazard(sel, data, v, base) {
  const b = binner(data, v);
  const h = hazardBy(data, b.key, b.levels);
  const F = frame(sel, { h: 260, m: { t: 12, r: 16, b: b.type === "num" ? 60 : 40, l: 52 } });
  const x = d3.scalePoint(b.levels, [0, F.iw]).padding(0.5);
  const y = d3.scaleLinear([0, Math.max(0.32, d3.max(h, d => d.hi) * 1.05)], [F.ih, 0]);
  yAxis(F.g, y, F.iw);
  const ax = xAxis(F.g, x, F.ih, "", F.iw);
  if (b.type === "num") ax.selectAll("text").attr("transform", "rotate(-30)").attr("text-anchor", "end").attr("dx", "-4").attr("dy", "6");
  baseline(F.g, y, F.iw, base);
  F.g.selectAll("line.ci").data(h).join("line").attr("class", "ci").attr("x1", d => x(d.level)).attr("x2", d => x(d.level))
    .attr("y1", d => y(d.lo)).attr("y2", d => y(d.hi)).attr("stroke", "var(--s1)").attr("stroke-width", 2).attr("opacity", 0.45);
  if (b.type === "num") F.g.append("path").datum(h).attr("fill", "none").attr("stroke", "var(--s1)").attr("stroke-width", 2)
    .attr("d", d3.line().x(d => x(d.level)).y(d => y(d.rate)));
  F.g.selectAll("circle").data(h).join("circle").attr("cx", d => x(d.level)).attr("cy", d => y(d.rate)).attr("r", 5)
    .attr("fill", "var(--s1)").attr("stroke", "var(--surface)").attr("stroke-width", 2)
    .on("mousemove", (ev, d) => showTip(ev, `<b>${d.level}</b><br>conversion rate ${pct(d.rate)} (CI ${pct(d.lo)}–${pct(d.hi)})<br>${num(d.clients)} clients · ${num(d.expo)} client-months · ${num(d.conv)} conversions`))
    .on("mouseleave", hideTip);
}

// ---------- 3. integrity ----------
function interShare(sel, eda) {
  const d = eda.interaccion.share_equal_by_t;
  const F = frame(sel, { h: 260 });
  const x = d3.scaleLinear([1, 11], [0, F.iw]), y = d3.scaleLinear([0, 1], [F.ih, 0]);
  yAxis(F.g, y, F.iw, d3.format(".0%"));
  xAxis(F.g, x, F.ih, "calendar month", F.iw);
  F.g.append("path").datum(d).attr("fill", "none").attr("stroke", "var(--s2)").attr("stroke-width", 2).attr("d", d3.line().x(d => x(d.t)).y(d => y(d.share)));
  F.g.selectAll("circle").data(d).join("circle").attr("cx", d => x(d.t)).attr("cy", d => y(d.share)).attr("r", 4)
    .attr("fill", "var(--s2)").attr("stroke", "var(--surface)").attr("stroke-width", 2)
    .on("mousemove", (ev, d) => showTip(ev, `<b>Month ${d.t}</b><br>${pct(d.share)} of rows still equal to dias_ultima_transaccion`)).on("mouseleave", hideTip);
}
function interGini(sel, eda) {
  const d = eda.interaccion.gini_by_t;
  const F = frame(sel, { h: 260 });
  const x = d3.scaleLinear([1, 11], [0, F.iw]), y = d3.scaleLinear([0, 0.15], [F.ih, 0]);
  yAxis(F.g, y, F.iw, d3.format(".2f"));
  xAxis(F.g, x, F.ih, "calendar month", F.iw);
  const s = [{ k: "gini_trans", label: "dias_ultima_transaccion", c: "var(--s1)" }, { k: "gini_inter", label: "dias_ultima_interaccion", c: "var(--s2)" }];
  s.forEach(se => {
    F.g.append("path").datum(d).attr("fill", "none").attr("stroke", se.c).attr("stroke-width", 2).attr("d", d3.line().x(d => x(d.t)).y(d => y(d[se.k])));
    F.g.selectAll(null).data(d).join("circle").attr("cx", d => x(d.t)).attr("cy", d => y(d[se.k])).attr("r", 4)
      .attr("fill", se.c).attr("stroke", "var(--surface)").attr("stroke-width", 2)
      .on("mousemove", (ev, d) => showTip(ev, `<b>Month ${d.t}</b><br>transaction Gini ${f2(d.gini_trans)}<br>interaction Gini ${f2(d.gini_inter)}`)).on("mouseleave", hideTip);
  });
  legend(sel, s.map(se => ({ label: se.label, color: se.c })));
}

function drawIntegrity(eda) {
  // correlation heatmap
  const { cols, matrix } = eda.corr;
  const short = c => c.replace("_ultimos_90_dias", "").replace("antiguedad_", "ant_").replace("_meses", "").replace("tiene_", "").replace("_ingresos", "").replace("dias_ultima_", "dias_").replace("_promedio", "").replace("distancia_sucursal_km", "distancia").replace("_ord", "");
  const F = frame("#ch-corr", { w: 620, h: 560, m: { t: 10, r: 10, b: 130, l: 130 } });
  const x = d3.scaleBand(cols, [0, F.iw]), y = d3.scaleBand(cols, [0, F.ih]);
  const col = d3.scaleLinear([-0.5, 0, 0.5], [cssVar("--div-neg"), cssVar("--div-mid"), cssVar("--div-pos")]).clamp(true);
  const cells = cols.flatMap((a, i) => cols.map((b, j) => ({ a, b, v: matrix[i][j], diag: i === j })));
  F.g.selectAll("rect").data(cells).join("rect").attr("x", d => x(d.b) + 1).attr("y", d => y(d.a) + 1)
    .attr("width", x.bandwidth() - 2).attr("height", y.bandwidth() - 2).attr("rx", 2)
    .attr("fill", d => d.diag ? "var(--grid)" : col(d.v))
    .on("mousemove", (ev, d) => showTip(ev, `<b>${d.a}</b> × <b>${d.b}</b><br>ρ = ${d3.format("+.3f")(d.v)}`)).on("mouseleave", hideTip);
  F.g.append("g").attr("class", "axis").call(d3.axisLeft(y).tickFormat(short).tickSize(0)).select("path").remove();
  F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x).tickFormat(short).tickSize(0))
    .call(g => g.select("path").remove()).selectAll("text").attr("transform", "rotate(-50)").attr("text-anchor", "end").attr("dx", "-6").attr("dy", "4");
  d3.select("#ch-corr").append("p").attr("class", "note").text("Color scale spans ρ = −0.5 (red) to +0.5 (blue). Every off-diagonal cell is gray: |ρ| ≤ 0.021.");

  d3.select("#tb-consistency").html("<tr><th>Check</th><th>Expected</th><th>Observed</th></tr>" +
    eda.consistency.map(c => `<tr><td>${c.check}</td><td>${c.expected}</td><td>${c.observed}</td></tr>`).join(""));
  d3.select("#tb-clips").html("<tr><th>Variable</th><th>Bound</th><th>Clients at bound</th><th>Hazard there</th></tr>" +
    eda.clips.map(c => `<tr><td>${c.var}</td><td>${c.side} = ${fmtV(c.value)}</td><td>${pct(c.share_clients)}</td><td>${pct(c.rate)}</td></tr>`).join(""));

  interShare("#ch-inter", eda);
  interGini("#ch-inter-gini", eda);

  // frozen
  const fr = Object.entries(eda.frozen).map(([k, v]) => ({ k, v })).sort((a, b) => b.v - a.v);
  const G = frame("#ch-frozen", { w: 900, h: 420, m: { t: 6, r: 50, b: 30, l: 200 } });
  const yx = d3.scaleBand(fr.map(d => d.k), [0, G.ih]).padding(0.25), xx = d3.scaleLinear([0, 1], [0, G.iw]);
  G.g.append("g").attr("class", "gridline").attr("transform", `translate(0,${G.ih})`).call(d3.axisBottom(xx).ticks(5).tickSize(-G.ih).tickFormat(""));
  G.g.append("g").attr("class", "axis").attr("transform", `translate(0,${G.ih})`).call(d3.axisBottom(xx).ticks(5).tickFormat(d3.format(".0%")));
  G.g.append("g").attr("class", "axis").call(d3.axisLeft(yx).tickSize(0)).select("path").remove();
  G.g.selectAll("rect").data(fr).join("rect").attr("y", d => yx(d.k)).attr("height", yx.bandwidth()).attr("x", 0)
    .attr("width", d => Math.max(d.v > 0 ? 2 : 0, xx(d.v))).attr("rx", 2).attr("fill", d => d.v > 0 ? "var(--s2)" : "var(--s1)");
  G.g.selectAll("text.lab").data(fr).join("text").attr("class", "lab").attr("x", d => xx(d.v) + 6).attr("y", d => yx(d.k) + yx.bandwidth() / 2 + 4).text(d => pct(d.v));
}

// ---------- 4. explorer ----------
function drawExplore(clients, base) {
  const allVars = [...NUM, ...CAT, ...BOOL];
  const splitVars = ["(none)", ...CAT, ...BOOL, "productos_grupo"];
  clients.forEach(d => d.productos_grupo = d.numero_productos >= 3 ? "3+" : String(d.numero_productos));
  LEVEL_ORDER.productos_grupo = ["1", "2", "3+"];
  const state = { x: "dias_ultima_transaccion", split: "banda_riesgo", risk: new Set(["low", "medium", "high"]), prod: new Set(["1", "2", "3+"]) };
  d3.select("#sel-x").selectAll("option").data(allVars).join("option").attr("value", d => d).text(d => d).property("selected", d => d === state.x);
  d3.select("#sel-split").selectAll("option").data(splitVars).join("option").attr("value", d => d).text(d => d).property("selected", d => d === state.split);
  d3.select("#sel-x").on("change", function () { state.x = this.value; render(); });
  d3.select("#sel-split").on("change", function () { state.split = this.value; render(); });
  function chips(sel, values, set) {
    d3.select(sel).selectAll("button").data(values).join("button").attr("class", "chip on").text(d => d)
      .on("click", function (ev, d) { set.has(d) ? set.delete(d) : set.add(d); d3.select(this).classed("on", set.has(d)); render(); });
  }
  chips("#chips-risk", ["low", "medium", "high"], state.risk);
  chips("#chips-prod", ["1", "2", "3+"], state.prod);

  function render() {
    const data = clients.filter(d => state.risk.has(d.banda_riesgo) && state.prod.has(d.productos_grupo));
    const b = binner(clients, state.x);
    let groups;
    if (state.split === "(none)") groups = [{ label: "all", rows: data }];
    else {
      const lv = LEVEL_ORDER[state.split] || [...new Set(clients.map(d => String(d[state.split])))].sort();
      groups = lv.map(l => ({ label: l, rows: data.filter(d => String(d[state.split]) === l) })).filter(g => g.rows.length);
    }
    groups.forEach((g, i) => { g.color = SERIES[i % SERIES.length]; g.h = hazardBy(g.rows, b.key, b.levels); });
    d3.select("#ex-title").text(`Monthly conversion rate by ${state.x}${state.split !== "(none)" ? ` split by ${state.split}` : ""}`);
    const F = frame("#ch-explore", { w: 960, h: 380, m: { t: 14, r: 20, b: b.type === "num" ? 70 : 40, l: 56 } });
    const x = d3.scalePoint(b.levels, [0, F.iw]).padding(0.5);
    const maxHi = d3.max(groups.flatMap(g => g.h.filter(d => d.expo >= 200).map(d => d.hi))) || 0.3;
    const y = d3.scaleLinear([0, Math.max(0.3, maxHi * 1.05)], [F.ih, 0]).clamp(true);
    yAxis(F.g, y, F.iw);
    const ax = xAxis(F.g, x, F.ih, "", F.iw);
    if (b.type === "num") ax.selectAll("text").attr("transform", "rotate(-30)").attr("text-anchor", "end").attr("dx", "-4").attr("dy", "6");
    baseline(F.g, y, F.iw, base);
    const off = (i) => (i - (groups.length - 1) / 2) * Math.min(6, x.step() / (groups.length + 2));
    groups.forEach((g, i) => {
      const pts = g.h.filter(d => d.expo > 0);
      const gg = F.g.append("g").attr("transform", `translate(${off(i)},0)`);
      gg.selectAll("line").data(pts).join("line").attr("x1", d => x(d.level)).attr("x2", d => x(d.level))
        .attr("y1", d => y(d.lo)).attr("y2", d => y(d.hi)).attr("stroke", g.color).attr("stroke-width", 2).attr("opacity", 0.4);
      gg.append("path").datum(pts).attr("fill", "none").attr("stroke", g.color).attr("stroke-width", 2)
        .attr("d", d3.line().x(d => x(d.level)).y(d => y(d.rate)));
      gg.selectAll("circle").data(pts).join("circle").attr("cx", d => x(d.level)).attr("cy", d => y(d.rate)).attr("r", 4.5)
        .attr("fill", g.color).attr("stroke", "var(--surface)").attr("stroke-width", 2)
        .on("mousemove", (ev, d) => showTip(ev, `<b>${state.split !== "(none)" ? state.split + " = " + g.label + "<br>" : ""}${state.x}: ${d.level}</b><br>conversion rate ${pct(d.rate)} (CI ${pct(d.lo)}–${pct(d.hi)})<br>${num(d.clients)} clients · ${num(d.conv)} conversions`))
        .on("mouseleave", hideTip);
      if (groups.length <= 4) {
        const last = pts[pts.length - 1];
        if (last) gg.append("text").attr("x", x(last.level) + 8).attr("y", y(last.rate) + 4).style("fill", "var(--ink2)").text(g.label);
      }
    });
    if (groups.length > 1) legend("#ch-explore", groups.map(g => ({ label: `${g.label} (${num(g.rows.length)})`, color: g.color })));
    d3.select("#ex-note").text(`${num(data.length)} clients after filters. Rates use every client-month in each bin; whiskers are 95% Wilson intervals.`);
  }
  render();
}

// ---------- 5. personas ----------
function drawPersonas(eda, base) {
  const ORDER = { risk: ["low", "medium", "high"], products: ["1", "2", "3+"], recency: ["<=120d", "121-240d", ">240d"], app: [1, 0] };
  const lab = { risk: v => `${v} risk`, products: v => `${v} prod`, recency: v => `txn ${v}`, app: v => v ? "app active" : "no app" };
  const riskColor = { low: "var(--s1)", medium: "var(--s2)", high: "var(--s3)" };
  function render(keys) {
    const roll = d3.rollups(eda.segments, a => ({ rows: d3.sum(a, d => d.rows), clients: d3.sum(a, d => d.clients), conv: d3.sum(a, d => d.rate * d.rows) }), d => keys.map(k => d[k]).join("|"));
    const data = roll.map(([key, r]) => {
      const parts = key.split("|"); const o = {}; keys.forEach((k, i) => o[k] = k === "app" ? +parts[i] : parts[i]);
      const [lo, hi] = wilson(r.conv, r.rows);
      return { ...o, ...r, rate: r.conv / r.rows, lo, hi, label: keys.map(k => lab[k](o[k])).join(" · ") };
    }).sort((a, b) => b.rate - a.rate);
    const rowH = 22, F = frame("#ch-persona", { w: 960, h: data.length * rowH + 50, m: { t: 6, r: 170, b: 34, l: 290 } });
    const y = d3.scaleBand(data.map(d => d.label), [0, F.ih]).padding(0.2);
    const x = d3.scaleLinear([0, Math.max(0.35, d3.max(data, d => d.hi))], [0, F.iw]);
    F.g.append("g").attr("class", "gridline").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x).ticks(7).tickSize(-F.ih).tickFormat(""));
    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`).call(d3.axisBottom(x).ticks(7).tickFormat(d3.format(".0%")));
    F.g.append("g").attr("class", "axis").call(d3.axisLeft(y).tickSize(0)).select("path").remove();
    F.g.selectAll("rect").data(data).join("rect").attr("y", d => y(d.label)).attr("height", y.bandwidth()).attr("x", 0)
      .attr("width", d => x(d.rate)).attr("rx", 3).attr("fill", d => riskColor[d.risk])
      .on("mousemove", (ev, d) => showTip(ev, `<b>${d.label}</b><br>conversion rate ${pct(d.rate)} (CI ${pct(d.lo)}–${pct(d.hi)})<br>${num(d.clients)} clients · ${num(d.rows)} client-months<br>≈ ${f2(1 / d.rate)} months to convert on average`))
      .on("mouseleave", hideTip);
    F.g.selectAll("line.ci").data(data).join("line").attr("class", "ci").attr("y1", d => y(d.label) + y.bandwidth() / 2).attr("y2", d => y(d.label) + y.bandwidth() / 2)
      .attr("x1", d => x(d.lo)).attr("x2", d => x(d.hi)).attr("stroke", "var(--ink2)").attr("stroke-width", 1.5).attr("opacity", 0.6);
    F.g.selectAll("text.lab").data(data).join("text").attr("class", "lab").attr("x", d => x(Math.max(d.hi, d.rate)) + 6)
      .attr("y", d => y(d.label) + y.bandwidth() / 2 + 4).text(d => `${pct(d.rate)} · ${num(d.clients)} clients · ~${d3.format(".1f")(1 / d.rate)} mo`);
    F.g.append("line").attr("x1", x(base)).attr("x2", x(base)).attr("y1", 0).attr("y2", F.ih).attr("stroke", "var(--muted)").attr("stroke-dasharray", "4 4");
    legend("#ch-persona", Object.entries(riskColor).map(([k, c]) => ({ label: `${k} risk`, color: c })));
  }
  const sel = d3.select("#sel-persona").on("change", function () { render(this.value.split(",")); });
  render(sel.property("value").split(","));
}

// ---------- 6. dias_ultima_interaccion ----------
const MONTH = "JFMAMJJASON";
function drawInteraccion(eda) {
  const I = eda.interaccion;
  const wrap = d3.select("#ch-examples");
  wrap.selectAll("*").remove();
  I.examples.forEach(ex => {
    const cell = wrap.append("div").attr("class", "sm");
    cell.append("div").attr("class", "sm-title").text(`client ${ex.id} · dias_ultima_transaccion = ${ex.rows[0].trans}`);
    const F = frame(cell.append("div").node(), { w: 380, h: 190, m: { t: 8, r: 10, b: 28, l: 40 }, fixed: true });
    const x = d3.scaleLinear([1, 11], [0, F.iw]), y = d3.scaleLinear([0, 365], [F.ih, 0]);
    yAxis(F.g, y, F.iw, d3.format("d"));
    F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`)
      .call(d3.axisBottom(x).ticks(11).tickFormat(d => MONTH[d - 1]));
    F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(ex.rows[0].trans)).attr("y2", y(ex.rows[0].trans))
      .attr("stroke", "var(--muted)").attr("stroke-width", 2);
    F.g.append("path").datum(ex.rows).attr("fill", "none").attr("stroke", "var(--s2)").attr("stroke-width", 1).attr("opacity", 0.4)
      .attr("d", d3.line().x(d => x(d.t)).y(d => y(d.inter)));
    F.g.selectAll("circle").data(ex.rows).join("circle").attr("cx", d => x(d.t)).attr("cy", d => y(d.inter)).attr("r", 5)
      .attr("fill", d => d.inter === d.trans ? "var(--s2)" : "var(--surface)").attr("stroke", "var(--s2)").attr("stroke-width", 2)
      .on("mousemove", (ev, d) => showTip(ev, `<b>month ${d.t}</b><br>transaction ${d.trans} · interaction ${d.inter}<br>${d.inter === d.trans ? "intact copy" : "overwritten"}`))
      .on("mouseleave", hideTip);
  });
  legend("#ch-examples", [{ label: "dias_ultima_transaccion (frozen)", color: "var(--muted)" },
    { label: "dias_ultima_interaccion (filled = intact copy, hollow = overwritten)", color: "var(--s2)" }]);

  // observed share of intact copies vs formula
  const F = frame("#ch-formula", { h: 280 });
  const x = d3.scaleLinear([1, 12], [0, F.iw]), y = d3.scaleLinear([0, 1], [F.ih, 0]);
  yAxis(F.g, y, F.iw, d3.format(".0%"));
  F.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F.ih})`)
    .call(d3.axisBottom(x).ticks(12).tickFormat(d => d === 12 ? "Dec" : MONTH[d - 1]));
  F.g.append("path").datum(I.formula).attr("fill", "none").attr("stroke", "var(--muted)").attr("stroke-width", 2).attr("stroke-dasharray", "5 4")
    .attr("d", d3.line().x(d => x(d.t)).y(d => y(d.share)));
  F.g.selectAll("circle").data(I.share_equal_by_t).join("circle").attr("cx", d => x(d.t)).attr("cy", d => y(d.share)).attr("r", 5)
    .attr("fill", "var(--s2)").attr("stroke", "var(--surface)").attr("stroke-width", 2)
    .on("mousemove", (ev, d) => showTip(ev, `<b>month ${d.t}</b><br>observed ${pct(d.share)} intact<br>formula ${pct(I.formula[d.t - 1].share)}`))
    .on("mouseleave", hideTip);
  F.g.append("text").attr("x", x(12)).attr("y", y(0.3)).attr("text-anchor", "end").text("Dec (test): 0% intact →");
  legend("#ch-formula", [{ label: "observed", color: "var(--s2)" }, { label: "formula (12 − month) ÷ 11", color: "var(--muted)" }]);

  // conversion rate: intact vs overwritten, within transaction ranges
  const R = frame("#ch-samevsrep", { h: 280 });
  const bins = I.rate_same_vs_replaced.map(d => d.bin);
  const label = b => b.replace(/[(\]]/g, "").replace(", ", "–") + " days";
  const x0 = d3.scaleBand(bins, [0, R.iw]).padding(0.3);
  const x1 = d3.scaleBand(["intact", "replaced"], [0, x0.bandwidth()]).padding(0.1);
  const yy = d3.scaleLinear([0, 0.25], [R.ih, 0]);
  yAxis(R.g, yy, R.iw);
  R.g.append("g").attr("class", "axis").attr("transform", `translate(0,${R.ih})`).call(d3.axisBottom(x0).tickFormat(label));
  R.g.append("text").attr("x", R.iw / 2).attr("y", R.ih + 34).attr("text-anchor", "middle").text("dias_ultima_transaccion");
  const cols = { intact: "var(--s1)", replaced: "var(--s2)" };
  I.rate_same_vs_replaced.forEach(d => ["intact", "replaced"].forEach(k => {
    R.g.append("rect").attr("x", x0(d.bin) + x1(k)).attr("width", x1.bandwidth())
      .attr("y", yy(d[k])).attr("height", R.ih - yy(d[k])).attr("rx", 2).attr("fill", cols[k])
      .on("mousemove", ev => showTip(ev, `<b>${label(d.bin)}</b><br>${k === "intact" ? "intact copy" : "overwritten"}: ${pct(d[k])}`))
      .on("mouseleave", hideTip);
  }));
  legend("#ch-samevsrep", [{ label: "intact copy", color: cols.intact }, { label: "overwritten", color: cols.replaced }]);

  interGini("#ch-inter-gini2", eda);
}

// ---------- 7. groups vs reality ----------
const PAY_MONTHS = [{ t: [2, 3], label: "school" }, { t: [5], label: "CTS" }, { t: [7], label: "grati" }, { t: [11], label: "CTS" }];
function payBands(g, x, ih) {
  PAY_MONTHS.forEach(p => {
    const x0 = x(d3.min(p.t) - 0.4), x1 = x(d3.max(p.t) + 0.4);
    g.append("rect").attr("x", x0).attr("width", x1 - x0).attr("y", 0).attr("height", ih).attr("fill", "var(--grid)").attr("opacity", 0.7);
    g.append("text").attr("x", (x0 + x1) / 2).attr("y", 10).attr("text-anchor", "middle").style("fill", "var(--muted)").style("font-size", "10px").text(p.label);
  });
}
const pfmt = p => p < 0.001 ? "<0.001" : d3.format(".3f")(p);

function drawGroups() {
  d3.json("data/groups.json").then(G => {
    const fp = G.fingerprint, se = G.seasonality;
    const sigGM = G.group_month.tests.filter(t => t.p < 0.05).length;
    d3.select("#g-tiles").selectAll(".tile").data([
      [`${d3.format(".1%")(fp.share_p05)}`, `of ${fp.n} attribute-pair tests have p < 0.05 (5% = pure chance)`],
      [`${G.reality.filter(r => r.strength === "strong").length} + ${G.reality.filter(r => r.strength === "moderate").length}`, `of ${G.reality.length} Peru reality checks fail: strong + moderate (has an innocent explanation)`],
      [`${sigGM} / ${G.group_month.tests.length}`, "groups with their own seasonal pattern"],
      [d3.format("+.3f")(se.calendar.grati.coef), `July (gratificación) effect, p = ${se.calendar.grati.p.toFixed(2)}`],
    ]).join("div").attr("class", "tile").html(d => `<div class="v">${d[0]}</div><div class="l">${d[1]}</div>`);

    // fingerprint histogram
    const F = frame("#ch-fp", { h: 280 });
    const bins = d3.bin().domain([0, 1]).thresholds(d3.range(0.05, 1, 0.05))(fp.tests.map(t => t.p));
    const x = d3.scaleLinear([0, 1], [0, F.iw]), y = d3.scaleLinear([0, 30], [F.ih, 0]);
    yAxis(F.g, y, F.iw, d3.format("d"));
    xAxis(F.g, x, F.ih, "p-value of the dependence test", F.iw);
    F.g.selectAll("rect").data(bins).join("rect").attr("x", d => x(d.x0) + 1).attr("width", d => x(d.x1) - x(d.x0) - 2)
      .attr("y", d => y(d.length)).attr("height", d => F.ih - y(d.length)).attr("rx", 2).attr("fill", (d, i) => i === 0 ? "var(--s2)" : "var(--s1)")
      .on("mousemove", (ev, d) => showTip(ev, `<b>p in ${d.x0.toFixed(2)}–${d.x1.toFixed(2)}</b><br>${d.length} tests<br>${fp.tests.filter(t => t.p >= d.x0 && t.p < d.x1).slice(0, 4).map(t => `${t.a} × ${t.b}`).join("<br>")}`))
      .on("mouseleave", hideTip);
    const exp = fp.n / 20;
    F.g.append("line").attr("x1", 0).attr("x2", F.iw).attr("y1", y(exp)).attr("y2", y(exp)).attr("stroke", "var(--muted)").attr("stroke-dasharray", "4 4");
    F.g.append("text").attr("x", F.iw).attr("y", 8).attr("text-anchor", "end").style("fill", "var(--muted)").text(`expected if independent: ${exp.toFixed(1)} per bar`);
    legend("#ch-fp", [{ label: "p < 0.05 (\"significant\")", color: "var(--s2)" }, { label: "p ≥ 0.05", color: "var(--s1)" }]);

    // seasonality ratio
    const S = frame("#ch-season", { h: 280 });
    const sx = d3.scaleLinear([0.5, 11.5], [0, S.iw]), sy = d3.scaleLinear([0.8, 1.15], [S.ih, 0]);
    payBands(S.g, sx, S.ih);
    yAxis(S.g, sy, S.iw, d3.format(".2f"));
    S.g.append("g").attr("class", "axis").attr("transform", `translate(0,${S.ih})`).call(d3.axisBottom(sx).ticks(11).tickFormat(d => MONTH[d - 1]));
    S.g.append("line").attr("x1", 0).attr("x2", S.iw).attr("y1", sy(1)).attr("y2", sy(1)).attr("stroke", "var(--muted)").attr("stroke-dasharray", "4 4");
    const md = se.monthly.map(d => { const [lo, hi] = wilson(d.obs * d.n, d.n); return { ...d, lo: lo / d.pred, hi: hi / d.pred }; });
    S.g.selectAll("line.ci").data(md).join("line").attr("class", "ci").attr("x1", d => sx(d.t)).attr("x2", d => sx(d.t))
      .attr("y1", d => sy(d.lo)).attr("y2", d => sy(d.hi)).attr("stroke", "var(--s1)").attr("stroke-width", 2).attr("opacity", 0.4);
    S.g.append("path").datum(md).attr("fill", "none").attr("stroke", "var(--s1)").attr("stroke-width", 2).attr("d", d3.line().x(d => sx(d.t)).y(d => sy(d.ratio)));
    S.g.selectAll("circle").data(md).join("circle").attr("cx", d => sx(d.t)).attr("cy", d => sy(d.ratio)).attr("r", 4.5)
      .attr("fill", "var(--s1)").attr("stroke", "var(--surface)").attr("stroke-width", 2)
      .on("mousemove", (ev, d) => showTip(ev, `<b>month ${d.t}</b><br>observed ${pct(d.obs)} · expected ${pct(d.pred)}<br>ratio ${f2(d.ratio)}`)).on("mouseleave", hideTip);

    // group x month lines
    const groups = G.group_month.tests.map(t => t.group);
    const sel = d3.select("#sel-gm").on("change", function () { drawGM(this.value); });
    sel.selectAll("option").data(groups).join("option").attr("value", d => d).text(d => d);
    function drawGM(g) {
      const t = G.group_month.tests.find(d => d.group === g);
      d3.select("#gm-test").html(`group × month test: χ² = ${t.lr.toFixed(1)} on ${t.df} df, p = ${pfmt(t.p)} ${t.p < 0.05 ? "→ own calendar" : "→ no own calendar"}`);
      const lines = G.group_month.lines[g];
      const F2 = frame("#ch-gm", { w: 960, h: 320 });
      const gx = d3.scaleLinear([0.5, 11.5], [0, F2.iw]), gy = d3.scaleLinear([0.7, 1.3], [F2.ih, 0]).clamp(true);
      payBands(F2.g, gx, F2.ih);
      yAxis(F2.g, gy, F2.iw, d3.format(".1f"));
      F2.g.append("g").attr("class", "axis").attr("transform", `translate(0,${F2.ih})`).call(d3.axisBottom(gx).ticks(11).tickFormat(d => MONTH[d - 1]));
      F2.g.append("line").attr("x1", 0).attr("x2", F2.iw).attr("y1", gy(1)).attr("y2", gy(1)).attr("stroke", "var(--muted)").attr("stroke-dasharray", "4 4");
      lines.forEach((l, i) => {
        const c = SERIES[i % SERIES.length];
        F2.g.append("path").datum(l.points).attr("fill", "none").attr("stroke", c).attr("stroke-width", 2).attr("d", d3.line().x(d => gx(d.t)).y(d => gy(d.ratio)));
        F2.g.selectAll(null).data(l.points).join("circle").attr("cx", d => gx(d.t)).attr("cy", d => gy(d.ratio)).attr("r", 4)
          .attr("fill", c).attr("stroke", "var(--surface)").attr("stroke-width", 2)
          .on("mousemove", (ev, d) => showTip(ev, `<b>${g} = ${l.level}</b><br>month ${d.t}: ratio ${f2(d.ratio)}<br>${num(d.n)} rows`)).on("mouseleave", hideTip);
      });
      legend("#ch-gm", lines.map((l, i) => ({ label: l.level, color: SERIES[i % SERIES.length] })));
    }
    drawGM("tipo_trabajador");

    d3.select("#tb-reality").html("<tr><th>Topic</th><th>Expected in Peru / real banking</th><th>Observed in train.csv</th><th>Strength of evidence</th></tr>" +
      G.reality.map(r => `<tr><td><b>${r.topic}</b></td><td>${r.expected}</td><td>${r.observed}</td><td><span class="badge ${r.strength === "strong" ? "corrupted" : "weak"}">${r.strength}</span><br><span class="note">${r.caveat}</span></td></tr>`).join(""));

    d3.select("#tb-gs").html("<tr><th>Group</th><th>Main effect</th><th>× drivers</th></tr>" +
      G.group_signal.map(r => `<tr><td>${r.group}</td><td>χ² ${r.main[0].toFixed(1)} / ${r.main[1]} df, p ${pfmt(r.main[2])}</td><td${r.interaction[2] < 0.001 ? ' style="font-weight:600"' : ""}>χ² ${r.interaction[0].toFixed(1)} / ${r.interaction[1]} df, p ${pfmt(r.interaction[2])}</td></tr>`).join(""));

    // branch x high risk
    const B = frame("#ch-branch", { h: 280 });
    const recs = ["<=180 days", ">180 days"], chans = ["branch", "other channels"];
    const bx0 = d3.scaleBand(recs, [0, B.iw]).padding(0.3), bx1 = d3.scaleBand(chans, [0, bx0.bandwidth()]).padding(0.1);
    const by = d3.scaleLinear([0, 0.2], [B.ih, 0]);
    yAxis(B.g, by, B.iw);
    B.g.append("g").attr("class", "axis").attr("transform", `translate(0,${B.ih})`).call(d3.axisBottom(bx0).tickFormat(d => "last transaction " + d));
    const bc = { branch: "var(--s2)", "other channels": "var(--s1)" };
    G.branch_high.forEach(d => {
      const [lo, hi] = wilson(d.rate * d.rows, d.rows);
      B.g.append("rect").attr("x", bx0(d.rec) + bx1(d.canal)).attr("width", bx1.bandwidth()).attr("y", by(d.rate)).attr("height", B.ih - by(d.rate)).attr("rx", 2).attr("fill", bc[d.canal])
        .on("mousemove", ev => showTip(ev, `<b>high risk · ${d.canal} · ${d.rec}</b><br>conversion ${pct(d.rate)} (95%: ${pct(lo)}–${pct(hi)})<br>${num(d.clients)} clients, ${num(d.rows)} rows`)).on("mouseleave", hideTip);
      B.g.append("text").attr("x", bx0(d.rec) + bx1(d.canal) + bx1.bandwidth() / 2).attr("y", by(d.rate) - 5).attr("text-anchor", "middle").style("fill", "var(--ink2)").text(pct(d.rate));
    });
    legend("#ch-branch", chans.map(c => ({ label: c, color: bc[c] })));
  });
}
