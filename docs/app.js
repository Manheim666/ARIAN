/* ═══════════════════════════════════════════════════════════
   ARIAN Wildfire & Weather Intelligence — app.js
   ═══════════════════════════════════════════════════════════ */

const RISK = {
  Low:      { color: "#22a66e", bg: "rgba(34,166,110,.08)" },
  Moderate: { color: "#daa520", bg: "rgba(218,165,32,.08)" },
  High:     { color: "#e06730", bg: "rgba(224,103,48,.08)" },
  Extreme:  { color: "#c0392b", bg: "rgba(192,57,43,.08)" },
};
const riskColor = (lvl) => (RISK[lvl] || RISK.Low).color;

let forecast = [], hourlyForecast = [], metrics = {};
let selectedRegion = "Baku", selectedDate = "";
let forecastMode = "daily";
let map, markers = [];

const fmt  = (v) => `${Math.round(v * 100)}%`;
const f1   = (v) => Number(v).toFixed(1);
const CIRC = 2 * Math.PI * 52; // SVG ring circumference

/* ─── DATA LOADING ────────────────────────────────────── */
async function loadData() {
  const [fRes, mRes, hRes] = await Promise.all([
    fetch("./data/forecast_30_days.json"),
    fetch("./data/metrics.json"),
    fetch("./data/hourly_forecast_168h.json"),
  ]);
  if (!fRes.ok || !mRes.ok || !hRes.ok) throw new Error("Data file not found (HTTP " + [fRes.status,mRes.status,hRes.status].join("/") + ")");
  forecast       = await fRes.json();
  metrics        = await mRes.json();
  hourlyForecast = await hRes.json();
  selectedDate   = forecast[0].date;
  selectedRegion = forecast[0].region;
  initControls();
  initMap();
  renderAll();
}

/* ─── CONTROLS ────────────────────────────────────────── */
function syncAllRegionSelects() {
  ["heroRegionSelect","regionSelect","panelRegionSelect"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = selectedRegion;
  });
}

function initControls() {
  const dates   = [...new Set(forecast.map(d => d.date))];
  const regions = [...new Set(forecast.map(d => d.region))].sort();
  const regOpts = regions.map(r => `<option value="${r}">${r}</option>`).join("");

  // populate all region selects
  ["heroRegionSelect","regionSelect","panelRegionSelect"].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.innerHTML = regOpts; el.value = selectedRegion; }
  });

  // date picker
  const dp = document.getElementById("datePicker");
  dp.innerHTML = dates.map(d => {
    const lbl = new Date(d+"T00:00").toLocaleDateString("en-US",{month:"short",day:"numeric"});
    return `<option value="${d}">${lbl}</option>`;
  }).join("");
  dp.value = selectedDate;

  // table date filter
  const tdf = document.getElementById("tableDateFilter");
  tdf.innerHTML = '<option value="">All dates</option>' + dates.map(d => {
    const lbl = new Date(d+"T00:00").toLocaleDateString("en-US",{month:"short",day:"numeric"});
    return `<option value="${d}">${lbl}</option>`;
  }).join("");

  // listeners
  dp.onchange = e => { selectedDate = e.target.value; renderAll(); };
  ["heroRegionSelect","regionSelect","panelRegionSelect"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.onchange = e => { selectedRegion = e.target.value; syncAllRegionSelects(); renderAll(); };
  });

  document.getElementById("toggleDaily").onclick  = () => setMode("daily");
  document.getElementById("toggleHourly").onclick = () => setMode("hourly");
  document.getElementById("riskFilter").onchange  = renderTable;
  tdf.onchange = renderTable;

  document.getElementById("modelStatus").textContent =
    `${metrics.selected_model} · ${metrics.prediction_horizon_days}-day forecast`;
}

/* ─── MAP ─────────────────────────────────────────────── */
function initMap() {
  map = L.map("map", { zoomControl: true }).setView([40.35, 47.8], 7);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 18, attribution: "CartoDB"
  }).addTo(map);
}

function dayRows() { return forecast.filter(d => d.date === selectedDate); }
function currentRow() {
  return forecast.find(d => d.region === selectedRegion && d.date === selectedDate)
      || forecast.find(d => d.region === selectedRegion)
      || forecast[0];
}

function renderMap() {
  markers.forEach(m => m.remove()); markers = [];
  dayRows().forEach(row => {
    const sel = row.region === selectedRegion;
    const c = L.circle([row.Latitude, row.Longitude], {
      radius: 22000, color: riskColor(row.risk_level),
      weight: sel ? 3 : 1, fillColor: riskColor(row.risk_level),
      fillOpacity: sel ? 0.22 : 0.10,
    }).addTo(map);
    const m = L.marker([row.Latitude, row.Longitude], {
      icon: L.divIcon({ className: "", iconSize: [22,22], iconAnchor: [11,11],
        html: `<div style="width:${sel?22:16}px;height:${sel?22:16}px;border-radius:50%;background:${riskColor(row.risk_level)};border:3px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,.25)"></div>` })
    }).addTo(map);
    m.bindTooltip(`<b>${row.region}</b><br>${row.risk_level} · ${fmt(row.probability)}`);
    const pick = () => { selectedRegion = row.region; syncAllRegionSelects(); renderAll(); };
    m.on("click", pick); c.on("click", pick);
    markers.push(c, m);
  });
}

/* ─── HERO ────────────────────────────────────────────── */
function renderHero() {
  const row = currentRow();
  document.getElementById("heroCity").textContent = row.region;
  document.getElementById("heroTemp").textContent     = `${f1(row.temperature)}°C`;
  document.getElementById("heroWind").textContent     = `${f1(row.wind)} km/h`;
  document.getElementById("heroHumidity").textContent = `${f1(row.humidity)}%`;
  document.getElementById("heroRain").textContent     = `${Number(row.rain).toFixed(2)} mm`;
  document.getElementById("heroSummary").textContent  = row.climate_summary || "";
  document.getElementById("heroWarning").textContent  = row.warning || "";

  // risk ring
  const pct = row.probability;
  const col = riskColor(row.risk_level);
  document.getElementById("heroRiskPct").textContent   = fmt(pct);
  document.getElementById("heroRiskLevel").textContent = row.risk_level;
  document.getElementById("heroRiskLevel").style.color = col;
  const arc = document.getElementById("riskArc");
  arc.style.strokeDashoffset = CIRC - (CIRC * Math.min(pct, 1));
  arc.style.stroke = col;
}

/* ─── DETAIL PANEL ────────────────────────────────────── */
function renderPanel() {
  const row = currentRow();
  document.getElementById("panelCity").textContent = row.region;
  document.getElementById("panelRegionSelect").value = row.region;
  const chip = document.getElementById("dpRiskChip");
  chip.textContent = row.risk_level;
  chip.style.background = riskColor(row.risk_level);
  document.getElementById("dpRiskPct").textContent = fmt(row.probability);
  document.getElementById("dpRiskRow").style.background = (RISK[row.risk_level]||RISK.Low).bg;
  document.getElementById("dpTemp").textContent     = `${f1(row.temperature)}°C`;
  document.getElementById("dpWind").textContent     = `${f1(row.wind)} km/h`;
  document.getElementById("dpHumidity").textContent = `${f1(row.humidity)}%`;
  document.getElementById("dpRain").textContent     = `${Number(row.rain).toFixed(2)} mm`;
  document.getElementById("dpSummary").textContent  = row.climate_summary || "";
  document.getElementById("dpWarning").textContent  = row.warning || "";

  // mini trend
  const rr = forecast.filter(d => d.region === selectedRegion);
  Plotly.react("dpTrendChart", [{
    x: rr.map(d => d.date), y: rr.map(d => d.probability*100),
    type:"scatter", mode:"lines", line:{color:"#4a9eda",width:2},
    fill:"tozeroy", fillcolor:"rgba(74,158,218,.10)",
    hovertemplate:"%{x}<br>%{y:.1f}%<extra></extra>",
  }], {
    margin:{t:4,r:4,b:24,l:30}, paper_bgcolor:"transparent", plot_bgcolor:"transparent",
    yaxis:{ticksuffix:"%",gridcolor:"#eef2f6",tickfont:{size:10}},
    xaxis:{gridcolor:"#eef2f6",tickfont:{size:9}},
  }, {displayModeBar:false, responsive:true});
}

/* ─── FORECAST STRIP ──────────────────────────────────── */
function renderStrip() {
  const strip = document.getElementById("forecastStrip");
  if (forecastMode === "daily") {
    document.getElementById("stripTitle").textContent = "30-Day Outlook";
    const rows = forecast.filter(d => d.region === selectedRegion);
    strip.innerHTML = rows.map(row => {
      const dt = new Date(row.date+"T00:00");
      const active = row.date === selectedDate ? " active" : "";
      const col = riskColor(row.risk_level);
      return `<div class="fc-card${active}" data-date="${row.date}">
        <div class="fc-date">${dt.toLocaleDateString("en",{weekday:"short"})}<br>${dt.toLocaleDateString("en",{month:"short",day:"numeric"})}</div>
        <div class="fc-risk-dot" style="background:${col}">${Math.round(row.probability*100)}</div>
        <div class="fc-temp">${f1(row.temperature)}°</div>
        <div class="fc-wind">${f1(row.wind)} km/h</div>
        <div class="fc-label" style="background:${col}22;color:${col}">${row.risk_level}</div>
      </div>`;
    }).join("");
    strip.querySelectorAll(".fc-card").forEach(c => {
      c.onclick = () => { selectedDate = c.dataset.date; document.getElementById("datePicker").value = selectedDate; renderAll(); };
    });
  } else {
    document.getElementById("stripTitle").textContent = "168-Hour Outlook";
    const rows = hourlyForecast.filter(d => d.region === selectedRegion);
    // show every 3rd hour for readability
    const sampled = rows.filter((_,i) => i % 3 === 0);
    strip.innerHTML = sampled.map(row => {
      const ts = new Date(row.timestamp);
      const prob = row.probability || 0;
      const lvl  = row.risk_level || "Low";
      const col  = riskColor(lvl);
      return `<div class="fc-card">
        <div class="fc-date">${ts.toLocaleDateString("en",{weekday:"short"})}<br>${ts.getHours().toString().padStart(2,"0")}:00</div>
        <div class="fc-risk-dot" style="background:${col}">${Math.round(prob*100)}</div>
        <div class="fc-temp">${f1(row.temperature)}°</div>
        <div class="fc-wind">${f1(row.wind)} km/h</div>
        <div class="fc-label" style="background:${col}22;color:${col}">${lvl}</div>
      </div>`;
    }).join("");
  }
}

/* ─── DAILY CHARTS ────────────────────────────────────── */
const LAYOUT = (extra={}) => ({
  margin:{t:8,r:12,b:40,l:46}, paper_bgcolor:"transparent", plot_bgcolor:"transparent",
  yaxis:{gridcolor:"#eef2f6", ...extra.yaxis},
  xaxis:{gridcolor:"#eef2f6", ...extra.xaxis},
  legend:{orientation:"h",y:1.12,font:{size:12}},
  ...extra,
});
const PCFG = {displayModeBar:false, responsive:true};

function renderDailyCharts() {
  const rr = forecast.filter(d => d.region === selectedRegion);
  Plotly.react("mainRiskChart", [{
    x:rr.map(d=>d.date), y:rr.map(d=>d.probability*100),
    type:"bar", marker:{color:rr.map(d=>riskColor(d.risk_level)), opacity:0.85},
    hovertemplate:"%{x}<br>Risk: %{y:.1f}%<extra></extra>",
  }], LAYOUT({yaxis:{ticksuffix:"%",gridcolor:"#eef2f6"}}), PCFG);

  Plotly.react("mainWeatherChart", [
    {x:rr.map(d=>d.date), y:rr.map(d=>d.temperature), name:"Temp °C", type:"scatter", mode:"lines+markers", line:{color:"#4a9eda",width:2}, marker:{size:4}},
    {x:rr.map(d=>d.date), y:rr.map(d=>d.wind),        name:"Wind km/h", type:"scatter", mode:"lines+markers", line:{color:"#e06730",width:2}, marker:{size:4}},
    {x:rr.map(d=>d.date), y:rr.map(d=>d.humidity),     name:"Humidity %", type:"scatter", mode:"lines", line:{color:"#22a66e",width:1.5,dash:"dot"}, yaxis:"y2"},
  ], LAYOUT({yaxis:{gridcolor:"#eef2f6"}, yaxis2:{overlaying:"y",side:"right",showgrid:false,ticksuffix:"%"}}), PCFG);
}

/* ─── HOURLY CHARTS ───────────────────────────────────── */
function renderHourlyCharts() {
  const rr = hourlyForecast.filter(d => d.region === selectedRegion);
  if (!rr.length) return;
  const hasRisk = rr[0].probability !== undefined;

  const hl = (title) => LAYOUT({yaxis:{title,gridcolor:"#eef2f6"}});

  if (hasRisk) {
    const cols = rr.map(d => riskColor(d.risk_level||"Low"));
    Plotly.react("hourlyRiskChart", [
      {x:rr.map(d=>d.timestamp), y:rr.map(d=>(d.probability||0)*100), type:"scatter", mode:"lines",
       line:{color:"#4a9eda",width:2}, fill:"tozeroy", fillcolor:"rgba(74,158,218,.06)",
       name:"Fire Risk", hovertemplate:"%{x}<br>%{y:.1f}%<extra></extra>"},
      {x:rr.map(d=>d.timestamp), y:rr.map(d=>(d.probability||0)*100), type:"bar",
       marker:{color:cols,opacity:0.25}, showlegend:false, hoverinfo:"skip"},
    ], LAYOUT({yaxis:{title:"Fire Risk %",ticksuffix:"%",gridcolor:"#eef2f6"},barmode:"overlay"}), PCFG);
  }

  Plotly.react("hourlyTempChart", [{
    x:rr.map(d=>d.timestamp), y:rr.map(d=>d.temperature), type:"scatter", mode:"lines",
    line:{color:"#4a9eda",width:2}, fill:"tozeroy", fillcolor:"rgba(74,158,218,.06)",
    hovertemplate:"%{x}<br>%{y:.1f}°C<extra></extra>",
  }], hl("°C"), PCFG);

  Plotly.react("hourlyHumidityChart", [{
    x:rr.map(d=>d.timestamp), y:rr.map(d=>d.humidity), type:"scatter", mode:"lines",
    line:{color:"#22a66e",width:2}, fill:"tozeroy", fillcolor:"rgba(34,166,110,.06)",
    hovertemplate:"%{x}<br>%{y:.1f}%<extra></extra>",
  }], hl("%"), PCFG);

  Plotly.react("hourlyWindChart", [{
    x:rr.map(d=>d.timestamp), y:rr.map(d=>d.wind), type:"scatter", mode:"lines",
    line:{color:"#e06730",width:2}, fill:"tozeroy", fillcolor:"rgba(224,103,48,.06)",
    hovertemplate:"%{x}<br>%{y:.1f} km/h<extra></extra>",
  }], hl("km/h"), PCFG);
}

/* ─── TABLE ───────────────────────────────────────────── */
function renderTable() {
  const rf = document.getElementById("riskFilter").value;
  const df = document.getElementById("tableDateFilter").value;
  let rows = forecast.filter(d => d.region === selectedRegion);
  if (rf !== "All") rows = rows.filter(d => d.risk_level === rf);
  if (df) rows = rows.filter(d => d.date === df);
  document.getElementById("forecastTable").innerHTML = rows.map(r => `
    <tr>
      <td>${r.date}</td><td>${r.region}</td>
      <td><span class="risk-chip" style="background:${riskColor(r.risk_level)}">${r.risk_level}</span></td>
      <td>${fmt(r.probability)}</td>
      <td>${f1(r.temperature)}°C</td><td>${f1(r.wind)} km/h</td><td>${f1(r.humidity)}%</td>
    </tr>`).join("");
}

/* ─── MODE TOGGLE ─────────────────────────────────────── */
function setMode(mode) {
  forecastMode = mode;
  document.getElementById("toggleDaily").classList.toggle("active", mode==="daily");
  document.getElementById("toggleHourly").classList.toggle("active", mode==="hourly");
  document.getElementById("chartsSection").style.display        = mode==="daily" ? "" : "none";
  document.getElementById("hourlyChartsSection").style.display  = mode==="hourly" ? "" : "none";
  document.getElementById("tableSection").style.display         = mode==="daily" ? "" : "none";
  document.getElementById("chartRiskTitle").textContent = mode==="daily" ? "Risk Trend — 30 Days" : "Fire Risk — 168 Hours";
  renderAll();
}

/* ─── RENDER ALL ──────────────────────────────────────── */
function renderAll() {
  renderHero();
  renderMap();
  renderPanel();
  renderStrip();
  if (forecastMode === "daily") {
    renderDailyCharts();
    renderTable();
  } else {
    renderHourlyCharts();
  }
}

/* ─── BOOT ────────────────────────────────────────────── */
loadData().catch(err => {
  document.querySelector("main").innerHTML =
    `<div style="padding:60px 28px;text-align:center"><h2>Could not load data</h2><p>${err.message}</p></div>`;
});
