import { WIDTH, fixedPlanCdf, freePulls, quantile, statistics } from './planner.mjs';

const $ = id => document.getElementById(id);
const names = ['我是欧皇', '我是非酋', '我是超级无敌宇宙天上地下唯我独尊大非酋'];
const percent = (value, digits = 2) => `${(value * 100).toFixed(digits)}%`;
const potential = copies => copies === 1 ? '0 潜 · 本体' : `${copies - 1} 潜${copies === 6 ? ' · 满潜' : ''}`;
const dataCache = new Map();
let manifest, submitted, plans, first, second, activeTab = 'character', detail, jointData;
let detailRevision = 0, lastUnit;

$('potential-options').innerHTML = Array.from({ length: 6 }, (_, i) => `<label><input type="radio" name="target" value="${i + 1}" ${i === 5 ? 'checked' : ''}><span>${i}<small>${i === 0 ? '本体' : i === 5 ? '满潜' : '潜'}</small></span></label>`).join('');
for (const id of ['joint-character', 'joint-weapon']) {
  $(id).innerHTML = Array.from({ length: 6 }, (_, i) => `<option value="${i + 1}">${i + 1} 份${i === 5 ? ' · 满潜' : ''}</option>`).join('');
}

function status(message, type = '') {
  $('status').textContent = message;
  $('status').className = `status ${type}`;
}

function integer(id) {
  const input = $(id);
  input.required = true;
  if (!input.checkValidity()) {
    input.reportValidity();
    throw new Error('请填写有效范围内的整数。');
  }
  return Number(input.value);
}

async function loadManifest() {
  if (manifest) return manifest;
  const response = await fetch('./data/manifest.json');
  if (!response.ok) throw new Error('概率数据加载失败，请刷新或重试。');
  manifest = await response.json();
  return manifest;
}

async function loadData(name) {
  if (!dataCache.has(name)) {
    dataCache.set(name, (async () => {
      const meta = await loadManifest();
      const response = await fetch(`./data/${name}`);
      if (!response.ok) throw new Error('概率数据加载失败，请重试。');
      const buffer = await response.arrayBuffer();
      const expected = meta.files[name].shape.reduce((a, b) => a * b, 1);
      if (buffer.byteLength !== expected * 8) throw new Error('概率数据不完整，请刷新重试。');
      const view = new DataView(buffer);
      return Float64Array.from({ length: expected }, (_, i) => view.getFloat64(i * 8, true));
    })().catch(error => { dataCache.delete(name); throw error; }));
  }
  return dataCache.get(name);
}

function computePlans(current, rerun, config) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./planner-worker.mjs', import.meta.url), { type: 'module' });
    worker.onmessage = ({ data }) => {
      worker.terminate();
      if (data.error) reject(new Error(data.error));
      else resolve(data.result);
    };
    worker.onerror = () => { worker.terminate(); reject(new Error('规划计算加载失败，请刷新重试。')); };
    worker.postMessage({ first: current, second: rerun, config });
  });
}

function readConfig() {
  return {
    mode: document.querySelector('[name=mode]:checked').value,
    target: Number(document.querySelector('[name=target]:checked').value),
    budget: $('budget').disabled ? Number($('budget').value || 0) : integer('budget'),
    initialPity: integer('initial-pity'), includeRerun: $('include-rerun').checked,
    rerunPity: $('include-rerun').checked ? integer('rerun-pity') : 0,
  };
}

function updateForm() {
  const budgetMode = document.querySelector('[name=mode]:checked').value === 'budget';
  $('potential-field').hidden = budgetMode;
  $('budget-field').hidden = !budgetMode;
  $('budget').disabled = !budgetMode;
  $('rerun-pity-field').hidden = !$('include-rerun').checked;
  $('rerun-pity').disabled = !$('include-rerun').checked;
  if (submitted) status('条件已改变，点击「计算寻访计划」更新结果。', 'stale');
}
$('plan-form').addEventListener('input', updateForm);
updateForm();

$('plan-form').addEventListener('submit', async event => {
  event.preventDefault();
  const button = $('calculate');
  try {
    const config = readConfig();
    button.disabled = true;
    button.firstElementChild.textContent = '正在计算…';
    status('正在寻找各档达成概率下的最佳分配…');
    await loadManifest();
    const [currentData, rerunData] = await Promise.all([
      loadData(`chartered-${config.initialPity}.bin`),
      config.includeRerun ? loadData(`rerun-${config.rerunPity}.bin`) : null,
    ]);
    const result = await computePlans(currentData, rerunData, config);
    submitted = config; plans = result; first = currentData; second = rerunData;
    renderPlans();
    $('detail-source').value = 'plan';
    if (activeTab === 'character') $('detail-target').value = String(Math.max(1, plans[Number($('detail-confidence').value)].copies));
    status('计算完成。修改条件后，再点击计算更新。');
    if (JSON.stringify(readConfig()) !== JSON.stringify(submitted)) updateForm();
    await updateDetails();
  } catch (error) {
    status(error.message, 'error');
  } finally {
    button.disabled = false;
    button.firstElementChild.textContent = '计算寻访计划';
  }
});

function renderPlans() {
  $('results-section').hidden = false;
  $('result-summary').textContent = `${submitted.mode === 'potential' ? `目标 ${potential(submitted.target)}` : `总预算 ${submitted.budget} 抽`} · ${submitted.includeRerun ? '首发 + 首次复刻' : '仅本池'} · 本池六星水位 ${submitted.initialPity}${submitted.includeRerun ? ` / 复刻 ${submitted.rerunPity}` : ''}`;
  $('results').innerHTML = plans.map((item, i) => {
    const missing = item.unavailable || item.copies === 0;
    const value = item.unavailable ? '范围内未达标' : item.copies === 0 ? '本体尚未达标' : submitted.mode === 'potential' ? `${item.total}<small>抽</small>` : `${item.copies - 1}<small>潜${item.copies === 1 ? ' · 本体' : item.copies === 6 ? ' · 满潜' : ''}</small>`;
    return `<article class="result-card ${Number($('detail-confidence').value) === i ? 'selected' : ''}">
      <div class="card-body"><div class="card-top"><span class="confidence">${percent(item.threshold, 0)} 达成概率</span></div>
      <h3>${names[i]}</h3><p class="card-caption">${submitted.mode === 'potential' ? '所需最少总预算' : '该概率下至少可达'}</p><p class="card-value ${missing ? 'message' : ''}">${value}</p>
      <p class="actual-probability">${item.unavailable ? '1200 抽内未找到满足条件的分配' : item.copies === 0 ? `最优分配获得本体的概率 ${percent(item.probability)}` : `实际达成概率 ${percent(item.probability)}`}</p>
      <div class="allocation"><div><span class="allocation-label">本池</span><strong>${item.current ?? '—'}</strong><small>抽</small></div><div><span class="allocation-label">${submitted.includeRerun ? '复刻池' : '复刻池 · 未启用'}</span><strong>${item.rerun ?? '—'}</strong><small>抽</small></div></div>
      <p class="bonus-note">${item.unavailable ? '可调整目标后重新计算。' : `按预算上限可触发：加急本池 ${freePulls('chartered', item.current)} 抽${submitted.includeRerun ? ` / 复刻 ${freePulls('rerun', item.rerun)} 抽` : ''}；${Math.floor(item.current / 240) + Math.floor(item.rerun / 240)} 份信物。`}</p></div>
      <button class="card-action" data-plan="${i}" ${item.unavailable ? 'disabled' : ''}>查看这份方案的概率<span aria-hidden="true">↗</span></button></article>`;
  }).join('');
}
$('results').addEventListener('click', event => {
  const button = event.target.closest('[data-plan]');
  if (!button) return;
  $('detail-confidence').value = button.dataset.plan;
  $('detail-source').value = 'plan';
  $('detail-target').value = String(Math.max(1, plans[Number(button.dataset.plan)].copies));
  switchTab('character');
  renderPlans();
  $('details').scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
});

function metric(label, value, unit = '') {
  return `<div class="metric"><span class="metric-label">${label}</span><span class="metric-value">${value}<small>${unit}</small></span></div>`;
}

async function updateDetails() {
  const revision = ++detailRevision;
  if (activeTab === 'joint') return updateJoint();
  if (!submitted && activeTab === 'character') return;
  $('distribution-content').hidden = true;
  $('detail-description').textContent = '正在加载对应的概率详情…';
  try {
    const copies = Number($('detail-target').value);
    let cdf, description, guaranteed;
    const characterGuarantee = k => k <= 0 ? 0 : k === 1 ? 120 : (k - 1) * 240;
    if (activeTab === 'weapon') {
      const data = await loadData('weapons.bin');
      cdf = Float64Array.from({ length: 101 }, (_, n) => data[n * WIDTH + copies]);
      description = `武器目标 ${copies} 份；以十连次数计费，包含直接赠送武器，忽略武库箱价值。`;
      guaranteed = copies === 1 ? 8 : 18 + (copies - 2) * 16;
    } else {
      const source = $('detail-source').value;
      if (source === 'plan') {
        const selected = plans[Number($('detail-confidence').value)];
        if (selected.unavailable) throw new Error('该档位在计算范围内未达标，请选择其他方案。');
        cdf = fixedPlanCdf(first, second, copies, selected.current);
        description = submitted.includeRerun ? `固定执行方案：本池最多 ${selected.current} 抽，此后转入复刻，达到 ${potential(copies)} 即停止。曲线继续展示该执行方案直到目标完成。` : `仅本池，达到 ${potential(copies)} 即停止；六星初始水位 ${submitted.initialPity}。`;
        const inFirst = characterGuarantee(copies);
        const guaranteedFirstCopies = Math.floor(selected.current / 240) + (selected.current >= 120 ? 1 : 0);
        guaranteed = second ? Math.min(inFirst <= selected.current ? inFirst : Infinity, selected.current + characterGuarantee(copies - guaranteedFirstCopies)) : inFirst;
      } else {
        const pity = source === 'chartered' ? submitted.initialPity : submitted.rerunPity;
        const data = await loadData(`${source}-${pity}.bin`);
        cdf = fixedPlanCdf(data, null, copies, 0);
        description = `仅${source === 'chartered' ? '本池' : '首次复刻池'}，目标 ${potential(copies)}；六星初始水位 ${pity}，累计奖励进度从零开始。`;
        guaranteed = characterGuarantee(copies);
      }
    }
    if (revision !== detailRevision) return;
    const unit = activeTab === 'weapon' ? '次十连' : '抽';
    const previousCopies = detail?.copies;
    cdf = cdf.slice(0, guaranteed + 1);
    detail = { cdf, copies, guaranteed, ...statistics(cdf), unit };
    if (lastUnit !== unit || previousCopies !== copies) {
      $('chart-max').value = activeTab === 'weapon' ? 60 : 800;
      $('range-start').value = 0;
      $('range-end').value = activeTab === 'weapon' ? 60 : 300;
      lastUnit = unit;
    }
    for (const id of ['chart-max', 'range-start', 'range-end']) {
      $(id).max = cdf.length - 1;
      if (Number($(id).value) > cdf.length - 1) $(id).value = cdf.length - 1;
    }
    $('chart-unit').textContent = activeTab === 'weapon' ? '十连' : '抽';
    $('table-unit').textContent = unit;
    $('detail-description').textContent = description;
    $('distribution-content').hidden = false;
    $('distribution-metrics').innerHTML = metric('平均完成花费', detail.mean.toFixed(2), unit) + metric('标准差', detail.std.toFixed(2), unit) + metric('均值 ± 1σ 内完成', percent(detail.withinStd)) + metric('完整分布覆盖率', percent(detail.mass));
    const rows = [[Math.ceil(detail.mean), '均值向上取整'], [Math.ceil(detail.mean + detail.std), '均值 + 标准差'], [quantile(cdf, .99), '99% 达成线']];
    $('probability-table').innerHTML = `<table><thead><tr><th>${unit}</th><th>累计达成概率</th></tr></thead><tbody>${rows.map(([n, label]) => `<tr><td>${n ?? '未覆盖'} <small>· ${label}</small></td><td>${n === null ? '—' : percent(cdf[Math.min(n, cdf.length - 1)], 6)}</td></tr>`).join('')}</tbody></table>`;
    $('distribution-rows').innerHTML = Array.from(cdf, (value, n) => `<tr><td>${n}</td><td>${percent(detail.pmf[n], 6)}</td><td>${percent(value, 6)}</td></tr>`).join('');
    updateQueries();
    drawDistributionCharts();
  } catch (error) {
    if (revision === detailRevision) {
      $('detail-description').textContent = error.message;
      $('distribution-content').hidden = true;
    }
  }
}

function updateQueries() {
  if (!detail) return;
  const probability = Number($('percent-input').value) / 100;
  if ($('percent-input').value === '' || !$('percent-input').checkValidity()) $('percent-result').textContent = '请输入 0–100 的概率百分比。';
  else {
    const n = probability === 1 ? detail.guaranteed : quantile(detail.cdf, probability);
    $('percent-result').textContent = n === null ? '计算范围内未达到该概率。' : `${percent(probability)} 达成概率 → ${n} ${detail.unit}`;
  }
  const start = Number($('range-start').value), end = Number($('range-end').value);
  if (['range-start', 'range-end'].some(id => $(id).value === '' || !$(id).checkValidity())) {
    $('range-result').textContent = `请输入 0–${detail.cdf.length - 1} 的整数。`;
  } else {
    const lower = Math.min(start, end), upper = Math.max(start, end);
    const mass = Math.max(0, detail.cdf[upper] - (detail.cdf[lower - 1] ?? 0));
    $('range-result').textContent = `[${lower}, ${upper}] ${detail.unit} → ${percent(mass, 6)}`;
  }
}

const chartLayout = {
  font: { family: 'Segoe UI, Microsoft YaHei, sans-serif', color: '#adbac8', size: 13 },
  paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
  hoverlabel: { bgcolor: '#202c37', bordercolor: '#526474', font: { color: '#e6edf4' } },
  margin: { t: 28, r: 16, b: 52, l: 58 }, showlegend: false,
  xaxis: { gridcolor: '#2b3845', zeroline: false },
  yaxis: { gridcolor: '#2b3845', zeroline: false, tickformat: '.0%' },
};
const chartConfig = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] };

function drawDistributionCharts() {
  if (!detail || activeTab === 'joint') return;
  if (!window.Plotly) {
    $('pmf-chart').textContent = '图表组件加载失败，请刷新页面；数值计算仍可查看。';
    return;
  }
  if ($('chart-max').value === '' || !$('chart-max').checkValidity()) return;
  const end = Number($('chart-max').value), x = Array.from({ length: end + 1 }, (_, n) => n);
  const shapes = [{ type: 'rect', x0: detail.lower, x1: detail.upper, y0: 0, y1: 1, yref: 'paper', fillcolor: '#82d8ba', opacity: .08, line: { width: 0 }, layer: 'below' }];
  for (const [q, color] of [[.5, '#bdcf86'], [.9, '#82d8ba'], [.99, '#c6b08c']]) {
    const n = quantile(detail.cdf, q);
    if (n !== null && n <= end) shapes.push({ type: 'line', x0: n, x1: n, y0: 0, y1: 1, yref: 'paper', line: { color, width: 1, dash: 'dot' } });
  }
  shapes.push({ type: 'line', x0: detail.mean, x1: detail.mean, y0: 0, y1: 1, yref: 'paper', line: { color: '#e0bd74', width: 1, dash: 'dash' } });
  const layout = { ...chartLayout, shapes, xaxis: { ...chartLayout.xaxis, title: { text: detail.unit, font: { size: 14 } }, range: [0, end] } };
  Plotly.react('pmf-chart', [{ x, y: Array.from(detail.pmf.slice(0, end + 1)), type: 'bar', marker: { color: '#82d8ba' }, hovertemplate: `%{x} ${detail.unit}<br>恰好完成 %{y:.6%}<extra></extra>` }], layout, chartConfig);
  Plotly.react('cdf-chart', [{ x, y: Array.from(detail.cdf.slice(0, end + 1)), type: 'scatter', mode: 'lines', line: { color: '#82d8ba', width: 2.5 }, fill: 'tozeroy', fillcolor: '#82d8ba12', hovertemplate: `%{x} ${detail.unit}<br>累计达成 %{y:.6%}<extra></extra>` }], { ...layout, yaxis: { ...layout.yaxis, range: [0, 1.03] } }, chartConfig);
}

async function updateJoint() {
  try {
    if (!jointData) {
      $('joint-coverage').textContent = '正在加载联合目标的概率数据…';
      const [probabilities, metrics] = await Promise.all([loadData('joint-probabilities.bin'), loadData('joint-metrics.bin')]);
      jointData = { probabilities, metrics };
    }
    const n = integer('joint-pulls'), a = Number($('joint-character').value), b = Number($('joint-weapon').value);
    const probability = (draw, char, weapon) => jointData.probabilities[(draw * WIDTH + char) * WIDTH + weapon];
    const m = jointData.metrics.slice(n * 5, n * 5 + 5);
    $('joint-metrics').innerHTML = metric('达成干员目标', percent(probability(n, a, 0), 6)) + metric('达成武器目标', percent(probability(n, 0, b), 6)) + metric('同时达成两项目标', percent(probability(n, a, b), 6)) + metric('六星获得数量期望', m[0].toFixed(4), '份') + metric('武库配额期望', m[1].toFixed(2)) + metric('武器十连次数期望', m[2].toFixed(4), '次');
    $('joint-coverage').textContent = `计算覆盖率 ${percent(m[4], 9)}；保留状态内最多可兑换 ${Math.round(m[3])} 次武器十连。配额按实际整数次数兑换，不将平均配额直接当作抽数。`;
    $('joint-inverse-result').textContent = '';
    if (activeTab === 'joint' && window.Plotly) {
      const x = Array.from({ length: 721 }, (_, draw) => draw);
      const curves = [[a, 0, '干员达成', '#bdcf86'], [0, b, '武器达成', '#e0bd74'], [a, b, '同时达成', '#82d8ba']].map(([char, weapon, name, color]) => ({ x, y: x.map(draw => probability(draw, char, weapon)), name, mode: 'lines', line: { color, width: 2 }, hovertemplate: `%{x} 抽<br>${name} %{y:.6%}<extra></extra>` }));
      Plotly.react('joint-chart', curves, { ...chartLayout, showlegend: true, legend: { orientation: 'h', x: 0, y: 1.03, yanchor: 'bottom' }, margin: { ...chartLayout.margin, t: 85 }, xaxis: { ...chartLayout.xaxis, title: { text: '本池干员常规寻访次数' } }, shapes: [{ type: 'line', x0: n, x1: n, y0: 0, y1: 1, yref: 'paper', line: { dash: 'dot', color: '#bdcf86' } }] }, chartConfig);
    }
  } catch (error) { $('joint-coverage').textContent = error.message; }
}

async function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('[data-tab]').forEach(button => {
    const selected = button.dataset.tab === tab;
    button.setAttribute('aria-selected', String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  $('distribution-panel').hidden = tab === 'joint';
  $('distribution-panel').setAttribute('aria-labelledby', `tab-${tab}`);
  $('joint-panel').hidden = tab !== 'joint';
  $('detail-source-field').hidden = tab !== 'character';
  $('detail-confidence-field').hidden = tab !== 'character' || $('detail-source').value !== 'plan';
  await updateDetails();
}
document.querySelector('.detail-tabs').addEventListener('click', event => {
  const button = event.target.closest('[data-tab]');
  if (button) switchTab(button.dataset.tab);
});
document.querySelector('.detail-tabs').addEventListener('keydown', event => {
  const buttons = [...document.querySelectorAll('[data-tab]')];
  const current = buttons.indexOf(document.activeElement);
  if (current < 0 || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  const index = event.key === 'Home' ? 0 : event.key === 'End' ? 2 : (current + (event.key === 'ArrowRight' ? 1 : 2)) % 3;
  buttons[index].focus(); switchTab(buttons[index].dataset.tab);
});
for (const id of ['detail-source', 'detail-confidence', 'detail-target']) $(id).addEventListener('change', () => {
  if (id === 'detail-confidence' && plans) {
    $('detail-target').value = String(Math.max(1, plans[Number($('detail-confidence').value)].copies));
    renderPlans();
  }
  $('detail-confidence-field').hidden = activeTab !== 'character' || $('detail-source').value !== 'plan';
  updateDetails();
});
for (const id of ['percent-input', 'range-start', 'range-end']) $(id).addEventListener('input', updateQueries);
$('chart-max').addEventListener('change', drawDistributionCharts);
$('joint-form').addEventListener('submit', event => { event.preventDefault(); updateJoint(); });
$('joint-form').addEventListener('input', () => {
  $('joint-coverage').textContent = '条件已改变，点击「计算联合概率」更新结果。';
  $('joint-inverse-result').textContent = '';
});
$('joint-inverse-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!jointData) await updateJoint();
  if (!jointData) return;
  const target = Number($('joint-percent').value) / 100;
  const a = Number($('joint-character').value), b = Number($('joint-weapon').value);
  let found = null;
  for (let n = 1; n <= 720; n++) {
    if (jointData.probabilities[(n * WIDTH + a) * WIDTH + b] >= target) { found = n; break; }
  }
  $('joint-inverse-result').textContent = found === null ? '1–720 抽内未达到该综合概率。' : `至少 ${found} 抽干员寻访，综合达成概率达到 ${percent(target)}。`;
});

try {
  await loadManifest();
  $('calculate').disabled = false;
  $('calculate').firstElementChild.textContent = '计算寻访计划';
  $('plan-form').requestSubmit();
} catch (error) {
  status(error.message, 'error');
  $('calculate').disabled = false;
  $('calculate').firstElementChild.textContent = '重试加载并计算';
}
