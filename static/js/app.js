/* hcu-train-web front end.
   All numbers rendered here come straight from the backend, which delegates
   every estimate to hcu_train_simulator in-process. */

const state = {
  task: 'sft_full',   // selected training mode (v1 enum value)
  modes: [],          // /api/v1/training/modes — the six training modes
  boot: null,         // legacy /api/bootstrap payload
  model: null,        // resolved preset model object
  modelRef: null,     // model slug / reference string
  modelTab: 'preset',
  searchText: '',
  customSummary: null,
  modelConfig: null,  // raw config.json of the resolved model
  modelConfigPath: null,
  modelSlug: null,
  modelSource: null,  // provenance { repo_id, revision, source, config_hash }
  modelFacts: null,   // normalised architecture facts
  result: null,
  lastResult: null,
  plans: null,
  issues: [],
  busy: false,
};

// ---------------------------------------------------------------- utilities
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined && v !== false) node.setAttribute(k, v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    // Numbers arrive straight from JSON (expert counts, ranks, ...); handing
    // one to appendChild throws "parameter 1 is not of type 'Node'".
    if (typeof child === 'string' || typeof child === 'number') {
      node.appendChild(document.createTextNode(String(child)));
    } else {
      node.appendChild(child);
    }
  }
  return node;
}

const fmt = (v, d = 2) =>
  (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });

const fmtInt = (v) => (v === null || v === undefined) ? '—' : Number(v).toLocaleString('en-US');

function toast(message, kind = '') {
  const node = $('#toast');
  node.textContent = message;
  node.className = 'toast' + (kind ? ` toast--${kind}` : '');
  node.hidden = false;
  clearTimeout(node._timer);
  node._timer = setTimeout(() => { node.hidden = true; }, kind === 'bad' ? 7000 : 3000);
}

function setStatus(text, kind = '') {
  const node = $('#status-line');
  node.className = 'status' + (kind ? ` status--${kind}` : '');
  node.innerHTML = kind === 'busy' ? `<span class="spinner"></span>${text}` : text;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let payload;
  try { payload = await response.json(); }
  catch { throw new Error(`服务端返回了非 JSON 响应（HTTP ${response.status}）`); }
  if (!payload.ok) {
    const err = new Error(payload.error || `请求失败（HTTP ${response.status}）`);
    // The v1 API returns structured reasons for an unsupported model.
    err.reasons = payload.reasons || payload.error?.details || [];
    err.suggestions = payload.suggestions || payload.error?.suggestions || [];
    err.code = payload.error?.code;
    throw err;
  }
  return payload;
}

/** Hide the model-compatibility error box before a new run. */
function clearModelError() {
  const box = $('#model-error-box');
  const card = $('#model-error-card');
  if (box) { box.hidden = true; box.innerHTML = ''; }
  if (card) card.hidden = true;
}

/**
 * Render an error that carries structured reasons.
 *
 * A model the kernel cannot represent comes back with a list of reasons
 * and suggestions; showing them as a list is far more useful than one
 * long sentence in a toast.
 */
function showModelError(error) {
  const reasons = error.reasons || [];
  const suggestions = error.suggestions || [];
  if (!reasons.length) {
    toast(error.message, 'bad');
    return;
  }
  const box = $('#model-error-box');
  const card = $('#model-error-card');
  if (!box) {
    toast(error.message, 'bad');
    return;
  }
  if (card) card.hidden = false;
  box.hidden = false;
  box.innerHTML = '';
  box.appendChild(el('div', { class: 'notice notice--error' },
    el('strong', {}, error.message || '内核无法为该模型给出可信估算')));
  const list = el('ul', { class: 'reason-list' });
  reasons.forEach(r => list.appendChild(el('li', {}, r)));
  box.appendChild(list);
  if (suggestions.length) {
    const s = el('div', { class: 'notice notice--info' });
    s.appendChild(el('strong', {}, '可以这样做'));
    const sl = el('ul', { class: 'reason-list' });
    suggestions.forEach(x => sl.appendChild(el('li', {}, x)));
    s.appendChild(sl);
    box.appendChild(s);
  }
  switchTab('validate');
  toast('该模型暂不支持估算', 'bad');
}

function busy(isBusy) {
  $$('#btn-simulate, #btn-solve, #btn-search, #btn-resolve-hf, #btn-resolve-upload, #btn-resolve-paste')
    .forEach(b => { b.disabled = isBusy; });
}

// ---------------------------------------------------------------- bootstrap
async function init() {
  try {
    state.boot = await api('/api/bootstrap');
  } catch (error) {
    setStatus('后端不可用', 'bad');
    toast(`初始化失败：${error.message}`, 'bad');
    return;
  }

  // The v1 API owns the training-mode catalogue (six modes, with the
  // modelled / unmodelled split). The legacy bootstrap stays as the
  // fallback so the page still renders if the new API is unavailable.
  try {
    const modes = await api('/api/v1/training/modes');
    state.modes = modes.modes || [];
  } catch (_) {
    state.modes = [];
  }

  renderTopbar();
  renderTasks();
  renderHardware();
  renderParallelFields();
  renderDataFields();
  renderLoraFields();
  updateLoraVisibility();
  bindEvents();
  restoreDefaultModel();

  // The image catalogue and the training templates are independent of the
  // current model, so they load in the background and do not block the page.
  loadImages();
  loadTemplates();
  loadSupportMatrix();
}

function renderTopbar() {
  const sim = state.boot.simulator || {};
  const repoPath = (sim.repo || '').split('\\').pop();
  $('#pill-sim').textContent = `simulator: ${repoPath || 'source-checkout'}`;
  const verified = state.boot.models.filter(m => m.verified).length;
  $('#pill-models').textContent = `模型: ${verified}/${state.boot.models.length} 已校验`;
}

// ---------------------------------------------------------------- training modes
// The product defines six training modes (plan §2). Only the ones the
// estimator kernel actually models may produce numbers; the rest say so
// explicitly rather than borrowing another mode's figures.
function renderTasks() {
  const grid = $('#task-grid');
  grid.innerHTML = '';
  const modes = state.modes.length ? state.modes : (state.boot.tasks || []);

  modes.forEach(mode => {
    const key = mode.mode || mode.key;
    const label = mode.label || mode.name;
    const supported = mode.modelled !== false && mode.enabled !== false;
    const btn = el('button', {
      class: 'task-btn' + (key === state.task ? ' is-active' : ''),
      title: mode.summary || mode.description || '',
      onclick: () => selectTask(key),
    }, el('span', {}, label));
    if (!supported) btn.appendChild(el('span', { class: 'badge' }, '未建模'));
    grid.appendChild(btn);
  });
  updateTaskNote();
}

function selectTask(key) {
  state.task = key;
  renderTasks();
  updateTaskNote();
  updateLoraVisibility();
  loadTemplates();

  // Pretraining presets full recompute on; fine-tuning leaves it off.
  const recompute = $('#t-full_recompute');
  if (recompute) {
    recompute.checked = key === 'pretrain';
    recompute.closest('.toggle')?.classList.toggle('is-on', recompute.checked);
  }
}

function updateTaskNote() {
  const mode = state.modes.find(m => (m.mode || m.key) === state.task)
    || (state.boot.tasks || []).find(t => t.key === state.task);
  if (!mode) return;

  $('#task-note').textContent = mode.summary || mode.description || '';

  // Modes the kernel cannot model must say so plainly (plan §1.3).
  const warn = $('#mode-warning');
  const modelled = mode.modelled !== false && mode.enabled !== false;
  if (!modelled) {
    warn.hidden = false;
    warn.textContent = mode.unmodelled_reason
      || '该训练类型尚未建立独立的资源模型，直接套用全参训练数字会与真实训练不一致，因此不提供估算。';
  } else {
    warn.hidden = true;
    warn.textContent = '';
  }

  // Modes that share the full-parameter kernel say so, so the user does not
  // read two identical numbers as a bug.
  const note = $('#mode-kernel-note');
  if (note) {
    if (modelled && mode.kernel_note) {
      note.hidden = false;
      note.textContent = mode.kernel_note;
    } else {
      note.hidden = true;
      note.textContent = '';
    }
  }
}

// ---------------------------------------------------------------- hardware
function renderHardware() {
  const box = $('#hardware-box');
  box.innerHTML = '';

  state.boot.hardware.forEach(hw => {
    const card = el('div', { class: 'card', style: 'margin-bottom:10px' });
    card.appendChild(el('h3', {}, `${hw.name}（${hw.vendor}）`));
    card.appendChild(el('dl', { class: 'kv' },
      el('dt', {}, 'FP16 / FP8 算力'), el('dd', {}, `${hw.fp16_tflops} / ${hw.fp8_tflops} TFLOPS`),
      el('dt', {}, '单卡显存'), el('dd', {}, `${hw.hbm_gib} GiB`),
      el('dt', {}, '单节点卡数'), el('dd', {}, `${hw.gpus_per_node} 卡`),
      el('dt', {}, '通信带宽来源'), el('dd', { style: 'font-family:var(--sans);text-align:right' }, hw.bandwidth_source),
    ));
    (hw.notes || []).forEach(n => card.appendChild(el('p', { class: 'hint' }, n)));
    box.appendChild(card);
  });

  const wrapper = el('div');
  wrapper.appendChild(el('label', { class: 'label' }, '算子 Profile（决定计算时间）'));
  const select = el('select', { class: 'input', id: 'profile-select' });
  state.boot.operator_profiles.forEach(p => {
    select.appendChild(el('option', { value: p.key, selected: p.key === 'bw1000' },
      p.name + (p.recommended ? '（推荐）' : '')));
  });
  wrapper.appendChild(select);
  const note = el('p', { class: 'hint', id: 'profile-note' }, '');
  wrapper.appendChild(note);
  box.appendChild(wrapper);
  updateProfileNote();

  renderHardwareOverrides();
}

function updateProfileNote() {
  const key = $('#profile-select')?.value || 'bw1000';
  const profile = state.boot.operator_profiles.find(p => p.key === key);
  const note = $('#profile-note');
  if (note && profile) note.textContent = profile.note;
}

const HARDWARE_OVERRIDE_FIELDS = [
  ['fp16_tflops', 'FP16 TFLOPS', 'number'],
  ['fp8_tflops', 'FP8 TFLOPS', 'number'],
  ['hbm_gib', '单卡显存 GiB', 'number'],
  ['gpus_per_node', '单节点卡数', 'number'],
  ['gemm_efficiency', 'GEMM 效率', 'number'],
  ['non_gemm_efficiency', '非 GEMM 效率', 'number'],
  ['optimizer_efficiency', '优化器效率', 'number'],
];

function renderHardwareOverrides() {
  const box = $('#hardware-override');
  box.innerHTML = '';
  HARDWARE_OVERRIDE_FIELDS.forEach(([key, label, type]) => {
    box.appendChild(el('div', { class: 'field' },
      el('label', { class: 'field__label' }, label),
      el('input', { class: 'input', type, step: 'any', id: `hw-${key}`, placeholder: '默认' })));
  });
}

// ---------------------------------------------------------------- data volume
// Plan C07: token/sample/epoch/step conversions. These drive the total
// training time estimate, which is why the block warns when nothing is
// filled in.
const DATA_FIELDS = [
  { key: 'total_tokens', label: '总 Token 数', type: 'number', hint: '训练语料总 token' },
  { key: 'total_samples', label: '总样本数', type: 'number', hint: '样本条数' },
  { key: 'epochs', label: 'Epoch 数', type: 'number', step: 'any', hint: '遍历次数' },
  { key: 'max_steps', label: '最大步数', type: 'number', hint: '直接指定训练步数' },
  { key: 'checkpoint_interval', label: '检查点间隔(步)', type: 'number', hint: '留空=不保存' },
  { key: 'checkpoint_keep', label: '保留检查点数', type: 'number', hint: '滚动保留份数' },
];

function renderDataFields() {
  const box = $('#data-grid');
  if (!box) return;
  box.innerHTML = '';
  DATA_FIELDS.forEach(field => {
    const { key, label, type, hint, step } = field;
    const input = el('input', { class: 'input', type, id: `d-${key}`, placeholder: '留空' });
    if (step) input.setAttribute('step', step);
    input.addEventListener('input', updateDataDerived);
    box.appendChild(el('div', { class: 'field' },
      el('label', { class: 'field__label' }, label),
      input,
      hint ? el('p', { class: 'hint' }, hint) : null));
  });
  updateDataDerived();
}

/** Read the data block into the payload shape the API expects. */
function dataPayload() {
  const get = (key) => {
    const node = document.getElementById(`d-${key}`);
    if (!node || node.value === '') return null;
    const num = Number(node.value);
    return Number.isFinite(num) ? num : null;
  };
  return {
    total_tokens: get('total_tokens'),
    total_samples: get('total_samples'),
    epochs: get('epochs'),
    max_steps: get('max_steps'),
    checkpoint_interval: get('checkpoint_interval'),
    checkpoint_keep: get('checkpoint_keep') ?? 1,
  };
}

/**
 * Derive the training step count and total wall-clock time.
 *
 * steps = total_tokens / (GBS × seq_length) per epoch, times epochs.
 * Total time = steps × iteration time from the last estimate, when there is
 * one — without an iteration time we can only report the step count.
 */
function updateDataDerived() {
  const box = $('#data-derived');
  if (!box) return;
  const data = dataPayload();
  const gbs = Number($('#t-global_batch_size')?.value || 256);
  const seq = Number($('#t-seq_length')?.value || 4096);

  const rows = [];
  let steps = null;

  if (data.max_steps) {
    steps = data.max_steps;
    rows.push(['最大步数（直接指定）', fmtInt(steps)]);
  } else if (data.total_tokens && gbs > 0 && seq > 0) {
    const perEpoch = data.total_tokens / (gbs * seq);
    const epochs = data.epochs || 1;
    steps = Math.ceil(perEpoch * epochs);
    rows.push(['每 Epoch 步数', fmtInt(Math.ceil(perEpoch))]);
    rows.push(['Epoch 数', String(epochs)]);
    rows.push(['总训练步数', fmtInt(steps)]);
  } else if (data.total_samples && gbs > 0) {
    const perEpoch = data.total_samples / gbs;
    const epochs = data.epochs || 1;
    steps = Math.ceil(perEpoch * epochs);
    rows.push(['每 Epoch 步数', fmtInt(Math.ceil(perEpoch))]);
    rows.push(['总训练步数', fmtInt(steps)]);
    rows.push(['提示', '未填总 Token 数，总时长按样本数粗算']);
  }

  const iteration = state.result?.result?.iteration_ms
    || state.result?.result?.iteration_time_ms;
  if (steps && iteration) {
    const hours = (steps * iteration) / 1000 / 3600;
    rows.push(['预计总时长', hours >= 48
      ? `${(hours / 24).toFixed(1)} 天`
      : `${hours.toFixed(1)} 小时`]);
  } else if (steps) {
    rows.push(['预计总时长', '需先运行一次估算才能给出']);
  }

  if (!rows.length) {
    box.innerHTML = '<p class="hint">填写数据量后可换算训练步数与预计总时长。</p>';
    return;
  }
  box.innerHTML = '<dl class="kv">' +
    rows.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`).join('') +
    '</dl>';
}

// ---------------------------------------------------------------- parallel form
const PARALLEL_FIELDS = [
  { key: 'tp_size', label: 'TP 张量并行', hint: '需整除 heads/KV组' },
  { key: 'pp_size', label: 'PP 流水并行', hint: '需整除层数' },
  { key: 'cp_size', label: 'CP 上下文并行', hint: '' },
  { key: 'ep_size', label: 'EP 专家并行', hint: 'MoE 专用' },
  { key: 'etp_size', label: 'ETP 专家张量', hint: '留空=TP' },
  { key: 'num_layers_per_vp_stage', label: 'VPP 每虚流水层数', hint: '需 pp>1' },
];

const BATCH_FIELDS = [
  { key: 'num_gpus', label: '总卡数 num_gpus', hint: '单节点倍数' },
  { key: 'micro_batch_size', label: 'micro batch', hint: '' },
  { key: 'global_batch_size', label: 'global batch', hint: '' },
  { key: 'seq_length', label: '序列长度', hint: '' },
  { key: 'decoder_first_pipeline_num_layers', label: '首 stage 层数', hint: '可空' },
  { key: 'decoder_last_pipeline_num_layers', label: '末 stage 层数', hint: '可空' },
];

const TOGGLES = [
  ['use_fp8_training', 'FP8 训练'],
  ['full_recompute', '全量重计算'],
  ['use_distributed_optimizer', '分布式优化器'],
  ['sequence_parallel', '序列并行'],
  ['ep_overlap_enabled', 'EP 通信掩盖'],
  ['overlap_p2p_comm', 'PP P2P 掩盖'],
];

const ADVANCED_FIELDS = [
  { key: 'overlap_mode', label: '通信掩盖模式', type: 'select', options: ['auto', 'manual', 'none'], value: 'auto' },
  { key: 'ep_communication_backend', label: 'EP 后端', type: 'select', options: ['alltoall', 'deepep'], value: 'alltoall' },
  { key: 'pp_schedule', label: '流水调度', type: 'select', options: ['1f1b', 'interleaved_1f1b'], value: '1f1b' },
  { key: 'tp_overlap_ratio', label: 'TP 掩盖比', type: 'number', value: 0 },
  { key: 'dp_overlap_ratio', label: 'DP 掩盖比', type: 'number', value: 0 },
  { key: 'ep_overlap_ratio', label: 'EP 掩盖比', type: 'number', value: 0 },
  { key: 'cp_overlap_ratio', label: 'CP 掩盖比', type: 'number', value: 0 },
];

function renderParallelFields() {
  const grid = $('#parallel-grid');
  grid.innerHTML = '';
  [...PARALLEL_FIELDS, ...BATCH_FIELDS].forEach(field => {
    grid.appendChild(el('div', { class: 'field' },
      el('label', { class: 'field__label', for: `p-${field.key}` }, field.label),
      el('input', { class: 'input', type: 'number', min: '1', id: `p-${field.key}`, placeholder: '默认' }),
      el('span', { class: 'field__hint', id: `h-${field.key}` }, field.hint)));
  });

  // Only the distributed optimizer and sequence parallelism default to on;
  // every overlap / recompute / fp8 switch must start off, because several of
  // them are only legal under specific schedules (e.g. overlap_p2p_comm
  // requires an interleaved VPP schedule and would fail validation).
  const DEFAULT_ON = new Set(['use_distributed_optimizer', 'sequence_parallel']);
  const toggles = $('#toggle-grid');
  toggles.innerHTML = '';
  TOGGLES.forEach(([key, label]) => {
    const checked = DEFAULT_ON.has(key) || (key === 'full_recompute' && state.task === 'pretrain');
    const input = el('input', { type: 'checkbox', id: `t-${key}`, checked });
    const wrap = el('label', { class: 'toggle' + (checked ? ' is-on' : ''), for: `t-${key}` }, input, label);
    input.addEventListener('change', () => wrap.classList.toggle('is-on', input.checked));
    toggles.appendChild(wrap);

    // The estimator kernel declares `full_recompute` but does not consume it
    // — activation memory and compute time do not branch on it. Say so,
    // otherwise toggling it looks like it should change the numbers.
    if (key === 'full_recompute') {
      toggles.appendChild(el('p', { class: 'hint hint--inline' },
        '当前估算内核尚未对重计算建模：勾选后显存与单步时间不会变化，此处仅记录训练配置意图。'));
    }
  });

  const advanced = $('#advanced-grid');
  advanced.innerHTML = '';
  ADVANCED_FIELDS.forEach(field => {
    let input;
    if (field.type === 'select') {
      input = el('select', { class: 'input', id: `a-${field.key}` });
      field.options.forEach(o => input.appendChild(el('option', { value: o, selected: o === field.value }, o)));
    } else {
      input = el('input', { class: 'input', type: 'number', step: 'any', id: `a-${field.key}`, value: field.value });
    }
    advanced.appendChild(el('div', { class: 'field' },
      el('label', { class: 'field__label' }, field.label), input));
  });
}

// ---------------------------------------------------------------- model picker
function restoreDefaultModel() {
  renderFamilies();
  renderModelOptions();
  const first = state.boot.models.find(m => m.verified) || state.boot.models[0];
  if (first) selectPreset(first.slug);
}

function renderFamilies() {
  const box = $('#family-list');
  box.innerHTML = '';
  const all = el('span', {
    class: 'family-chip' + (state.family === null ? ' is-active' : ''),
    onclick: () => { state.family = null; state.searchText = ''; $('#model-search').value = ''; renderFamilies(); renderModelOptions(); },
  }, `全部 ${state.boot.models.length}`);
  box.appendChild(all);
  (state.boot.families || []).forEach(f => {
    box.appendChild(el('span', {
      class: 'family-chip' + (state.family === f.name ? ' is-active' : ''),
      onclick: () => { state.family = f.name; renderFamilies(); renderModelOptions(); },
    }, `${f.name} ${f.count}`));
  });
}

function filteredModels() {
  const text = state.searchText.trim().toLowerCase();
  return state.boot.models.filter(model => {
    if (state.family && model.family !== state.family) return false;
    if (!text) return true;
    return (model.name + ' ' + model.hf_id + ' ' + (model.model_type || '') + ' ' + (model.family || ''))
      .toLowerCase().includes(text);
  });
}

function renderModelOptions() {
  const select = $('#model-select');
  select.innerHTML = '';
  const models = filteredModels();
  if (!models.length) {
    select.appendChild(el('option', { value: '' }, '（无匹配模型）'));
    return;
  }
  models.forEach(model => {
    const params = model.params_b ? `${model.params_b}B` : '—';
    const badge = model.verified ? '' : '（未校验）';
    select.appendChild(el('option', { value: model.slug },
      `${model.name} · ${params} · ${model.adapter || '?'}${badge}`));
  });
  if (state.model && models.some(m => m.slug === state.model.slug)) {
    select.value = state.model.slug;
  }
}

function selectPreset(slug) {
  const model = state.boot.models.find(m => m.slug === slug);
  if (!model) return;
  state.model = model;
  state.customSummary = null;
  state.modelRef = model.slug;
  // The v1 API needs the model identified by slug *or* config path. Preset
  // models live in the catalogue, so pass the slug and let the server
  // resolve the config; the path is filled in for display only.
  state.modelSlug = model.slug;
  state.modelConfigPath = null;
  state.modelSource = null;
  state.modelFacts = null;
  $('#model-select').value = slug;
  applyModelDefaults(model);
  renderPresetSummary();
  clearRemoteSummaries();
  updateDerived();
}

function applyModelDefaults(model) {
  if (model.suggested_tp) $('#p-tp_size').value = model.suggested_tp;
  if (model.suggested_pp) $('#p-pp_size').value = model.suggested_pp;
  if (model.suggested_ep) $('#p-ep_size').value = model.suggested_ep;
  if (model.suggested_num_gpus) $('#p-num_gpus').value = model.suggested_num_gpus;
  if (model.baseline) {
    $('#p-global_batch_size').value = model.baseline.global_batch_size;
    $('#p-seq_length').value = model.baseline.seq_length;
    $('#p-micro_batch_size').value = model.baseline.micro_batch_size;
  }
  updateFieldHints(model);
}

function currentNumExperts() {
  const source = state.model || state.customSummary;
  if (!source) return null;
  return source.num_experts || source.num_experts || null;
}

// Mirror of engine.config_builder.valid_ep_options: the repo asserts
// EP*ETP*EDP*PP == num_gpus, and MoE models default ETP = TP.
function validEpOptions(numGpus, tp, pp, numExperts) {
  const denominator = tp * pp;
  if (!denominator || numGpus % denominator !== 0) return [];
  const limit = numGpus / denominator;
  let options = [8, 4, 2, 1].filter(ep => limit % ep === 0);
  if (numExperts) options = options.filter(ep => numExperts % ep === 0);
  return options;
}

function updateFieldHints(model) {
  const set = (key, text, bad = false) => {
    const node = $(`#h-${key}`);
    if (!node) return;
    node.textContent = text;
    node.className = 'field__hint' + (bad ? ' field__hint--bad' : '');
  };
  const validTp = model.valid_tp || [];
  const validPp = model.valid_pp || [];
  set('tp_size', validTp.length ? `可选 ${validTp.join('/')}` : '无合法 TP');
  set('pp_size', validPp.length ? `可选 ${validPp.join('/')}` : '无合法 PP');
  set('cp_size', '上下文并行');
  updateEpHint();
}

// EP legality depends on num_gpus / TP / PP, so it is recomputed whenever any
// of those change rather than being baked into the catalog entry.
function updateEpHint() {
  const numGpus = Number($('#p-num_gpus').value || 8);
  const tp = Number($('#p-tp_size').value || 1);
  const pp = Number($('#p-pp_size').value || 1);
  const experts = currentNumExperts();
  const node = $('#h-ep_size');
  const field = $('#p-ep_size');
  if (!node || !field) return;

  if (!experts) {
    node.textContent = '稠密模型不适用';
    node.className = 'field__hint';
    field.value = 1;
    return;
  }

  const options = validEpOptions(numGpus, tp, pp, experts);
  if (!options.length) {
    node.textContent = `ETP=${tp}×PP=${pp} 无法整除 ${numGpus} 卡，请调整 TP/PP`;
    node.className = 'field__hint field__hint--bad';
    field.value = 1;
    return;
  }
  node.textContent = `ETP=${tp}·可选 ${options.join('/')}`;
  node.className = 'field__hint';
  const current = Number(field.value || 1);
  if (!options.includes(current)) field.value = options[0];
}

function renderPresetSummary() {
  const model = state.model;
  if (!model) { $('#preset-summary').innerHTML = ''; return; }
  const box = $('#preset-summary');
  box.innerHTML = '';

  box.appendChild(el('div', {},
    el('span', { class: 'adapter-tag' + (model.adapter === 'generic_gpt' ? ' adapter-tag--fallback' : '') },
      `adapter: ${model.adapter || '未知'}`),
    model.verified
      ? el('span', { class: 'tag tag--ok', style: 'margin-left:6px' }, '已校验')
      : el('span', { class: 'tag tag--warn', style: 'margin-left:6px' }, '未校验')));

  box.appendChild(el('dl', { class: 'kv', style: 'margin-top:8px' },
    el('dt', {}, 'HF 仓库'), el('dd', {}, model.hf_id),
    el('dt', {}, '参数量'), el('dd', {}, model.params_b ? `${model.params_b} B` : '—'),
    el('dt', {}, '层数 / hidden'), el('dd', {}, `${model.num_layers} / ${model.hidden_size}`),
    el('dt', {}, 'heads / KV 组'), el('dd', {}, `${model.num_attention_heads} / ${model.num_query_groups}`),
    el('dt', {}, '专家数'), el('dd', {}, model.num_experts ? `${model.num_experts}` : '—（稠密）'),
    el('dt', {}, 'model_type'), el('dd', {}, model.model_type || '—'),
  ));

  if (model.baseline_result) {
    const r = model.baseline_result;
    box.appendChild(el('p', { class: 'hint' },
      `基线（${model.baseline.num_gpus}卡 TP${model.baseline.tp_size}）：` +
      `迭代 ${fmt(r.iteration_ms, 1)} ms · TGS ${fmtInt(r.tgs)} · MFU ${fmt(r.mfu * 100, 2)}%`));
  }
  if (model.verify_error) {
    box.appendChild(el('p', { class: 'hint', style: 'color:var(--bad)' }, `校验失败：${model.verify_error}`));
  }
}

function clearRemoteSummaries() {
  ['#hf-summary', '#upload-summary', '#paste-summary'].forEach(sel => { $(sel).innerHTML = ''; });
}

function renderCustomSummary(sel, summary, label) {
  const box = $(sel);
  box.innerHTML = '';
  box.appendChild(el('div', {},
    el('span', { class: 'adapter-tag' + (summary.adapter_is_fallback ? ' adapter-tag--fallback' : '') },
      `adapter: ${summary.adapter || '未匹配'}`),
    el('span', { class: 'tag tag--info', style: 'margin-left:6px' }, label)));
  if (summary.adapter_is_fallback) {
    box.appendChild(el('p', { class: 'hint', style: 'color:var(--warn)' },
      '该模型族没有专用适配器，参考仓库会回退到通用 Transformer/MoE 近似计算。'));
  }
  box.appendChild(el('dl', { class: 'kv', style: 'margin-top:8px' },
    el('dt', {}, 'model_type'), el('dd', {}, summary.model_type || '—'),
    el('dt', {}, '层数 / hidden'), el('dd', {}, `${summary.num_hidden_layers} / ${summary.hidden_size}`),
    el('dt', {}, 'heads / KV 组'), el('dd', {}, `${summary.num_attention_heads} / ${summary.num_query_groups}`),
    el('dt', {}, '合法 TP'), el('dd', {}, (summary.tp_divisors || []).join(' / ')),
    el('dt', {}, '合法 PP'), el('dd', {}, (summary.pp_divisors || []).join(' / ')),
  ));
  box.appendChild(el('div', { class: 'traits' },
    (summary.traits || []).map(t => el('span', { class: 'trait' }, t))));

  // Apply the discovered dimensions as sensible parallel defaults.
  const tp = (summary.tp_divisors || [1])[(summary.tp_divisors || [1]).length - 1];
  $('#p-tp_size').value = tp;
  $('#p-pp_size').value = (summary.pp_divisors || [1])[0];
  $('#p-ep_size').value = 1;
  if (!$('#p-num_gpus').value) $('#p-num_gpus').value = 8;
  updateDerived();
}

// ---------------------------------------------------------------- derived info
function updateDerived() {
  const numGpus = Number($('#p-num_gpus').value || 8);
  const tp = Number($('#p-tp_size').value || 1);
  const cp = Number($('#p-cp_size').value || 1);
  const pp = Number($('#p-pp_size').value || 1);
  const ep = Number($('#p-ep_size').value || 1);
  const mbs = Number($('#p-micro_batch_size').value || 1);
  const gbs = Number($('#p-global_batch_size').value || 256);

  const modelParallel = tp * cp * pp;
  const dp = modelParallel ? Math.floor(numGpus / modelParallel) : 0;
  const microbatches = dp ? gbs / mbs / dp : 0;

  const issues = [];
  if (modelParallel && numGpus % modelParallel !== 0) {
    issues.push(`num_gpus=${numGpus} 不是 TP×CP×PP=${modelParallel} 的整数倍`);
  }
  if (microbatches && microbatches < 1) issues.push('global_batch_size 太小，每个 DP 副本至少需要 1 个 micro batch');

  const experts = currentNumExperts();
  if (experts) {
    const etp = Number($('#p-etp_size')?.value || 0) || tp;
    const edpDenominator = ep * etp * pp;
    if (edpDenominator && numGpus % edpDenominator !== 0) {
      issues.push(`EP=${ep}×ETP=${etp}×PP=${pp} 无法整除 ${numGpus} 卡`);
    }
  }

  $('#derived-box').innerHTML =
    `DP = ${dp} · 微批数 = ${microbatches.toFixed(2)} · 模型并行度 = ${modelParallel}` +
    (experts ? ` · EDP = ${Math.floor(numGpus / (ep * (Number($('#p-etp_size')?.value || 0) || tp) * pp))}` : '') +
    (issues.length ? `<br><span style="color:var(--bad)">⚠ ${issues.join('；')}</span>` : '');
}

// ---------------------------------------------------------------- payload
function buildPayload() {
  const parallel = {};
  [...PARALLEL_FIELDS, ...BATCH_FIELDS].forEach(field => {
    const raw = $(`#p-${field.key}`)?.value;
    parallel[field.key] = raw === '' || raw === null ? null : Number(raw);
  });
  TOGGLES.forEach(([key]) => { parallel[key] = $(`#t-${key}`).checked; });
  ADVANCED_FIELDS.forEach(field => {
    const node = $(`#a-${field.key}`);
    if (!node) return;
    parallel[field.key] = field.type === 'select' ? node.value : (node.value === '' ? null : Number(node.value));
  });

  const hardwareOverrides = {};
  HARDWARE_OVERRIDE_FIELDS.forEach(([key]) => {
    const raw = $(`#hw-${key}`)?.value;
    if (raw !== '' && raw !== null) hardwareOverrides[key] = Number(raw);
  });

  return {
    task: state.task,
    hardware: 'bw1000',
    operator_profile: $('#profile-select')?.value || 'bw1000',
    model: state.modelRef,
    parallel,
    hardware_overrides: hardwareOverrides,
    search_candidates: {
      tp_size: [1, 2, 4, 8], cp_size: [1, 2], pp_size: [1, 2, 4],
      ep_size: [1, 2, 4, 8], micro_batch_size: [1, 2],
    },
  };
}

function requireModel() {
  if (!state.modelRef) { toast('请先选择模型', 'bad'); return false; }

  // Modes the kernel cannot model must be refused up front with the reason
  // (plan §1.3: no borrowed numbers).
  const mode = state.modes.find(m => (m.mode || m.key) === state.task);
  if (mode && mode.modelled === false) {
    toast(mode.unmodelled_reason || '该训练类型尚未建模，不提供估算', 'bad');
    return false;
  }
  return true;
}

/** Map the UI task key to the v1 training-mode enum value. */
function modeValue() {
  const map = {
    full: 'sft_full',
    pretrain: 'pretrain',
    sft_full: 'sft_full',
    sft_lora: 'sft_lora',
    dpo: 'dpo',
    grpo: 'grpo',
    ppo: 'ppo',
  };
  return map[state.task] || 'sft_full';
}

/** Build the v1 scenario payload from the current form state. */
function buildV1Payload() {
  const payload = buildPayload();
  const data = dataPayload();
  return {
    mode: modeValue(),
    hardware_id: payload.hardware,
    model_config_path: state.modelConfigPath || null,
    model_slug: state.modelSlug || null,
    model_revision: state.modelSource?.revision || null,
    model_config_hash: state.modelSource?.config_hash || null,
    parallel: {
      tp_size: payload.parallel.tp_size ?? 1,
      cp_size: payload.parallel.cp_size ?? 1,
      pp_size: payload.parallel.pp_size ?? 1,
      ep_size: payload.parallel.ep_size ?? 1,
      etp_size: payload.parallel.etp_size ?? null,
      num_gpus: payload.parallel.num_gpus ?? 8,
      micro_batch_size: payload.parallel.micro_batch_size ?? 1,
      global_batch_size: payload.parallel.global_batch_size ?? 256,
      seq_length: payload.parallel.seq_length ?? 4096,
    },
    use_distributed_optimizer: payload.parallel.use_distributed_optimizer ?? true,
    full_recompute: payload.parallel.full_recompute ?? false,
    use_fp8: payload.parallel.use_fp8_training ?? false,
    lora: {
      r: loraPayload().r,
      alpha: loraPayload().alpha,
      dropout: loraPayload().dropout,
      target_modules: loraPayload().target === 'attn_only'
        ? ['q_proj', 'k_proj', 'v_proj', 'o_proj'] : [],
    },
    lora_ds_config: loraPayload().ds_config,
    lora_liger: loraPayload().liger,
    ...data,
  };
}

// ---------------------------------------------------------------- actions
async function runSimulate() {
  if (!requireModel()) return;
  clearModelError();
  busy(true); setStatus('仿真计算中…', 'busy'); switchTab('summary');
  try {
    const payload = await api('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildPayload()),
    });
    state.lastResult = payload.result;
    renderSimulation(payload.result);
    $('#result-empty').hidden = true;
    setStatus(`完成 · ${payload.result.elapsed_ms} ms`, 'ok');
  } catch (error) {
    setStatus('仿真失败', 'bad');
    if (error.reasons && error.reasons.length) {
      showModelError(error);
    } else {
      toast(error.message, 'bad');
      $('#log-box').textContent = error.message;
      switchTab('log');
    }
  } finally { busy(false); }
}

async function runSolve() {
  if (!requireModel()) return;
  clearModelError();
  busy(true); setStatus('求解三类方案中…', 'busy'); switchTab('plans');
  try {
    // The v1 search endpoint returns the three plan types the product
    // defines (minimum / stable / performance) and runs each estimate in an
    // isolated process, so concurrent requests cannot corrupt each other.
    const payload = await api('/api/v1/training/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildV1Payload()),
    });
    state.plans = payload.plans;
    renderPlans(payload.plans, payload.plans?._meta, payload.mode_kernel_note);
    renderIssues(payload.issues);
    renderMeasured(payload.measured);
    $('#result-empty').hidden = true;
    updateDataDerived();
    setStatus(`求解完成 · ${payload.elapsed_ms} ms`, 'ok');
  } catch (error) {
    setStatus('求解失败', 'bad');
    if (error.reasons && error.reasons.length) {
      showModelError(error);
    } else {
      toast(error.message, 'bad');
      const box = $('#plans-box');
      if (box) box.innerHTML = `<div class="note">${escapeHtml(error.message)}</div>`;
      switchTab('log');
    }
  } finally { busy(false); }
}

/** Validate the current parameters without running an estimate. */
async function runValidate() {
  busy(true); setStatus('校验参数中…', 'busy'); switchTab('validate');
  try {
    const payload = await api('/api/v1/training/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildV1Payload()),
    });
    state.issues = payload.issues || [];
    renderIssues(state.issues);
    $('#result-empty').hidden = true;
    setStatus(payload.ok ? '参数合法' : '发现参数问题', payload.ok ? 'ok' : 'warn');
  } catch (error) {
    setStatus('校验失败', 'bad');
    toast(error.message, 'bad');
  } finally { busy(false); }
}

async function runSearch() {
  if (!requireModel()) return;
  busy(true); setStatus('并行策略搜索中…（组合较多，请稍候）', 'busy'); switchTab('solve');
  try {
    const payload = await api('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildPayload()),
    });
    renderSearch(payload.result);
    $('#result-empty').hidden = true;
    setStatus(`搜索完成 · 评估 ${payload.result.evaluated} 组 · ${payload.result.elapsed_ms} ms`, 'ok');
  } catch (error) {
    setStatus('搜索失败', 'bad');
    toast(error.message, 'bad');
    $('#solve-box').innerHTML = `<div class="note">${escapeHtml(error.message)}</div>`;
  } finally { busy(false); }
}

// ---------------------------------------------------------------- lora parameters
// LoRA has its own resource model, so it needs its own parameters: the
// adapter rank and alpha, the target modules, and the training stack
// (DeepSpeed stage, Liger's fused kernels) that the calibration was fitted
// against.
const LORA_FIELDS = [
  { key: 'lora_r', label: 'LoRA rank (r)', type: 'number', value: 8,
    hint: '实测数据为 r=8' },
  { key: 'lora_alpha', label: 'LoRA alpha', type: 'number', value: 16,
    hint: '缩放系数，通常为 r 的 2 倍' },
  { key: 'lora_dropout', label: 'LoRA dropout', type: 'number', step: 'any', value: 0.05,
    hint: '' },
];

function renderLoraFields() {
  const grid = $('#lora-grid');
  if (!grid) return;
  grid.innerHTML = '';
  LORA_FIELDS.forEach(field => {
    const input = el('input', {
      class: 'input', type: field.type, id: `lora-${field.key}`,
      value: field.value,
    });
    if (field.step) input.setAttribute('step', field.step);
    grid.appendChild(el('div', { class: 'field' },
      el('label', { class: 'field__label', for: `lora-${field.key}` }, field.label),
      input,
      field.hint ? el('p', { class: 'hint' }, field.hint) : null));
  });

  const stack = el('div', { class: 'field' },
    el('label', { class: 'field__label', for: 'lora-ds' }, 'DeepSpeed 阶段'));
  const dsSelect = el('select', { class: 'input input--select', id: 'lora-ds' });
  [['none', '不用 DeepSpeed'], ['z2', 'ZeRO-2'], ['z3', 'ZeRO-3']].forEach(([v, t]) => {
    dsSelect.appendChild(el('option', { value: v, selected: v === 'z3' }, t));
  });
  stack.appendChild(dsSelect);
  stack.appendChild(el('p', { class: 'hint' }, '实测库覆盖 none 与 z3'));
  grid.appendChild(stack);

  const ligerWrap = el('div', { class: 'field' },
    el('label', { class: 'field__label' }, 'Liger 融合算子'));
  const ligerInput = el('input', { type: 'checkbox', id: 'lora-liger', checked: true });
  const ligerLabel = el('label', { class: 'toggle is-on', for: 'lora-liger' }, ligerInput, '启用');
  ligerInput.addEventListener('change', () => {
    ligerLabel.classList.toggle('is-on', ligerInput.checked);
  });
  ligerWrap.appendChild(ligerLabel);
  ligerWrap.appendChild(el('p', { class: 'hint' }, '实测库覆盖 0 与 1；启用后显存显著下降'));
  grid.appendChild(ligerWrap);
}

/** Show the LoRA block only for the LoRA mode. */
function updateLoraVisibility() {
  const block = $('#lora-block');
  if (!block) return;
  const isLora = modeValue() === 'sft_lora';
  block.hidden = !isLora;
  if (isLora && !$('#lora-grid').children.length) renderLoraFields();
}

/** The LoRA block's values, in the shape the API expects. */
function loraPayload() {
  return {
    r: Number($('#lora-lora_r')?.value || 8),
    alpha: Number($('#lora-lora_alpha')?.value || 16),
    dropout: Number($('#lora-lora_dropout')?.value || 0.05),
    target: $('#lora-target')?.value || 'all',
    ds_config: $('#lora-ds')?.value || 'z3',
    liger: $('#lora-liger')?.checked ? 1 : 0,
  };
}

// ---------------------------------------------------------------- measured data
// The local bench.db holds real LoRA fine-tuning runs on BW1000. It is
// reference data from the same hardware — not a validation of the estimate,
// because the estimator models full-parameter training while LoRA freezes
// the base weights. The card says so instead of implying the numbers agree.
const MATCH_QUALITY_LABEL = {
  exact: '卡数 / 序列长度 / micro-batch 均一致',
  cards_and_seq: '卡数与序列长度一致，micro-batch 不同',
  cards_only: '仅卡数一致',
  model_only: '同模型，配置不同',
};

function renderMeasured(measured) {
  const card = $('#measured-card');
  const box = $('#measured-box');
  if (!card || !box) return;

  if (!measured || !measured.available) {
    card.hidden = true;
    box.innerHTML = '';
    return;
  }

  card.hidden = false;
  box.innerHTML = '';

  if (!measured.matched) {
    box.appendChild(el('div', { class: 'notice' },
      measured.reason || '实测库中没有该模型的数据'));
    return;
  }

  const rows = [];
  rows.push(['实测模型', (measured.model_names || []).join(' / ')]);
  rows.push(['实测记录数', `${measured.row_count} 条成功${measured.oom_count ? `，${measured.oom_count} 条 OOM` : ''}`]);
  rows.push(['匹配程度', MATCH_QUALITY_LABEL[measured.match_quality] || measured.match_quality]);

  const kv = el('dl', { class: 'kv' });
  rows.forEach(([k, v]) => {
    kv.appendChild(el('dt', {}, k));
    kv.appendChild(el('dd', {}, String(v)));
  });
  box.appendChild(kv);

  // Throughput: the one dimension where the two are broadly comparable.
  if (measured.estimate_tgs && measured.measured_tgs) {
    const dev = measured.tgs_deviation_percent;
    const tone = Math.abs(dev) <= 20 ? 'kpi--ok' : Math.abs(dev) <= 50 ? 'kpi--warn' : 'kpi--bad';
    const grid = el('div', { class: 'kpi-grid kpi-grid--compact' });
    grid.appendChild(kpi('估算 TGS', fmtInt(measured.estimate_tgs), '', '每卡 tokens/s'));
    grid.appendChild(kpi('实测 TGS', fmtInt(measured.measured_tgs), '', '每卡 tokens/s', tone));
    if (dev !== null && dev !== undefined) {
      grid.appendChild(kpi('偏差', `${dev > 0 ? '+' : ''}${fmt(dev, 1)}`, '%',
        dev > 0 ? '估算高于实测' : '估算低于实测', tone));
    }
    box.appendChild(grid);
  }

  // Memory: shown, but explicitly marked as not comparable.
  if (measured.estimate_peak_gib || measured.measured_peak_gib) {
    const memKv = el('dl', { class: 'kv' });
    memKv.appendChild(el('dt', {}, '估算峰值显存'));
    memKv.appendChild(el('dd', {}, measured.estimate_peak_gib != null
      ? `${fmt(measured.estimate_peak_gib, 2)} GiB（全参训练）` : '—'));
    memKv.appendChild(el('dt', {}, '实测峰值显存'));
    memKv.appendChild(el('dd', {}, measured.measured_peak_gib != null
      ? `${fmt(measured.measured_peak_gib, 2)} GiB（LoRA，基座冻结）` : '—'));
    box.appendChild(memKv);
  }

  const best = measured.best_match || {};
  const cfg = el('p', { class: 'hint' },
    `最佳实测配置：${best.gpu_count ?? '—'} 卡 · bs ${best.bs ?? '—'} · ga ${best.grad_accum ?? '—'} · `
    + `seq ${best.cutoff_len ?? '—'} · LoRA r=${best.lora_rank ?? '—'} · ${best.ds_config || 'none'} · `
    + `单步 ${best.step_time_s ?? '—'} s · 峰值 ${best.peak_mem_mib ?? '—'} MiB`);
  box.appendChild(cfg);

  box.appendChild(el('div', { class: 'notice notice--info' }, measured.comparability_note));

  if (measured.rows && measured.rows.length > 1) {
    const details = el('details', { class: 'measured-details' });
    details.appendChild(el('summary', {}, `查看全部 ${measured.rows.length} 条实测记录`));
    const table = el('table', { class: 'table' });
    table.appendChild(el('thead', {}, el('tr', {},
      el('th', {}, '卡数'), el('th', {}, 'bs'), el('th', {}, 'ga'), el('th', {}, 'seq'),
      el('th', {}, 'r'), el('th', {}, 'DS'), el('th', {}, '单步 s'),
      el('th', {}, '每卡 tok/s'), el('th', {}, '峰值 MiB'))));
    const tbody = el('tbody');
    measured.rows.forEach(r => {
      tbody.appendChild(el('tr', {},
        el('td', {}, String(r.gpu_count ?? '—')),
        el('td', {}, String(r.bs ?? '—')),
        el('td', {}, String(r.grad_accum ?? '—')),
        el('td', {}, String(r.cutoff_len ?? '—')),
        el('td', {}, String(r.lora_rank ?? '—')),
        el('td', {}, String(r.ds_config || 'none')),
        el('td', {}, String(r.step_time_s ?? '—')),
        el('td', {}, String(r.tokens_per_s ?? '—')),
        el('td', {}, String(r.peak_mem_mib ?? '—'))));
    });
    table.appendChild(tbody);
    details.appendChild(table);
    box.appendChild(details);
  }
}

// ---------------------------------------------------------------- images & launch
// The product hands the user three things: a model to download, an image to
// pull, and a script to start training. This section renders all three.
async function loadImages() {
  const select = $('#img-framework');
  const box = $('#images-box');
  if (!box) return;

  const framework = select?.value || '';
  box.innerHTML = '<p class="hint">加载中…</p>';
  try {
    const payload = await api(`/api/v1/images${framework ? `?framework=${encodeURIComponent(framework)}` : ''}`);
    renderImageCatalogue(payload);

    if (select && !select.dataset.filled) {
      select.dataset.filled = '1';
      select.appendChild(el('option', { value: '' }, '全部框架'));
      (payload.frameworks || []).forEach(f => {
        select.appendChild(el('option', { value: f.framework }, `${f.framework} (${f.tag_count})`));
      });
    }
  } catch (error) {
    box.innerHTML = `<div class="notice notice--error">镜像目录加载失败：${escapeHtml(error.message)}</div>`;
  }
}

function renderImageCatalogue(payload) {
  const box = $('#images-box');
  box.innerHTML = '';
  const images = payload.images || [];
  if (!images.length) {
    box.appendChild(el('p', { class: 'hint' }, '没有匹配的镜像。'));
    return;
  }

  const source = payload.source || {};
  box.appendChild(el('p', { class: 'hint' },
    `镜像来源：${source.path || '未知'}${source.fetched_at ? `（更新于 ${source.fetched_at}）` : ''}`));

  images.forEach(image => {
    const card = el('div', { class: 'card', style: 'margin-bottom:10px' });
    card.appendChild(el('h4', {}, image.repo));
    if (image.abstract) card.appendChild(el('p', { class: 'hint' }, image.abstract));

    const table = el('table', { class: 'table' });
    const thead = el('thead');
    thead.appendChild(el('tr', {},
      el('th', {}, 'Tag'), el('th', {}, '框架'), el('th', {}, 'DTK / Python / OS'),
      el('th', {}, '大小'), el('th', {}, '拉取命令')));
    table.appendChild(thead);

    const tbody = el('tbody');
    image.tags.slice(0, 6).forEach(tag => {
      tbody.appendChild(el('tr', {},
        el('td', {}, tag.name),
        el('td', {}, tag.framework || '—'),
        el('td', {}, [tag.dtk_version, tag.python_version, tag.os_version].filter(Boolean).join(' / ') || '—'),
        el('td', {}, tag.size || '—'),
        el('td', {}, el('code', {}, tag.pull_url || '—'))));
    });
    table.appendChild(tbody);
    card.appendChild(table);

    if (image.detail_url) {
      card.appendChild(el('p', { class: 'hint' },
        el('a', { href: image.detail_url, target: '_blank', rel: 'noopener' }, '查看镜像详情 →')));
    }
    box.appendChild(card);
  });
}

/** Load the training templates for the selected mode. */
async function loadTemplates() {
  const select = $('#tpl-select');
  if (!select) return;
  select.innerHTML = '<option value="">加载中…</option>';
  try {
    const payload = await api(`/api/v1/templates?mode=${encodeURIComponent(modeValue())}`);
    select.innerHTML = '';
    if (!payload.templates.length) {
      select.appendChild(el('option', { value: '' }, '该训练类型暂无模板'));
      return;
    }
    payload.templates.forEach(t => {
      select.appendChild(el('option', { value: t.id },
        `${t.name}${t.verified ? '' : '（未在目标环境验证）'}`));
    });
  } catch (error) {
    select.innerHTML = '<option value="">模板加载失败</option>';
    toast(error.message, 'bad');
  }
}

// ------------------------------------------------------------- support matrix
// The kernel models a fixed set of architectures. This table says, for every
// preset model, whether it can be estimated and what is missing if not.
let supportMatrix = null;

async function loadSupportMatrix() {
  const box = $('#support-box');
  const summary = $('#support-summary');
  if (!box) return;
  try {
    const data = await api('/api/v1/model-support');
    supportMatrix = data;
    renderSupportFamilies();
    renderSupportSummary(data);
    renderSupportRows();
  } catch (error) {
    box.innerHTML = '';
    box.appendChild(el('p', { class: 'hint hint--bad' },
      `加载支持矩阵失败：${error.message}`));
    if (summary) summary.innerHTML = '';
  }
}

function renderSupportSummary(data) {
  const box = $('#support-summary');
  if (!box) return;
  const totals = data.totals || {};
  box.innerHTML = '';

  // The upstream simulator ships the same six adapters, so the limit is
  // the kernel's architecture coverage — not this tool's. Show both
  // behaviours so the difference is visible.
  const upstream = data.upstream;
  if (upstream) {
    box.appendChild(el('div', { class: 'notice notice--info' }, [
      el('strong', {}, `上游 hcu-train-simulator 同样是 ${upstream.adapter_count} 个适配器`),
      el('span', {}, `（${upstream.adapters.join('、')}）`),
      el('p', { class: 'hint' }, upstream.fallback_behaviour),
      el('p', { class: 'hint' }, upstream.gate_behaviour),
      el('p', { class: 'hint' }, upstream.gate_note),
    ]));
  }
  const row = el('div', { class: 'support-totals' });
  [
    ['可精确估算', totals.estimable, 'ok'],
    ['可估算（近似）', totals.approximate, 'warn'],
    ['不可估算', totals.not_estimable, 'bad'],
  ].forEach(([label, count, kind]) => {
    row.appendChild(el('div', { class: `support-total support-total--${kind}` }, [
      el('strong', {}, String(count ?? 0)),
      el('span', {}, label),
    ]));
  });
  box.appendChild(row);

  // The family modules this project adds on top of the kernel.
  if (Array.isArray(data.family_modules) && data.family_modules.length) {
    const list = el('div', { class: 'support-families' });
    list.appendChild(el('h4', {}, '系列通用模块'));
    data.family_modules.forEach(module => {
      list.appendChild(el('div', { class: 'support-family' }, [
        el('strong', {}, module.adapter),
        el('span', { class: 'hint' }, module.covers || ''),
        el('span', { class: 'hint' }, module.capability || ''),
      ]));
    });
    box.appendChild(list);
  }

  // The adapter table explains what each adapter can represent.
  if (Array.isArray(data.adapters) && data.adapters.length) {
    const list = el('div', { class: 'support-adapters' });
    data.adapters.forEach(adapter => {
      list.appendChild(el('div', { class: 'support-adapter' }, [
        el('code', {}, adapter.name),
        el('span', { class: 'hint' }, adapter.description || ''),
      ]));
    });
    box.appendChild(list);
  }
}

function renderSupportRows() {
  const box = $('#support-box');
  if (!box || !supportMatrix) return;
  const family = $('#support-family') ? $('#support-family').value : '';
  const rows = (supportMatrix.models || []).filter(
    row => !family || (row.family || '其他') === family);

  box.innerHTML = '';
  if (!rows.length) {
    box.appendChild(el('p', { class: 'hint' }, '该系列没有预设模型。'));
    return;
  }

  const table = el('table', { class: 'support-table' });
  table.appendChild(el('thead', {}, el('tr', {}, [
    el('th', {}, '模型'),
    el('th', {}, '状态'),
    el('th', {}, '适配器'),
    el('th', {}, '说明'),
  ])));
  const tbody = el('tbody');
  rows.forEach(row => {
    const kind = row.status === 'estimable' ? 'ok'
      : row.status === 'approximate' ? 'warn' : 'bad';
    const detail = row.status === 'not_estimable'
      ? (row.reasons || []).join('；')
      : (row.warnings || []).join('；') || row.adapter_description || '';
    tbody.appendChild(el('tr', {}, [
      el('td', {}, el('code', {}, row.reference)),
      el('td', {}, el('span', { class: `badge badge--${kind}` }, row.status_label)),
      el('td', {}, row.adapter || '—'),
      el('td', { class: 'support-detail' }, detail),
    ]));
  });
  table.appendChild(tbody);
  box.appendChild(table);
}

function renderSupportFamilies() {
  const select = $('#support-family');
  if (!select || !supportMatrix) return;
  const families = Object.keys(supportMatrix.families || {}).sort();
  select.innerHTML = '';
  select.appendChild(el('option', { value: '' }, '全部系列'));
  families.forEach(name => {
    const counts = supportMatrix.families[name] || {};
    select.appendChild(el('option', { value: name },
      `${name}（${counts.estimable || 0} 可算 / ${counts.not_estimable || 0} 不可算）`));
  });
}

/** Render the YAML / shell / docker artefacts for the selected template. */
async function renderLaunch() {
  const box = $('#launch-box');
  const templateId = $('#tpl-select')?.value;
  if (!templateId) { toast('请先选择训练模板', 'bad'); return; }

  const plan = state.plans?.performance || state.plans?.stable || state.plans?.minimum;
  if (!plan) { toast('请先求解方案，再生成启动配置', 'bad'); return; }

  box.innerHTML = '<p class="hint">生成中…</p>';
  try {
    const payload = await api('/api/v1/launch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        template_id: templateId,
        plan,
        model_path: $('#tpl-model')?.value.trim() || 'YOUR_MODEL_PATH',
        output_dir: '/workspace/output',
        work_dir: '/workspace',
        image: $('#tpl-image')?.value.trim() || null,
        nodes: 1,
      }),
    });

    box.innerHTML = '';
    if (!payload.verified) {
      box.appendChild(el('div', { class: 'notice notice--warn' },
        '该模板尚未在目标海光环境验证，命令仅供参考，请勿直接用于生产训练。'));
    }

    [['YAML 训练配置', payload.yaml], ['Shell 启动脚本', payload.shell], ['Docker 命令', payload.docker]]
      .forEach(([title, content]) => {
        const card = el('div', { class: 'card', style: 'margin-top:10px' });
        card.appendChild(el('h4', {}, title));
        const pre = el('pre', { class: 'log' }, content);
        card.appendChild(pre);
        const copy = el('button', { class: 'btn btn--ghost' },
          '复制');
        copy.addEventListener('click', async () => {
          try {
            await navigator.clipboard.writeText(content);
            toast('已复制到剪贴板', 'ok');
          } catch (_) { toast('复制失败，请手动选择文本', 'bad'); }
        });
        card.appendChild(copy);
        box.appendChild(card);
      });
  } catch (error) {
    box.innerHTML = `<div class="notice notice--error">生成失败：${escapeHtml(error.message)}</div>`;
  }
}

// ---------------------------------------------------------------- rendering
function renderSimulation(result) {
  const s = result.summary;
  const memClass = result.peak_memory_gib > result.hbm_gib ? 'kpi--bad'
    : result.peak_memory_gib > result.hbm_gib * 0.9 ? 'kpi--warn' : 'kpi--ok';

  const grid = $('#kpi-grid');
  grid.innerHTML = '';
  grid.appendChild(kpi('迭代时间', fmt(s.iteration_ms, 1), 'ms',
    `计算 ${fmt(s.compute_ms, 0)} + 通信 ${fmt(s.communication_ms, 0)}`));
  grid.appendChild(kpi('TGS 吞吐', fmtInt(s.tgs), '', `tokens / p / s · ${s.tokens_per_iteration.toLocaleString()} tokens/iter`));
  grid.appendChild(kpi('MFU', fmt(s.mfu_percent, 2), '%', `峰值算力 ${s.actual_tflops} TFLOPS`));
  grid.appendChild(kpi('参数量', fmt(s.model_size_b, 2), 'B', result.model_adapter));
  grid.appendChild(kpi('峰值显存', fmt(result.peak_memory_gib, 2), `GiB / ${result.hbm_gib} GiB`,
    `占用 ${fmt(result.peak_memory_gib / result.hbm_gib * 100, 1)}%`, memClass));
  grid.appendChild(kpi('通信占比', fmt(s.comm_percent, 1), '%', `计算占比 ${fmt(s.compute_percent, 1)}%`,
    s.comm_percent > 50 ? 'kpi--warn' : ''));

  renderMeasured(result.measured);
  renderParallelCard(result);
  renderMemoryTable(result);
  renderComputeTable(result);
  renderCommTable(result);
  renderAdvice(result);
  $('#log-box').textContent = result.log || '(无日志输出)';
}

function kpi(label, value, unit, foot, cls = '') {
  return el('div', { class: 'kpi ' + cls },
    el('div', { class: 'kpi__label' }, label),
    el('div', { class: 'kpi__value' }, value, unit ? el('span', { class: 'kpi__unit' }, unit) : null),
    foot ? el('div', { class: 'kpi__foot' }, foot) : null);
}

function renderParallelCard(result) {
  const p = result.parallel;
  const box = $('#parallel-card');
  box.innerHTML = '';
  box.appendChild(el('dl', { class: 'kv' },
    el('dt', {}, '总卡数 / 节点'), el('dd', {}, `${p.num_gpus} 卡 · ${Math.ceil(p.num_gpus / 8)} 节点`),
    el('dt', {}, 'TP / CP / PP'), el('dd', {}, `${p.tp_size} / ${p.cp_size} / ${p.pp_size}`),
    el('dt', {}, 'DP / EP / ETP'), el('dd', {}, `${p.dp_size} / ${p.ep_size} / ${p.etp_size}`),
    el('dt', {}, 'micro / global batch'), el('dd', {}, `${p.micro_batch_size} / ${p.global_batch_size}`),
    el('dt', {}, '序列长度'), el('dd', {}, fmtInt(p.seq_length)),
    el('dt', {}, '每个 DP 的微批数'), el('dd', {}, fmt(p.num_microbatches, 2)),
    el('dt', {}, '流水调度'), el('dd', {}, p.pp_schedule + (p.num_layers_per_vp_stage ? ` (VPP 每虚流水 ${p.num_layers_per_vp_stage} 层)` : '')),
    el('dt', {}, 'FP8 / 全量重计算'), el('dd', {}, `${p.use_fp8_training ? '开' : '关'} / ${p.full_recompute ? '开' : '关'}`),
    el('dt', {}, '通信掩盖模式'), el('dd', {}, p.overlap_mode),
  ));
  $('#model-card').innerHTML = '';
  const summary = state.model || state.customSummary;
  const card = $('#model-card');
  card.innerHTML = '';
  if (state.model) {
    card.appendChild(el('dl', { class: 'kv' },
      el('dt', {}, '模型'), el('dd', {}, state.model.name),
      el('dt', {}, 'HF 仓库'), el('dd', {}, state.model.hf_id),
      el('dt', {}, '参数量'), el('dd', {}, `${state.lastResult.summary.model_size_b} B`),
      el('dt', {}, '适配器'), el('dd', {}, result.model_adapter),
      el('dt', {}, '层数 / hidden'), el('dd', {}, `${state.model.num_layers} / ${state.model.hidden_size}`),
      el('dt', {}, 'heads / KV 组'), el('dd', {}, `${state.model.num_attention_heads} / ${state.model.num_query_groups}`),
      el('dt', {}, '专家数'), el('dd', {}, state.model.num_experts || '—（稠密）'),
    ));
  } else if (summary) {
    card.appendChild(el('dl', { class: 'kv' },
      el('dt', {}, '模型'), el('dd', {}, state.modelRef?.startsWith('path:') ? '自定义 config.json' : state.modelRef),
      el('dt', {}, '参数量'), el('dd', {}, `${result.summary.model_size_b} B`),
      el('dt', {}, '适配器'), el('dd', {}, result.model_adapter),
      el('dt', {}, '层数 / hidden'), el('dd', {}, `${summary.num_hidden_layers} / ${summary.hidden_size}`),
      el('dt', {}, 'heads / KV 组'), el('dd', {}, `${summary.num_attention_heads} / ${summary.num_query_groups}`),
    ));
  }
}

function renderMemoryTable(result) {
  const box = $('#memory-table');
  box.innerHTML = '';
  box.appendChild(el('div', { class: 'note note--info' },
    `可行性判定：峰值显存 ${fmt(result.peak_memory_gib, 2)} GiB ${result.peak_memory_gib <= result.hbm_gib ? '≤' : '>'} 单卡容量 ${result.hbm_gib} GiB` +
    (result.peak_memory_gib <= result.hbm_gib ? '，可放下。' : '，放不下，需要增加卡数或降低批次/序列长度。')));

  const table = el('table', { class: 'data' });
  table.appendChild(el('thead', {}, el('tr', {},
    el('th', {}, 'PP Stage'), el('th', {}, '权重/梯度/优化器'),
    el('th', {}, '激活值'), el('th', {}, '合计'), el('th', {}, '占用率'), el('th', {}, ''))));
  const tbody = el('tbody');
  result.memory.forEach(stage => {
    const ratio = stage.total_gib / stage.hbm_gib;
    const cls = ratio > 1 ? 'bad' : ratio > 0.9 ? 'warn' : 'ok';
    tbody.appendChild(el('tr', {},
      el('td', { class: 'name' }, `#${stage.pp_rank}`),
      el('td', { class: 'num' }, `${fmt(stage.wgo_gib)} GiB`),
      el('td', { class: 'num' }, `${fmt(stage.activation_gib)} GiB`),
      el('td', { class: 'num' }, `${fmt(stage.total_gib)} GiB`),
      el('td', { class: 'num' }, `${fmt(stage.utilization, 1)}%`),
      el('td', {}, el('div', { class: `bar bar--${cls}` }, el('i', { style: `width:${Math.min(100, ratio * 100)}%` })))));
  });
  table.appendChild(tbody);
  box.appendChild(el('div', { class: 'scroll-y' }, table));
}

function renderComputeTable(result) {
  const box = $('#compute-table');
  box.innerHTML = '';
  const rows = [...result.compute_details].sort((a, b) => (b.time_ms || 0) - (a.time_ms || 0));
  const total = rows.reduce((sum, r) => sum + (r.time_ms || 0), 0) || 1;

  const table = el('table', { class: 'data' });
  table.appendChild(el('thead', {}, el('tr', {},
    el('th', {}, '算子模块'), el('th', {}, '类型'), el('th', {}, '调用次数'),
    el('th', {}, '耗时 ms'), el('th', {}, '占比'), el('th', {}, '来源'))));
  const tbody = el('tbody');
  rows.forEach(row => {
    const share = (row.time_ms || 0) / total * 100;
    const source = row.benchmark_source || 'theoretical';
    const tagCls = source === 'theoretical' ? 'tag--warn' : 'tag--ok';
    const label = source === '/' ? '理论值' : String(source);
    tbody.appendChild(el('tr', {},
      el('td', { class: 'name' }, row.model_part || '—'),
      el('td', { class: 'name' }, row.op_type || '—'),
      el('td', { class: 'num' }, fmtInt(row.count)),
      el('td', { class: 'num' }, fmt(row.time_ms, 3)),
      el('td', { class: 'num' }, `${fmt(share, 1)}%`),
      el('td', {}, el('span', { class: `tag ${tagCls}` }, label))));
  });
  table.appendChild(tbody);
  box.appendChild(el('div', { class: 'scroll-y' }, table));
}

function renderCommTable(result) {
  const box = $('#comm-table');
  box.innerHTML = '';
  const table = el('table', { class: 'data' });
  table.appendChild(el('thead', {}, el('tr', {},
    el('th', {}, '通信域'), el('th', {}, '规模'), el('th', {}, '原始耗时'),
    el('th', {}, '掩盖比例'), el('th', {}, '暴露耗时'), el('th', {}, '占比'))));
  const tbody = el('tbody');
  result.communication.forEach(row => {
    const size = row.size ? `${row.size}` : '—';
    tbody.appendChild(el('tr', {},
      el('td', { class: 'name' }, row.domain),
      el('td', { class: 'num' }, size),
      el('td', { class: 'num' }, `${fmt(row.raw_time_ms)} ms`),
      el('td', { class: 'num' }, `${fmt(row.overlap_percent, 1)}%`),
      el('td', { class: 'num' }, `${fmt(row.time_ms)} ms`),
      el('td', { class: 'num' }, `${fmt(row.percent, 2)}%`)));
  });
  table.appendChild(tbody);
  box.appendChild(table);
}

function renderAdvice(result) {
  const box = $('#advice-box');
  box.innerHTML = '';
  const s = result.summary;
  const items = [];

  const ratio = result.peak_memory_gib / result.hbm_gib;
  if (ratio > 1) {
    items.push(['bad', '✖', `峰值显存 ${fmt(result.peak_memory_gib)} GiB 超出单卡 ${result.hbm_gib} GiB，当前配置无法训练。请增加卡数（TP/PP/EP）、减小 micro batch 或序列长度，或改用「配置求解」获取可行方案。`]);
  } else if (ratio > 0.9) {
    items.push(['warn', '⚠', `显存占用已达 ${fmt(ratio * 100, 1)}%，余量不足 ${fmt(result.hbm_gib - result.peak_memory_gib)} GiB，建议预留 5 GiB 安全余量。`]);
  } else {
    items.push(['ok', '✔', `显存占用 ${fmt(ratio * 100, 1)}%，剩余 ${fmt(result.hbm_gib - result.peak_memory_gib)} GiB，配置可放下。`]);
  }

  const dominant = [...result.communication].sort((a, b) => b.time_ms - a.time_ms)[0];
  if (dominant && dominant.time_ms > 0) {
    if (dominant.percent > 50) {
      items.push(['warn', '⚠', `通信以 ${dominant.domain} 为主，占迭代时间 ${fmt(dominant.percent, 1)}%（暴露 ${fmt(dominant.time_ms)} ms）。可尝试减小 ${dominant.domain} 通信域规模、开启 VPP / EP overlap，或把 overlap_mode 切到 manual 并调高掩盖比例。`]);
    }
  }
  if (s.mfu < 0.3) {
    items.push(['warn', '⚠', `MFU 仅 ${fmt(s.mfu_percent, 1)}%，算力利用偏低。若瓶颈在计算，可增大 batch 或选择让算子 shape 更大的并行切分；若瓶颈在通信，参考上一条。`]);
  } else if (s.mfu > 0.5) {
    items.push(['ok', '✔', `MFU ${fmt(s.mfu_percent, 1)}%，算力利用良好。`]);
  }

  const p = result.parallel;
  if (p.pp_size > 1 && !p.num_layers_per_vp_stage) {
    const bubbles = (p.pp_size - 1) / (p.num_microbatches || 1);
    if (bubbles > 0.15) {
      items.push(['warn', '⚠', `PP=${p.pp_size} 且每个 DP 仅 ${fmt(p.num_microbatches, 1)} 个微批，流水气泡约 ${fmt(bubbles * 100, 1)}%（>15%）。可开启 VPP（interleaved_1f1b）或增大 global_batch_size。`]);
    }
  }
  items.push(['', 'ℹ', '迭代时间 = 计算时间 + 通信暴露时间（已扣除 overlap 掩盖与 PP 气泡）。该值为方案对比与瓶颈分析的规划估算，不等于真实训练作业耗时。']);

  items.forEach(([cls, icon, text]) => {
    box.appendChild(el('div', { class: `advice__item${cls ? ' advice__item--' + cls : ''}` },
      el('span', { class: 'advice__icon' }, icon),
      el('span', {}, text)));
  });
}

// ---------------------------------------------------------------- three plans
// The product defines three plan types (plan §3.1). They come from the v1
// search endpoint, which returns them already separated.
function renderPlans(plans, meta, kernelNote) {
  const box = $('#plans-box');
  if (!box) return;
  box.innerHTML = '';

  // If two modes share one kernel, the numbers come out identical — say so
  // next to the numbers rather than letting the user think it is a bug.
  const note = $('#plans-kernel-note');
  if (note) {
    if (kernelNote) {
      note.hidden = false;
      note.textContent = kernelNote;
    } else {
      note.hidden = true;
      note.textContent = '';
    }
  }

  const order = [
    ['minimum', '最低可训练方案', '以最小卡数为目标；必须预留安全显存余量，不能按 100% 显存占满计算。'],
    ['stable', '推荐稳定方案', (meta && meta.stable_label) || '保留更高显存余量，避免极端 Batch、严重 PP 不均衡与过高通信占比，适合持续训练。'],
    ['performance', '推荐性能方案', '在指定资源上限内按 TGS 排序；相对最低方案的提速与瓶颈见下方说明。'],
  ];

  const grid = el('div', { class: 'solve-grid' });
  order.forEach(([key, title, desc]) => {
    grid.appendChild(planCard(title, desc, (plans || {})[key]));
  });
  box.appendChild(grid);

  if (meta && meta.stable_reasons && meta.stable_reasons.length) {
    box.appendChild(el('div', { class: 'note' },
      '推荐稳定方案已降级：' + meta.stable_reasons.join('；')));
  }

  const min = (plans || {}).minimum;
  const perf = (plans || {}).performance;
  if (min && perf && perf.tgs && min.tgs) {
    const speedup = perf.tgs / min.tgs;
    const extra = (perf.num_gpus || 0) - (min.num_gpus || 0);
    const rows = [
      ['吞吐提升', `${fmt(speedup, 2)}×（${fmtInt(min.tgs)} → ${fmtInt(perf.tgs)} token/s）`],
      ['卡数变化', extra === 0 ? '相同卡数，仅并行策略不同' : `${extra > 0 ? '+' : ''}${extra} 卡`],
    ];
    const comm = perf.communication_time_ms && perf.iteration_ms
      ? (perf.communication_time_ms / perf.iteration_ms) * 100 : null;
    if (comm !== null) {
      rows.push(['性能方案通信占比', `${fmt(comm, 1)}%`]);
    }
    const box2 = el('dl', { class: 'kv' });
    rows.forEach(([k, v]) => {
      box2.appendChild(el('dt', {}, k));
      box2.appendChild(el('dd', {}, v));
    });
    box.appendChild(el('div', { class: 'card', style: 'margin-top:12px' },
      el('h3', {}, '方案对比'), box2));
  }
}

function planCard(title, desc, plan) {
  const card = el('div', { class: 'solve-card' });
  card.appendChild(el('h4', {}, title));
  card.appendChild(el('div', { class: 'desc' }, desc));
  if (!plan || !plan.num_gpus) {
    card.appendChild(el('p', { class: 'hint' }, '无可行方案'));
    return card;
  }
  const kv = el('dl', { class: 'kv' });
  const rows = [
    ['总卡数', `${plan.num_gpus} 卡`],
    ['TP / PP / CP / EP', `${plan.tp_size} / ${plan.pp_size} / ${plan.cp_size} / ${plan.ep_size}`],
    ['峰值显存', `${plan.peak_gib} GiB`],
    ['　权重+梯度+优化器', `${plan.wgo_gib} GiB`],
    ['　激活值', `${plan.activation_gib} GiB`],
    ['单步时间', plan.iteration_ms ? `${fmtInt(plan.iteration_ms)} ms` : '—'],
    ['通信占比', (plan.communication_time_ms && plan.iteration_ms)
      ? `${fmt((plan.communication_time_ms / plan.iteration_ms) * 100, 1)}%` : '—'],
    ['MFU', plan.mfu != null ? `${fmt(plan.mfu * 100, 2)}%` : '—'],
    ['吞吐 TGS', plan.tgs != null ? `${fmtInt(plan.tgs)} token/s` : '—'],
  ];
  rows.forEach(([k, v]) => {
    kv.appendChild(el('dt', {}, k));
    kv.appendChild(el('dd', {}, v));
  });
  card.appendChild(kv);
  return card;
}

// ---------------------------------------------------------------- validation
function renderIssues(issues) {
  const box = $('#validate-box');
  if (!box) return;
  box.innerHTML = '';
  const list = issues || [];
  if (!list.length) {
    box.appendChild(el('p', { class: 'hint' }, '未发现参数问题。'));
    return;
  }
  list.forEach(issue => {
    const cls = issue.severity === 'error' ? 'notice notice--error' : 'notice notice--warn';
    const node = el('div', { class: cls },
      el('strong', {}, issue.field), '：', issue.message);
    if (issue.suggestion) {
      node.appendChild(el('div', { class: 'hint' }, '建议：' + issue.suggestion));
    }
    box.appendChild(node);
  });
}

function renderSolve(result) {
  const box = $('#solve-box');
  box.innerHTML = '';
  if (result.assumptions && result.assumptions.length) {
    box.appendChild(el('div', { class: 'note' },
      '求解器降级说明：' + result.assumptions.join('；')));
  }
  const grid = el('div', { class: 'solve-grid' });
  grid.appendChild(solveCard(
    '最小卡数方案',
    '在 5 GiB 显存余量与 15% 流水气泡约束下的最少卡数配置。该方案按「峰值显存最省」择优，所以常落在切分最细的 TP 上——它省的是卡，不是时间；要吞吐请看右侧方案。',
    result.minimum));
  grid.appendChild(solveCard('相对高性能方案', '参考仓库按 2 倍世界规模搜索出的高性能配置', result.performance));
  box.appendChild(grid);

  if (!result.minimum && !result.performance) {
    box.appendChild(el('div', { class: 'note' }, '求解器未返回可行方案。'));
  }
}

function solveCard(title, desc, row) {
  const card = el('div', { class: 'solve-card' });
  card.appendChild(el('h4', {}, title));
  card.appendChild(el('div', { class: 'desc' }, desc));
  if (!row) {
    card.appendChild(el('p', { class: 'hint' }, '无可行方案'));
    return card;
  }
  const p = row.parallel;
  const perf = row.performance || {};
  card.appendChild(el('dl', { class: 'kv' },
    el('dt', {}, '总卡数'), el('dd', {}, `${p.num_gpus} 卡`),
    el('dt', {}, 'TP / CP / PP / DP'), el('dd', {}, `${p.tp_size} / ${p.cp_size} / ${p.pp_size} / ${p.dp_size}`),
    el('dt', {}, 'EP / ETP'), el('dd', {}, `${p.ep_size} / ${p.etp_size}`),
    el('dt', {}, 'VPP'), el('dd', {}, p.num_layers_per_vp_stage ? `${p.num_layers_per_vp_stage} 层 × ${p.vp_size || '?'}` : '未启用'),
    el('dt', {}, 'micro / global batch'), el('dd', {}, `${p.micro_batch_size} / ${p.global_batch_size}`),
    el('dt', {}, '峰值显存'), el('dd', {}, `${fmt(row.peak_memory_gib)} GiB${row.peak_pp_rank != null ? ` (stage #${row.peak_pp_rank})` : ''}`),
    el('dt', {}, '迭代时间'), el('dd', {}, perf.iteration_ms != null ? `${fmt(perf.iteration_ms)} ms` : '—'),
    el('dt', {}, 'TGS'), el('dd', {}, perf.tgs != null ? fmtInt(perf.tgs) : '—'),
    el('dt', {}, 'MFU'), el('dd', {}, perf.mfu != null ? `${fmt(perf.mfu * 100, 2)}%` : '—'),
  ));
  return card;
}

function renderSearch(result) {
  const box = $('#solve-box');
  box.innerHTML = '';
  box.appendChild(el('div', { class: 'note note--info' },
    `共评估 ${result.evaluated} 个组合，其中 ${result.feasible} 个可行（峰值显存 ≤ ${fmt(result.memory_limit_gib)} GiB，已扣 5 GiB 余量）。按 TGS 排序，搜索使用与主仿真一致的算子 Profile。`));

  if (!result.top.length) {
    box.appendChild(el('div', { class: 'note' }, '没有可行组合，请放宽候选范围或降低显存余量。'));
    return;
  }

  const table = el('table', { class: 'data' });
  table.appendChild(el('thead', {}, el('tr', {},
    el('th', {}, '#'), el('th', {}, '卡数'), el('th', {}, 'TP'), el('th', {}, 'CP'),
    el('th', {}, 'PP'), el('th', {}, 'DP'), el('th', {}, 'EP'), el('th', {}, 'ETP'),
    el('th', {}, 'VPP'), el('th', {}, 'micro'), el('th', {}, '峰值显存'),
    el('th', {}, '迭代 ms'), el('th', {}, 'TGS'), el('th', {}, 'MFU'))));
  const tbody = el('tbody');
  result.top.forEach((row, index) => {
    tbody.appendChild(el('tr', {},
      el('td', { class: 'num' }, index + 1),
      el('td', { class: 'num' }, row.num_gpus),
      el('td', { class: 'num' }, row.tp_size),
      el('td', { class: 'num' }, row.cp_size),
      el('td', { class: 'num' }, row.pp_size),
      el('td', { class: 'num' }, row.dp_size),
      el('td', { class: 'num' }, row.ep_size),
      el('td', { class: 'num' }, row.etp_size ?? '—'),
      el('td', { class: 'num' },
        row.num_layers_per_vp_stage ? `${row.vp_size}×${row.num_layers_per_vp_stage}层` : '—'),
      el('td', { class: 'num' }, row.micro_batch_size),
      el('td', { class: 'num' }, `${fmt(row.peak_memory_gib)} GiB`),
      el('td', { class: 'num' }, fmtInt(row.iteration_ms)),
      el('td', { class: 'num' }, fmtInt(row.tgs)),
      el('td', { class: 'num' }, `${fmt(row.mfu * 100, 2)}%`)));
  });
  table.appendChild(tbody);
  box.appendChild(el('div', { class: 'scroll-y' }, table));
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

// ---------------------------------------------------------------- tabs
function switchTab(name) {
  $$('.tab').forEach(t => t.classList.toggle('is-active', t.dataset.tab === name));
  $$('.tab-pane').forEach(p => p.classList.toggle('is-active', p.dataset.pane === name));
}

// ---------------------------------------------------------------- resolve actions
async function resolveModel(source) {
  busy(true);
  try {
    // ModelScope goes through the v1 API, which returns the config plus its
    // provenance (repo, revision, config hash) so the estimate stays
    // reproducible and auditable.
    if (source === 'modelscope') {
      await resolveModelScope();
      return;
    }

    let body, options = { method: 'POST' };
    if (source === 'hf') {
      const ref = $('#hf-ref').value.trim();
      if (!ref) { toast('请输入模型名或链接', 'bad'); return; }
      options.headers = { 'Content-Type': 'application/json' };
      body = JSON.stringify({ source: 'hf', hf_ref: ref });
    } else if (source === 'upload') {
      const file = $('#file-input').files[0];
      if (!file) { toast('请选择 config.json 文件', 'bad'); return; }
      const form = new FormData();
      form.append('source', 'upload');
      form.append('file', file);
      body = form;
    } else {
      const text = $('#paste-area').value.trim();
      if (!text) { toast('请粘贴 config.json 内容', 'bad'); return; }
      options.headers = { 'Content-Type': 'application/json' };
      body = JSON.stringify({ source: 'paste', config_text: text });
    }

    const payload = await api('/api/resolve-model', options.body ? { ...options, body } : options);
    state.customSummary = payload.summary;
    state.model = null;
    state.modelRef = payload.model_ref;
    state.modelSlug = null;
    state.modelConfigPath = null;
    const selMap = { hf: '#hf-summary', upload: '#upload-summary', paste: '#paste-summary' };
    renderCustomSummary(selMap[source], payload.summary, payload.label);
    toast(`已解析：适配器 ${payload.summary.adapter || '未匹配'}`, 'ok');
  } catch (error) {
    toast(error.message, 'bad');
  } finally { busy(false); }
}

/**
 * Resolve a model config from ModelScope (plan §8).
 *
 * The server enforces the domain allowlist, size cap and timeout; the page
 * only reports what came back. On failure the user can still paste or
 * upload a config.json by hand.
 */
async function resolveModelScope() {
  const ref = $('#ms-ref').value.trim();
  if (!ref) { toast('请输入 ModelScope 仓库名或链接', 'bad'); return; }

  const payload = await api('/api/v1/model-configs/resolve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source: 'modelscope', reference: ref }),
  });

  state.modelConfig = payload.config;
  state.modelConfigPath = null;
  state.modelSource = payload.provenance || null;
  state.modelFacts = payload.facts || null;
  state.modelRef = payload.provenance?.repo_id || ref;
  // The backend resolves a bare reference that is not a catalogue slug as
  // a ModelScope repository, and caches the config on its side.
  state.modelSource = payload.provenance || null;
  state.modelSlug = null;
  state.model = null;
  state.customSummary = null;

  renderModelScopeSummary(payload);
  toast('已从 ModelScope 解析 config.json', 'ok');
}

/** Render the ModelScope pane: architecture facts plus provenance. */
function renderModelScopeSummary(payload) {
  const box = $('#ms-summary');
  if (!box) return;
  const facts = payload.facts || {};
  const prov = payload.provenance || {};
  const rows = [
    ['模型', prov.repo_id || '—'],
    ['Revision', prov.revision || '—'],
    ['配置哈希', prov.config_hash ? prov.config_hash.slice(0, 16) + '…' : '—'],
    ['层数', facts.num_hidden_layers || '—'],
    ['隐藏维度', facts.hidden_size || '—'],
    ['注意力头', facts.num_attention_heads || '—'],
    ['KV 头', facts.num_key_value_heads || '—'],
    ['词表大小', facts.vocab_size || '—'],
    ['MoE', facts.is_moe ? `是（${facts.n_routed_experts} 专家）` : '否'],
  ];
  const kv = el('dl', { class: 'kv' });
  rows.forEach(([k, v]) => {
    kv.appendChild(el('dt', {}, k));
    kv.appendChild(el('dd', {}, String(v)));
  });
  box.innerHTML = '';
  box.appendChild(kv);
}

// ---------------------------------------------------------------- events
function bindEvents() {
  $$('[data-model-tab]').forEach(btn => {
    btn.addEventListener('click', () => {
      state.modelTab = btn.dataset.modelTab;
      $$('[data-model-tab]').forEach(b => b.classList.toggle('is-active', b === btn));
      $$('.model-pane').forEach(p => p.classList.toggle('is-active', p.dataset.modelPane === state.modelTab));
      state.modelRef = state.modelTab === 'preset' ? state.model?.slug || null : state.modelRef;
      if (state.modelTab === 'preset' && state.model) {
        state.modelRef = state.model.slug;
        state.modelSlug = state.model.slug;
        state.customSummary = null;
        state.modelSource = null;
        renderPresetSummary();
      }
    });
  });

  $$('.tab').forEach(tab => tab.addEventListener('click', () => switchTab(tab.dataset.tab)));

  $('#model-search').addEventListener('input', (event) => {
    state.searchText = event.target.value;
    renderModelOptions();
  });
  $('#model-select').addEventListener('change', (event) => {
    if (event.target.value) selectPreset(event.target.value);
  });

  $('#btn-resolve-ms')?.addEventListener('click', () => resolveModel('modelscope'));
  $('#btn-resolve-hf').addEventListener('click', () => resolveModel('hf'));
  $('#btn-resolve-upload').addEventListener('click', () => resolveModel('upload'));
  $('#btn-resolve-paste').addEventListener('click', () => resolveModel('paste'));

  $('#btn-simulate').addEventListener('click', runSimulate);
  $('#btn-solve').addEventListener('click', runSolve);
  $('#btn-search').addEventListener('click', runSearch);
  $('#btn-validate')?.addEventListener('click', runValidate);
  $('#img-framework')?.addEventListener('change', loadImages);
  $('#support-family')?.addEventListener('change', renderSupportRows);
  $('#btn-render-launch')?.addEventListener('click', renderLaunch);

  $('#profile-select').addEventListener('change', updateProfileNote);
  ['p-num_gpus', 'p-tp_size', 'p-pp_size', 'p-cp_size', 'p-micro_batch_size', 'p-global_batch_size']
    .forEach(id => $(`#${id}`).addEventListener('input', () => { updateEpHint(); updateDerived(); updateDataDerived(); }));
}

document.addEventListener('DOMContentLoaded', init);