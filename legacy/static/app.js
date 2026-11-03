"use strict";

/* ========== small helpers ========== */
const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function setStatus(text, klass) {
  const el = $("#status");
  el.textContent = text;
  el.className = "status " + (klass || "");
}

function fmtMs(ms) {
  if (ms == null || ms === "" || isNaN(parseFloat(ms))) return "?";
  const v = parseFloat(ms);
  if (v > 1000) return (v / 1000).toFixed(2) + " s";
  return v.toFixed(1) + " ms";
}

function fmtPct(s) {
  if (!s) return "?";
  const v = parseFloat(s);
  if (!isNaN(v)) return (v * (v > 1 ? 1 : 100)).toFixed(2) + "%";
  return s;
}

/* ========== tabs ========== */
$$(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    $$(".tab").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
    $$(".tab-panel").forEach(p => p.classList.toggle("hidden", p.dataset.panel !== tab));
  });
});

/* ========== task radio (toggles LoRA card) ========== */
function selectedTask() {
  const el = document.querySelector('input[name="task"]:checked');
  return el ? el.value : "full_ft";
}
function syncTaskUi() {
  const isLora = selectedTask() === "lora";
  const card = document.getElementById("lora-card");
  if (card) card.classList.toggle("hidden", !isLora);
}
document.querySelectorAll('input[name="task"]').forEach(r => {
  r.addEventListener("change", syncTaskUi);
});
syncTaskUi();

/* ========== preset loaders ========== */
let HW_PRESETS = {};
let MODEL_PRESETS = [];

function buildPresetCard(item, kind) {
  const div = document.createElement("div");
  div.className = "preset";
  div.dataset.key = item.key;
  const sub = kind === "hardware"
    ? `${item.fp16_tflops} TF FP16 · ${item.hbm_gib} GiB HBM`
    : `${(item.rough_params_B || "").toString()} B params · ${item.kind}`;
  div.innerHTML =
    `<div class="label">${escapeHtml(item.label)}</div>` +
    `<div class="sub">${escapeHtml(sub)}</div>`;
  div.addEventListener("click", () => {
    const root = div.parentElement;
    $$(".preset", root).forEach(x => x.classList.toggle("active", x === div));
    if (kind === "hardware") applyHwOverrides(item);
  });
  return div;
}

function applyHwOverrides(item) {
  $("#opt_fp16").value  = item.fp16_tflops;
  $("#opt_fp8").value   = item.fp8_tflops;
  $("#opt_hbm").value   = item.hbm_gib;
  $("#opt_intra").value = item.intra_bw_gbps;
  $("#opt_gemm").value  = item.gemm_efficiency;
  $("#opt_p2p").value   = item.p2p_intra_efficiency;
}

async function loadPresets() {
  try {
    const r = await fetch("/api/presets");
    if (!r.ok) throw new Error(r.statusText);
    const j = await r.json();
    HW_PRESETS = Object.fromEntries(j.hardware.map(h => [h.key, h]));
    MODEL_PRESETS = j.models;

    const hwGrid = $("#hw-preset-grid");
    j.hardware.forEach(item => hwGrid.appendChild(buildPresetCard(item, "hardware")));
    const mGrid = $("#model-preset-grid");
    j.models.forEach(item => mGrid.appendChild(buildPresetCard(item, "model")));

    $$(".preset", hwGrid).find(x => x.dataset.key === "hygon_dcu_bw1000")?.click();
    $$(".preset", mGrid).find(x => x.dataset.key === "qwen3_8b")?.click();
  } catch (e) { console.error("failed to load presets", e); }
}

async function loadHealth() {
  const el = $("#health");
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    const cls = j.simulator_exists ? (j.ok ? "ok" : "warn") : "err";
    el.classList.add(cls);
    el.textContent = j.simulator_exists
      ? `Python ${escapeHtml(j.python_version.split(' ')[0])} · 模拟器：${escapeHtml(j.simulator_dir)}`
      : `⚠ 模拟器目录不存在：${escapeHtml(j.simulator_dir)}`;
  } catch (e) {
    el.classList.add("err");
    el.textContent = "⚠ 后端未启动";
  }
}

/* ========== result rendering ========== */

function renderSummary(s, source) {
  if (!s) return "<i>(无 config，没有执行架构自动识别)</i>";
  const tags = [];
  tags.push(["arch", s.arch || "?"]);
  if (s.vlm) tags.push(["VLM", "vlm"]);
  if (s.moe) tags.push(["MoE", "moe"]);
  if (s.mla) tags.push(["MLA", "mla"]);
  if (s.attention_type) tags.push([s.attention_type, s.attention_type.toLowerCase()]);

  const tagHtml = tags
    .map(([lbl]) => `<span class="v tag">${escapeHtml(lbl)}</span>`)
    .join(" ");

  const card = (k, v) => v == null ? "" :
    `<div class="kv"><div class="k">${escapeHtml(k)}</div>` +
    `<div class="v">${escapeHtml(v)}</div></div>`;

  const src = source ? `<p class="hint">来源：${escapeHtml(source)}</p>` : "";

  return (
    `${src}` +
    `<div class="arch-card">` +
    card("Model Type",  s.model_type) +
    card("架构",         tagHtml) +
    card("参数量 ≈",     (s.params_B ?? "?") + " B") +
    card("Hidden Size",  s.hidden_size) +
    card("层数",          s.num_layers) +
    card("Q/KV Heads",   (s.num_heads ?? "?") + " / " + (s.num_kv_heads ?? "?")) +
    card("Intermediate", s.intermediate) +
    card("Vocab",         s.vocab) +
    card("Rope θ",       s.rope_theta) +
    `</div>` +
    (s.moe
      ? `<p class="hint">MoE: ${escapeHtml(JSON.stringify(s.moe))}</p>`
      : "")
  );
}

function renderSimulation(p) {
  const sim = p.simulation;
  const mem = p.memory;
  if (!sim && !mem) return "<i>无</i>";
  const card = (k, v) =>
    `<div class="kv"><div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(v || "?")}</div></div>`;
  const tgs = sim && sim.tgs ? `${parseFloat(sim.tgs).toFixed(0)} tok/s` : "?";
  return `<div class="arch-card">` +
    card("Iteration 时长", sim ? fmtMs(sim.iteration_ms) : "?") +
    card("Compute",         sim ? fmtMs(sim.compute_ms) : "?") +
    card("Communication",   sim ? fmtMs(sim.communication_ms) : "?") +
    card("Throughput (TGS)", tgs) +
    card("MFU",             sim ? fmtPct(sim.mfu) : "?") +
    card("Per-GPU memory",  mem ? mem.total_gib + " GiB" : "?") +
    card("HBM / 卡",         mem ? mem.hbm_gib + " GiB" : "?") +
    card("显存利用率",        mem ? mem.utilization : "?") +
    `</div>`;
}

function renderSolve(p) {
  const cfgs = p.solve_configs || [];
  const perf = p.solve_perf || {};
  if (!cfgs.length && !Object.keys(perf).length)
    return "<i>solve 未产生有效结果（架构不在内置 adapter 中，或配置不合法）</i>";

  let html = "";
  if (cfgs.length) {
    html += '<div class="arch-card">';
    for (const cfg of cfgs) {
      const label = cfg.label === "min_gpus" ? "最小 GPU 方案" : "高性能方案";
      const color = cfg.label === "min_gpus" ? "" : " tag perf";
      html +=
        `<div class="kv"><div class="k">${escapeHtml(label)}</div>` +
        `<div class="v tag${color}">` +
        `${escapeHtml(cfg.gpus)} 卡 · TP=${escapeHtml(cfg.tp)} PP=${escapeHtml(cfg.pp)} DP=${escapeHtml(cfg.dp)} EP=${escapeHtml(cfg.ep)}` +
        ` · mbs=${escapeHtml(cfg.mbs)} gbs=${escapeHtml(cfg.gbs)}` +
        `</div></div>`;
    }
    html += '</div>';
  }

  if (Object.keys(perf).length) {
    html += '<div class="arch-card">';
    for (const [k, v] of Object.entries(perf)) {
      const label = k === "min_gpus" ? "最小方案 性能" : "高性能方案 性能";
      html +=
        `<div class="kv"><div class="k">${escapeHtml(label)}</div>` +
        `<div class="v">` +
        `${fmtMs(v.iter_ms)} / MFU ${fmtPct(v.mfu)} / ${parseFloat(v.tgs).toFixed(0)} tok/s` +
        `</div></div>`;
    }
    html += '</div>';
  }
  return html;
}

/* ========== run the simulator ========== */

function getActivePreset(tabPanel) {
  const a = $(".preset.active", tabPanel);
  return a ? a.dataset.key : null;
}

function collectOverrides() {
  const f = (id) => {
    const v = $("#" + id).value;
    return v === "" || v == null ? null : Number(v);
  };
  return {
    fp16_tflops: f("opt_fp16"),
    fp8_tflops:  f("opt_fp8"),
    hbm_gib:     f("opt_hbm"),
    intra_bw_gbps: f("opt_intra"),
    gemm_efficiency: f("opt_gemm"),
    p2p_intra_efficiency: f("opt_p2p"),
  };
}

async function runSimulator() {
  const tab = $(".tab.active").dataset.tab;
  const task = selectedTask();
  const payload = {
    input_type: tab,
    task: task,
    hardware: getActivePreset($("#hw-preset-grid")),
    hw_overrides: collectOverrides(),
    tp_size: Number($("#tp").value) || 1,
    pp_size: Number($("#pp").value) || 1,
    ep_size: Number($("#ep").value) || 1,
    cp_size: Number($("#cp").value) || 1,
    micro_batch_size: Number($("#mbs").value) || 1,
    global_batch_size: Number($("#gbs").value) || 1,
    seq_length: Number($("#seq").value) || 4096,
    num_gpus: Number($("#ng").value) || 8,
    use_fp8: $("#fp8").checked,
  };
  if (task === "lora") {
    payload.lora_r       = Number($("#lora_r").value) || 16;
    payload.lora_alpha   = Number($("#lora_alpha").value) || 32;
    payload.lora_dropout = Number($("#lora_dropout").value) || 0.05;
    payload.lora_targets = ($("#lora_targets").value || "").trim() || "q_proj,k_proj,v_proj,o_proj";
  }

  if (tab === "preset") {
    payload.preset = getActivePreset($("#model-preset-grid"));
    if (!payload.preset) { setStatus("请选一个模型", "err"); return; }
  } else if (tab === "hf_id") {
    payload.hf_id = ($("#hf_id").value || "").trim();
    if (!payload.hf_id) { setStatus("请填 HF id", "err"); return; }
  } else if (tab === "upload") {
    const fileInput = $("#cfg_file");
    const textEl = $("#cfg_text");
    if (fileInput.files[0]) {
      try { payload.config_json = JSON.parse(await fileInput.files[0].text()); }
      catch (e) { setStatus("JSON 错误：" + e.message, "err"); return; }
    } else if (textEl.value.trim()) {
      try { payload.config_json = JSON.parse(textEl.value); }
      catch (e) { setStatus("JSON 错误：" + e.message, "err"); return; }
    } else {
      setStatus("请上传或粘贴 config.json", "err"); return;
    }
  }

  setStatus("跑模拟中 …", "working");
  $("#run").disabled = true;
  $("#results").classList.add("hidden");

  try {
    const r = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const j = await r.json();

    if (j.error) {
      setStatus("失败：" + j.error, "err");
      $("#r_summary").innerHTML = "<i>" + escapeHtml(j.error) + "</i>";
      $("#results").classList.remove("hidden");
      return;
    }

    $("#results").classList.remove("hidden");
    $("#r_summary").innerHTML = renderSummary(j.summary, j.source);
    $("#r_simulation").innerHTML = renderSimulation(j.parsed || {});
    $("#r_solve").innerHTML = renderSolve(j.parsed || {});
    $("#r_stdout").textContent = j.simulator.stdout || "(empty)";
    $("#r_stderr").textContent = j.simulator.stderr || "(no stderr)";
    $("#r_yaml").textContent = j.yaml_text;
    $("#r_command").innerHTML = "命令：<code>" + escapeHtml(j.simulator.command) + "</code>";

    if (j.simulator.returncode === 0) {
      setStatus("跑完 ✓", "ok");
    } else {
      setStatus("退出码 " + j.simulator.returncode + "（看 stderr）", "err");
    }
  } catch (e) {
    setStatus("请求失败：" + e.message, "err");
  } finally {
    $("#run").disabled = false;
  }
}

$("#run").addEventListener("click", runSimulator);

loadHealth();
loadPresets();
