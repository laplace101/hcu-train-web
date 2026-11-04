/**
 * Frontend integration test for the layered UI (plan §4, C/D coverage).
 *
 * Loads templates/index.html into jsdom, stubs `fetch` with the real
 * /api/v1 responses captured from a live server, and drives the page the way
 * a user would: pick a training mode, fill the data block, solve, and read
 * the three plans.
 *
 *     node tests/frontend_test.js [baseUrl]
 */

const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const ROOT = path.resolve(__dirname, '..');
const BASE = process.argv[2] || 'http://127.0.0.1:7801';

const failures = [];
function check(name, cond, detail = '') {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${detail ? ' — ' + detail : ''}`);
  if (!cond) failures.push(name);
}

const html = fs.readFileSync(path.join(ROOT, 'templates/index.html'), 'utf8');
const appJs = fs.readFileSync(path.join(ROOT, 'static/js/app.js'), 'utf8');

/** Fixtures mirroring the live /api/v1 payloads. */
// Fixtures are captured from the live server so the test exercises the
// same payload shapes the UI actually receives.
const FIXTURES = {
  '/api/v1/model-support': {
    ok: true,
    totals: { estimable: 29, approximate: 5, not_estimable: 1, total: 35 },
    families: {
      'Qwen3': { estimable: 8, approximate: 0, not_estimable: 0 },
      'GLM-5': { estimable: 3, approximate: 0, not_estimable: 0 },
      'DeepSeek': { estimable: 4, approximate: 0, not_estimable: 1 },
    },
    adapters: [
      { name: 'qwen3', description: 'Qwen3 稠密与 MoE' },
      { name: 'glm5', description: 'GLM-5，DSA 索引注意力' },
      { name: 'generic_gpt', description: '通用稠密解码器兜底' },
    ],
    models: [
      { reference: 'qwen3-8b', status: 'estimable', status_label: '可精确估算',
        adapter: 'qwen3', adapter_description: 'Qwen3 稠密与 MoE',
        family: 'Qwen3', reasons: [], warnings: [] },
      { reference: 'qwen2-5-7b', status: 'approximate', status_label: '可估算（近似值）',
        adapter: 'generic_gpt', adapter_description: '通用稠密解码器兜底',
        family: 'Qwen3', reasons: [],
        warnings: ['该模型的结构特征在内核里没有专用适配器'] },
      { reference: 'deepseek-v3-2-exp', status: 'not_estimable', status_label: '不可估算',
        adapter: 'deepseek_v3', adapter_description: '',
        family: 'DeepSeek', warnings: [],
        reasons: ['该模型包含内核无法表示的结构：DSA 索引注意力结构'] },
    ],
  },
  '/api/bootstrap': {
    ok: true,
    simulator: { repo: '/path/to/hcu-train-simulator', version: 'source-checkout' },
    families: [],
    proxy: {},
    operator_profiles: [],
    hardware: [
      { key: 'bw1000', name: '海光 DCU BW1000', hbm_gib: 64, gpus_per_node: 8,
        fp16_tflops: 480, fp8_tflops: 960, gemm_efficiency: 0.4,
        non_gemm_efficiency: 0.08, collective_intra_efficiency: 0.7,
        collective_inter_efficiency: 0.8, notes: ['实测带宽表'] },
    ],
    models: Array.from({ length: 35 }, (_, i) => ({
      slug: i === 0 ? 'qwen3-8b' : `model-${i}`, name: `Model ${i}`,
      family: 'qwen', verified: i < 20,
      adapter: 'qwen3', baseline: { num_gpus: 8, tp_size: 1, pp_size: 1,
        micro_batch_size: 1, global_batch_size: 256, seq_length: 4096,
        operator_profile: 'bw1000', task: 'full' },
    })),
    tasks: [
      { key: 'full', name: '全参微调', description: '全部权重参与训练。', enabled: true, status: 'ready' },
      { key: 'pretrain', name: '预训练', description: '同为全参训练。', enabled: true, status: 'ready' },
      { key: 'lora', name: 'LoRA 微调', description: '规划中', enabled: false, status: 'planned' },
    ],
  },
  '/api/v1/training/modes': {
    ok: true,
    modes: [
      { mode: 'pretrain', label: '预训练', modelled: true, summary: '从零开始训练',
        kernel_note: '预训练与「全参数 SFT」当前共用同一套计算内核，资源估算数值一致。' },
      { mode: 'sft_full', label: '全参 SFT', modelled: true, summary: '全参数更新',
        kernel_note: '预训练与「全参数 SFT」当前共用同一套计算内核，资源估算数值一致。' },
      { mode: 'sft_lora', label: 'LoRA SFT', modelled: true, summary: '低秩适配',
        kernel_note: 'LoRA 使用独立的资源模型，系数由实测基准库拟合得出。' },
      { mode: 'dpo', label: 'DPO', modelled: false, summary: '偏好对齐',
        unmodelled_reason: 'DPO 需要同时前向参考模型，显存与吞吐模型尚未建立，不提供估算。' },
      { mode: 'grpo', label: 'GRPO', modelled: false, summary: '强化学习',
        unmodelled_reason: 'GRPO 包含 rollout 与训练双阶段，资源模型尚未建立，不提供估算。' },
      { mode: 'ppo', label: 'PPO', modelled: false, summary: '强化学习',
        unmodelled_reason: 'PPO 包含 actor/critic 双模型，资源模型尚未建立，不提供估算。' },
    ],
  },
  '/api/v1/training/search': {
    ok: true,
    elapsed_ms: 1234.5,
    action: 'full',
    mode: 'sft_full',
    mode_kernel_note: '预训练与「全参数 SFT」当前共用同一套计算内核，资源估算数值一致。',
    measured: { available: true, matched: true, slug: 'qwen3-5-2b',
                estimate_tgs: 6988.34, measured_tgs: 6476.69,
                tgs_deviation_percent: 7.9, match_quality: 'exact',
                comparability_note: '实测为 LoRA，估算为全参。', row_count: 14,
                best_match: { gpu_count: 8, bs: 4, cutoff_len: 2048 },
                rows: [] },
    issues: [
      { field: 'global_batch_size', severity: 'error',
        message: 'Global Batch 101 不能被 MBS(1) × DP(4) = 4 整除（余数 1）',
        suggestion: '将 Global Batch 改为 4 的整数倍（最近可用值 100）' },
    ],
    plans: {
      minimum: { num_gpus: 2, tp_size: 2, pp_size: 1, cp_size: 1, ep_size: 1,
        peak_gib: 41.2, wgo_gib: 30.1, activation_gib: 11.1, iteration_ms: 820,
        communication_time_ms: 200, mfu: 0.41, tgs: 125000 },
      stable: { num_gpus: 4, tp_size: 2, pp_size: 1, cp_size: 1, ep_size: 1,
        peak_gib: 42.0, wgo_gib: 30.1, activation_gib: 11.9, iteration_ms: 790,
        communication_time_ms: 180, mfu: 0.44, tgs: 158000 },
      performance: { num_gpus: 8, tp_size: 4, pp_size: 1, cp_size: 1, ep_size: 1,
        peak_gib: 40.5, wgo_gib: 30.1, activation_gib: 10.4, iteration_ms: 640,
        communication_time_ms: 150, mfu: 0.55, tgs: 250000 },
      _meta: { stable_label: '推荐稳定方案', stable_reasons: [] },
    },
  },
  '/api/v1/training/validate': { ok: false, issues: [] },
  '/api/simulate': {
    ok: true,
    result: {
      summary: { iteration_ms: 9377.91, compute_ms: 8200, communication_ms: 1177,
                 tgs: 6988.34, mfu: 0.31, mfu_percent: 31, comm_percent: 12.5,
                 compute_percent: 87.5, tokens_per_iteration: 1048576,
                 actual_tflops: 480, model_size_b: 2.21 },
      peak_memory_gib: 63.72, hbm_gib: 64, model_adapter: 'qwen3_5',
      parallel: { num_gpus: 8, tp_size: 1, pp_size: 1, cp_size: 1, dp_size: 8,
                  ep_size: 1, etp_size: 1, seq_length: 2048, micro_batch_size: 4,
                  global_batch_size: 256, use_fp8_training: false,
                  full_recompute: false, overlap_mode: 'auto' },
      memory: [{ pp_rank: 0, wgo_gib: 40.1, activation_gib: 23.6, total_gib: 63.72,
                 hbm_gib: 64, utilization: 99.6, feasible: true }],
      communication: [], compute_details: [], log: 'sim log',
      elapsed_ms: 1234,
      measured: {
        available: true, matched: true, slug: 'qwen3-5-2b',
        model_names: ['Qwen/Qwen3.5-2B'], measured_is_lora: true,
        estimate_tgs: 6988.34, measured_tgs: 6476.69,
        tgs_deviation_percent: 7.9,
        estimate_peak_gib: 63.72, measured_peak_gib: 8.77,
        match_quality: 'exact', match_bs: 4, estimate_bs: 4,
        row_count: 14, oom_count: 0, gpu_counts: [8],
        comparability_note: '实测数据为 LoRA 微调（基座冻结），估算为全参训练内核。',
        best_match: { gpu_count: 8, bs: 4, grad_accum: 8, cutoff_len: 2048,
                      lora_rank: 8, ds_config: 'none', step_time_s: 10.119,
                      peak_mem_mib: 8978, tokens_per_s: 6476.69 },
        rows: [
          { gpu_count: 8, bs: 4, grad_accum: 8, cutoff_len: 2048, lora_rank: 8,
            ds_config: 'none', step_time_s: 10.119, tokens_per_s: 6476.69, peak_mem_mib: 8978 },
          { gpu_count: 8, bs: 8, grad_accum: 8, cutoff_len: 2048, lora_rank: 8,
            ds_config: 'none', step_time_s: 18.587, tokens_per_s: 7051.62, peak_mem_mib: 11607 },
        ],
      },
    },
  },
  '/api/v1/images': {
    ok: true,
    count: 1,
    source: { path: '/path/to/sourcefind_dcu_images.json',
               fetched_at: '2026-09-02' },
    frameworks: [{ framework: 'megatron', tag_count: 2 }],
    images: [{
      repo: 'admin/base/custom', project: 'hcu', abstract: '基础镜像',
      detail_url: 'https://developer.sourcefind.cn/x',
      updated: '2026-09-02',
      tags: [{ name: 'megatron-ubuntu22.04-dtk26.04-py3.10', framework: 'megatron',
               dtk_version: 'DTK 26.04', python_version: 'py3.10',
               os_version: 'ubuntu22.04', size: '8.21GB',
               pull_url: 'docker pull harbor.sourcefind.cn:5443/hcu/admin/base/custom:megatron-ubuntu22.04-dtk26.04-py3.10' }],
    }],
  },
  '/api/v1/templates': {
    ok: true,
    templates: [
      { id: 'megatron-sft-full', mode: 'sft_full', name: 'Megatron-LM 全参 SFT',
        framework: 'megatron', verified: true, verified_on: 'BW1000 / DTK 26.04 / 8 卡',
        notes: '全参微调模板' },
      { id: 'verl-grpo', mode: 'grpo', name: 'verl GRPO 强化学习',
        framework: 'verl', verified: false, verified_on: null, notes: '未验证' },
    ],
  },
  '/api/v1/launch': {
    ok: true,
    template_id: 'megatron-sft-full', template_name: 'Megatron-LM 全参 SFT',
    // The verified flag mirrors the template that was requested, so the UI's
    // "unverified template" warning can be exercised.
    verified: true, verified_on: 'BW1000 / DTK 26.04 / 8 卡', notes: '全参微调模板',
    image: null,
    yaml: '# YAML\ntensor_model_parallel_size: 2\n',
    shell: '#!/usr/bin/env bash\nexport WORLD_SIZE=8\n',
    docker: 'docker run --rm harbor.sourcefind.cn:5443/hcu/admin/base/custom:megatron',
    env: ['WORLD_SIZE=8'],
    rendezvous: 'torchrun --nnodes=1 --nproc_per_node=8',
  },
  '/api/v1/model-configs/resolve': {
    ok: true,
    config: { model_type: 'qwen3', num_hidden_layers: 36 },
    facts: { num_hidden_layers: 36, hidden_size: 4096, is_moe: false },
    provenance: { repo_id: 'Qwen/Qwen3-8B', revision: 'master', config_hash: 'a'.repeat(64) },
  },
};

const requests = [];

function makeDom() {
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    url: BASE,
    pretendToBeVisual: true,
    beforeParse(window) {
      window.fetch = async (url, options = {}) => {
        // Resolve relative URLs against the page origin so the query string
        // is stripped and the path can be matched against the fixtures.
        const parsed = new URL(String(url), BASE);
        const path = parsed.pathname;
        requests.push({ path, body: options.body ? JSON.parse(options.body) : null });
        const fixture = FIXTURES[path];
        if (!fixture) {
          return { ok: false, status: 404, json: async () => ({ error: { message: 'no fixture' } }) };
        }
        let body = JSON.parse(JSON.stringify(fixture));
        if (path === '/api/v1/launch' && options.body) {
          const requested = JSON.parse(options.body).template_id;
          if (requested === 'verl-grpo') {
            body.verified = false;
            body.verified_on = null;
            body.template_name = 'verl GRPO 强化学习';
          }
        }
        return { ok: true, status: 200, json: async () => body };
      };
      window.__TEST__ = { requests };
    },
  });
  // Inject the app script into the parsed document.
  const script = dom.window.document.createElement('script');
  script.textContent = appJs;
  dom.window.document.body.appendChild(script);
  return dom;
}

async function main() {
  const dom = makeDom();
  const { window } = dom;
  const doc = window.document;

  // init() is async — wait for it to settle.
  await new Promise(r => setTimeout(r, 300));

  console.log('=== ① 训练类型渲染（6 种）===');
  const modeButtons = [...doc.querySelectorAll('#task-grid .task-btn')];
  check('渲染 6 个训练类型', modeButtons.length === 6, String(modeButtons.length));
  const labels = modeButtons.map(b => b.textContent.replace(/未建模/, '').trim());
  check('包含 LoRA/DPO/GRPO/PPO',
    ['LoRA SFT', 'DPO', 'GRPO', 'PPO'].every(l => labels.includes(l)),
    labels.join(', '));
  const badged = modeButtons.filter(b => b.querySelector('.badge'));
  // LoRA is now modelled (own resource model); DPO / GRPO / PPO are not.
  check('3 个未建模类型打了标记', badged.length === 3, String(badged.length));
  const unmodelledLabels = badged.map(b => b.textContent.replace(/未建模/, '').trim());
  check('未建模的是 DPO/GRPO/PPO',
    unmodelledLabels.every(l => ['DPO', 'GRPO', 'PPO'].includes(l)),
    unmodelledLabels.join(', '));

  console.log('\n=== ② 未建模模式给出原因 ===');
  const dpoBtn = modeButtons.find(b => b.textContent.includes('DPO'));
  dpoBtn.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  await new Promise(r => setTimeout(r, 50));
  const warn = doc.querySelector('#mode-warning');
  check('警告区可见', warn && !warn.hidden);
  check('警告文案说明了原因',
    warn && (warn.textContent.includes('资源模型') || warn.textContent.includes('模型尚未建立')),
    warn ? warn.textContent.slice(0, 60) : '');

  console.log('\n=== ③ 数据量字段存在 ===');
  const dataInputs = ['d-total_tokens', 'd-total_samples', 'd-epochs', 'd-max_steps'];
  check('数据量字段已渲染', dataInputs.every(id => doc.getElementById(id)),
    dataInputs.filter(id => !doc.getElementById(id)).join(',') || '全部存在');

  console.log('\n=== ④ 切换到已建模模式并求解 ===');
  const sftBtn = modeButtons.find(b => b.textContent.includes('全参'));
  sftBtn.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  await new Promise(r => setTimeout(r, 50));

  // Fill the data block so the step count can be derived.
  const tokens = doc.getElementById('d-total_tokens');
  tokens.value = '100000000';
  tokens.dispatchEvent(new window.Event('input', { bubbles: true }));
  await new Promise(r => setTimeout(r, 50));
  const derived = doc.querySelector('#data-derived');
  check('换算出总训练步数', derived && derived.textContent.includes('总训练步数'),
    derived ? derived.textContent.replace(/\s+/g, ' ').slice(0, 80) : '');

  const solveBtn = doc.getElementById('btn-solve');
  check('求解按钮存在', !!solveBtn);
  solveBtn.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  await new Promise(r => setTimeout(r, 300));

  const searchReq = requests.find(r => r.path === '/api/v1/training/search');
  check('调用了 v1 search 接口', !!searchReq);
  if (searchReq) {
    check('请求带上训练模式', searchReq.body.mode === 'sft_full', searchReq.body.mode);
    check('请求带上数据量', searchReq.body.total_tokens === 100000000,
      String(searchReq.body.total_tokens));
    check('请求带上并行方案', searchReq.body.parallel && searchReq.body.parallel.tp_size > 0,
      JSON.stringify(searchReq.body.parallel || {}).slice(0, 90));
  }

  console.log('\n=== ⑤ 三类方案渲染 ===');
  const plansBox = doc.querySelector('#plans-box');
  check('方案区有内容', plansBox && plansBox.innerHTML.length > 100);
  check('包含最低可训练方案', plansBox.textContent.includes('最低可训练方案'));
  check('包含推荐稳定方案', plansBox.textContent.includes('推荐稳定方案'));
  check('包含推荐性能方案', plansBox.textContent.includes('推荐性能方案'));
  check('显示卡数 2/4/8',
    plansBox.textContent.includes('2 卡') && plansBox.textContent.includes('8 卡'));
  check('显示吞吐 TGS', plansBox.textContent.includes('token/s'));
  check('显示吞吐提升对比', plansBox.textContent.includes('吞吐提升'));

  console.log('\n=== ⑥ 校验问题渲染 ===');
  const validateBox = doc.querySelector('#validate-box');
  check('校验区有内容', validateBox && validateBox.innerHTML.length > 50);
  check('显示错误级别', validateBox.querySelector('.notice--error') !== null);
  check('显示修改建议', validateBox.textContent.includes('最近可用值'));

  console.log('\n=== ⑦ 页签与面板一一对应 ===');
  const tabs = [...doc.querySelectorAll('.tab')].map(t => t.dataset.tab);
  const panes = [...doc.querySelectorAll('.tab-pane')].map(p => p.dataset.pane);
  check('页签数与面板数一致', tabs.length === panes.length, `${tabs.length} vs ${panes.length}`);
  check('三类方案页签存在', tabs.includes('plans'));
  check('参数校验页签存在', tabs.includes('validate'));
  check('无孤立页签', tabs.every(t => panes.includes(t)));
  check('无孤立面板', panes.every(p => tabs.includes(p)));

  console.log('\n=== ⑧ ModelScope 解析 ===');
  const msTab = doc.querySelector('[data-model-tab="modelscope"]');
  msTab.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  await new Promise(r => setTimeout(r, 50));
  check('ModelScope 面板可见',
    doc.querySelector('[data-model-pane="modelscope"]').classList.contains('is-active'));
  doc.getElementById('ms-ref').value = 'Qwen/Qwen3-8B';
  doc.getElementById('btn-resolve-ms').dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  await new Promise(r => setTimeout(r, 300));
  const msReq = requests.find(r => r.path === '/api/v1/model-configs/resolve');
  check('调用了 ModelScope 解析接口', !!msReq);
  if (msReq) check('传了仓库引用', msReq.body.reference === 'Qwen/Qwen3-8B');
  const msSummary = doc.querySelector('#ms-summary');
  check('显示仓库与 revision', msSummary.textContent.includes('Qwen/Qwen3-8B')
    && msSummary.textContent.includes('master'));
  check('显示配置哈希', msSummary.textContent.includes('aaaa'));

  console.log('\n=== ⑨ 镜像目录渲染 ===');
  const imagesBox = doc.querySelector('#images-box');
  check('镜像区有内容', imagesBox && imagesBox.innerHTML.length > 100);
  check('显示仓库名', imagesBox.textContent.includes('admin/base/custom'));
  check('显示拉取命令', imagesBox.textContent.includes('docker pull harbor.sourcefind.cn'));
  check('显示框架与版本', imagesBox.textContent.includes('megatron')
    && imagesBox.textContent.includes('DTK 26.04'));

  console.log('\n=== ⑩ 模板下拉框 ===');
  const tplSelect = doc.querySelector('#tpl-select');
  check('模板已加载', tplSelect && tplSelect.options.length >= 1,
    tplSelect ? [...tplSelect.options].map(o => o.textContent).join(' | ') : '');
  check('未验证模板带标注',
    [...tplSelect.options].some(o => o.textContent.includes('未在目标环境验证')));

  console.log('\n=== ⑪ 生成启动配置 ===');
  tplSelect.value = 'megatron-sft-full';
  const imgInput = doc.getElementById('tpl-image');
  imgInput.value = 'harbor.sourcefind.cn:5443/hcu/admin/base/custom:megatron';
  doc.getElementById('tpl-model').value = '/models/Qwen3-8B';
  doc.getElementById('btn-render-launch')
    .dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  await new Promise(r => setTimeout(r, 300));

  const launchReq = requests.find(r => r.path === '/api/v1/launch');
  check('调用了 launch 接口', !!launchReq);
  if (launchReq) {
    check('传了模板 ID', launchReq.body.template_id === 'megatron-sft-full');
    check('传了镜像', launchReq.body.image === 'harbor.sourcefind.cn:5443/hcu/admin/base/custom:megatron');
    check('传了模型路径', launchReq.body.model_path === '/models/Qwen3-8B');
    check('传了方案', !!launchReq.body.plan);
  }
  const launchBox = doc.querySelector('#launch-box');
  check('生成了 YAML 区块', launchBox.textContent.includes('YAML 训练配置'));
  check('生成了 Shell 区块', launchBox.textContent.includes('Shell 启动脚本'));
  check('生成了 Docker 区块', launchBox.textContent.includes('Docker 命令'));
  check('有复制按钮', launchBox.querySelectorAll('button').length === 3,
    String(launchBox.querySelectorAll('button').length));

  console.log('\n=== ⑫ 未验证模板给出警告 ===');
  tplSelect.value = 'verl-grpo';
  doc.getElementById('btn-render-launch')
    .dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  await new Promise(r => setTimeout(r, 300));
  check('显示未验证警告',
    doc.querySelector('#launch-box').querySelector('.notice--warn') !== null);

  console.log('\n=== ⑬ 预设模型带 slug 求解（回归）===');
  // Regression: picking a preset model must send model_slug, otherwise the
  // server has no config.json to work from and the solve fails with 400.
  const presetOption = [...doc.querySelectorAll('#model-select option')]
    .find(o => o.value === 'qwen3-8b');
  check('预设模型在列表中', !!presetOption);
  if (presetOption) {
    doc.getElementById('model-select').value = 'qwen3-8b';
    doc.getElementById('model-select')
      .dispatchEvent(new window.Event('change', { bubbles: true }));
    await new Promise(r => setTimeout(r, 80));

    doc.getElementById('btn-solve')
      .dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
    await new Promise(r => setTimeout(r, 300));

    const presetReq = requests.filter(r => r.path === '/api/v1/training/search').pop();
    check('预设模型求解发出请求', !!presetReq);
    if (presetReq) {
      check('请求带上 model_slug', presetReq.body.model_slug === 'qwen3-8b',
        String(presetReq.body.model_slug));
      check('请求不带空的 config_path',
        !presetReq.body.model_config_path, String(presetReq.body.model_config_path));
    }
    check('方案区有内容', doc.querySelector('#plans-box').innerHTML.length > 100);
  }

  console.log('\n=== ⑭ 内核共享标注（如实告知）===');
  const modeNote = doc.querySelector('#mode-kernel-note');
  check('标注区存在', !!modeNote);
  // The default mode after init is sft_full, which carries a note.
  check('已建模模式显示标注', modeNote && !modeNote.hidden);
  // The LoRA mode must carry its own note; switching to it shows it.
  const loraModeBtn2 = [...doc.querySelectorAll('#task-grid .task-btn')]
    .find(b => b.textContent.includes('LoRA'));
  if (loraModeBtn2) {
    loraModeBtn2.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
    await new Promise(r => setTimeout(r, 60));
    check('LoRA 模式显示独立说明',
      modeNote && !modeNote.hidden && modeNote.textContent.includes('独立'),
      modeNote ? modeNote.textContent.slice(0, 40) : '');
  }
  // The default mode is full SFT, which shares the kernel with pretraining.
  const sftModeBtn = [...doc.querySelectorAll('#task-grid .task-btn')]
    .find(b => b.textContent.includes('全参'));
  if (sftModeBtn) {
    sftModeBtn.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
    await new Promise(r => setTimeout(r, 60));
  }
  check('标注说明了共用内核',
    modeNote && modeNote.textContent.includes('共用同一套计算内核'),
    modeNote ? modeNote.textContent.slice(0, 40) : '');
  check('标注使用 info 样式', modeNote && modeNote.classList.contains('notice--info'));

  // Switching to an unmodelled mode must hide the note.
  const grpoBtn = [...doc.querySelectorAll('#task-grid .task-btn')]
    .find(b => b.textContent.includes('GRPO'));
  if (grpoBtn) {
    grpoBtn.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
    await new Promise(r => setTimeout(r, 50));
    check('未建模模式隐藏该标注', modeNote.hidden);
  }

  console.log('\n=== ⑮ 结果页标注 ===');
  const sftBtn2 = [...doc.querySelectorAll('#task-grid .task-btn')]
    .find(b => b.textContent.includes('全参'));
  sftBtn2.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  await new Promise(r => setTimeout(r, 80));
  doc.getElementById('btn-solve')
    .dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  await new Promise(r => setTimeout(r, 300));
  const plansNote = doc.querySelector('#plans-kernel-note');
  check('结果页有标注区', !!plansNote);
  check('结果页标注可见', plansNote && !plansNote.hidden);
  check('结果页标注内容正确',
    plansNote && plansNote.textContent.includes('共用同一套计算内核'),
    plansNote ? plansNote.textContent.slice(0, 40) : '');

  console.log('\n=== ⑯ 重计算开关提示 ===');
  const recomputeHint = [...doc.querySelectorAll('.hint--inline')]
    .find(n => n.textContent.includes('重计算'));
  check('重计算旁有提示', !!recomputeHint,
    recomputeHint ? recomputeHint.textContent.slice(0, 40) : '');
  check('提示说明内核不建模',
    recomputeHint && recomputeHint.textContent.includes('尚未对重计算建模'));

  console.log('\n=== ⑰ 实测数据卡片 ===');
  // The legacy /api/simulate response carries the measured comparison.
  doc.getElementById('btn-simulate')
    .dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  await new Promise(r => setTimeout(r, 300));

  const measuredCard = doc.querySelector('#measured-card');
  check('实测卡片可见', measuredCard && !measuredCard.hidden);
  const measuredBox = doc.querySelector('#measured-box');
  check('有实测内容', measuredBox && measuredBox.innerHTML.length > 100);
  check('显示实测模型名', measuredBox.textContent.includes('Qwen/Qwen3.5-2B'));
  // fmtInt uses en-US grouping, so 6988.34 renders as "6,988".
  check('显示估算 TGS', measuredBox.textContent.includes('6,988'));
  check('显示实测 TGS', measuredBox.textContent.includes('6,476'));
  check('显示偏差', measuredBox.textContent.includes('7.9'),
    measuredBox.textContent.match(/[-+]?\d+\.\d%/)?.[0] || '');
  check('显示匹配程度',
    measuredBox.textContent.includes('micro-batch 均一致'),
    measuredBox.textContent.match(/卡数[^\n]{0,30}/)?.[0] || '');
  check('标注 LoRA 不可比',
    measuredBox.textContent.includes('LoRA') && measuredBox.textContent.includes('全参'));
  check('显示估算显存(全参)', measuredBox.textContent.includes('63.72'));
  check('显示实测显存(LoRA)', measuredBox.textContent.includes('8.77'));
  check('显示最佳配置',
    measuredBox.textContent.includes('最佳实测配置') && measuredBox.textContent.includes('bs 4'));
  check('有实测记录明细表',
    measuredBox.querySelectorAll('details tbody tr').length === 2,
    String(measuredBox.querySelectorAll('details tbody tr').length));

  console.log('\n=== ⑱ LoRA 参数表单 ===');
  const loraModeBtn = [...doc.querySelectorAll('#task-grid .task-btn')]
    .find(b => b.textContent.includes('LoRA'));
  check('有 LoRA 模式按钮', !!loraModeBtn);

  const loraBlock = doc.querySelector('#lora-block');
  check('LoRA 参数区块存在', !!loraBlock);
  check('非 LoRA 模式时隐藏', loraBlock && loraBlock.hidden);

  if (loraModeBtn) {
    loraModeBtn.dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
    await new Promise(r => setTimeout(r, 80));
    check('切到 LoRA 后显示参数区', loraBlock && !loraBlock.hidden);
    check('有 rank 输入', !!doc.getElementById('lora-lora_r'));
    check('有 alpha 输入', !!doc.getElementById('lora-lora_alpha'));
    check('有 dropout 输入', !!doc.getElementById('lora-lora_dropout'));
    check('有目标模块选择', !!doc.getElementById('lora-target'));
    check('有 DeepSpeed 阶段选择', !!doc.getElementById('lora-ds'));
    check('有 Liger 开关', !!doc.getElementById('lora-liger'));
    check('默认 rank 为 8', doc.getElementById('lora-lora_r').value === '8',
      doc.getElementById('lora-lora_r').value);
    check('默认 ZeRO-3', doc.getElementById('lora-ds').value === 'z3',
      doc.getElementById('lora-ds').value);

    // The request must carry the LoRA parameters.
    doc.getElementById('lora-lora_r').value = '16';
    doc.getElementById('btn-solve')
      .dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
    await new Promise(r => setTimeout(r, 300));
    const loraReq = requests.filter(r => r.path === '/api/v1/training/search').pop();
    check('请求带上 LoRA 参数', !!loraReq && !!loraReq.body.lora);
    if (loraReq && loraReq.body.lora) {
      check('rank 传入请求', loraReq.body.lora.r === 16, String(loraReq.body.lora.r));
      check('目标模块传入请求', 'target_modules' in loraReq.body.lora);
    }
    check('请求带上 ZeRO 阶段', !!loraReq && !!loraReq.body.lora_ds_config,
      loraReq ? String(loraReq.body.lora_ds_config) : '');
    check('请求带上 Liger 开关', !!loraReq && loraReq.body.lora_liger !== undefined,
      loraReq ? String(loraReq.body.lora_liger) : '');
  }

  console.log('\n=== ⑲ 模型兼容性错误展示 ===');
  const errCard = doc.querySelector('#model-error-card');
  const errBox = doc.querySelector('#model-error-box');
  check('模型错误卡片存在', !!errCard && !!errBox);

  // Make the next simulate call fail with structured reasons.
  const origFetch = window.fetch;
  window.fetch = async (url, options = {}) => {
    if (String(url).includes('/api/simulate')) {
      return {
        ok: false, status: 422,
        json: async () => ({
          ok: false,
          error: '内核无法为该模型给出可信估算',
          reasons: ['该模型是多模态模型，而内核把它匹配到了文本适配器',
                    '该模型包含内核未建模的结构：DSA 索引注意力结构'],
          suggestions: ['内核尚未提供 DeepSeek 系多模态适配器',
                        '如需评估该模型，请等待内核支持'],
          model_summary: 'model_type=deepseek_v41 · 匹配到适配器 deepseek_v3',
        }),
      };
    }
    return origFetch(url, options);
  };

  doc.getElementById('btn-simulate')
    .dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  await new Promise(r => setTimeout(r, 300));

  check('错误卡片可见', errCard && !errCard.hidden);
  check('显示通用说明',
    errBox && errBox.textContent.includes('无法为该模型给出可信估算'));
  check('逐条列出原因', errBox.querySelectorAll('.reason-list li').length >= 2,
    String(errBox.querySelectorAll('.reason-list li').length));
  check('列出建议',
    errBox.textContent.includes('内核尚未提供'),
    errBox.textContent.match(/内核尚未提供[^\n]{0,20}/)?.[0] || '');

  // A successful run must clear the box.
  window.fetch = origFetch;
  doc.getElementById('btn-simulate')
    .dispatchEvent(new window.MouseEvent('click', { bubbles: true }));
  await new Promise(r => setTimeout(r, 300));
  check('成功后清空错误区', errCard.hidden);

  console.log('\n=== ⑳ 模型支持矩阵 ===');
  const supportBox = doc.querySelector('#support-box');
  const supportSummary = doc.querySelector('#support-summary');
  const supportFamily = doc.querySelector('#support-family');
  check('支持矩阵容器存在', !!supportBox && !!supportSummary);
  check('系列筛选存在', !!supportFamily);
  check('请求了支持矩阵',
    requests.some(r => r.path === '/api/v1/model-support'));

  // The matrix renders after init() resolves its fetch.
  await new Promise(r => setTimeout(r, 200));
  const supportRows = supportBox.querySelectorAll('.support-table tbody tr');
  check('渲染出模型行', supportRows.length === 3, String(supportRows.length));
  check('显示统计', supportSummary.textContent.includes('29'));
  check('列出适配器', supportSummary.textContent.includes('qwen3'));
  check('显示可精确估算', supportBox.textContent.includes('可精确估算'));
  check('显示近似标注', supportBox.textContent.includes('可估算（近似值）'));
  check('显示不可估算', supportBox.textContent.includes('不可估算'));
  check('不可估算时给出原因',
    supportBox.textContent.includes('DSA 索引注意力结构'));

  // Filtering by family narrows the table.
  const famSelect = doc.querySelector('#support-family');
  const optionValues = Array.from(famSelect.options).map(o => o.value);
  check('系列下拉包含 DeepSeek', optionValues.includes('DeepSeek'),
    JSON.stringify(optionValues));
  if (optionValues.includes('DeepSeek')) {
    famSelect.value = 'DeepSeek';
    famSelect.dispatchEvent(new window.Event('change', { bubbles: true }));
    await new Promise(r => setTimeout(r, 100));
    const filtered = supportBox.querySelectorAll('.support-table tbody tr');
    check('按系列筛选生效', filtered.length === 1, String(filtered.length));
    check('筛选后只剩该系列',
      supportBox.textContent.includes('DSA 索引注意力结构'));

    famSelect.value = '';
    famSelect.dispatchEvent(new window.Event('change', { bubbles: true }));
    await new Promise(r => setTimeout(r, 100));
    check('恢复全部',
      supportBox.querySelectorAll('.support-table tbody tr').length === 3);
  }

  console.log('\n' + '='.repeat(46));
  if (failures.length) {
    console.log(`FAILED ${failures.length}: ${failures.join(', ')}`);
    process.exit(1);
  }
  console.log('ALL FRONTEND CHECKS PASSED');
  process.exit(0);
}

main().catch(err => { console.error(err); process.exit(1); });
