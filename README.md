# hcu-train-web

海光 DCU 训练资源规划平台：选模型 → 选并行策略 → 得到三类推荐方案、显存/通信/性能明细，并导出可直接使用的训练配置。

**所有数值都来自 [`hcu-train-simulator`](https://github.com/HYGON-AI/hcu-train-simulator)，本项目不重写任何计算公式。**

---

## 1. 这个项目解决什么问题

给一个训练任务（模型 + 卡 + 训练类型 + 数据量），回答三个问题：

1. **需要几张卡？怎么摆并行策略？** —— 三类推荐方案（最低可训练 / 推荐稳定 / 推荐性能）
2. **跑起来显存够不够、快不快？** —— 单步时间、TGS、MFU、通信占比、PP Stage 显存明细
3. **怎么把它跑起来？** —— 训练镜像拉取命令 + YAML / Shell / Docker 启动配置

---

## 2. 准确率是怎么保证的

这是本项目的第一约束。三层保障：

1. **进程内直接调用，而不是解析命令行输出。**
   后端 `import hcu_train_simulator` 后直接调用 `run_simulation()` /
   `solve_training_config()`，拿的是估计器返回的结构化对象。

2. **汇总口径逐行对齐 `report_data_processing`。**
   迭代时间 = 计算时间 + 通信暴露时间；TGS / MFU / 通信占比的公式与
   `hcu_train_simulator/simulator.py` 中的实现完全一致。

3. **每个预设模型都有「网页 vs 官方 CLI」的逐项比对。**
   `verification/verify_accuracy.py` 对同一份配置分别跑网页后端和官方 CLI，
   比对 compute / communication / iteration / TGS / MFU 以及每个 PP stage 的显存表。

   **当前结果：35 / 35 预设模型全部通过（见 `verification/baseline.json`）。**

```bash
python verification/verify_accuracy.py              # 全量校验
python verification/verify_accuracy.py --only qwen3-8b deepseek-v3
```

新增的 `/api/v1` 分层接口同样遵守这条约束：它复用 `engine/config_builder.build_config()`
构造配置，数值与旧路由**逐位一致**（`tests/test_e2e_live.py` 对基线做逐项比对）。

---

## 3. 快速开始

前置：`hcu-train-simulator` 源码放在本项目同级目录
（`../hcu-train-simulator`），或设置环境变量 `HCU_SIM_SRC` 指向其 `src`。

```bash
pip install -r requirements.txt
python app.py                              # http://127.0.0.1:7766
python app.py --host 0.0.0.0 --port 7801  # 对外提供服务
```

前端测试需要 jsdom：

```bash
npm install jsdom
```

---

## 4. 架构

分层是硬约束：**接入层不含计算，估算层是唯一与模拟器交互的地方。**

```
hcu-train-web/
├── api/routes.py              接入层：解析请求、校验、格式化响应
├── services/
│   ├── estimates.py            估算编排：场景 → 三类方案
│   ├── validation.py           参数校验：并行整除、Batch 关系、数据量换算
│   ├── launcher.py             命令渲染：YAML / Shell / Docker / 分布式变量
│   └── modelscope.py           ModelScope 接入：白名单、SSRF 防护、revision 记录
├── estimators/adapter.py       估算适配层：唯一调用 simulator 的地方（子进程隔离）
├── repositories/
│   ├── hardware.py             硬件目录与可估算性判定
│   ├── images.py               训练镜像目录 + 提取码哈希存储
│   ├── templates.py            训练配置模板
│   └── measured.py             实测基准库（只读）
├── services/
│   ├── lora_model.py           LoRA 资源模型（解析部分）
│   └── lora_calibration.py     LoRA 系数校准（从实测拟合）
├── domain/scenarios.py         领域模型：六种训练模式、场景、并行方案
├── engine/                     已验证的估算内核（35/35 基线依赖，不要改）
├── catalog/                    35 个预设模型的 config.json（随仓库保存，离线可用）
├── static/, templates/         前端
├── tests/                      分层接口 / 镜像 / 启动命令 / 前端 DOM / 实时端到端
├── tools/run_tests.sh          一键跑全部测试
└── verification/               35/35 精度基线
```

### 两个容易踩的坑（已在代码中处理）

1. **导入顺序**：`import hcu_train_simulator.models.registry` 会触发
   qwen_vl / qwen3_5 benchmark 模块之间的循环导入。必须先
   `import hcu_train_simulator.simulator`（见 `engine/paths.py:bootstrap`）。

2. **进程级全局状态**：`context._CONFIG` 是模块级全局变量。
   Web 层每次估算都在**独立子进程**中执行（`estimators/adapter.py`），
   因此并发请求不会互相污染。

---

## 5. 六种训练类型与能力边界

| 训练类型 | 状态 | 说明 |
|---------|------|------|
| **预训练** | 可估算 | 与全参 SFT 共用同一计算内核 |
| **全参数 SFT** | 可估算 | 同上 |
| **LoRA-SFT** | 可估算 | 独立资源模型，系数由本机实测基准库拟合 |
| DPO | 未建模 | 需要策略 / 参考双模型的显存与偏好对计算模型 |
| GRPO | 未建模 | 缺少 rollout 生成、规则/奖励模型与组内相对优势计算 |
| PPO | 未建模 | 需要 Actor/Critic 双向更新、GAE 优势估计与分阶段角色放置的峰值显存模型 |

**未建模的类型会明确拒绝并给出原因，不会套用全参训练的数字。**

### LoRA 资源模型

LoRA 不套用全参训练的数字，而是有自己的资源模型：

- **解析部分**（`services/lora_model.py`）：基座冻结后没有基座梯度、没有基座优化器状态，
  但反向仍需穿过基座，所以激活与反向计算不能按可训练参数比例缩减。
  适配器自身的参数、梯度与优化器状态按 `r × (in + out)` 计算。
- **校准部分**（`services/lora_calibration.py`）：吞吐按
  `step_time = fixed + per_sample × micro_batch` 从实测拟合，
  显存按同组实测点插值。

结果带完整来源标注：置信度等级（多样本拟合 / 单样本低置信 / 跨模型外推）、
拟合样本数、拟合误差、显存来源（实测插值 / 实测外推 / 解析基线）。

**实测库是只读的**（`mode=ro` 由 SQLite 强制拒绝写入），校准以数据库文件与
WAL sidecar 的大小和修改时间为指纹，新数据入库后自动重建。

> 校准以 `tokens_per_s`（实测吞吐）反推单步时间，而不是直接用库里的
> `step_time_s` 字段——该字段在不同采集批次间口径不一致（有的记单步、
> 有的记一个梯度累积周期），反推值与字段值的比值正好是累积步数的整数倍。
> 改用吞吐反推后，多样本拟合组的单步时间误差从 24% 降到 9%。

当前实测覆盖：6 个模型 × {none, z3} × {liger 开/关}，bs 1–32，seq 2048/4096，
均为 BW1000 真机 LLaMA-Factory + DeepSpeed 记录。

### 预训练与全参 SFT 的差异（如实说明）

两者当前**共用同一套计算内核，资源估算数值一致**——实测单步时间、TGS、峰值显存均相同。
内核声明了 `full_recompute` 字段但并未消费它，因此勾选重计算不会改变显存或步时。

实际差别在于：① 训练步数与总时长口径（预训练按总 Token 数折算，SFT 按样本数与 packing 折算）；
② 数据量与有效 Token 口径；③ 推荐的重计算策略不同。

`tests/test_task_difference.py` 会实测这一点——若内核将来开始区分两者，该测试会失败并提醒更新文案。

---

## 6. 功能

### 6.1 模型来源

- **预设模型**：35 个，覆盖 6 个适配器，`config.json` 随仓库保存，离线可用
- **ModelScope**：仓库名或链接，读取 `config.json` 并记录 revision 与配置哈希
- **HF 链接 / 上传文件 / 粘贴 JSON**

ModelScope 接入有白名单（`modelscope.cn` / `modelers.cn` 等）、SSRF 防护、
2 MiB 响应上限、30 秒超时，并记录 revision 以保证结果可复现。外部服务失败时
回落到本地上传路径，不阻断整个流程。

**ModelScope 模型可直接参与仿真**：`_build_from_request` 会把不是预设 slug 的
引用当作 ModelScope 仓库处理，拉取 config 并缓存到 `runtime/modelscope/<repo>-<hash>/`，
后续同一模型不再重复下载。

### 6.2 模型兼容性门禁

内核只有固定的模型适配器集合。某些模型落到不适配的适配器上——例如多模态模型
落到纯文本适配器、缺少 MLA 字段、带有内核未建模的结构。强行计算会得到失真数字，
所以 `services/compatibility.py` 在**估算之前**判定并拒绝，返回结构化的原因与建议：

- 该模型是多模态模型，而内核把它匹配到了文本适配器 `deepseek_v3`
- 缺少 MLA 字段（`kv_lora_rank` / `qk_nope_head_dim`），注意力布局无法确定
- 含有内核未建模的结构（DSA 索引注意力、dspark、engram 等）
- 缺少 MoE 层分布参数（`moe_layer_freq` / `first_k_dense_replace`）

界面在「参数校验」页逐条列出原因与可选做法，而不是给一句断言消息。

### 6.3 模型支持矩阵

内核只有 6 个适配器，且有一个通用兜底适配器 `generic_gpt`——任何模型都会
落到某个适配器上。所以失败从来不是"没有适配器"，而是"落到的适配器无法
表示这个架构"。

| 适配器 | 覆盖 |
|---|---|
| `qwen3` | Qwen3 稠密与 MoE（GQA） |
| `qwen3_vl` | Qwen3-VL 多模态（视觉塔 + 文本解码器） |
| `qwen3_5` | Qwen3.5 混合线性注意力（layer_types + MTP） |
| `deepseek_v3` | DeepSeek V2/V3/R1（MLA 低秩 KV + MoE） |
| `glm5` | GLM-5（DSA 索引注意力 + MoE） |
| `generic_gpt` | 通用稠密解码器兜底（标准注意力 + 标准 FFN） |

矩阵区分三种状态：

- **可精确估算**——内核有该架构的专用适配器
- **可估算（近似值）**——落到 `generic_gpt`。对普通稠密解码器成立；对注意力
  或专家布局不同的模型只是量级参考
- **不可估算**——内核无法表示，列出具体缺什么

界面「参数校验 → 模型支持矩阵」按系列列出每个预设模型的状态、适配器和
缺失项。接口：

    GET  /api/v1/model-support              全量矩阵
    POST /api/v1/model-support/check        单模型查询
    GET  /api/v1/model-support/adapters     每个适配器读哪些配置字段

**适配器选择由内核自己回答。** `services/compatibility.py` 调用内核的
`resolve_model_adapter`，不复制它的匹配逻辑——复刻的版本错了两次（漏掉兜底
适配器、把 `glm4_moe` 误配成 `glm5`）。`tests/test_adapter_agreement.py` 把
"预测的适配器"和"内核实际使用的适配器"逐个对比，8 个模型全部一致。

**阻断与提示是分开的。** 内核在 `config/models.py` 统一读取配置字段，再由
适配器决定用哪些。所以某个字段适配器不读，只意味着该结构按通用公式处理，
不构成拒绝理由；只有内核根本没有对应字段的结构才会阻断。

### 6.4 三类推荐方案

| 方案 | 目标 |
|------|------|
| 最低可训练方案 | 满足显存、并行整除、硬件约束，卡数最少，预留 5 GiB 安全余量 |
| 推荐稳定方案 | 更高显存余量，通信占比不超 50%，适合持续训练与 Checkpoint |
| 推荐性能方案 | 给定资源上限内按 TGS 排序，并给出相对最低方案的提速与瓶颈 |

三类方案由 simulator 的 `solve` 得出，最小卡数方案在显存受限时可能仍是同样的卡数
（例如 Qwen3-8B 单卡 64 GiB 装不下 TP=1），页面会如实显示。

### 6.5 训练镜像与启动配置

- **镜像目录**：来自光合开发者社区 Harbor 仓库快照，16 个仓库 / 192 个训练相关 tag，
  标注框架、DTK / Python / OS 版本与 `docker pull` 命令
- **提取码**：按 G02 要求哈希存储，列表只显示掩码
- **训练模板**：预训练 / 全参 SFT / LoRA-SFT / DPO / GRPO / PPO 各一套，
  未在目标环境验证的模板在 YAML 头部和页面上都有明确警告
- **导出**：YAML 训练配置、Shell 启动脚本、Docker 命令、分布式环境变量、Rendezvous 命令

### 6.6 参数校验

并行整除性（`num_gpus % (tp*cp*pp)`、KV 头数、层数、MoE expert placement）、
Batch 关系（`GBS = MBS × DP × 梯度累积`）、数据量与步数换算、检查点设置。
每个问题都带原因和可行建议。

---

## 7. 硬件支持

**BW1000** 是唯一完整可用的卡型：同时具备实测带宽表、算子 Profile 与完整硬件参数。

其他卡（如 BW1100 / BW500SM）会保留在目录中并标注不支持原因，但**不提供估算**——
模拟器内核对 `gpu_model` 写死为 BW1000，硬算会直接破坏与参考仓库一致这条底线。

---

## 8. 测试

```bash
bash tools/run_tests.sh            # 全量（含 35/35 基线）
SKIP_BASELINE=1 bash tools/run_tests.sh   # 跳过基线

python tests/test_api_v1.py        # 分层接口
python tests/test_launch.py        # 镜像与启动命令
node tests/frontend_test.js        # 前端 DOM 流程
python tests/test_e2e_live.py http://127.0.0.1:7801   # 实时端到端
python tests/test_task_difference.py                 # 预训练 vs SFT 实测差异
python tests/test_lora_model.py                      # LoRA 模型精度
python tests/test_lora_api.py                        # LoRA 接口
python tests/test_calibration_refresh.py             # 校准随数据库更新重建
python tests/test_model_sources.py                   # 模型来源与兼容性门禁
python tests/test_adapter_agreement.py               # 适配器选择与内核一致
python tests/test_support_matrix.py                  # 模型支持矩阵
python tools/probe_model_support.py                  # 批量探测各系列模型
```

---

## 9. 已知边界

- 迭代时间是**规划估算**，用于方案对比与瓶颈分析；参考仓库自身也说明
  数据搬移、kernel launch、框架调度等未被精确建模。
- 仓库的内置带宽表只实测到 2 节点。当通信域跨节点布局与实测表不一致时，
  仓库会打印 `Bandwidth table topology mismatch` 警告并做近似——
  这是参考仓库自身的行为，网页与 CLI 表现一致。
- 提取码只存哈希，无法反查明文。
- 未建模的训练类型（见第 5 节）不提供估算。
- 模型支持按架构划分，不按发布批次划分（见第 6.3 节）。已实测确认的边界：
  - **可精确估算**：Qwen3 / Qwen3-VL / Qwen3.5 全系列、DeepSeek-V2/V3/R1、
    GLM-5 全系列。Kimi-K2 的架构声明就是 `DeepseekV3ForCausalLM`，
    走 `deepseek_v3` 适配器。
  - **可估算（近似值）**：Qwen2.5、Llama、Mistral、Phi、Gemma 等标准稠密解码器
    （走 `generic_gpt`），以及 GLM-4.5。
  - **不可估算**：
    - DeepSeek-V3.2 及以后——含 DSA 索引注意力，内核只有 `glm5` 适配器会
      应用 `index_*` 字段
    - DeepSeek-V4.1-Flash-1017——多模态视觉塔 + 缺 MLA 字段 + 缺 MoE 层分布
      参数 + DSA / candidate / dspark / engram 结构
    - GLM-4.6V-Flash——多模态视觉塔
    - MiniMax 全系（M2 / M1 / Text-01）——专家数写在 `num_local_experts`，
      内核只读 `num_experts` 与 `n_routed_experts`，会当成稠密模型算，
      显存与算力被大幅低估；另有 `attn_type_list`（Lightning 混合注意力）
      与 `use_mtp` 无对应建模
  - 兼容性门禁在估算**之前**拦截。这很重要：DeepSeek-V3.2 若直接调内核，
    它照样能算出数，而且与 DeepSeek-V3 的结果完全一致——那是个看着正常、
    实际失真的数字。
  - 门禁拦下的模型，内核其实也能算出数——但那个数是错的。MiniMax-M2 是
    230B 总参 / 10B 激活的 MoE，内核按稠密模型算出的峰值显存是 67.48 GiB，
    而同为 MoE 的 Qwen3-30B-A3B 是 239.65 GiB，低了 3.5 倍：256 个专家被
    当成了一个稠密 FFN。更直接的证据是 MiniMax-M1 与 MiniMax-Text-01
    算出的数字完全相同（127768.23 ms / 270.05 GiB）——两者层数与隐藏维度
    不同，本该有差异。
  - 门禁的方向与内核一致，不是臆断。内核自己在
    `config/models.py` 有一道防线：多模态模型配纯文本适配器时直接抛
    `accepted a vision_config but built a text-only ModuleSpec; refusing to
    drop the vision model`。门禁只是把这一步提前到拉取 config 之后、
    估算之前，让用户看到「缺什么」而不是一个 Python 异常。

---

## 10. 授权

本项目使用 Apache License 2.0，与 [`hcu-train-simulator`](https://github.com/HYGON-AI/hcu-train-simulator)
保持一致。`engine/` 目录下的代码派生自该仓库，版权归 Hygon Information Technology Co., Ltd. 所有，
详见 `NOTICE`。
