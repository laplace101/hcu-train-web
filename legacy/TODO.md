# TODO — hcu-train-web 后续开发需求

会话结束时间：2026-10-09。下一回来直接照着这份 TODO 干。


## 本轮完成记录（2026-10-10）

- ✅ **P0-1 后端 bug 修复**：`_build_config_yaml` 里 `TASK_DEFAULTS.get(task, ...)` 拿到内层 dict（key 是 `full_recompute` / `recompute_note`），再 `.get(task, default)` 永远 miss，回退到默认 `false`。
  改用**一次** `.get()` 拿内层 dict，task=`pretrain` 现在能正确输出 `full_recompute: true` + 注释；task=`lora` 输出完整 lora 块 + simulator 提示。
  顺手把多余的 f-string `{{full_recompute}}` 替换为直接占位符 `{full_recompute}`（f-string 本身是对的，但语义不清晰）。

- ✅ **P0-1 前端**：templates/index.html 顶部加 `<h2>① 选择任务</h2>`，3 选 1 radio；
  原 ①②③ 顺延为 ②③④；新增 `<section id="lora-card" class="card hidden">` 放 `lora_r / lora_alpha / lora_dropout / lora_targets`。
  static/app.js：新增 `selectedTask() / syncTaskUi()`，切到 `lora` 时自动展开卡片，`runSimulator()` 把 `task` + 4 个 lora 字段塞进 payload。
  static/styles.css：加 `.task-group` 网格 + 选中态高亮 + `.grid.three`（3 列 LoRA 表单）。

- ✅ **P0-2 MODEL_PRESETS 重组**：29 条按内置 adapter 分 6 组（qwen3×8 / qwen3_5×3 / qwen3_vl×3 / deepseek_v3×2 / glm5×1 / generic_gpt×12），每条都带 `adapter` 字段。
  /tmp/build_presets.py 离线跑过 HF config.json 抓取 + 跑 `match_score` 验证，验过的才会写进 backend.py。
  受 gated（meta-llama / google / 部分 THUDM）影响的部分通过 declared_group 回退标记，stderr 里能看到。

- ✅ **P0-3 硬件**：hardware_presets.py 缩到 4 条（hygon_dcu_bw1000 / nvidia_a100 / nvidia_h100 / nvidia_h200），删 `hygon_dcu_bw1000_empirical`。
  默认选 hygon_dcu_bw1000。

- ✅ **P0-4 LoRA**：见上面 P0-1。后端 _build_config_yaml 已有完整 lora 块；前端 LoRA 卡片可见性已接通。

- ✅ **P1-5 tests/run_presets.py**：写完，4×4 sweep qwen3_8b 在 nvidia_a100 上 56.9s 跑完 16 runs，落 16 行 CSV 到 /tmp/sweep_4x4.csv。
  失败组合（tp=1,pp=1 / num_gpus>16）以 `simulator_rc=1` + 空指标 形式保留在 CSV 里，方便人肉对比。
  入口：```python3 tests/run_presets.py --only <preset> --tp 1 2 4 8 --pp 1 2 4 8 --out sweep.csv```

- **重启**：服务现在跑在 PID `/tmp/hcu-train-web.pid`，入口仍是 http://127.0.0.1:7766/，但**改完代码必须重启**（`python3 /tmp/launch_backend.py` 双 fork 守护，否则老进程占着 Python 内存不会 reload）。

## 已知边界（next session 可选做）

- `_summarize_config()` 只看顶层字段，Qwen3.5+ 这种 `text_config` 嵌套结构识别不到 num_attention_heads/num_key_value_heads。修法：在 `_fetch_hf_config` 之后把 `text_config` / `vision_config` 拍平到顶层。
- simulator 的 `num_query_groups must be divisible by tp_size` 会在某些 Qwen3.5 + 小 TP 组合炸，但 solve 阶段还能跑出结果；前端要明确告诉用户「sim 失败 / solve 成功」。
- run_presets 的 sweep 在 num_gpus > 16 时几乎全炸（simulator 内部断言），后续要么限定 num_gpus 上限、要么记到「lab infeasible」表里。

## 当前已经完成（不要再动）

✅ `backend.py` — Flask 后端，封装 hcu-train-simulator  
✅ `templates/index.html` + `static/styles.css` + `static/app.js` — Web UI  
✅ `hardware_presets.py` — 5 个硬件预设  
✅ 服务在 `127.0.0.1:7766` 跑着（pid 写 `/tmp/hcu-train-web.pid`）  
✅ `/root/hcu-train-web/start.sh` — 管理脚本（start/stop/restart/status）  
✅ Cloudflare 公网隧道已 **关掉**（用户要求切回本地）  
✅ `git` 仓库存放于 `/root/hcu-train-simulator`  
✅ `_summarize_config()` — 规则式架构识别（arch / MoE / MLA / VLM / GQA…）  
✅ `_parse_results()` — 从 simulator stdout 抽出结构化数字给前端展示  

## 下一会话要做的事（按优先级排序）

### P0 — 必须先做的

#### 1. 任务分类拆三栏（最优先） ✅ DONE 2026-10-10

UI 顶部添加 3 个 radio 按钮 / tab，**全参 / 预训练 / lora微调**，下面所有 YAML 生成逻辑按所选任务分支：

| 任务 | 关键 YAML 字段差异 |
|------|------|
| **全参 (full FT)** | 当前默认逻辑（`full_recompute=false` + `use_distributed_optimizer=true`）|
| **预训练 (pretrain)** | 同全参，但 **`decoder_*` pipeline 层数 = num_layers**、`<bos>` token 打开、`use_distributed_optimizer=true` |
| **lora微调 (lora)** | 留空先用 TODO 形式提示「LoRA 模式暂未实装」，模型自带 LoRA 层数计算公式参考 https://github.com/HYGON-AI/hcu-train-simulator 仓内 `modeling/lora.py`（如果有） |

实现位置：
- `templates/index.html`：`<h2>① 选择任务</h2>` tab，加在原 ① 之前
- `static/app.js`：`runSimulator()` 把 `task` 加进 payload
- `backend.py：_build_config_yaml()` 按 `task` 切三个分支

#### 2. 预设模型按 hcu-train-simulator 内置 adapter 全列 ✅ DONE 2026-10-10

盘点后端仓库存放的 adapter 覆盖范围（截至此次会话末）：

```
qwen3     → HF types: qwen3, qwen3_moe            prefix=qwen3_    archs=Qwen3For,Qwen3MoeFor
qwen3_5   → qwen3_5, qwen3_5_moe, _text, _moe_text  prefix=qwen3_5  archs=Qwen3_5
qwen3_vl  → qwen3_vl, qwen3_vl_moe, _text, _moe_text prefix=qwen3_vl archs=Qwen3VL
glm5      → glm_moe_dsa                          prefix=glm_moe_dsa archs=GlmMoeDsa
deepseek_v3 → deepseek_v3, deepseek_v2           prefix=deepseek   archs=Deepseek,DeepSeek
generic_gpt → 上面没覆盖到的全部兜底
```

文件路径：`/root/hcu-train-simulator/src/hcu_train_simulator/models/*/`

要做的事：
1. 把 `backend.py:67 MODEL_PRESETS` 改成**按 adapter 分组**组织，每个模型附 `adapter: <name>` 字段
2. 至少塞这些 HF id（用户能直接拿来用）：
   - **qwen3**: Qwen/Qwen3-{0.6B, 1.7B, 4B, 8B, 14B, 32B}, Qwen/Qwen3-{30B-A3B, 235B-A22B}
   - **qwen3_5**: Qwen/Qwen3.5-*（按仓内已记录的 HF id）
   - **qwen3_vl**: Qwen/Qwen3-VL-*  
   - **glm5**: THUDM/GLM-5-*（最新 GLM5 DSA 系列）
   - **deepseek_v3**: deepseek-ai/DeepSeek-V3, deepseek-ai/DeepSeek-V2, deepseek-ai/DeepSeek-V2-Lite-Chat, deepseek-ai/deepseek-llm-{7B,67B}
   - **generic_gpt**（兜底，可选塞几个方便人用）：meta-llama/Llama-3.{1,2}-{1B,3B,8B,70B}, meta-llama/Meta-Llama-3-{8B,70B}, Qwen/Qwen2.5-{7B,72B}, google/gemma-2-{9B,27B}, microsoft/Phi-3-{mini,medium,14B}
3. `summary` 块顺手把 `adapter` 也显示出来，让用户能确认路由到哪个内置 adapter

#### 3. 硬件默认只保留 DCU BW1000 ✅ DONE 2026-10-10

`hardware_presets.py` 里海光 (Hygon) 那个 `hygon_dcu_bw1000_empirical` **删掉**，保留 `hygon_dcu_bw1000` 作为默认。

`hardware_presets.py` 改完后只剩下：
```
hygon_dcu_bw1000   ← 用户进站默认选项
nvidia_a100         ← 保留（之前留的，便于横向对比）
nvidia_h100
nvidia_h200
```

具体实现：把 `hygon_dcu_bw1000` 作为 `hardware_presets` 字典的第 0 项 + 前端 `loadPresets()` 自动选第一项。

#### 4. LoRA 计算留空 ✅ DONE 2026-10-10

`backend.py：_build_config_yaml()` 加一个分支：

```python
if task == "lora":
    # TODO: 仓内 modeling/lora.py 还没有公开 API，LoRA 计算暂留空
    yaml_text = yaml_text + (
        "# NOTE: LoRA mode currently shows full-FT numbers; "
        "# real LoRA-aware compute is not implemented yet.\n"
    )
    # 同时把 LoRA 关键参数暴露给用户：
    #   lora_r=16, lora_alpha=32, lora_dropout=0.05, target_modules
    # 前端加一个 lora_r / lora_alpha / target_modules 输入
```

注释里写清为什么留空（仓内没有 lora.py），并 TODO 链接到 https://github.com/HYGON-AI/hcu-train-simulator。

### P1 — 自动测试矩阵（每次加模型必跑）

#### 5. 加模型后自动跑测试 ✅ DONE 2026-10-10

写个 `tests/run_presets.py`，自动遍历 `MODEL_PRESETS` 里的每个 preset，跑下面的矩阵：

```
TP ∈ {1, 2, 4, 8}
PP ∈ {1, 2, 4, 8}  (但 TP × PP ≤ num_gpus=8)
EP ∈ {1, 8}
seq_length ∈ {2048, 4096, 8192}
gbs ∈ {64, 256, 1024}
```

每个组合调 `/api/predict` 一次，统计：
- 跑通率（rc=0 比例）
- simulation 阶段 vs solve 阶段分别多久
- 跨组合的 iteration_ms / TGS / MFU 是否**单调合理**（TP↑ → iter↓；gbs↑ → iter↑）

输出 `/root/hcu-train-web/bench-results/<preset>-<timestamp>.csv`，列：

```
preset,task,hardware,tp,pp,ep,cp,mbs,gbs,seq,sim_rc,sim_iter_ms,sim_mfu,sim_tgs,solve_rc,solve_best_iter_ms,solve_best_tgs
```

下一步细节：跑成功后，把结果附在每次新加模型 PR 的 commit message 里，方便用户肉眼对比。

### P2 — 可选优化

#### 6. 架构识别的 fallback 改进

仓内 `generic_gpt` 没覆盖到 `qwen2_vl`、`chatglm`、`llama` 这些。`/api/predict` 现在如果识别为 unknown，就 fallback 走 generic（之前测试 `chatglm` 模型就是 simulation 跑挂了但 solve 仍出）。可以考虑：
- 让用户选「强制走 generic adapter」 checkbox
- 提示「simulation 阶段可能失败，仅 solve 结果可信」

#### 7. 海光卡的 FP8 / DS 自动提示

仓内 `profile_config.builtin_selector` 支持 `accelerator: bw1000 | bw1100`。前端让用户选硬件时自动把 bw1100 选为默认（最新海光卡）。

## 复盘这个会话踩过的坑（避免再踩）

- **stdlib `logging` 与仓内 `logging.py` 同名**，脚本模式 `python3 cli.py` 会自踩。  
  → 必须 `python3 -m hcu_train_simulator.cli <subcmd>` 走 module 模式
- **PEP 604 (`X | Y`) 和 PEP 585 (`tuple[X]`) 语法** Python 3.7 跑不动。  
  → 用 uv 自带的 Python 3.14（`/root/.local/share/uv/python/cpython-3.14.7-.../bin/python3.14`）
- **ParallelConfig 缺字段**，YAML 里 `etp_size / decoder_first/last_pipeline_num_layers` 必须有，否则 simulator `from_dict` 抛 `missing 3 required positional arguments`
- **`gpus_per_node=8` 与 `num_gpus=4` 冲突**，必须 `gpn=min(hw["gpus_per_node"], num_gpus)`
- **后台运行的进程经常被 exec_command session end 时 SIGKILL**。解决办法：`setsid + </dev/null + disown` 三件套，或者用双 fork 大法（在 `/tmp/launch_backend.py` 里）。模板在 `/tmp/launch_backend.py`
- **Qwen3.5 / chatglm / qwen3_5_moe 模型 simulation stage 阶段会失败**（`num_query_groups` assert / adapter 不全），但 **solve 阶段还能出结果**，前端要让用户看到这个区别

## 启动 / 关闭

```bash
# 启动 backend
/root/hcu-train-web/start.sh start

# 关掉
/root/hcu-train-web/start.sh stop

# 起一个公网隧道（仅调试用，域名每次换）
/root/hcu-train-web/start-tunnel.sh
cat /tmp/cf-public-url
```

## 跑测试矩阵

```bash
# TODO: 未实现，先 stub
python3 /root/hcu-train-web/tests/run_presets.py --preset qwen3_8b --task full_ft
```
