# hcu-train-web 部署包

自包含。`hcu-train-simulator` 已经放在包里，目标机器**不需要 clone、不需要联网下代码**，
只要有 Python 3.9+ 和能装 pip 依赖即可。

```
hcu-train-web-deploy/
├── deploy.sh              # Linux / macOS
├── deploy.cmd             # Windows
├── DEPLOY.md              # 本文件
└── hcu-train-simulator/   # 参考仓库（计算逻辑来源，已内置）
    └── hcu-train-web/     # Web 应用
```

---

## 0. 环境要求（重要）

**必须 Python ≥ 3.9。** 参考仓库在导入时就会求值 PEP 585 / 604 注解
（例如 `ModuleRef = Union[type, tuple[str, str]]`），Python 3.7 / 3.8 会直接
报 `TypeError: 'type' object is not subscriptable`。

`deploy.sh` 会**自动探测**可用的解释器，依次尝试：
`python3.14 → … → python3.9 → python3`，
再回退到常见自装位置（`~/.local/share/uv/python/*/bin/python3`、
`/opt/conda/bin/python3`、`/root/miniconda3/bin/python3` 等）。

如果机器上只有 3.7，先装一个新的：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # uv，最省事
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
```

或者手动指定：`PY=/path/to/python3.12 ./deploy.sh`。

**pip 源**：不少内网/大环境机器连不上 `pypi.org`，但能连国内镜像。
脚本默认走清华源；换源用：

```bash
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple ./deploy.sh
```

---

## 1. 传到工作站

任选一种：

**A. scp（Linux 工作站）**
```bash
scp -r hcu-train-web-deploy user@<工作站IP>:/opt/
```

**B. U 盘 / 共享目录**
把整个 `hcu-train-web-deploy` 文件夹拷过去，解压即可。

**C. 打包传输**
```bash
# 在当前这台机器上
python tools/make_bundle.py --zip     # 生成 hcu-train-web-deploy.zip
```

---

## 2. 一条命令启动

**Linux**
```bash
cd /opt/hcu-train-web-deploy
chmod +x deploy.sh
./deploy.sh
```

**Windows**
```bat
cd C:\hcu-train-web-deploy
deploy.cmd
```

脚本会自动：建 venv → 装依赖 → 起服务（监听 `0.0.0.0:7766`）→ 健康检查 → 打印可访问地址。

常用操作：

```bash
./deploy.sh stop        # 停止
./deploy.sh restart     # 重启
./deploy.sh status      # 查看状态
PORT=8080 ./deploy.sh   # 换端口
tail -f hcu-train-web/runtime/server.log
```

---

## 3. 在你自己的电脑上打开

部署脚本最后会打印类似：

```
本机自检   : http://127.0.0.1:7766/
已校验模型 : 35/35
局域网访问 : http://10.x.x.x:7766/
```

**在你自己的 PC 浏览器里打开那个「局域网访问」地址**，前提是两台机器在同一内网。

打不开按顺序排查：

1. **网络不通** —— 先 `ping <工作站IP>`。不通说明不在同一内网，或需要 VPN。
2. **防火墙挡住** —— 工作站上执行：
   ```bash
   # CentOS / RHEL / 欧拉
   firewall-cmd --add-port=7766/tcp --permanent && firewall-cmd --reload
   ```
   ```bat
   :: Windows
   netsh advfirewall firewall add rule name="hcu-train-web 7766" dir=in action=allow protocol=TCP localport=7766
   ```
3. **内网隔离 / 不方便放端口** —— 改用 SSH 隧道，不需要开任何端口：
   ```bash
   # 在你自己的 PC 上执行
   ssh -L 7766:127.0.0.1:7766 user@<工作站IP>
   ```
   然后本地浏览器打开 <http://127.0.0.1:7766/>。

---

## 4. 部署后自检

```bash
curl http://127.0.0.1:7766/api/health
# {"ok":true,"simulator_src":".../hcu-train-simulator/src"}
```

想再跑一遍 35 个模型的准确率校验（比对官方 `hcu-train-sim` CLI）：

```bash
cd hcu-train-web
python verification/verify_accuracy.py
# 预期：通过 35 / 35，失败 0
```

---

## 5. 注意事项

- **离线也能用**：35 个预设模型的 `config.json` 已随包内置，不依赖 HuggingFace。
  只有「HF 链接」这一种输入方式需要工作站能访问外网。
- **代理**：如果工作站需要走代理才能上外网，启动前设置 `HTTP_PROXY` / `HTTPS_PROXY`。
  Linux 端 `deploy.sh` 会自动读取 Windows 注册表里的代理设置（若从 Windows 同步过去）。
- **算力与硬件**：BW1000 预设来自参考仓库 `templates/config.yaml` 与内置实测带宽表，
  部署到别的卡上需要改 `engine/config_builder.py` 里的 `BW1000` 定义。
- **开机自启**（可选，Linux）：
  ```bash
  cat >/etc/systemd/system/hcu-train-web.service <<'EOF'
  [Unit]
  Description=hcu-train-web
  After=network.target
  [Service]
  WorkingDirectory=/opt/hcu-train-web-deploy/hcu-train-web
  ExecStart=/opt/hcu-train-web-deploy/hcu-train-web/.venv/bin/python -u app.py --host 0.0.0.0 --port 7766
  Restart=always
  [Install]
  WantedBy=multi-user.target
  EOF
  systemctl enable --now hcu-train-web
  ```