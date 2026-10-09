# AttnRes-ViT Phase 0 实验运行指南

> 本文档说明如何在 **2 台服务器 × 8×A40 (48GB)** 集群上运行 Phase 0 实验（共 16 卡）。  
> 每个实验使用 **4 张 GPU（单机内 DDP）**，两台机器同时跑时可 **并行 4 个实验**。

---

## 1. 目录结构

```
attnres/
├── attnres_vit/          # 我们的 AttnRes ViT 实现
├── configs/phase0/       # 8 组实验配置 (E01-E08)
├── scripts/              # 一键运行脚本
├── repos/                # 第三方 baseline 代码 (timm, dinov2, DiT, ...)
├── train_imagenet.py     # 训练入口
├── validate_imagenet.py  # 验证入口
├── tests/test_forward.py # 冒烟测试
└── output/phase0/        # 训练输出 (自动创建)
```

---

## 2. 一次性环境配置

```bash
cd /path/to/attnres
bash scripts/setup_env.sh          # 建 ./.venv（torch 2.10.0+cu128、vendored timm、attnres_vit）并跑冒烟测试
source .venv/bin/activate          # 之后每次训练/评测前激活
```

为什么不用 conda / 直接 pip（147 上 2026-10 实测）：conda.anaconda.org 连不上、defaults 源需要接受服务条款；
阿里云镜像对 HTTP/1.1（pip、uv 都走这个）限速到约 0.24 MB/s，而 HTTP/2（curl）有约 25 MB/s。
所以脚本用 curl 下载 torch、torchvision 和 torch 锁定的 nvidia/triton 包，再用 uv 从本地文件安装。
需要 CPython 3.13（默认 `~/miniconda3/bin/python3`，可用 `PYTHON=...` 指定）。

测试：

```bash
python tests/test_forward.py         # 所有模型前向 + 初始化时 AttnRes 与同权重 baseline 输出一致
python tests/test_kimi_reference.py  # float64 下与 Kimi 论文 Fig. 2 伪代码逐行移植的实现一致
```

---

## 3. 修改数据路径

**所有实验配置文件**中的 `data_dir` 默认为 `/data/imagenet`，请改为服务器上的实际路径。

ImageNet 目录结构应为：

```
/imagenet/
├── train/
│   ├── n01440764/
│   └── ...
└── val/          # 或 validation/ — 若目录名不同，改 yaml 中 val_split
    ├── n01440764/
    └── ...
```

批量替换示例：

```bash
# 假设数据在 /mnt/datasets/imagenet
sed -i 's|/data/imagenet|/mnt/datasets/imagenet|g' configs/phase0/*.yaml
```

若验证集目录名为 `validation` 而非 `val`：

```bash
sed -i 's|val_split: val|val_split: validation|g' configs/phase0/*.yaml
```

### 3.1 数据放本地盘（推荐）

数据在 NAS 上时，每读一张图都要一次网络往返，147 上两个实验各只有约 350 张/秒（GPU 能跑约 1750）。
推荐做法：`data_dir` 指向项目内的软链接目录，数据先用 NAS 上的，再拷一份到本地盘后无缝切换：

```bash
mkdir -p data/imagenet
ln -s /mnt/a100b/default/wangjin/data/imagenet/train data/imagenet/train        # 147 上的 NAS 路径
ln -s /mnt/a100b/default/wujiannan/data/imagenet-1k/val data/imagenet/val
sed -i "s|^data_dir: /data/imagenet\$|data_dir: $PWD/data/imagenet|" configs/phase0/*.yaml

# 拷到本地盘（在 tmux 里跑；训练集从整包 tar 流式解压，约 30 分钟）
python scripts/data/extract_imagenet_train.py /mnt/a100b/default/wangjin/data/imagenet/ILSVRC2012_img_train.tar /mnt/d/bench/imagenet/train
mkdir -p /mnt/d/bench/imagenet/val && (cd /mnt/a100b/default/wujiannan/data/imagenet-1k/val && ls | xargs -P 32 -I{} cp -r {} /mnt/d/bench/imagenet/val/)

# 核对文件清单 + 抽查内容，一致才切换软链接；正在跑的实验下一次读文件就读本地，无需重启
bash scripts/data/switch_to_local_data.sh /mnt/d/bench/imagenet \
  /mnt/a100b/default/wangjin/data/imagenet/train /mnt/a100b/default/wujiannan/data/imagenet-1k/val
```

---

## 4. Phase 0 实验矩阵

| ID | 配置文件 | 模型 | Depth | Dim | AttnRes | 预期参数量 |
|----|----------|------|-------|-----|---------|-----------|
| E01 | `e01_d12_baseline.yaml` | baseline | 12 | 768 | ✗ | ~87M |
| E02 | `e02_d12_attnres.yaml` | +BlockAttnRes | 12 | 768 | ✓ (4 blocks) | ~87M |
| E03 | `e03_d24_baseline.yaml` | baseline | 24 | 768 | ✗ | ~172M |
| E04 | `e04_d24_attnres.yaml` | +BlockAttnRes | 24 | 768 | ✓ (8 blocks) | ~172M |
| E05 | `e05_d48_narrow_baseline.yaml` | deep-narrow | 48 | 512 | ✗ | ~152M |
| E06 | `e06_d48_narrow_attnres.yaml` | deep-narrow | 48 | 512 | ✓ (8 blocks) | ~152M |
| E07 | `e07_d48_wide_baseline.yaml` | deep-wide | 48 | 768 | ✗ | ~342M |
| E08 | `e08_d48_wide_attnres.yaml` | deep-wide | 48 | 768 | ✓ (8 blocks) | ~342M |

训练配置：DeiT-style，300 epochs，AdamW lr=1e-3，batch=128/GPU (global 512 for E01-E06)，RandAug + Mixup/CutMix + EMA。

---

## 5. 运行单个实验

所有训练都在 tmux 里跑，SSH 断开不影响。

```bash
tmux new -s e01                  # 进入 tmux 后先 source .venv/bin/activate

# 语法: bash scripts/train_4gpu.sh <config.yaml> <gpu_ids> [额外参数]
# 示例: E01 在 GPU 0-3 上训练
bash scripts/train_4gpu.sh configs/phase0/e01_d12_baseline.yaml 0,1,2,3

# 示例: 快速验证 (100 epochs)，另开一个 tmux 会话跑；同机第二个任务换端口
MASTER_PORT=29501 bash scripts/train_4gpu.sh configs/phase0/e02_d12_attnres.yaml 4,5,6,7 --epochs 100

# 离开但不中断: Ctrl-b d；回来: tmux attach -t e01
```

### 5.1 随机种子与可复现性

| 项目 | 设置 |
|------|------|
| **默认 seed** | `42`（所有 phase0 配置已显式写入 `seed: 42`） |
| **多卡 seed** | timm 对每个 rank 使用 `seed + rank`（DDP 标准做法） |
| **DataLoader** | `worker_seeding: all`，worker 从主 seed 派生 |
| **严格确定性** | `ATTNRES_DETERMINISTIC=1 bash scripts/train_4gpu.sh ...`（更慢，不保证所有 GPU 上 bit-exact） |

> Phase 0 scaling 实验通常 `seed=42` + timm 默认即可；严格确定性仅用于 debug / 小规模 ablation。

### 5.2 输出目录命名（防覆盖 + 一眼识别配置）

每次运行自动创建**带时间戳**的目录，不会覆盖旧结果：

```
output/phase0/{YYYYMMDD_HHMMSS}_{experiment_slug}_s{seed}/
```

示例：
```
output/phase0/20261008_153045_e01_d12_baseline_s42/
output/phase0/20261008_153045_e02_d12_attnres_s42/
```

从目录名可直接读出：**日期时间 · 实验 ID · seed**。

若目录已存在且未设置 `RESUME`，脚本会**拒绝启动**（防误覆盖）。强制重用：`FORCE=1 bash scripts/train_4gpu.sh ...`

固定时间戳（调试）：`RUN_ID=20261008_120000 bash scripts/train_4gpu.sh ...`

### 5.3 实验记录文件（每个 run 目录内）

| 文件 | 内容 |
|------|------|
| `run_manifest.json` | 启动时间、hostname、seed、GPU、git commit、额外 CLI 参数 |
| `config.yaml` | 本次运行使用的配置副本 |
| `args.yaml` | timm 解析后的完整训练参数 |
| `summary.csv` | 每个 epoch 的 train loss、lr 和 eval top1/top5（开了 EMA 时为 EMA 模型的指标） |
| `last.pth.tar` | 最新 epoch 的完整状态，每个 epoch 覆盖 |
| `checkpoint-{epoch}.pth.tar` | 按验证指标保留最好的 10 个 |
| `model_best.pth.tar` | 验证集（EMA）最佳 checkpoint（评测、续训用） |

Checkpoint **文件名**沿用 timm 惯例（`checkpoint-290.pth.tar`），配置信息在**父目录名**中；完整配置见 `config.yaml` / `args.yaml`。

查看训练日志（vendored timm 不支持 TensorBoard）：

```bash
tail -f logs/phase0/e01_d12_baseline.log                                          # 实时输出
column -s, -t < output/phase0/20261008_153045_e01_d12_baseline_s42/summary.csv    # 每 epoch 指标
```

---

## 6. 双机 8+8 卡并行（推荐）

### 6.1 集群拓扑与排队

每台机器开两个 tmux 队列，每个队列占 4 张卡、按顺序跑两个实验，前一个结束后立即接下一个：

```
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│  Server A（8×A40）                    │     │  Server B（8×A40）                    │
│  GPU 0-3  p0_a_g0:  E01 → E07        │     │  GPU 0-3  p0_b_g0:  E03 → E05        │
│  GPU 4-7  p0_a_g1:  E02 → E08        │     │  GPU 4-7  p0_b_g1:  E04 → E06        │
└──────────────────────────────────────┘     └──────────────────────────────────────┘
         各实验独立 4 卡 DDP，不需要跨机 torchrun
```

- baseline / AttnRes 成对实验始终在同一台机器上同时跑。
- 最长的 d48 wide（E07/E08）排在最短的 d12 后面，两台机器负载接近（见 §9）；d12、d24 的结果最先出来。
- 每个实验启动前，队列会等这 4 张卡的显存占用都低于 2000 MiB（每 5 秒查一次，可用 `GPU_FREE_MIB` / `GPU_POLL_SEC` 调整）：卡被别人占着就等，不会 OOM。
- 某个实验失败时队列继续跑下一个，日志末尾记录 `END ... exit=<非 0>`；失败的实验之后用 §8 续训。

每台服务器本地 GPU 编号均为 **0–7**。单实验只在**一台机器内**跑 4 卡 DDP，**不要**跨两台做 8 卡 DDP（Phase 0 不需要）。

### 6.2 启动

**在 Server A 上**（SSH 登录并 `source .venv/bin/activate` 后）：
```bash
cd /path/to/attnres
NODE=a bash scripts/run_phase0_node.sh
```

**在 Server B 上**：
```bash
cd /path/to/attnres
NODE=b bash scripts/run_phase0_node.sh
```

脚本只创建 tmux 会话后立即返回，之后断开 SSH 不影响训练。

若部分卡被别人占着，可按实际空闲情况指定两组卡，被占的那组会等卡空出后自动开始，例如：
```bash
GPUS_G0=0,5,6,7 GPUS_G1=1,2,3,4 NODE=a bash scripts/run_phase0_node.sh
```

| 服务器 | tmux 会话 | GPU | 实验（依次） | 日志 |
|--------|-----------|-----|--------------|------|
| A | `p0_a_g0` | 0-3 | E01 → E07 | `logs/phase0/e01_d12_baseline.log`、`e07_d48_wide_baseline.log` |
| A | `p0_a_g1` | 4-7 | E02 → E08 | `logs/phase0/e02_d12_attnres.log`、`e08_d48_wide_attnres.log` |
| B | `p0_b_g0` | 0-3 | E03 → E05 | `logs/phase0/e03_d24_baseline.log`、`e05_d48_narrow_baseline.log` |
| B | `p0_b_g1` | 4-7 | E04 → E06 | `logs/phase0/e04_d24_attnres.log`、`e06_d48_narrow_attnres.log` |

查看 / 停止：

```bash
tmux ls
tmux attach -t p0_a_g0           # 退出查看但不中断: Ctrl-b d
tmux kill-session -t p0_a_g0     # 停掉整条队列（含还没开始的实验）
```

### 6.3 可选：从一台机器 SSH 分发

若已配置免密 SSH，编辑 `scripts/cluster_dispatch.sh` 中的 `SERVER_A` / `SERVER_B` / `REMOTE_PROJECT` / `REMOTE_ENV`（非交互 SSH 下激活环境的命令，默认 `source $REMOTE_PROJECT/.venv/bin/activate`），然后：

```bash
bash scripts/cluster_dispatch.sh   # 两台机器各起两个 tmux 队列
```

### 6.4 输出目录与共享存储

- **推荐**：两台机器挂载**同一 NFS**，`output/phase0/` 路径一致，结果自动汇总。
- **若无共享盘**：各机输出在本地，`run_manifest.json` 里有 `hostname` 字段可区分；训练完后 rsync 汇总：

```bash
rsync -av server-a:/path/to/attnres/output/phase0/ ./output/phase0_from_a/
rsync -av server-b:/path/to/attnres/output/phase0/ ./output/phase0_from_b/
```

### 6.5 同机双队列的端口隔离

同一台机器上的两个队列分别使用 `MASTER_PORT` 29500（GPU 0-3）和 29501（GPU 4-7），队列内的实验依次复用该端口，避免 torchrun 冲突。

### 6.6 改代码后替换正在跑的实验

先挂新任务、后停旧任务，中间不留空卡：

1. 代码检查、`py_compile`、小规模验证在旧任务还占着卡时做完；
2. 新开 tmux 会话，挂上修复后的队列（它会一直等这 4 张卡空出来），队列里写上旧队列还没跑完的实验：
   ```bash
   tmux new -s fix_a_g1             # 进入后 source .venv/bin/activate
   bash scripts/run_queue.sh 4,5,6,7 29511 configs/phase0/e02_d12_attnres.yaml configs/phase0/e08_d48_wide_attnres.yaml
   # Ctrl-b d 离开
   ```
3. 确认新队列停在 `[WAIT]` 后，`tmux kill-session -t p0_a_g1` 停旧队列；新队列 5 秒内检测到空卡并启动。

---

## 7. 验证 / 评估 checkpoint

评测的是 **EMA 权重**（`--use-ema`），与 `summary.csv` 里的 eval 指标、`model_best.pth.tar` 的选择口径一致。在 tmux 里运行：

```bash
bash scripts/validate.sh \
  configs/phase0/e02_d12_attnres.yaml \
  output/phase0/20261008_153045_e02_d12_attnres_s42/model_best.pth.tar \
  0
```

---

## 8. 断点续训

续训从**最佳 epoch**（`model_best.pth.tar`，按 EMA 验证指标选出）的下一个 epoch 继续，恢复 model + optimizer + scaler + epoch + EMA。需要从最新 epoch 接着跑时，第二个参数直接给 `last.pth.tar` 的路径。

```bash
# 语法: bash scripts/resume_4gpu.sh <config.yaml> <run 目录或 checkpoint> [gpu_ids] [master_port]
bash scripts/resume_4gpu.sh \
  configs/phase0/e01_d12_baseline.yaml \
  output/phase0/20261008_153045_e01_d12_baseline_s42 \
  0,1,2,3 29500
```

脚本新建 tmux 会话 `resume_<run 目录名>`，等指定的卡空出后在原 run 目录里继续训练，日志追加到 `logs/phase0/<实验名>.log`。同机另一组卡上若有任务在跑，换一个端口（如 GPU 4-7 用 29501）。

---

## 9. 预期训练时间 (粗估)

每个实验 300 epoch + 10 cooldown epoch = 310 epoch，ImageNet 每 epoch 1.28M 张：

| 模型 | 4×A40 吞吐（估计） | 310 epoch 耗时 |
|------|-------------------|---------------|
| d12 (E01/E02) | ~1200 img/s | ~3.8 天 |
| d24 (E03/E04) | ~800 img/s | ~5.7 天 |
| d48 narrow (E05/E06) | ~600 img/s | ~7.7 天 |
| d48 wide (E07/E08) | ~350 img/s | ~13.1 天 |

吞吐为估计值，开跑后以日志里第一个 epoch 的实际速度为准；前提是读数据不是瓶颈（数据在本地盘，见 §3.1）。147 上数据在 NAS 时实测每个实验只有约 350 张/秒，d12 一个实验就要约 13 天。按 §6 的队列：

- **Server A**：E01/E02（~3.8 天）→ E07/E08（~13.1 天），约 **17 天**
- **Server B**：E03/E04（~5.7 天）→ E05/E06（~7.7 天），约 **13.5 天**
- **总计约 17 天**；d12 结果约第 4 天出，d24 约第 6 天出

---

## 10. 结果汇报模板

每个实验完成后，请记录：

```
实验 ID:
Top-1 Acc (best):
Top-5 Acc:
Final epoch:
Train loss (last):
GPU hours:
Checkpoint path:
Notes:
```

---

## 11. 常见问题

**Q: OOM on d48 wide (E07/E08)**  
A: 配置已设 batch=96/GPU。若仍 OOM，降至 `--batch-size 64` 并线性缩放 lr：`--lr 0.000667`。

**Q: GitHub clone 失败**  
A: 使用镜像：`git clone https://ghfast.top/https://github.com/xxx/yyy.git`

**Q: `All smoke tests passed` 但训练报 model not found**  
A: 确保通过 `train_imagenet.py` 启动（它会注册 AttnRes 模型），不要直接用 `repos/timm/train.py`。

**Q: 如何确认 AttnRes 生效？**  
A: 每个子层（attn、MLP）前各一个 AttnRes，再加输出层一个，共 2×depth+1 个（d12 为 25）；`python tests/test_forward.py` 还会检查初始化时（pseudo-query 为 0）AttnRes 模型与同权重 baseline 输出一致：
```python
python -c "
import attnres_vit
from timm.models import create_model
m = create_model('attnres_vit_d12_base_attnres')
print('AttnRes modules:', len(m.attn_res))
"
```

---

