# 改动说明（相对最初的 attnres.zip，2026-10-09）

本包 = 最初的 `attnres.zip` + 下列改动。`repos/` 原样保留；去掉了原包里的 `__pycache__/`、`attnres_vit.egg-info/`
（安装时自动生成）。逐行差异见同目录的 `CHANGES.diff`。

| 部分 | 文件 | 改动 |
|---|---|---|
| 模型 | `attnres_vit/vit.py`、`modules.py`、`factory.py` | 按 Kimi 原版重写 Block AttnRes；删除 routing/additive；不堆叠的计算方式；块内和 fp32 累加 |
| 测试 | `tests/test_forward.py`、`tests/test_kimi_reference.py`（新） | 初始化时与 baseline 等价；float64 下与 Kimi Fig. 2 一致 |
| 运行 | `scripts/` | tmux 队列、等卡空、新排布、续训用 model_best、评测用 EMA、新的环境与数据工具 |
| 配置 | `configs/phase0/*.yaml` | 删除不生效的 `tensorboard` 键，其余不变 |
| 文档 | `RUN.md` | 同步以上所有改动 |

---

## 一、模型（`attnres_vit/`）

### 1. 按 Kimi 原版重写 Block AttnRes

**原实现的问题**：本意是实现 Kimi 的 Block AttnRes（注释引用了论文，聚合函数照抄论文 Fig. 2），但 `vit.py` 里的调用方式写错了，
把 "Block" 理解成了"每个块只做一次"：

- 每块结束后只做 1 次深度注意力（12 层共 4 次），而论文是**每个子层（attention、MLP）之前各做一次**；
- 加权的对象是**累积隐状态**，而论文是**每个块内各子层输出之和**（第 0 项为 embedding）；
- 加权结果直接**替换残差主干**，而论文里结果只作为该子层的输入，最后输出层再对所有块聚合一次；
- pseudo-query 是随机初始化，而论文要求**必须零初始化**（初始化时等价于标准 PreNorm 网络）。

后果：初始化时就和同权重 baseline 不等价（最终特征相对差 0.83 / 1.03，d12 / d24）；8 块的模型里第 4–7 块的输出在初始化时只剩 2–5% 的权重。
实际做出来的结构更接近 DenseFormer 式的"块尾深度加权平均"，结论无法归因到 AttnRes。

**新实现**（`vit.py` 的 `_forward_attnres`）：

- 每个子层前一个 `BlockAttnRes`，再加输出层一个，共 `2 × depth + 1` 个（d12 为 25 个）；
- 来源 = `[embedding, 各已完成块的子层输出之和, 当前块已累加的部分和]`（论文 Eq. 5–6）；
- 加权结果只作为下一个子层的输入，残差主干不再被替换；最终 norm 读所有块的加权和；
- pseudo-query 零初始化（`init_weights` 里在通用初始化之后重新置零）；
- `ViTBlock` 拆出 `attn_branch` / `mlp_branch`，baseline 的前向计算不变。

### 2. 删除 `routing`（replace / additive）选项

原 `additive` 分支 `partial_block + (aggregated - partial_block)` 恒等于 `aggregated`，是空操作。
按 Kimi 原版结构，聚合结果只作为子层输入，不存在"替换还是相加"的选择，所以这个选项整个删掉（`vit.py`、`modules.py`、`factory.py`）。
原来 8 组实验都用默认的 replace，不受影响。

### 3. 不堆叠的计算方式（与论文数学上完全相同）

按论文伪代码直接写（每次调用把所有来源 `stack` 成 `[S, B, T, D]` 再做 RMSNorm），在 147 上实测：

- E02 显存 31.4 GB / 卡（baseline 11.6 GB），纯计算 1.24 s / 步（baseline 0.29 s）；
- 48 层的 AttnRes 外推需要 100 GB 以上，A40（46 GB）必然 OOM。

新写法（`modules.py`）：把 RMSNorm 的增益并入 query，`w · (g ⊙ v / rms(v)) = ((w ⊙ g) · v) / rms(v)`；每个来源的 `1/rms` 只算一次
（`inv_rms`，反向时重算，不常驻 fp32 副本）；用逐元素加权求和代替 `einsum`。保存的激活只比 baseline 多约 5%（d12：153.5 vs 145.7 MB/样本）。

### 4. 块内和用 fp32 累加（修复 E02 崩溃）

**现象**：147 上第一次跑的 E02 从 epoch 5 起退化，EMA top1 从 7.64% 掉到 0.13%（随机水平），epoch 10 训练 loss 为 NaN；同期 E01 正常。

**原因**：fp16 溢出。块内和由 fp16 子层输出直接相加、以 fp16 存，训练中最大值长到 2.08 万（fp16 上限 6.55 万）；聚合的 fp16 `einsum`
在反向时溢出。GradScaler 的放大系数从 32768 一路减半到 0.03（E01 为 1,048,576），几乎每一步都被跳过。

**修复**：块内和按 embedding 的数据类型累加（AMP 下为 fp32，和 baseline 的残差主干一致）；logits 与加权求和都是 fp32 逐元素运算。
AMP 设置不变，attention / MLP 里的矩阵乘法仍是 fp16。

**验证**：在崩溃时（epoch 6）的权重上，旧写法在 loss scale 65536 和 1024 下梯度溢出，新写法都正常，两者 loss 相同（6.423）。

### 验证汇总

```bash
python tests/test_forward.py         # 10 个模型前向；初始化时 AttnRes 与同权重 baseline 相对差 ~6e-5
python tests/test_kimi_reference.py  # float64 + 随机 query：与 Kimi Fig. 2 逐行移植一致（相对差 1e-14）
```

---

## 二、运行脚本（`scripts/`）

| 文件 | 改动 |
|---|---|
| `run_queue.sh`（新） | 一组 4 卡按顺序跑多个实验；每个实验启动前等这 4 张卡空出来（显存 < 2000 MiB，5 秒查一次，可用 `GPU_FREE_MIB` / `GPU_POLL_SEC` 调）；某个实验失败就接着跑下一个；日志追加到 `logs/phase0/<实验名>.log` |
| `run_phase0_node.sh`（重写） | 每台机器两个 tmux 队列，A：E01→E07、E02→E08；B：E03→E05、E04→E06（成对实验仍同机同时跑，最长的 d48 宽排在最短的 d12 后面）；`GPUS_G0` / `GPUS_G1` 可按实际空闲卡分组；不再分批人工衔接 |
| `lib/run_utils.sh` | 新增 `wait_gpus_free`；新增 `tmux_launch`（显式传 PATH / LD_LIBRARY_PATH，避免已有 tmux 服务用错 Python 环境；会话重名时拒绝启动） |
| `resume_4gpu.sh`（重写） | 在 tmux 里续训；传 run 目录时用 `model_best.pth.tar`（从最佳 epoch 续训），需要从最新 epoch 续时直接传 `last.pth.tar`；可指定卡和端口 |
| `validate.sh` | 加 `--use-ema`，评测 EMA 权重，与 `summary.csv` 和 `model_best` 的选择口径一致 |
| `cluster_dispatch.sh`（重写） | 两台机器各起 tmux 队列；去掉会把两批同时启动的 `all` 模式；`REMOTE_ENV` 默认激活 `.venv` |
| `setup_env.sh`（重写） | venv + curl + uv 安装（原因见 RUN.md §2：147 上 conda 源连不上，镜像对 HTTP/1.1 限速） |
| `lib/fetch_torch_deps.py`（新） | 读 torch 包里锁定的 nvidia / triton 依赖版本，用 curl 下载 |
| `data/extract_imagenet_train.py`（新） | 从 NAS 上的 ILSVRC2012 训练集整包 tar 流式解压到本地盘，不产生中间文件 |
| `data/switch_to_local_data.sh`（新） | 核对本地副本与 NAS 的文件清单、抽查内容，一致才原子地切换数据软链接；正在跑的实验无需重启 |
| `run_phase0_parallel.sh` | 删除（单机 16 卡的旧脚本，你们没有这种机器） |

---

## 三、配置（`configs/phase0/*.yaml`）

删除 `tensorboard: true`（vendored timm 不支持 TensorBoard，这个键被静默忽略）。其余超参全部不变；
`data_dir` 仍是占位的 `/data/imagenet`，部署时按 RUN.md §3 修改（147 上已改为 `~/attnres/data/imagenet`）。

---

## 四、`RUN.md`

同步了环境配置（§2）、数据放本地盘（§3.1）、tmux 运行方式（§5、§6）、改代码后替换任务的流程（§6.6）、评测用 EMA（§7）、
续训用 model_best（§8）、排期（§9：按 310 epoch 计算，并注明读数据不能是瓶颈）和 FAQ。

---

## 五、未改动

训练配方（DeiT 式，AdamW lr 1e-3、batch、AMP fp16、EMA、数据增强）、实验矩阵 E01–E08、`idea.txt`、`train_imagenet.py`、
`validate_imagenet.py`、`repos/` 都保持原样。

---

## 六、147 上的当前状态（2026-10-09 傍晚）

- 部署在 `~/attnres`，环境为 `~/attnres/.venv`（torch 2.10.0+cu128）；数据目录 `~/attnres/data/imagenet` 下的软链接目前指向 NAS，
  本地副本在 `/mnt/d/bench/imagenet`，拷完后由 tmux 会话 `attnres_data_switch` 自动核对并切换（日志 `logs/setup/switch_data.log`）。
- 卡 0–3，tmux `p0_a_g0`：E01 运行中（epoch 12，EMA top1 23.4%），之后自动接 E07。
- 卡 4–7，tmux `p0_a_g1b`：新 E02（修复后的代码）17:39 从头开始，之后自动接 E08。
  第一次跑的 E02（已崩溃）目录 `output/phase0/20261009_040117_e02_d12_attnres_s42` 保留但作废。
- 142：尚未部署。
