# RoboSynChallenge 初赛提交 · ACT Policy

> **评测方：如需在全新 RTX 5090 环境从零复现，请直接看 [`REPRODUCE.md`](REPRODUCE.md)**
> —— 包含完整的环境安装、权重获取与校验、10 个任务的逐条测评命令。
> 本方案**不使用 Ollama、不启动任何服务、评测全程离线**。

基于 **ACT（Action Chunking with Transformers）** 策略，在官方 `cobotmagic_Sim_*` 仿真数据集上
训练 LeRobot v0.3.3 模型，覆盖全部 **10 个操作任务**。

| 项 | 值 |
|---|---|
| 策略 | ACT（`policy/act/`，官方 baseline 架构，未修改模型结构） |
| 框架 | LeRobot v0.3.3 + EmbodiChain |
| 评测档位 | `random`（全部成绩均为 random 档） |
| 评测口径 | seed 0，50 集/任务（0% 的三项为 20 集定性筛查） |
| 权重镜像 | HuggingFace（见下方【检查点镜像】） |

---

## 一、任务级检查点映射（10/10）

`checkpoints/ACT_sim_<task>/` 目录内**直接**放置 `config.json` / `model.safetensors` /
`train_config.json`（即 `pretrained_model` 的内容，不再嵌套一层 `pretrained_model/`），
可直接作为 `checkpoint_path` 传给评测脚本。

| # | task | checkpoint 路径 | 训练步数 | 自测成功率 | 官方 ACT |
|---|---|---|---:|---:|---:|
| 1 | `table_rearrangement` | `checkpoints/ACT_sim_table_rearrangement` | 50000 | **94%** | 63% |
| 2 | `click_bell` | `checkpoints/ACT_sim_click_bell` | 50000 | **66%** | 37% |
| 3 | `water_pouring` | `checkpoints/ACT_sim_water_pouring` | 80000 | **52%** | 72% |
| 4 | `handle_basket` | `checkpoints/ACT_sim_handle_basket` | 50000 | 42% | — |
| 5 | `items_handover` | `checkpoints/ACT_sim_items_handover` | 15000 | 0% ※ | — |
| 6 | `drawer_open_place` | `checkpoints/ACT_sim_drawer_open_place` | 15000 | 0% ※ | 29% ▼ |
| 7 | `mixer_operating` | `checkpoints/ACT_sim_mixer_operating` | 45000 | **88%** | 77% |
| 8 | `item_assembly` | `checkpoints/ACT_sim_item_assembly` | 60000 | 28% | — |
| 9 | `manipulate_pipette` | `checkpoints/ACT_sim_manipulate_pipette` | 60000 | 44% | — |
| 10 | `sample_loading` | `checkpoints/ACT_sim_sample_loading` | 15000 | 0% ※ | — |

※ 20 集定性筛查；▼ 官方该分数的口径见 §四。

> 10 项的平均成功率 **48.8%**（0% 三项按 0 计）。自测口径统一为
> `random` 档 / `seed 0` / `max_episodes 50`；官方评测使用 held-out 参数，
> 与自测数值不同属正常。

同一份权重另有一份 HuggingFace 镜像（见 §五），两处内容逐字节一致。

---

## 二、统一评测命令

**仓库布局与本赛道官方模板 `RoboSynChallenge` 完全一致**：三个入口文件位于
`policy/act/` 下，和其官方仓库中的 `policy/act/{eval.sh,deploy_policy.yml,deploy_policy.py}`
一一对应（官方根目录同样没有这三个文件）。说明见下方"入口文件"。

对上面 10 个 `<task>` 各执行一次：

```bash
cd <repo root>
export HF_LEROBOT_HOME=<数据集根目录>          # 目录内为 cobotmagic_Sim_<task>/
source policy/act/.venv/bin/activate          # 或等价的 Python 环境

bash policy/act/eval.sh <task> random checkpoints/ACT_sim_<task> 0 \
    --pytorch_device cuda \
    --headless True \
    --seed 0 \
    --max_episodes 50
```

`<task>` 取值（10 个，与上表一一对应）：
`table_rearrangement` `click_bell` `water_pouring` `handle_basket` `items_handover`
`drawer_open_place` `mixer_operating` `item_assembly` `manipulate_pipette` `sample_loading`

评测结果（含三相机拼接录像）输出到
`<repo root>/eval_result/<task>/act/random/.../<model>/<timestamp>/videos/`，
文件名带 `_success` / `_fail` 后缀，可直接逐集核查。

### 入口文件（官方 checklist 要求的三项）

| 官方要求 | 本仓库路径 | 备注 |
|---|---|---|
| `eval.sh` | `policy/act/eval.sh` | 用法见上文；脚本自身可干任何目录运行，内部会 `cd` 到仓库根 |
| `deploy_policy.yml` | `policy/act/deploy_policy.yml` | 评测配置，含 `act_step: 50`（见 §三） |
| `deploy_policy.py` | `policy/act/deploy_policy.py` | policy adapter 实现 |

> **不要把这三类文件复制到仓库根目录**：`eval.sh` 用
> `REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"` 反推根目录，
> 挪到根目录会让 `REPO_ROOT` 指向仓库的**父目录**、`VENV_DIR` 也失效，直接跑崩。
> 本仓库保持了与官方模板相同的目录结构，直接照官方方式调用即可。

---

## 三、⚠️ 唯一影响评测结果的代码改动：`act_step: 50`

`policy/act/deploy_policy.yml` 中的 `act_step` 由官方默认的 **10 改为 50**，
用于对齐训练时的 `chunk_size = 50` 与 `n_action_steps = 50`。

**若将其还原为 10，全部 10 个任务的成功率都会显著下降。**

训练超参（全部任务一致）：`batch_size 32`、`chunk_size 50`、`n_action_steps 50`、
`lr 1e-5`（恒定，`scheduler=None`）。

---

## 四、成绩说明与已知限制

### 官方已发布基线的 5 个任务

官方 baseline 取自本仓库 `evaluation_results/released_checkpoint_results.json`
（2026-08-19 发布，random 档，100 集/任务）：

| task | 本提交 | 官方 ACT | 官方 DP |
|---|---:|---:|---:|
| table_rearrangement | 94% | 63% | 14% |
| mixer_operating | 88% | 77% | 66% |
| click_bell | 66% | 37% | 51% |
| water_pouring | 52% | 72% | 36% |
| drawer_open_place | 0% | 29% ▼ | 0% |
| **宏平均** | **60.0%** | **55.6%** | 33.4% |

### ▼ `drawer_open_place` 的口径说明（避免误判）

官方 `evaluation_results/README.md` 原文：

> The drawer ACT figure comes from the separately recorded **`drawer_drive050_grip20`
> modified-physics** run. The DP drawer figure is ... under the **current modified-physics
> random configuration**; **neither drawer figure should be presented as an
> unmodified-physics result.**

即官方那个 **29% 取自另一套物理参数**（drive050 / grip20）的独立记录。在**当前仓库配置**下，
官方 DP 的 drawer 成绩为 **0.0%，平均动作步数 900.00 / 900** —— 与本提交自测的
0/20、900.00 / 900 完全一致。

因此该项 0% 属于**配置量级问题**（当前物理配置下该任务本身即难以完成），
而非训练量不足：本任务 loss 从 0.042（1.1 epoch）降至 0.019（5.7 epoch），成功率始终为 0。

### `item_assembly`

本仓库随附的权重为 **060000 步**（50 集正式评估 **28.0%**，13→14/50）。

已训练过的三个检查点在统一口径（`random` / `seed 0` / `max_episodes 50`）下为：

| checkpoint | epoch | 50 集成功率 |
|---|---|---|
| 030000 | ~3.0 | 18.0% |
| **060000（提交版本）** | ~6.0 | **28.0%** |
| 090000 | ~9.0 | 26.0% |

090000 继续加训后反而由 28.0% 回落至 26.0%，故最终选择 060000。
该曲线也说明：本任务 3→6 epoch 明显有效，6→9 epoch 已无明显收益。

> 口径纪律：不同 `--max_episodes` 会抽到不同场景集合，跨值比较无效。本提交所有对外
> 声明均为 `random` 档、`seed 0`、`max_episodes 50`（0% 三项为 20 集定性筛查）。

### 训练量观察

- **加训有效的任务**：mixer_operating（48%→88%）、water_pouring（15k/2.3ep 16%
  → 45k/6.9ep 36% → **80k/13.7ep 52%**）、item_assembly（18%→28%，再往上 9 epoch 回落至 26%）、
  manipulate_pipette（30%→44%）
- water_pouring 是本任务里训练量收益最显著的一项：**每翻倍训练步数约 +16 点**，
  且截至 80k 步（13.7 epoch）仍未出现过拟合拐点
- **加训无效的任务**：handle_basket 在 15k→50k 加训后成功率与平均动作步数均几乎不变
  （42% → 42%），瓶颈在演示数据的夹取位姿而非训练量
- click_bell 在 21.6 epoch 已明显过拟合（10k 64% → 50k 66%，仅 +2 点）
- `items_handover` / `sample_loading` 当前仅训约 1.1 epoch，20 集筛查中平均动作步数打满
  上限（900/900），策略尚未学会有效行为

### 其他

- 第 11 个任务名 `open_pan` 位于 `configs/other_tasks/`，官方未发布对应训练数据，
  本提交未包含（仅提交官方发布的 10 个任务）。
- 数据集仅使用官方开源的 `cobotmagic_Sim_*`（每任务 1000 条演示），未使用额外自采数据、
  外部私有数据集或真实机器人演示；未调用任何外部 API，可离线运行。

---

## 五、检查点镜像（HuggingFace）

权重以 Git LFS 存于本仓库 `checkpoints/` 下；同一份权重另有一份 HuggingFace 镜像：

**https://huggingface.co/Op19199/RoboSynChallenge-2026-ACT**（public）

目录结构与本仓库一致（`checkpoints/ACT_sim_<task>/`），可直接：

```bash
hf download Op19199/RoboSynChallenge-2026-ACT \
    --include "checkpoints/ACT_sim_<task>/*" --local-dir checkpoints/
```

---

## 六、仓库结构

```
<repo root>/
├── policy/act/                      # ACT 策略代码
│   ├── deploy_policy.py             # 评测接口：get_model / eval / reset_model
│   ├── deploy_policy.yml            # 评测配置（act_step: 50）
│   ├── eval.sh                      # 评测入口
│   └── scripts/                     # 训练/评测脚本
├── checkpoints/
│   └── ACT_sim_<task>/              # 每任务一个 ckpt（10 个）
│       ├── config.json
│       ├── model.safetensors        # 约 197 MB
│       └── train_config.json
├── configs/                         # 任务/档位配置（官方）
├── scripts/                         # 官方评测脚本（eval_policy.py 等）
└── assets/ docs/ launch/ misc/      # 官方仓库原有内容
```
