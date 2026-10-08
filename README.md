# RoboSynChallenge 初赛提交 · ACT Policy

基于 **ACT（Action Chunking with Transformers）** 策略，在官方 `cobotmagic_Sim_*` 仿真数据集上
微调 LeRobot v0.3.3 模型，覆盖全部 **10 个操作任务**。

| 项 | 值 |
|---|---|
| 策略 | ACT（`policy/act/`，官方 baseline 架构，未修改模型结构） |
| 框架 | LeRobot v0.3.3 + EmbodiChain |
| 档位 | `random`（全部成绩均为 random 档） |
| 评测 | seed 0，50 集/任务（0% 的三项为 20 集快筛定性） |

## 仓库结构

```
<repo root>/
├── policy/act/                      # ACT 策略代码
│   ├── deploy_policy.py             # 评测接口：get_model / eval / reset_model
│   ├── deploy_policy.yml            # 评测配置
│   ├── eval.sh                      # 评测入口
│   └── scripts/                     # 训练/评测脚本
├── checkpoints/
│   └── ACT_sim_<task_name>/         # 每任务一个 ckpt
│       ├── config.json
│       ├── model.safetensors        # 约 206 MB
│       └── train_config.json
├── configs/                         # 任务/档位配置（官方）
├── scripts/                         # 官方评测脚本（eval_policy.py 等）
└── assets/ docs/ launch/ misc/      # 官方仓库原有内容
```

`checkpoints/ACT_sim_<task_name>/` 目录内含**完整的 `pretrained_model` 内容**（不是再套一层
`pretrained_model/`），因此可直接作为 `checkpoint_path` 传给评测脚本。

## 评测命令

```bash
cd <repo root>
source policy/act/.venv/bin/activate          # 或你的等价环境
export HF_LEROBOT_HOME=<数据集所在目录>

bash policy/act/eval.sh <task_name> random checkpoints/ACT_sim_<task_name> 0 \
    --pytorch_device cuda \
    --headless True \
    --seed 0 \
    --max_episodes 50
```

`<task_name>` 取自：`click_bell` `handle_basket` `water_pouring` `table_rearrangement`
`items_handover` `drawer_open_place` `mixer_operating` `item_assembly`
`manipulate_pipette` `sample_loading`

评测结果（含视频）输出到 `<repo root>/eval_result/<task>/act/random/.../videos/`。

## ⚠️ 关键配置：`act_step: 50`

`policy/act/deploy_policy.yml` 中的 `act_step` 由官方默认的 **10 改为 50**，
用于对齐训练时的 `chunk_size = 50` 与 `n_action_steps = 50`。

**若将其还原为 10，所有任务的成功率都会显著下降。**

## 成绩总览

random 档 / seed 0 / 50 集：

| task | ckpt 训练步数 | epoch | 成功率 | 官方 ACT | 官方 DP |
|---|---|---|---|---|---|
| table_rearrangement | 050000 | 8.0 | **94%** | 63% | 14% |
| mixer_operating | 045000 | 4.6 | **88%** | 77% | 66% |
| click_bell | 050000 | 21.6 | **66%** | 37% | 51% |
| manipulate_pipette | 060000 | 6.7 | 44% | — | — |
| handle_basket | 050000 | 4.9 | 42% | — | — |
| water_pouring | 045000 | 6.9 | 36% | 72% | 36% |
| item_assembly | 060000 | 6.0 | 28% | — | — |
| drawer_open_place | 015000 | — | 0%（20 集） | **29%** | 0% |
| items_handover | 015000 | — | 0%（20 集） | — | — |
| sample_loading | 015000 | — | 0%（20 集） | — | — |

> 官方 baseline 取自本仓库 `evaluation_results/released_checkpoint_results.json`
> （2026-08-19 发布，random 档，100 集/任务，仅覆盖上表 5 个任务）。

**在官方发布了 baseline 的 5 个任务上，宏平均：本提交 56.8% vs 官方 ACT 55.6%。**

### 训练量说明

- **加训有效的任务**：mixer（48%→88%）、water_pouring（16%→36%）、
  item_assembly（18%→28%）、manipulate_pipette（30%→44%）
- **加训无效的任务**：handle_basket 在 15k→50k 加训后成功率与平均动作步数均几乎不变
  （42% → 42%），瓶颈在夹取位姿而非训练量
- click_bell 在 21.6 epoch 已明显过拟合（10k 64% → 50k 66%，仅 +2 点）

## 已知限制

- `docs/` 中出现的第 11 个任务名 **`open_pan`** 位于 `configs/other_tasks/`，
  官方未发布对应训练数据，本提交未包含。
- 三个 0% 任务（drawer_open_place / items_handover / sample_loading）当前均只训练了
  约 1.7 epoch，在 20 集快筛中平均动作步数打满上限，策略尚未学会有效行为。
  其中 **`drawer_open_place` 官方 ACT 在 100 集评测下可达 29%**，说明该任务可解，
  我们这一项的差距来自训练量不足，而非任务本身不可完成。
