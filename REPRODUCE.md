# 全新 RTX 5090 环境 · 从零复现说明（ACT · 10 任务）

提交编号 `sub_3f87a246dc6744` / 评测编号 `eval_96b41a90cbd84e`

---

## 0. 先说清本提交的资源构成（回应"完整模型链接 / 服务启动"）

本提交使用的是 **ACT（Action Chunking with Transformers）**，属于**纯行为克隆的轻量策略网络**，
与需要本地大模型/VLM 的方案不同：

| 项目 | 本提交的情况 |
|---|---|
| 是否使用 Ollama / 本地 LLM / VLM | **否。** 全流程不调用任何语言模型 |
| 是否需要在评测前启动服务 / 守护进程 / 端口监听 | **否。** 评测是一次性命令行调用，没有 server 阶段 |
| 评测时是否联网 | **否。** 权重与数据全部在本地，推理完全离线 |
| 需要哪些外部资源 | 仅两项：① 官方 `cobotmagic_Sim_*` 仿真数据集；② 本提交的 10 个检查点 |
| "模型导入"怎么做 | 无需转换 / 无需 import 步骤。把检查点目录路径作为 `--checkpoint_path` / `eval.sh` 第 3 个参数传入即可 |

因此本文的"安装 → 导入 → 启动 → 测评"对应为：
**安装依赖 → 放置权重（=导入）→ 无服务需要启动 → `bash eval.sh` 逐任务测评**。

---

## 1. 硬件与驱动前提（RTX 5090 专属）

RTX 5090 是 Blackwell 架构（**计算能力 sm_120**），这是最容易踩的坑：

- **必须 PyTorch ≥ 2.7.0 + CUDA 12.8**。低于 2.7 的 PyTorch 编译产物不含 sm_120 内核，
  会直接报 `NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the
  current PyTorch installation`，或运行时 `no kernel image is available for execution`。
  > 本仓库 `policy/act/pyproject.toml` 声明的是 `torch>=2.2.1,<2.8.0`，区间**包含** 2.7.x，
  > 但依赖解析器可能挑到更低的版本。**请在安装后显式核对版本**（见 §2 第 4 步）。
- NVIDIA 驱动 **≥ 570**（CUDA 12.8 的最低要求）。
- 显存：ACT 推理约 2–4 GB，5090 的 32 GB 完全够用，单卡可跑。

核对驱动：

```bash
nvidia-smi            # 右上角应显示 CUDA Version: 12.8 或更高
```

---

## 2. 安装（推荐 Docker，与官方安装文档一致）

### 目录布局（**必须这样放**，否则评测脚本找不到 EmbodiChain）

`policy/act/eval.sh` 里 `EMBODICHAIN_ROOT` 默认取 **仓库的兄弟目录** `../EmbodiChain`：

```
RoboSynChallenge_ws/            <- 工作区根
├── EmbodiChain/                <- 官方仿真内核（v0.2.4.post1）
└── RoboSynChallenge/           <- 本提交仓库（clone 后目录名即为这个）
    └── checkpoints/ACT_sim_*/
```

如要放在别处，可用环境变量覆盖：`export EMBODICHAIN_ROOT=/path/to/EmbodiChain`。

### 第 1 步 · 拉起官方基础镜像（已含 CUDA 12.8）

```bash
docker pull dexforce/embodichain:ubuntu22.04-cuda12.8
mkdir -p RoboSynChallenge_ws && cd RoboSynChallenge_ws
./docker/docker_run.sh <container_name> <data_path>
```

> 该镜像预置 CUDA 12.8 + Vulkan + Python 3.11，是官方安装文档推荐路径，
> 也是 5090 上最省事的一条路（不需要手工配 CUDA）。

### 第 2 步 · 在容器内建环境

```bash
conda create -n robosyn python=3.11 -y
conda activate robosyn
```

### 第 3 步 · 安装 EmbodiChain v0.2.4.post1

```bash
cd /root/workspace
git clone https://github.com/DexForce/EmbodiChain.git
cd EmbodiChain
git checkout tags/v0.2.4.post1
pip install -e . --extra-index-url http://pyp.open3dv.site:2345/simple/ --trusted-host pyp.open3dv.site
pip install "numpy<2.0"
```

### 第 4 步 · 安装本仓库并锁定 5090 可用的 torch

```bash
cd /root/workspace
git clone https://github.com/Op191995/RoboSynChallenge-2026.git RoboSynChallenge
cd RoboSynChallenge
pip install -e .

# —— RTX 5090 (sm_120) 关键一步：确保 torch 是 2.7.x + CUDA 12.8 ——
pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.7.0" "torchvision==0.22.0"

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0),
      "| capability", torch.cuda.get_device_capability(0))
PY
```

**判据（三行都要对）**：`torch 2.7.0` / `cuda 12.8` / `available True`
且 capability 打印为 `(12, 0)`。若 capability 能正常打印而没有报
`sm_120 is not compatible`，说明 5090 内核可用。

### 第 5 步 · 验证安装

```bash
python -c "import embodichain, dexsim; print(embodichain.__version__, dexsim.__version__)"
python -c "import robosynchallenge; print(robosynchallenge.__version__)"
```

---

## 3. 模型导入（= 放权重，10 个检查点）

两条路，**任选其一**，内容逐字节一致（已用 sha256 双向校验：GitHub git-lfs 指针 oid
与 HF `lfs.sha256` 对全部 10 个文件完全一致）。

### 方式 A · 直接用仓库内权重（推荐给已 clone 的评测机）

仓库内已有 `checkpoints/ACT_sim_<task>/`，但走 Git LFS，**clone 后必须 pull**：

```bash
cd /root/workspace/RoboSynChallenge
git lfs install
git lfs pull --include "checkpoints/**"
ls -l checkpoints/ACT_sim_*/model.safetensors
```

### 方式 B · 从 HuggingFace 镜像拉取（public，免登录）

完整模型仓库链接：
**https://huggingface.co/Op19199/RoboSynChallenge-2026-ACT**

```bash
pip install -U "huggingface_hub[cli]"
hf download Op19199/RoboSynChallenge-2026-ACT \
    --local-dir /root/workspace/RoboSynChallenge/checkpoints_root

# 镜像内路径是 checkpoints/ACT_sim_<task>/，平铺到仓库的 checkpoints/ 下
cp -r /root/workspace/RoboSynChallenge/checkpoints_root/checkpoints/ACT_sim_* \
      /root/workspace/RoboSynChallenge/checkpoints/
```

### 权重自检（**不要只看文件大小，10 个文件都是 206,563,032 字节，看大小分不出版本**）

```bash
cd /root/workspace/RoboSynChallenge/checkpoints
sha256sum ACT_sim_*/model.safetensors
```

应与下表完全一致（2026-10-11 实测，GitHub git-lfs `oid sha256` 与 HF `lfs.sha256` 双侧一致）：

| task | model.safetensors sha256（前 12 位） | 字节数 |
|---|---|---|
| ACT_sim_table_rearrangement | `92090ab80fee` | 206,563,032 |
| ACT_sim_click_bell | `cd7093615d77` | 206,563,032 |
| ACT_sim_water_pouring | `06ae6297dc5d` | 206,563,032 |
| ACT_sim_handle_basket | `386ac1a24efd` | 206,563,032 |
| ACT_sim_items_handover | `9f2291232cf7` | 206,563,032 |
| ACT_sim_drawer_open_place | `ef5f95cc6439` | 206,563,032 |
| ACT_sim_mixer_operating | `68baecbe3fa1` | 206,563,032 |
| ACT_sim_item_assembly | `a8357acb0ac9` | 206,563,032 |
| ACT_sim_manipulate_pipette | `8ad4548488d1` | 206,563,032 |
| ACT_sim_sample_loading | `ab92d0bbc51d` | 206,563,032 |

**关键判据**：每个文件必须恰好 **206,563,032 字节**。若只有一百多字节，说明拿到的是
未下载的 Git LFS 指针文件（见 §9 排错）。

> 注意：**不要只用文件大小判断版本** —— 10 个检查点大小完全相同，换过权重也看不出来。
> 必须用 sha256。

---

## 4. 数据集

评测需要官方的 `cobotmagic_Sim_<task>` 数据集，按 LeRobot v2.1 格式存放。
把 10 个 `cobotmagic_Sim_<task>/` 目录放在同一个根下，并导出：

```bash
export HF_LEROBOT_HOME=/path/to/data_root
ls $HF_LEROBOT_HOME | grep cobotmagic_Sim      # 应列出 10 个目录
```

---

## 5. 服务启动：**无**

本方案**没有任何需要启动的服务**。`deploy_policy.py` 是进程内直接加载的
policy adapter（`get_model` / `eval` / `reset_model`），由官方评测脚本 `scripts/eval_policy.py`
在**同一个 Python 进程内**调用。不涉及 HTTP/gRPC/Ollama，无端口、无后台常驻进程。

---

## 6. 测评命令（10 个任务，逐条直接复制执行）

前置：

```bash
conda activate robosyn
cd /root/workspace/RoboSynChallenge
export HF_LEROBOT_HOME=/path/to/data_root
export EMBODICHAIN_ROOT=/root/workspace/EmbodiChain   # 若目录布局已按 §2 摆放则可省略
export PYTHON_BIN=$(which python)                      # 若无 policy/act/.venv 则必设
```

`eval.sh` 签名：`bash eval.sh <task> <setting> <checkpoint_path> [gpu_id] [extra_opts...]`

```bash
bash policy/act/eval.sh table_rearrangement random checkpoints/ACT_sim_table_rearrangement 0 --headless True --seed 0 --max_episodes 50
bash policy/act/eval.sh click_bell          random checkpoints/ACT_sim_click_bell          0 --headless True --seed 0 --max_episodes 50
bash policy/act/eval.sh water_pouring       random checkpoints/ACT_sim_water_pouring       0 --headless True --seed 0 --max_episodes 50
bash policy/act/eval.sh handle_basket       random checkpoints/ACT_sim_handle_basket       0 --headless True --seed 0 --max_episodes 50
bash policy/act/eval.sh items_handover      random checkpoints/ACT_sim_items_handover      0 --headless True --seed 0 --max_episodes 50
bash policy/act/eval.sh drawer_open_place   random checkpoints/ACT_sim_drawer_open_place   0 --headless True --seed 0 --max_episodes 50
bash policy/act/eval.sh mixer_operating     random checkpoints/ACT_sim_mixer_operating     0 --headless True --seed 0 --max_episodes 50
bash policy/act/eval.sh item_assembly       random checkpoints/ACT_sim_item_assembly       0 --headless True --seed 0 --max_episodes 50
bash policy/act/eval.sh manipulate_pipette  random checkpoints/ACT_sim_manipulate_pipette  0 --headless True --seed 0 --max_episodes 50
bash policy/act/eval.sh sample_loading      random checkpoints/ACT_sim_sample_loading      0 --headless True --seed 0 --max_episodes 50
```

或一条循环：

```bash
for t in table_rearrangement click_bell water_pouring handle_basket items_handover \
         drawer_open_place mixer_operating item_assembly manipulate_pipette sample_loading; do
  bash policy/act/eval.sh $t random checkpoints/ACT_sim_$t 0 \
      --headless True --seed 0 --max_episodes 50
done
```

**耗时**：单卡 4090 实测约 **50 分钟 / 任务 / 50 集**（瓶颈在仿真渲染，不在网络推理）；
5090 预计同量级或更快。10 个任务串行共约 **8–10 小时**。

**输出位置**：

```
<repo root>/eval_result/<task>/act/random/<cfg>/<model>/<timestamp>/
├── ...（成功率等数值结果）
└── videos/episode_NNN_seed_0_{success|fail}.mp4     # 三相机 1920x480 拼接录像
```

录像文件名自带成败标签，可逐集人工复核。

---

## 7. 预期结果（本方自测，`random` 档 / `seed 0` / 50 集）

| # | task | 检查点步数 | 自测成功率 |
|---|---|---:|---:|
| 1 | table_rearrangement | 50000 | 94% |
| 2 | mixer_operating | 45000 | 88% |
| 3 | click_bell | 50000 | 66% |
| 4 | water_pouring | 80000 | 52% |
| 5 | manipulate_pipette | 60000 | 44% |
| 6 | handle_basket | 50000 | 42% |
| 7 | item_assembly | 60000 | 28% |
| 8 | drawer_open_place | 15000 | 0% |
| 9 | items_handover | 15000 | 0% |
| 10 | sample_loading | 15000 | 0% |

> 官方评测使用 held-out 参数（不同的随机化范围 / seed），与自测数值不同属正常。
> 三项 0% 的原因分析见仓库 `README.md` 第四节（属配置量级问题与训练量不足，非部署问题）。

---

## 8. 配置项：唯一影响结果的一项改动

`policy/act/deploy_policy.yml` 中 `act_step` 由官方默认 **10 改为 50**，
用于对齐训练时的 `chunk_size = 50` / `n_action_steps = 50`。
**还原为 10 会让全部 10 个任务成功率显著下降**，请勿改回。

训练超参（10 个任务一致）：`batch_size 32`、`chunk_size 50`、`n_action_steps 50`、
`lr 1e-5`（恒定，scheduler=None）。

---

## 9. 排错速查

| 现象 | 原因 / 处理 |
|---|---|
| `sm_120 is not compatible with the current PyTorch installation` | torch < 2.7。重做 §2 第 4 步，装 `torch==2.7.0+cu128` |
| `no kernel image is available for execution on the device` | 同上，torch 编译产物不含 Blackwell 内核 |
| `model.safetensors` 只有一百多字节 | 是未下载的 Git LFS 指针文件。执行 `git lfs pull --include "checkpoints/**"`，或改走 §3 方式 B |
| 找不到 `embodichain` 模块 | 目录布局不对。`eval.sh` 取 `EMBODICHAIN_ROOT` 默认为**仓库的兄弟目录**；用 `export EMBODICHAIN_ROOT=...` 覆盖 |
| `Error: cannot find Python command` | `eval.sh` 先找 `policy/act/.venv/bin/python`，没有就退回 `python`。用 `export PYTHON_BIN=/abs/path/to/python` 指定 |
| `size mismatch for model.vae_encoder_pos_enc ... [1,52,512] vs [1,18,512]` | 只在**训练**时会出现（chunk_size 没传 50）。**评测不会遇到**，评测只加载已有检查点 |
| 想核对权重版本 | 用 sha256，不要用文件大小（10 个都同为 206,563,032 字节） |

---

## 10. 一句话给评测方

> 本提交为 ACT（LeRobot v0.3.3），**不使用 Ollama、不启动任何服务、评测全程离线**。
> 按 §2 装环境 → §3 放权重 → §6 跑 10 条命令即可复现全部 10 个任务，
> 无需任何额外配置或调试。
