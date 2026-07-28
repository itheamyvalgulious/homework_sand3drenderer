# SAND 复现

复现论文 *SAND: Spatially Adaptive Network Depth for Fast Sampling of Neural Implicit Surfaces*：
用八叉树自适应地决定每个空间区域所需的网络推理深度，加速神经隐式表面（SDF）的采样与渲染。
在标准距离场之外，本项目还扩展了 RGB 纹理通道的联合拟合。详细介绍见 [ppt.md](ppt.md)

## 环境

- Python 3.12 + CUDA GPU（CPU 也可运行，把 `--device` 设为 `cpu`，会慢很多）
- 依赖已安装在项目自带的 `.venv/` 中（torch、trimesh、numpy、matplotlib、flask 等）

## 运行

### 一键完整流程（推荐）

```bash
bash script/run_all.sh                          # 默认 assets/sample.glb -> output/
bash script/run_all.sh --quick                  # 快速冒烟（iters=1500, res=96）
bash script/run_all.sh --model assets/r2.glb --out report/r2   # 指定模型与输出目录
```

依次执行：主流水线 → RGB 校准重渲染 → 512³ 网格重渲染 → 展示图渲染。

### 只跑主流水线

```bash
.venv/bin/python script/run_pipeline.py \
    --model assets/sample.glb --out output \
    --solid-union-res 512 --octree-depth 9 --iters 10000 --batch 65536
```

步骤：加载模型 → 构建八叉树 → 训练 3 个网络（SAND 无纹理 / SAND+纹理 / baseline）
→ 按叶子节点分配推理深度 → 网格评估 → Marching Cubes 提取网格 →
软件光栅化与 Ray Marching 渲染 → 输出 `output/report.md`（含全部计时与统计）。

常用参数：`--quick`（冒烟）、`--res N`（网格分辨率）、`--iters N`、`--device cpu`、
`--solid-union-res N`（非水密多部件模型先做实心化预处理）。

### 训练后的可视化

交互式 Ray Marching 查看器（加载 `output/` 下的 checkpoint）：

```bash
.venv/bin/python script/viz_server.py   # 打开 http://127.0.0.1:8080/
```

### 测试

```bash
PYTHONPATH=. .venv/bin/python src/tests/test_data.py
PYTHONPATH=. .venv/bin/python src/tests/test_octree.py
PYTHONPATH=. .venv/bin/python src/tests/test_infer.py
PYTHONPATH=. .venv/bin/python src/tests/test_raymarch.py
PYTHONPATH=. .venv/bin/python src/tests/test_render.py
PYTHONPATH=. .venv/bin/python src/tests/test_tmlp_train.py
```

## 代码结构

- `src/` — 核心库：`config.py`（配置）、`data.py`（网格加载与采样）、`octree.py`（八叉树）、
  `tmlp.py`（T-MLP / baseline 网络）、`train.py`（训练）、`infer.py`（网格评估与提取）、
  `render.py`（软件光栅化）、`raymarch.py`（Ray Marching）
- `script/` — 流水线与辅助脚本（重渲染、报告生成、可视化等）
- `assets/` — 输入模型（`.glb`）
- `output/` — 流水线产物（checkpoint、`.ply` 网格、渲染图、`report.md`）
- `report/` — 各模型的正式实验结果（Suzanne / R2 / Wall-E 等）
