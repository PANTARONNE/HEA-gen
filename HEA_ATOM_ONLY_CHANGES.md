# HEA 固定几何 · 仅生成元素种类 — 修改说明

将 MatterGen 从"联合扩散坐标+晶格+原子类型"改造为**固定 FCC(111) 几何模板下只扩散/生成原子种类**的任务。元素池 9 种，从零训练，训练数据归一化到统一网格。

## 任务目标

- **只生成元素种类**：位置（pos）与晶格（cell）固定，仅对原子类型（atomic_numbers）做扩散去噪。
- **元素池 9 种**（去掉仅出现 1 次的 In）：Fe(26) Co(27) Ni(28) Cu(29) Zn(30) Ga(31) Mo(42) Sn(50) W(74)。
- **几何模板**：`ase.build.fcc111`，size=(4,4,4)=64 位点，`DEFAULT_A=3.6 Å`，真空 15 Å（对应 `build_hea_surface.py`）。
- **训练数据归一化**：全部结构映射到同一 `DEFAULT_A` 标准网格，pos/cell 完全一致，仅 atomic_numbers 不同。
- **从零训练**，保留原始 64 位点超胞（不做 primitive/Niggli 对称约化）。
- **生成后处理**：由生成成分经 Vegard's law 计算真实晶格常数，重建 slab 并写 CIF。

## 数据事实（已验证）

- 原始 2000 个 CIF，全部 64 原子。
- 含 In 结构 1 个（`Cu15In4Mo18W15Zn12.cif`）→ 排除。
- 剩余 1999 个全部成功映射到标准网格，**0 失败**。
- 拆分：1799 train / 200 val。

---

## 一、核心代码修改（4 处）

### 1. 修复空 `sdes` 崩溃
**文件**：`mattergen/diffusion/diffusion_module.py`（`_get_device`）

移除 pos/cell 的连续 SDE 后 `corruption.sdes` 为空，原代码 `next(iter(空))` 会 `StopIteration`。

```python
# 改前
return next(batch[k].device for k in self.corruption.sdes.keys())
# 改后：改用 corrupted_fields（含离散腐蚀 atomic_numbers）
return next(batch[k].device for k in self.corruption.corrupted_fields)
```

### 2. 新增 HEA 元素常量
**文件**：`mattergen/common/utils/globals.py`

```python
# One-based atomic numbers: Fe, Co, Ni, Cu, Zn, Ga, Mo, Sn, W
HEA_ATOMIC_NUMBERS = [26, 27, 28, 29, 30, 31, 42, 50, 74]
```

### 3. 新增 HEA 元素硬掩码
**文件**：`mattergen/denoiser.py`

新增 `mask_to_hea_elements(logits, x, batch_idx, predictions_are_zero_based)`，复用现有 `mask_logits`/`atomic_numbers_to_mask`，把非池内元素 logits 置 `-inf`。仅在采样时（`training=False`）经 `element_mask_func` 生效，训练不受影响。保证生成 100% 只出这 9 种元素。

### 4. 新增 HEA 固定几何模板加载器
**文件**：`mattergen/common/data/condition_factory.py`

新增 `get_hea_template_loader(num_structures, batch_size)`：每个条件样本携带**同一份** DEFAULT_A 标准网格（真实 pos/cell），atomic_numbers 为占位（D3PM prior 会覆盖为全 MASK token）。采样时只有原子类型被去噪，pos/cell 保持不变。

---

## 二、新增文件

### 共享模块 `mattergen/common/data/hea_template.py`
模板构建、位点映射、Vegard 定律的公共逻辑，供数据脚本、模板加载器、生成脚本复用。

- `DEFAULT_A=3.6`、`SLAB_SIZE=(4,4,4)`、`VACUUM_TOTAL=15.0`
- `FCC_LATTICE_CONSTANTS`：9 元素的 FCC 等效晶格常数（取自 `build_hea_surface.py`）
- `build_template_slab(a)`：构建 fcc111 slab
- `canonical_template()`：返回标准网格的 pos(64,3)、cell(3,3)、位点键索引
- `map_atoms_to_canonical(...)`：把 CIF 原子按几何不变量映射到标准位点顺序
- `vegard_lattice_constant(symbols)`：成分加权 `a = Σ aᵢ / N`

**位点映射键**：`(层号按z排序, round(x%1,3), round(y%1,3))`。fcc111 网格的面内 (x,y) 分数坐标与晶格常数无关，故此键跨结构稳定。

> **浮点边界修复**：wrap 到 [0,1) 时必须 **先 round 再取模** —— `round(round(v,3) % 1.0, 3)`。若先取模再 round，`0.9995` 会被 round 进位成 `1.0` 且无法拉回，导致键匹配失败（初版曾因此有 173 个结构映射失败）。

### 数据准备脚本 `mattergen/scripts/prepare_hea_dataset.py`
读 CIF → 排除 In → 映射到标准网格 → 写 `.npy` 缓存到 `datasets/cache/hea/{train,val}`。
- 用 ASE 读取，**跳过** primitive/Niggli 约化（保留 64 位点超胞）。
- 所有结构共享同一 pos/cell，仅 atomic_numbers 逐结构不同。

```bash
python -m mattergen.scripts.prepare_hea_dataset \
    --raw-dir datasets/randomStructures --cache-folder datasets/cache \
    --dataset-name hea --val-fraction 0.1 --seed 42
```

### 生成脚本 `mattergen/scripts/generate_hea.py`
1. 加载 atom-only 训练模型，注入 HEA 元素掩码（`mask_to_hea_elements`）。
2. 用固定几何模板加载器 + `sampling_conf/atom_only.yaml` 采样原子类型（pos/cell 不变）。
3. **Vegard 后处理**：由生成成分算 `a = Σ xᵢaᵢ` → 用该 a 重建 fcc111 slab → 填入生成元素 → 写 CIF。

```bash
python -m mattergen.scripts.generate_hea \
    --model_path outputs/singlerun/<date>/<time> \
    --num_structures 100 --batch_size 64 --output_dir outputs/hea
```

### 冒烟测试 `scripts_test_hea.py`
6 项检查：模板/Vegard、数据缓存加载、训练配置组装、一步训练（loss 有限）、模板 prior 采样、元素掩码限定。

```bash
python scripts_test_hea.py
```

---

## 三、新增配置文件（5 个）

| 文件 | 作用 |
|---|---|
| `mattergen/conf/data_module/hea.yaml` | 指向 `datasets/cache/hea` 缓存；无 transforms/properties（数据已规范化） |
| `mattergen/conf/lightning_module/diffusion_module/corruption/atom_only.yaml` | 只含 `atomic_numbers` 的 D3PM MaskDiffusion（dim=101，1000 步），**无 sdes** |
| `mattergen/conf/lightning_module/diffusion_module/atom_only.yaml` | loss 仅 `include_atomic_numbers`；`pre_corruption_fn: null`（无条件） |
| `mattergen/conf/hea.yaml` | 顶层训练配置，`defaults` 选 hea 数据 + atom_only 扩散/腐蚀 |
| `sampling_conf/atom_only.yaml` | 采样只保留 `atomic_numbers` predictor；`condition_loader` 指向 HEA 模板加载器 |

**删除的无关字段**（对本任务无作用）：`average_density`（仅晶格 SDE 用）、`test_dataset`、`properties`、per-sample `transforms`、guidance 相关字段。

---

## 四、执行流程

```bash
# 1. 数据准备（已验证：1999 结构，0 映射失败）
python -m mattergen.scripts.prepare_hea_dataset

# 2. 冒烟测试（不需 checkpoint）
python scripts_test_hea.py

# 3. 从零训练（只扩散 atomic_numbers）
mattergen-train --config-name=hea

# 4. 固定模板生成 + Vegard 定晶格
python -m mattergen.scripts.generate_hea --model_path=<ckpt目录> --num_structures=100
```

## 五、维度/机制要点

- 训练时 batch 内所有原子拼成大图，逐原子量维度 N=Σ64，逐晶体量 B。GemNet 骨干不变。
- 只有 `atomic_numbers` 被 D3PM 腐蚀；pos/cell 作为干净上下文进入 GemNet（提供几何），其输出的 pos/cell score 在 loss 中被忽略、在采样中不被更新。
- 采样 prior 只对 `corruption.corruptions`（=atomic_numbers）采噪声（全 MASK 起点），pos/cell 直接沿用条件模板 → 几何天然冻结，无需 inpainting mask。
- 训练与生成锚定同一 DEFAULT_A 网格，几何完全一致，唯一变量是元素种类。

## 六、未改动

模型结构、GemNet、D3PM 核心算法均未改动。所有改造通过配置 + 4 处小改 + 新增脚本完成。
