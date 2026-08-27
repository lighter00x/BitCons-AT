# BitCons CIFAR-10 单 Seed 完整实验数据报告

> 报告日期：2026-08-28  
> 实验批次：`cifar10_r18_s4243_110e`  
> 原始汇总：[logs/suite_cifar10_r18_s4243_110e/results.tsv](logs/suite_cifar10_r18_s4243_110e/results.tsv)  
> 完成状态：[logs/suite_cifar10_r18_s4243_110e/summary.tsv](logs/suite_cifar10_r18_s4243_110e/summary.tsv)

## 1. 执行摘要

本批实验共完成 17 次训练和 17 次 best checkpoint 完整评估，所有任务均正常结束，没有训练失败、CUDA OOM 或 NaN。实验覆盖四种宿主对抗训练方法的 Base/Core/Full 对比，以及 PGD-AT 上 Mask-CE、Align、Contrast 三个模块的全部八种开关组合。

本批结果给出的核心结论是：

1. 当前 BitCons 配置没有提升标准对抗鲁棒性。四种宿主方法的 Core 和 Full 在 AutoAttack、PGD-20/50、C&W 上均低于各自 Base。
2. Core 在四种方法上平均提高 Clean Accuracy 2.58 个百分点，同时使 AutoAttack Accuracy 平均下降 2.47 个百分点。因此当前方法体现的是 clean-robust trade-off，而不是鲁棒性净增益。
3. Mask-CE 对 Clean Accuracy 有稳定正向作用，但损害标准鲁棒性。
4. Align 是当前最值得继续研究的模块。PGD-AT Align-only 的 AA 仅比 Base 低 1.17，其训练曲线最高标准 PGD-10 与 Base 基本持平，但仍未证明有正向鲁棒增益。
5. Contrast 在当前实现中明确有害。PGD-AT 八组平衡消融显示，开启 Contrast 平均使 AA 降低 2.27；TRADES Full 则发生严重鲁棒性坍塌。
6. 当前 checkpoint 选择协议不公平：Base 按标准 PGD-10 选择 best，BitCons 按非自适应 masked PGD-10 选择 best。这会扩大部分差距，但不足以推翻“当前方法没有提升标准鲁棒性”的结论。
7. 这批结果适合用于诊断与筛选，不能直接作为论文最终主表。原因包括单 seed、测试集参与逐 epoch 选模、best 语义不统一，以及 PGD/C&W 仅单次随机启动。

## 2. 实验设置

| 项目 | 设置 |
|---|---|
| 数据集 | CIFAR-10 |
| 模型 | ResNet18 |
| Seed | 4243 |
| Epoch | 110 |
| 优化器 | SGD，momentum 0.9，weight decay `5e-4` |
| 初始学习率 | 0.1 |
| 学习率计划 | epoch 100、105 乘 0.1 |
| 训练攻击 | `L_inf` PGD-10，epsilon `8/255`，alpha `2/255` |
| BitCons planes | `[3,4,5]` |
| Align | KL |
| BitCons alpha | 1.0，前 100 epoch 线性 warmup |
| Mask-CE | 权重 1.0，固定 `label_smoothing=0.5` |
| Contrast temperature | 0.5 |
| Contrast lambda | PGD-AT 为 0.001；TRADES/MART/RPAT 为 1.0 |
| 硬件 | 2 x NVIDIA A100-SXM4-40GB |
| 完整评估 | Clean、PGD-10/20/50、C&W-50、AutoAttack standard |
| checkpoint | 仅对 `best_model.pt` 做完整评估；`final_model.pt` 已保留 |

三种对比变体定义如下：

- Base：Mask-CE=0，Align=0，Contrast=0。
- Core：Mask-CE=1，Align=1，Contrast=0。
- Full：Mask-CE=1，Align=1，Contrast=1。

所有正式表格均使用标准无测试时掩码的模型推理。AutoAttack 为 `L_inf, epsilon=8/255, version=standard`，运行完整 CIFAR-10 测试集。

## 3. 主对比实验

### 3.1 完整绝对指标

单位均为准确率百分比。每种宿主方法内部的最高标准鲁棒指标加粗。

| Method | Variant | Clean | PGD-10 | PGD-20 | PGD-50 | C&W | AutoAttack |
|---|---|---:|---:|---:|---:|---:|---:|
| PGD-AT | Base | 82.96 | **53.56** | **52.69** | **52.52** | **51.05** | **48.82** |
| PGD-AT | Core | 83.71 | 51.21 | 50.26 | 50.02 | 48.54 | 46.60 |
| PGD-AT | Full | **83.76** | 49.78 | 48.84 | 48.59 | 47.67 | 45.62 |
| TRADES | Base | 81.23 | **53.84** | **53.03** | **52.93** | **50.27** | **49.27** |
| TRADES | Core | **84.98** | 49.89 | 48.49 | 48.25 | 47.15 | 45.58 |
| TRADES | Full | 80.22 | 27.79 | 25.63 | 24.73 | 24.98 | 21.44 |
| MART | Base | 77.29 | **57.63** | **57.18** | **56.91** | **50.11** | **48.56** |
| MART | Core | **81.04** | 55.50 | 54.87 | 54.71 | 48.69 | 47.01 |
| MART | Full | 80.66 | 52.94 | 52.33 | 52.21 | 47.37 | 46.05 |
| RPAT | Base | 82.98 | **53.49** | **52.54** | **52.26** | **50.49** | **48.29** |
| RPAT | Core | **85.06** | 50.03 | 49.05 | 48.79 | 48.00 | 45.86 |
| RPAT | Full | 83.61 | 49.13 | 48.22 | 47.85 | 47.16 | 45.23 |

### 3.2 相对各自 Base 的变化

正数表示提高，负数表示降低。

| Method | Variant | Delta Clean | Delta PGD-10 | Delta PGD-20 | Delta PGD-50 | Delta C&W | Delta AA |
|---|---|---:|---:|---:|---:|---:|---:|
| PGD-AT | Core | +0.75 | -2.35 | -2.43 | -2.50 | -2.51 | -2.22 |
| PGD-AT | Full | +0.80 | -3.78 | -3.85 | -3.93 | -3.38 | -3.20 |
| TRADES | Core | +3.75 | -3.95 | -4.54 | -4.68 | -3.12 | -3.69 |
| TRADES | Full | -1.01 | -26.05 | -27.40 | -28.20 | -25.29 | -27.83 |
| MART | Core | +3.75 | -2.13 | -2.31 | -2.20 | -1.42 | -1.55 |
| MART | Full | +3.37 | -4.69 | -4.85 | -4.70 | -2.74 | -2.51 |
| RPAT | Core | +2.08 | -3.46 | -3.49 | -3.47 | -2.49 | -2.43 |
| RPAT | Full | +0.63 | -4.36 | -4.32 | -4.41 | -3.33 | -3.06 |

### 3.3 跨方法平均结果

| Variant | Clean | PGD-10 | PGD-20 | PGD-50 | C&W | AutoAttack |
|---|---:|---:|---:|---:|---:|---:|
| Base | 81.12 | 54.63 | 53.86 | 53.66 | 50.48 | 48.73 |
| Core | 83.70 | 51.66 | 50.67 | 50.44 | 48.09 | 46.26 |
| Full | 82.06 | 44.91 | 43.75 | 43.34 | 41.80 | 39.59 |

Core 相对 Base 的跨方法平均变化为：Clean `+2.58`、PGD-10 `-2.97`、PGD-50 `-3.22`、C&W `-2.39`、AA `-2.47`。这种方向在四种宿主方法上完全一致，说明不是某一个 baseline 的偶然异常。

Full 的平均结果受 TRADES 严重坍塌影响。即使排除 TRADES，PGD-AT、MART、RPAT 的 Full AA 仍分别降低 3.20、2.51、3.06，因此 Full 的负面结论并不只来自 TRADES。

### 3.4 各宿主方法解读

#### PGD-AT

Core 和 Full 的 Clean 分别提高 0.75、0.80，但 AA 分别降低 2.22、3.20。Contrast 加入 Core 后，所有攻击指标继续下降。当前 PGD-AT Full 不优于 Base。

#### TRADES

Core 的 Clean 提高 3.75，但 AA 降低 3.69。Full 的 AA 从 49.27 降至 21.44，属于严重鲁棒性坍塌，而不是正常波动。其 Contrast lambda 为 1.0，最终总训练损失为 6.79，远高于 Base 的 1.06；对比目标已显著干扰 TRADES 主目标。

#### MART

MART 是 Core 损失最小的宿主：Clean `+3.75`、AA `-1.55`。这仍不是鲁棒增益，但说明 MART 对辅助分支相对更耐受。Full 加入 Contrast 后 AA 进一步下降至 46.05。

#### RPAT

Core 的 Clean 提高 2.08，AA 降低 2.43；Full 的 AA 降低 3.06。方向与 PGD-AT 一致，未观察到 BitCons 与 RPAT 的正向互补。

## 4. PGD-AT 完整组件消融

### 4.1 八种组合绝对指标

| Mask-CE | Align | Contrast | Variant | Clean | PGD-10 | PGD-20 | PGD-50 | C&W | AA |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | Base | 82.96 | **53.56** | **52.69** | **52.52** | **51.05** | **48.82** |
| 1 | 0 | 0 | Mask-only | 84.46 | 51.82 | 50.98 | 50.75 | 48.97 | 46.90 |
| 0 | 1 | 0 | Align-only | 83.30 | 51.97 | 51.34 | 51.01 | 49.67 | 47.65 |
| 1 | 1 | 0 | Core | 83.71 | 51.21 | 50.26 | 50.02 | 48.54 | 46.60 |
| 0 | 0 | 1 | Contrast-only | 82.99 | 49.15 | 48.49 | 48.24 | 47.58 | 45.48 |
| 1 | 0 | 1 | Mask+Contrast | **84.62** | 48.49 | 47.58 | 47.30 | 46.37 | 44.37 |
| 0 | 1 | 1 | Align+Contrast | 82.73 | 49.15 | 48.31 | 48.10 | 47.46 | 45.40 |
| 1 | 1 | 1 | Full | 83.76 | 49.78 | 48.84 | 48.59 | 47.67 | 45.62 |

### 4.2 相对 Base 的变化

| Variant | Delta Clean | Delta PGD-10 | Delta PGD-20 | Delta PGD-50 | Delta C&W | Delta AA |
|---|---:|---:|---:|---:|---:|---:|
| Mask-only | +1.50 | -1.74 | -1.71 | -1.77 | -2.08 | -1.92 |
| Align-only | +0.34 | -1.59 | -1.35 | -1.51 | -1.38 | -1.17 |
| Core | +0.75 | -2.35 | -2.43 | -2.50 | -2.51 | -2.22 |
| Contrast-only | +0.03 | -4.41 | -4.20 | -4.28 | -3.47 | -3.34 |
| Mask+Contrast | +1.66 | -5.07 | -5.11 | -5.22 | -4.68 | -4.45 |
| Align+Contrast | -0.23 | -4.41 | -4.38 | -4.42 | -3.59 | -3.42 |
| Full | +0.80 | -3.78 | -3.85 | -3.93 | -3.38 | -3.20 |

### 4.3 平衡因子平均效应

因为八种组合构成完整的 `2 x 2 x 2` 设计，可以将某模块开启的四组均值与关闭的四组均值相减。该结果描述本批设置下的平均关联效应，不代替多 seed 显著性检验。

| 模块 | Clean | PGD-10 | PGD-20 | PGD-50 | C&W | AA |
|---|---:|---:|---:|---:|---:|---:|
| Mask-CE | +1.14 | -0.63 | -0.79 | -0.80 | -1.05 | -0.96 |
| Align | -0.38 | -0.23 | -0.25 | -0.27 | -0.16 | -0.07 |
| Contrast | -0.08 | -3.00 | -3.01 | -3.02 | -2.29 | -2.27 |

模块结论：

- Mask-CE 对 Clean 最有用，平均提高 1.14，但对 AA 平均造成 0.96 的损失。它可以作为准确率正则研究，但目前不是鲁棒性增强模块。
- Align 的平均边际伤害最小，AA 平均效应仅为 -0.07；但 Align-only 相对 Base 的 AA 仍低 1.17。它是下一轮唯一值得优先保留和重新调参的模块。
- Contrast 是当前最明确的负面模块。它对 Clean 几乎没有平均收益，却使 PGD-10/20/50 各下降约 3 点、AA 下降 2.27。
- 三个模块之间没有出现足以超过 Base 的正向协同。八组中所有非 Base 组合的 AA 都低于 Base。

## 5. Masked 指标的机制分析

下表为各 BitCons 模型最后一个 epoch 的指标。Masked PGD 的流程是先对原模型生成攻击，再对对抗样本做位平面掩码后分类，并不是攻击 `f(mask(x))` 的自适应鲁棒评估。

| Variant | Final standard PGD-10 | Final masked PGD-10 | Masked - standard |
|---|---:|---:|---:|
| PGD-AT Mask-only | 50.31 | 53.43 | +3.12 |
| PGD-AT Align-only | 52.14 | 54.40 | +2.26 |
| PGD-AT Core | 51.22 | 53.98 | +2.76 |
| PGD-AT Contrast-only | 48.94 | 55.32 | +6.38 |
| PGD-AT Mask+Contrast | 46.53 | 53.63 | +7.10 |
| PGD-AT Align+Contrast | 48.85 | 55.31 | +6.46 |
| PGD-AT Full | 47.64 | 55.07 | +7.43 |
| TRADES Core | 49.91 | 53.10 | +3.19 |
| TRADES Full | 27.70 | 60.10 | +32.40 |
| MART Core | 55.19 | 55.18 | -0.01 |
| MART Full | 51.41 | 56.49 | +5.08 |
| RPAT Core | 49.66 | 53.48 | +3.82 |
| RPAT Full | 47.24 | 53.90 | +6.66 |

随着 Contrast 加入，standard 与 masked 指标的分裂普遍扩大。TRADES Full 的 32.40 点差距尤其表明当前训练目标和 checkpoint 选择强烈偏向 masked 路径，而没有形成标准推理路径上的鲁棒模型。该现象可以作为机制诊断，但不能作为防御有效性的证据。

如果论文要把测试时掩码作为正式防御的一部分，攻击必须直接针对包含掩码的完整模型，并使用 BPDA；存在随机混合时还需要 EOT。否则 masked PGD/AA 只能标记为 non-adaptive transfer-style evaluation。

## 6. Checkpoint 选择与结果公平性

当前训练代码使用不同语义选择 best：

- Base：按标准 PGD-10 保存 best。
- 任意 BitCons 组：按 masked PGD-10 保存 best。

这会让主表中的 best checkpoint 不可严格配对。例如：

| Variant | 保存的 best epoch | 保存依据 | 训练曲线最高标准 PGD-10 |
|---|---:|---:|---:|
| PGD-AT Base | 109 | standard PGD 53.52 | 53.52 |
| PGD-AT Core | 106 | masked PGD 54.55 | 52.46 |
| PGD-AT Align-only | 106 | masked PGD 54.81 | 53.55 |
| TRADES Base | 108 | standard PGD 53.77 | 53.77 |
| TRADES Core | 110 | masked PGD 53.10 | 50.04 |
| TRADES Full | 110 | masked PGD 60.10 | 38.05 |
| MART Base | 107 | standard PGD 57.66 | 57.66 |
| MART Core | 106 | masked PGD 55.87 | 56.87 |
| RPAT Base | 107 | standard PGD 53.47 | 53.47 |
| RPAT Core | 106 | masked PGD 53.86 | 51.70 |

即使忽略保存下来的 best，直接比较训练曲线中的最高标准 PGD-10，四种 Core 仍未超过各自 Base。PGD-AT Align-only 的最高标准 PGD-10 为 53.55，与 Base 53.52 基本持平。因此 checkpoint 偏差影响差距幅度，但不会反转本批总体结论。

更严重的是，目前每个 epoch 都使用 CIFAR-10 test loader 计算指标并选择 checkpoint。这构成测试集参与模型选择。最终论文必须从训练集固定划分 validation，在 validation 上用统一的标准 PGD-20 选择 checkpoint，测试集只能在模型和超参数冻结后使用。

## 7. 训练成本

| Method | Base 训练时间 | Core 训练时间 | Full 训练时间 | Core 开销 | Full 开销 |
|---|---:|---:|---:|---:|---:|
| PGD-AT | 1.81 h | 2.24 h | 2.29 h | +24.1% | +27.1% |
| TRADES | 1.84 h | 2.27 h | 2.32 h | +23.2% | +26.0% |
| MART | 1.98 h | 2.41 h | 2.47 h | +22.0% | +24.6% |
| RPAT | 2.15 h | 2.60 h | 2.65 h | +20.7% | +23.3% |

当前 Core 增加约 21%--24% 训练时间，Full 增加约 23%--27%。在没有标准鲁棒收益的情况下，这一开销目前无法由效果合理化。

## 8. 问题归因

### 8.1 Contrast 权重不统一且部分设置过大

PGD-AT 的 `bitcons_contrast_lam=0.001`，TRADES/MART/RPAT 为 1.0，相差 1000 倍。TRADES Full 从 epoch 20 起标准 PGD 已明显落后 Core，最终训练损失 6.79，说明对比目标强度过高并主导优化。

### 8.2 共享 BatchNorm 可能被辅助视图污染

ResNet18 的主流、masked view 和 unreliable-bit view 共用同一组 BatchNorm。训练模式下每次辅助前向都会更新 running mean/variance。Contrast 开启后额外执行 unreliable-bit view 前向，该输入分布与自然/对抗图像差异很大。

PGD-AT 中 Contrast 权重只有 0.001，Full 最终总损失与 Core 几乎相同，但标准鲁棒性仍显著降低。这不能仅由损失梯度解释，BatchNorm 统计污染是需要优先验证的实现机制。

### 8.3 位平面掩码强度大于威胁半径

删除 `[3,4,5]` 可造成每个通道最高 `56/255` 的像素变化，远大于训练威胁半径 `8/255`。辅助任务迫使模型适应强结构化失真，可能促进类别语义和 Clean Accuracy，却不一定改善 epsilon 邻域内的局部鲁棒性。

### 8.4 对齐参考不够一致

PGD-AT/RPAT 当前用 PGD 第一步随机初始化附近的 logits 作为对齐参考，而不是最终 adversarial logits。TRADES/MART 使用 clean logits。四种方法中“Align”实际对应不同教师目标，削弱了跨方法结论的一致性。

### 8.5 目标权重只按 epoch 变化，未按实际损失尺度校准

`bitcons_alpha` 在 100 epoch 内升到 1.0，但 Mask-CE、KL 和 NT-Xent 的原始尺度不同。相同的系数不代表相同的梯度贡献。当前日志只记录总损失，无法从历史实验直接恢复每个模块的梯度占比。

### 8.6 单 seed 与评估协议限制

本批只有 seed 4243，不能判断 1--2 个百分点差异是否稳定。PGD/C&W 当前只有一次随机启动。虽然 AA、PGD-50 和 C&W 对所有主要结论方向一致，降低了“单一弱攻击误判”的可能，但仍不能替代多 seed 和多 restart。

## 9. 工作点有效性判断

按照“标准无掩码推理下提高 AutoAttack 鲁棒准确率”的原始目标，当前工作点尚未被验证有效，不能宣称 BitCons 提升了对抗鲁棒性。

已经得到证据支持的较窄结论是：

- 位平面 masked classification 能提高 Clean Accuracy。
- 模型可以学习对掩码视图进行分类，并在非自适应 masked evaluation 中得到较高数值。
- Align 的负面影响远小于 Contrast，可能存在通过重新定义 reference、减弱权重和隔离 BN 后转化为正向增益的空间。

没有得到支持的结论是：

- 当前 Core/Full 能提升标准 PGD、C&W 或 AutoAttack 鲁棒性。
- Contrast 带来正向贡献。
- masked PGD/AA 数值能够证明测试时防御有效。

## 10. 下一步完善方案

### P0：先修复协议和可观测性

1. 从 CIFAR-10 训练集固定划出 5,000 张 validation。
2. 所有 Base/BitCons 统一按 validation standard PGD-20 保存 `best_standard_model.pt`。
3. masked best 如需保留，另存 `best_masked_model.pt`，不得用于标准主表。
4. 每 epoch 记录基础损失、Mask-CE、Align、Contrast 的原始值、加权值及梯度范数。
5. PGD/C&W 增加固定 seed 的多 restart；修正评估报告中 C&W 实际 50 steps、文本写 100 steps 的记录错误。

### P1：隔离 BatchNorm 后重新验证模块

优先比较以下三种实现：

1. 辅助 masked/unreliable 前向时冻结 BN running statistics。
2. 主流和辅助流使用独立 BN。
3. 使用不依赖 batch statistics 的归一化作为诊断对照。

最小诊断矩阵只需 PGD-AT 的 Base、Contrast-only、Align-only、Full。若冻结或独立 BN 能显著恢复 Contrast-only 的标准 PGD/AA，说明当前失败主要来自统计污染；否则说明对比目标本身存在问题。

### P2：缩小到 Align-only 做顺序筛选

不要进行全笛卡尔积，建议依次筛选：

1. reference：clean logits、final adversarial logits。
2. planes：`[0,1]`、`[0,1,2]`、`[2,3]`，与当前 `[3,4,5]` 对照。
3. align weight/alpha：0.05、0.1、0.25、0.5。
4. align 类型：KL、JS；只有前两步有效后再比较 MSE/KL-zscore。
5. warmup：20、60、100；优先在较优权重上比较。

筛选阶段使用 validation standard PGD，不看测试 AA 调参。候选配置冻结后再运行一次完整测试集 AA。

### P3：谨慎恢复 Mask-CE 和 Contrast

- Mask-CE 保持 `label_smoothing=0.5` 不变，但模块整体权重从 0.05、0.1、0.25 开始，不再默认 1.0。
- Contrast 只有在 BN 问题解决后再恢复。
- 所有方法统一 contrast lambda；先测 `1e-4、1e-3、1e-2`，禁止直接使用 1.0。
- 对比目标先去掉 unreliable-bit negatives，验证 same-sample positive 是否本身有效，再逐步加入负样本。

### P4：设置继续扩展的成功门槛

单 seed PGD-AT 筛选配置至少满足：

- standard AutoAttack 相对 Base 提高至少 0.5 个百分点；
- Clean Accuracy 下降不超过 1 个百分点；
- PGD-10/20/50 随攻击增强不异常上升；
- 不依赖测试时掩码；
- 训练开销可解释。

只有达到上述门槛，才扩展到 TRADES/MART/RPAT 和 3 个 seed。否则应继续修改方法，而不是扩大数据集和模型规模。

## 11. 最终结论

本批实验成功完成了方法否证和模块定位：当前 BitCons Full 不能作为论文最终方法，Contrast 应暂停，Mask-CE 只能被视为 Clean Accuracy 正则，Align-only 是唯一值得优先整改和复验的核心候选。

下一轮工作的重点不是增加更多 baseline，而是先统一 checkpoint、隔离 BatchNorm、降低辅助目标强度，并在 PGD-AT 上证明标准 AutoAttack 的净增益。只有这一最低因果结论成立，才有必要开展多 seed、CIFAR-100、WRN 和 TinyImageNet 的正式论文实验。
