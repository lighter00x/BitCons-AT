# BitCons 低位平面阶段探索报告

> 报告状态：阶段中期版  
> 数据截止：2026-08-28 10:10（Asia/Shanghai）  
> 当前完成度：8 组计划中 5 组已完成完整 best 评估，1 组正在评估，1 组正在训练，1 组等待运行  
> 本报告只把已经生成 `eval_results_best.txt` 的实验写入定量主表，未完成实验不作结果推断。

## 1. 本阶段目的

上一阶段在 CIFAR-10、ResNet18、seed 4243、110 epoch 上完成了四种宿主方法的 Base/Core/Full 对比和 PGD-AT 八组组件消融。结论是：旧 BitCons 可以提高部分 Clean Accuracy，但没有提高标准输入上的鲁棒性。

上一阶段跨四种宿主方法的平均结果如下：

| Variant | Clean | PGD-10 | PGD-20 | PGD-50 | C&W | AutoAttack |
|---|---:|---:|---:|---:|---:|---:|
| Base | 81.12 | 54.63 | 53.86 | 53.66 | 50.48 | 48.73 |
| Core | 83.70 | 51.66 | 50.67 | 50.44 | 48.09 | 46.26 |
| Full | 82.06 | 44.91 | 43.75 | 43.34 | 41.80 | 39.59 |

Core 相对 Base 平均表现为 Clean `+2.58`、AA `-2.47`；Contrast 在 PGD-AT 完整消融中的平均 AA 效应为 `-2.27`。因此本阶段不再扩大到多宿主方法，而是先在 PGD-AT 上回答三个问题：

1. 旧 `[3,4,5]` 是否因掩码幅度远大于 `epsilon=8/255` 而损害鲁棒性？
2. 降低辅助权重后，Align-only 能否超过可复现的 Base？
3. BN 统计隔离和统一 reference 后，Mask-CE、Contrast 是否能恢复作用？

## 2. 相对上一阶段的修改

### 2.1 位平面与量化

旧设置清除 `[3,4,5]`，单像素最大改变量为：

```text
8 + 16 + 32 = 56，即 56/255
```

本阶段主设置清除 `[0,1,2]`，掩码自身最大改变量为：

```text
1 + 2 + 4 = 7，即 7/255
```

另设 `[0,1]` 对照，其掩码自身最大改变量为 `3/255`。浮点输入先四舍五入到最近的 8-bit 值，再进行位运算，替代旧实现的直接向下取整，以减少额外量化偏差。

### 2.2 BitCons 输入和对齐目标

上一阶段不同方法语义不一致：PGD-AT 会随机混合 clean/adv，PGD-AT/RPAT 使用 PGD 第一步 logits，TRADES/MART 使用 clean logits。

本阶段统一为：

```text
clean -> PGD final adversarial image -> host loss
                                  |-> clear selected low bits
                                       |-> Mask-CE
                                       |-> align to final adversarial logits.detach()
                                       |-> optional Contrast
```

PGD-AT 不再随机混合 clean/adv，不再使用 PGD 第一步 logits。所有候选都对齐到同一次最终 adversarial forward 的 detached logits。

### 2.3 BN、权重与 checkpoint

本阶段进行了以下统一：

1. masked/unreliable 辅助前向不再更新 BN running mean/variance。
2. `label_smoothing=0.5` 保持不变。
3. `bitcons_alpha` 从旧默认 `1.0` 下调并筛选 `0.10 / 0.25 / 0.50`。
4. warmup 从 100 epoch 改为 60 epoch。
5. Contrast lambda 统一为 `0.001`。
6. Base 与 BitCons 全部按普通标准 PGD-10 选择 best；masked PGD 只作机制诊断。
7. 新增 `loss_components.csv`，逐 epoch 记录 host、Mask-CE、Align、Contrast 及实际加权辅助损失。

代码修改后通过 24 项单元测试，包括低位变化上界、BN running statistics 隔离、四种宿主方法、Cons-AT、八种 PGD-AT 消融组合、AWP/RWP 和本地数据集复用。

## 3. 实验设置

| 项目 | 设置 |
|---|---|
| 数据集 | CIFAR-10 |
| 模型 | ResNet18 |
| 宿主方法 | PGD-AT |
| seed | 4243，单 seed 探索 |
| epoch | 110 |
| optimizer | SGD，momentum 0.9，weight decay `5e-4` |
| 学习率 | 0.1；epoch 100、105 各乘 0.1 |
| 训练攻击 | PGD-10，`epsilon=8/255`，`alpha=2/255` |
| best 选择 | 每 epoch 标准 PGD-10 |
| 完整评估 | Clean、PGD-10/20/50、C&W、AutoAttack |
| checkpoint | 只完整评估 best，final 只保存 |
| 环境 | Conda `bit`，双 A100 40GB |

完整探索矩阵：

| 实验 | planes | alpha | Mask-CE | Align | Contrast |
|---|---|---:|---:|---:|---:|
| PGD-AT Base | - | 0 | 0 | 0 | 0 |
| Align P012 A010 | 0,1,2 | 0.10 | 0 | 1 | 0 |
| Align P012 A025 | 0,1,2 | 0.25 | 0 | 1 | 0 |
| Align P012 A050 | 0,1,2 | 0.50 | 0 | 1 | 0 |
| Core P012 A010 | 0,1,2 | 0.10 | 1 | 1 | 0 |
| Core P012 A025 | 0,1,2 | 0.25 | 1 | 1 | 0 |
| Align P01 A025 | 0,1 | 0.25 | 0 | 1 | 0 |
| Full P012 A025 | 0,1,2 | 0.25 | 1 | 1 | 1 |

## 4. 当前完整评估结果

### 4.1 绝对指标

| 实验 | Clean | PGD-10 | PGD-20 | PGD-50 | C&W | AA | best epoch（1-based） |
|---|---:|---:|---:|---:|---:|---:|---:|
| PGD-AT Base | **82.96** | **53.50** | **52.71** | **52.47** | **51.04** | **48.84** | 109 |
| Align P012 A010 | 81.48 | 52.04 | 51.46 | 51.42 | 49.18 | 47.18 | 107 |
| Align P012 A025 | 69.28 | 44.79 | 44.32 | 44.27 | 42.34 | 40.61 | 28 |
| Align P012 A050 | 70.10 | 43.83 | 43.47 | 43.42 | 41.04 | 39.14 | 26 |
| Core P012 A010 | 64.25 | 43.76 | 43.42 | 43.27 | 35.77 | 34.45 | 24 |

### 4.2 相对本阶段 Base 的变化

| 实验 | Delta Clean | Delta PGD-10 | Delta PGD-20 | Delta PGD-50 | Delta C&W | Delta AA |
|---|---:|---:|---:|---:|---:|---:|
| Align P012 A010 | -1.48 | -1.46 | -1.25 | -1.05 | -1.86 | -1.66 |
| Align P012 A025 | -13.68 | -8.71 | -8.39 | -8.20 | -8.70 | -8.23 |
| Align P012 A050 | -12.86 | -9.67 | -9.24 | -9.05 | -10.00 | -9.70 |
| Core P012 A010 | -18.71 | -9.74 | -9.29 | -9.20 | -15.27 | -14.39 |

到当前为止，没有一个 BitCons 配置超过 Base。Align `alpha=0.10` 是唯一没有发生严重坍塌的候选，但其 Clean、PGD、C&W、AA 全部低于 Base，因此尚不能认为新设计有效。

## 5. 与上一阶段的对应比较

本阶段重新训练的 Base AA 为 `48.84`，上一阶段 Base AA 为 `48.82`，只差 `0.02`。PGD-50 分别为 `52.47` 和 `52.52`，只差 `0.05`。这说明基线在相同 seed 下高度可复现，本阶段下降不能用 Base 漂移解释。

| PGD-AT 实验 | 上一阶段 AA | 本阶段对应 AA | 说明 |
|---|---:|---:|---|
| Base | 48.82 | 48.84 | 基线复现稳定 |
| Align-only 最好候选 | 47.65 | 47.18 | 从旧语义改为低位/最终 adv 后仍未超过 Base |
| Core | 46.60 | 34.45（A010） | 新 Core 发生更严重优化坍塌 |

该对比不是严格的单变量实验：两阶段同时改变了 planes、alpha、warmup、masked input、teacher、BN 和 checkpoint 语义。因此它只能说明“整套新候选没有解决问题”，不能单独归因于低三位。

## 6. 训练动态分析

### 6.1 Align 存在清晰的权重阈值

Base 在学习率衰减后达到最佳标准 PGD-10；Align A010 也能训练到 epoch 106，并保持正常形态。相比之下：

| 实验 | epoch 30 PGD-10 | epoch 40 PGD-10 | epoch 60 PGD-10 | final PGD-10 | best epoch（1-based） |
|---|---:|---:|---:|---:|---:|
| Base | 42.70 | 44.18 | 46.26 | 51.86 | 109 |
| Align A010 | 45.36 | 44.10 | 44.83 | 51.49 | 107 |
| Align A025 | 44.04 | 27.20 | 13.87 | 40.30 | 28 |
| Align A050 | 40.12 | 10.00 | 14.40 | 32.81 | 26 |

A025/A050 都在 warmup 尚未结束时坍塌，且 best 提前到 epoch 25--27。这不是通常的 robust overfitting，而是辅助梯度随权重增长破坏主优化。

### 6.2 loss 数值小不代表梯度无害

Align A025 在 epoch 59：

```text
host loss              = 1.6456
raw align loss         = 0.0698
alpha                  = 0.2458
weighted align loss    = 0.0172
```

加权 Align 只有 host loss 数值的约 1%，但此时标准 PGD-10 已降到 `13.87%`。A050 也有相同现象。因此退化不是辅助 loss 标量“压过”主 loss，而是梯度方向、归一化路径或 teacher/student 定义发生冲突。后续必须记录梯度范数和 cosine similarity，不能只根据 loss 数值调 alpha。

### 6.3 Mask-CE 明显过强

Core A010 在 epoch 59：

```text
host loss              = 1.5245
raw Mask-CE            = 2.1783
raw Align              = 0.1272
alpha                  = 0.0983
weighted BitCons total = 0.2267
```

辅助项达到 host loss 的约 15%。Core A025 同期加权辅助项为 `0.5705`，约为 host loss 的 27%。`label_smoothing=0.5` 保持不变时，Mask-CE 的目标本身具有很强的高熵倾向；它配合当前权重已不再是轻量正则，因此 Core 比 Align-only 更早、更严重地坍塌。

## 7. 根因定位

### 7.1 BN 统计隔离改变了 BN 的计算模式

本阶段为了不污染 running statistics，在辅助前向中把 BN 临时切换到 eval。结果是：

```text
teacher: final adversarial + train BN（当前 batch statistics）
student: masked adversarial + eval BN（running statistics）
```

Align 同时在消除位平面差异和 train/eval BN 差异，且辅助梯度会更新共享卷积和 BN affine 参数。这解释了为什么 Align loss 数值很小，仍可能产生持续的错误优化方向。

正确的隔离方式应保持 BN training mode 和 batch-statistics 计算，在辅助前向前保存 running mean、running variance、num_batches_tracked，前向后恢复它们。这样只隔离状态写入，不改变函数本身。

### 7.2 “7/255 与 8/255 匹配”在当前组合流程中不成立

`[0,1,2]` 掩码自身最多改变约 `7.5/255`，但当前是在最终 PGD adversarial image 上再次掩码：

```text
clean -> 最多 8/255 PGD -> 最多继续向下约 7.5/255
```

因此 masked adversarial view 相对 clean 最多可达到约 `15.5/255`。在构造的负向 PGD 边界情形中，约 91% 的随机像素会在清除低三位后超出 `8/255`。本阶段修正了单次掩码幅度，却没有保证组合后的样本仍在论文声明的威胁球内。

下一版应二选一：

1. 对 clean image 清除/重采样低位，使 BitCons view 自身位于 `8/255` 内，再对齐 final adversarial logits。
2. 对 adversarial image 修改低位后，显式投影回以 clean 为中心的 `8/255` 球。

### 7.3 清零低位产生确定性的变暗偏置

随机输入统计显示：

| planes | 平均有符号变化 | 平均绝对变化 | 最大绝对变化 | 向下变化比例 |
|---|---:|---:|---:|---:|
| `[0,1]` | `1.50/255` | `1.56/255` | `3.50/255` | 87.6% |
| `[0,1,2]` | `3.50/255` | `3.53/255` | `7.50/255` | 94.0% |

这里“有符号变化”为 `original - masked`，正值表示 masked 更暗。当前分支更像固定方向的低幅亮度变换，而不是覆盖低位扰动集合的不变性训练。

更合理的机制候选是随机重采样低 k 位，使变化覆盖正负方向，并在必要时投影回威胁球。确定性清零可保留为消融，而不应直接等同于“位平面鲁棒性”。

### 7.4 detached adversarial teacher 可能强化错误预测

训练早期 final adversarial logits 的置信度和正确率都有限。当前 KL 无条件把 masked student 对齐到 detached adversarial teacher，包括 teacher 预测错误的样本。高 alpha 会增强自蒸馏错误。

后续可以比较：

1. 只对 teacher 预测正确或置信度超过阈值的样本计算 Align。
2. 使用 true-label 条件下的 consistency，而不是无条件复制全部 teacher 分布。
3. 比较 KL 与对称 JS，但必须在 BN 和威胁球语义修复后进行。

## 8. 当前阶段结论

### 8.1 已经能够确认的结论

1. 新 Base 与旧 Base 高度一致，当前负结果不是基线复现失败。
2. `[0,1,2] + Align alpha=0.10` 没有严重坍塌，但 AA 仍比 Base 低 `1.66`。
3. Align 的 `alpha=0.25/0.50` 明显过大，导致训练中期优化坍塌。
4. 当前 Mask-CE 权重明显过强，Core A010 已比 Base 低 `14.39` AA 点。
5. 低三位选择在单次变换幅度上比 `[3,4,5]` 合理，但当前“PGD 后再掩码”的组合并未保持 `8/255` 威胁约束。
6. 当前 BN 隔离实现避免了 running-stat 写入，却错误地引入 train/eval BN 函数差异。

### 8.2 目前不能确认的结论

1. 不能据此断言低位平面本身无效，因为当前实现同时存在 BN 模式错位和威胁球越界。
2. 不能宣称 Align A010 与 Base 的 `1.66` 点差距具有统计显著性，因为目前只有单 seed；但所有强攻击方向一致，至少没有正增益证据。
3. 不能用 masked PGD 指标证明防御有效；本阶段主结论只依据标准输入的完整评估。

## 9. 下一阶段修改和最小实验

在继续扩大正式矩阵前，应先修改机制，再运行最小因果实验。

### 9.1 必须先改的代码

1. BN 辅助前向保持 train 计算模式，只保存并恢复 running statistics。
2. BitCons view 改为 `mask(clean)`，或在 `mask(adv)` 后投影回 clean 的 `8/255` 球。
3. 新增逐 batch 梯度诊断：host/Align/CE 的梯度范数及 cosine similarity。
4. Contrast 继续关闭。

### 9.2 建议的四组最小矩阵

| 实验 | BitCons view | planes | Align alpha | Mask-CE |
|---|---|---|---:|---:|
| Base | - | - | 0 | 0 |
| Clean-LSB Align A010 | mask(clean) | 0,1,2 | 0.01 | 0 |
| Clean-LSB Align A025 | mask(clean) | 0,1,2 | 0.025 | 0 |
| Clean-LSB Align A050 | mask(clean) | 0,1,2 | 0.05 | 0 |

只有 Align 在标准 AA 上不低于 Base，才恢复 Mask-CE。`label_smoothing=0.5` 保持不动，但 Mask-CE 独立权重应从 `0.01 / 0.025` 开始，而不是与 Align 共用同一个大 alpha。

### 9.3 进入正式实验的门槛

单 seed 候选至少需要：

1. AA 相对 Base 提高至少 `0.5` 点。
2. Clean 下降不超过 `1` 点。
3. PGD-10/20/50 随攻击步数增强不异常上升。
4. 不依赖 test-time bit mask。
5. host 与辅助梯度不长期负相关或数量级失衡。

未达到上述条件时，不应扩展到 TRADES、MART、RPAT、多数据集和多 seed。

## 10. 未完成实验与更新说明

报告生成时以下结果尚未完整产生：

| 实验 | 当前状态 |
|---|---|
| Core P012 A025 | 训练完成，best AutoAttack 进行中 |
| Align P01 A025 | 训练进行中 |
| Full P012 A025 | 等待运行 |

这些结果完成后，应补入第 4 节并更新阶段结论。预计它们不会改变“当前实现尚未有效”的最低结论，但最终报告不得在数据产生前填写预测值。

## 11. 数据位置

- 本阶段任务矩阵：`logs/lowbit_screen_lowbit_explore_20260828_015531/manifest.tsv`
- 本阶段启动日志：`logs/lowbit_screen_lowbit_explore_20260828_015531/launcher.log`
- 各实验训练曲线：对应输出目录下的 `metrics.csv`
- 各实验损失分量：对应输出目录下的 `loss_components.csv`
- 各实验完整评估：对应输出目录下的 `eval_results_best.txt`
- 上一阶段完整报告：`CIFAR10单Seed完整实验数据报告.md`
