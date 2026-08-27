# BitCons 低位平面改进实验方案

## 1. 改动目的

首批 CIFAR-10 单 seed 实验说明：旧 BitCons 主要提高 Clean，标准 AutoAttack 平均下降，Contrast 还会进一步退化。因此下一轮不能只把 `[3,4,5]` 替换为 `[0,1,2]` 后直接重跑全部正式实验，而应先验证退化来自位平面强度、参考分支、辅助权重还是 BN 统计污染。

旧配置清除 bit 3、4、5，单个 8-bit 像素最大改变为：

```text
8 + 16 + 32 = 56，即 56/255
```

这远大于训练威胁半径 `epsilon=8/255`。新配置清除 bit 0、1、2，最大改变为：

```text
1 + 2 + 4 = 7，即 7/255
```

因此 `[0,1,2]` 更符合“在对抗威胁邻域内建立低位不变性”的方法定义。连续 adversarial tensor 会先四舍五入到最近的 8-bit 值，整个变换仍不超过 `8/255`。

## 2. 新训练语义

所有宿主方法统一使用以下流程：

```text
clean image -> host attack -> final adversarial image
                              |-> host loss
                              |-> clear bits [0,1,2]
                                   |-> masked CE
                                   |-> align to detached final adversarial logits
                                   |-> optional feature contrast
```

具体约束：

1. 不再随机混合 clean/adv，也不再使用 PGD 第一步 logits。
2. PGD-AT、TRADES、MART、RPAT 都使用最终 adversarial image 作为 BitCons 输入和 reference；Cons-AT 使用对应增强视图的最终 adversarial 样本。
3. masked/unreliable 辅助前向冻结 BatchNorm running statistics，BN affine 参数仍可学习。
4. unreliable view 仅作为 detached contrast negative，不参与梯度。
5. `label_smoothing=0.5` 保持不变。
6. Base 和 BitCons 全部按普通 PGD-10 选择 best；masked PGD 仅作机制诊断。

当前 YAML 是待筛选的候选默认值，不代表已经验证为最优：

```yaml
bitcons_planes: [0, 1, 2]
bitcons_alpha: 0.25
bitcons_warmup: 60
bitcons_contrast_lam: 0.001
```

## 3. 第一阶段：PGD-AT 筛选

固定 CIFAR-10、ResNet18、seed 4243、110 epoch、`epsilon=8/255`、PGD-10 训练。只改变表中列出的 BitCons 参数。

| 组 | planes | alpha | Mask-CE | Align | Contrast | 用途 |
|---|---:|---:|---:|---:|---:|---|
| PGD-AT Base | - | 0 | 0 | 0 | 0 | 新语义基线 |
| Align P012 A010 | 0,1,2 | 0.10 | 0 | 1 | 0 | 低权重对齐 |
| Align P012 A025 | 0,1,2 | 0.25 | 0 | 1 | 0 | 主候选 |
| Align P012 A050 | 0,1,2 | 0.50 | 0 | 1 | 0 | 权重上界 |
| Core P012 A010 | 0,1,2 | 0.10 | 1 | 1 | 0 | 检验 Mask-CE 增益 |
| Core P012 A025 | 0,1,2 | 0.25 | 1 | 1 | 0 | Core 主候选 |
| Align P01 A025 | 0,1 | 0.25 | 0 | 1 | 0 | 位平面强度对照 |
| Full P012 A025 | 0,1,2 | 0.25 | 1 | 1 | 1 | BN 修复后的 Contrast 诊断 |

执行脚本为 `run_lowbit_screening.sh`。它使用 Conda 环境 `bit` 和 GPU 0、1，每张卡一次只运行一个任务。默认不启动训练，人工执行时使用：

```bash
chmod +x run_lowbit_screening.sh
DRY_RUN=1 RUN_ID=lowbit_v1 ./run_lowbit_screening.sh
RUN_ID=lowbit_v1 nohup ./run_lowbit_screening.sh > lowbit_v1_launcher.log 2>&1 &
```

默认只完成训练，完整评估关闭。训练结束后如需评估所有 best checkpoint：

```bash
RUN_ID=lowbit_v1 RUN_BEST_EVAL=1 RUN_AA=1 ./run_lowbit_screening.sh
```

## 4. 筛选判据

主判据必须是标准输入上的鲁棒性，而不是 masked 输入准确率：

1. 首先比较 best 标准 PGD-10，并检查最后 10 epoch 是否稳定。
2. 候选组再比较 Clean、PGD-20/50、C&W 和 AutoAttack。
3. 若 AA 不低于 Base，且 Clean 或 AA 至少一项有可重复的正增益，才进入正式实验。
4. masked PGD 高而标准 PGD/AA 低，视为模型依赖预处理，不算鲁棒性提升。
5. Contrast 只有在相对相同 Core 配置提高标准 AA 时才保留；否则正式方案删除 Contrast。

当前仍是单 seed 筛选。它可以淘汰明显无效配置，但不能作为论文最终显著性证据。正式定稿至少应对最终 Base/Core（以及保留时的 Full）补多 seed 均值和标准差。

## 5. 第二阶段：正式对比与消融

第一阶段只选择一个固定 BitCons 配置。选择完成后再运行：

1. 主对比：PGD-AT、TRADES、MART、RPAT 的 Base 与固定 BitCons Core。
2. 若 Contrast 通过筛选，再增加四种方法的 Full；否则不把 Full 作为主方法。
3. 消融只在 PGD-AT 上完成 `Mask-CE / Align / Contrast` 的必要组合，并补 `[0,1]` 对 `[0,1,2]`。
4. 最终报告 Clean、PGD-10/20/50、C&W、AutoAttack、best epoch、final gap、训练时间和显存。

旧 `[3,4,5] + alpha=1 + warmup=100` 结果应保留为失败设计证据，但不能与新语义结果混为同一组消融。新实验必须使用新的 `RUN_ID` 和独立日志目录。

## 6. 仍需注意的方法学问题

当前训练代码每个 epoch 使用 CIFAR-10 test loader 计算 PGD-10 并选择 best，这适合工程筛选，但正式论文存在测试集参与模型选择的问题。正式定稿前应从训练集固定划分 validation，用 validation PGD 选择 checkpoint，冻结配置后只在 test 上做一次完整评估。
