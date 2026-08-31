# BPDA 低位平面探索实验分析报告

## 1. 报告结论

本轮四组实验均完成 110 epoch 训练，并对各自按 PGD-10 选择的 best checkpoint 完成 Clean、PGD-10/20/50、C&W 和 AutoAttack 自适应评估。运行编号为 `bpda_explore_20260828_123058`。

结论分为三层：

1. **实现层面通过了有效性初筛。** 三个 BPDA 低位平面配置在 PGD-10/20/50、C&W、AutoAttack 上都没有低于本轮修正后的 PGD-AT Base。攻击步数增加时准确率正常下降，C&W 和 AutoAttack 进一步降低准确率，未出现最直接的梯度遮蔽异常。
2. **效果层面只有弱正向信号。** 最佳 AutoAttack 是 P01 的 `48.70%`，相对 Base 的 `48.39%` 提高 `0.31` 个百分点；最佳 PGD-50 是 P012 的 `52.08%`，提高 `0.47` 个百分点。提升较小，且 P0、P01、P012 之间没有稳定的单调关系。
3. **论文层面尚不能据此宣称方法有效。** 当前只有单 seed，提升量处于常见训练波动范围；当前模块本质上仍接近“确定性低位清零/量化预处理 + BPDA”，独立创新性不足。它可以作为后续方法的基础组件，但暂时不能作为论文核心贡献定稿。

若必须从本轮选择下一步候选：以 AutoAttack 和 Clean 为主，优先选择 **P01**；以 PGD/C&W 为主，P012 略好。由于 P01 与 P012 的 AutoAttack 只差 `0.10`，现阶段不应宣布其中一个稳定优于另一个。

## 2. 本轮目的与上一阶段修正

上一阶段低位辅助流存在四个关键问题：在 PGD 样本上再次掩码可能越出 `8/255` 威胁球、辅助前向的 BN 语义不一致、Mask-CE 权重过强、detached adversarial teacher 可能复制错误预测。上一阶段所有完整候选均低于 Base，不能继续靠调整辅助权重扩大实验。

本轮改为把低位平面变换直接纳入模型：

```text
g(x) = f(mask_low_bits(x))
```

前向执行真实的 8-bit 四舍五入和低位清零，反向使用 identity STE/BPDA。训练 PGD 与测试 PGD、C&W、AutoAttack 都直接攻击包装后的完整模型，因而评估是自适应的，不再使用“先攻击原模型、再掩码”的非自适应口径。

本轮没有启用旧 BitCons 辅助流：四组配置均为 `bitcons=false`、`bitcons_contrast=false`，因此 Mask-CE、Align、Contrast 的损失记录均为 `N/A`。代码中的 Mask-CE 仍保留固定 `label_smoothing=0.5`，没有修改，但因辅助流关闭而不参与本轮训练。主 PGD-AT 损失使用普通交叉熵。

此外，PGD 生成训练对抗样本时使用模型 eval mode，攻击结束后恢复训练状态，避免 PGD 内部多次前向污染 BatchNorm running statistics。

## 3. 实验设置

| 项目 | 设置 |
|---|---|
| 数据集 | CIFAR-10，本地数据复用 |
| 模型 | ResNet18 |
| seed | 4243（单 seed 探索） |
| epoch | 110 |
| batch size | 128 |
| optimizer | SGD，momentum `0.9`，weight decay `5e-4` |
| 学习率 | `0.1`，epoch 100、105 各乘 `0.1` |
| 训练攻击 | PGD-10，`epsilon=8/255`，`alpha=2/255` |
| best 选择 | 每 epoch 测得的 PGD-10 最高值 |
| 完整评估 | Clean、PGD-10/20/50、C&W-100、AutoAttack standard |
| checkpoint | 只完整评估 best；final 仅保存 |
| 环境 | Conda `bit`，2 x A100 40GB |

实验矩阵：

| 实验 | 方法 | 清零位平面 | 对 8-bit 输入最大改变量 | 作用 |
|---|---|---|---:|---|
| Corrected PGD Base | PGD-AT | 无 | 0 | 同轮修正基线 |
| BPDA P0 | BitPlane-AT | bit 0 | `1/255` | 最弱低位处理 |
| BPDA P01 | BitPlane-AT | bit 0,1 | `3/255` | 中等低位处理 |
| BPDA P012 | BitPlane-AT | bit 0,1,2 | `7/255` | 最强候选 |

需要注意：对 CIFAR-10 clean 图像，上表最大改变量成立；对连续值 adversarial tensor，代码先四舍五入到 8-bit 再清零，因此相对连续输入的理论最大变化分别约为 `1.5/255、3.5/255、7.5/255`。这说明当前变换同时包含“量化”和“清零指定低位”两部分。

## 4. Best checkpoint 完整结果

### 4.1 绝对指标

所有数值单位均为准确率百分点，表中 best epoch 使用 1-based 编号。

| 实验 | Clean | PGD-10 | PGD-20 | PGD-50 | C&W | AutoAttack | best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Corrected PGD Base | 83.97 | 53.00 | 51.86 | 51.61 | 50.79 | 48.39 | 106 |
| BPDA P0 | 83.62 | 53.30 | 52.23 | 51.95 | 50.98 | 48.59 | 107 |
| BPDA P01 | **83.99** | 53.35 | 52.22 | 51.94 | 50.86 | **48.70** | 106 |
| BPDA P012 | 83.84 | **53.55** | **52.30** | **52.08** | **51.02** | 48.60 | 108 |

### 4.2 相对 Corrected PGD Base 的变化

| 实验 | Delta Clean | Delta PGD-10 | Delta PGD-20 | Delta PGD-50 | Delta C&W | Delta AA |
|---|---:|---:|---:|---:|---:|---:|
| BPDA P0 | -0.35 | +0.30 | +0.37 | +0.34 | +0.19 | +0.20 |
| BPDA P01 | +0.02 | +0.35 | +0.36 | +0.33 | +0.07 | **+0.31** |
| BPDA P012 | -0.13 | **+0.55** | **+0.44** | **+0.47** | **+0.23** | +0.21 |

三个配置在五项鲁棒指标上全部为正增益，这是本轮最有价值的信号。但增益区间只有 `+0.07` 到 `+0.55`，不能忽略单 seed 方差。

## 5. 指标解读

### 5.1 P01 是当前综合候选，P012 是 PGD/C&W 候选

P01 同时获得最高 Clean `83.99%` 和最高 AutoAttack `48.70%`，相对 Base 没有 Clean 代价。因此，若下一轮只能保留一个配置，P01 是更稳妥的综合候选。

P012 在 PGD-10/20/50 和 C&W 上最好，但 AutoAttack 仅为 `48.60%`，比 P01 低 `0.10`；其 Clean 也比 P01 低 `0.15`。这说明更强的低位清零可能改善基于梯度的 PGD/C&W，却没有同步转化成最强综合攻击下的优势。

### 5.2 位数增加没有形成稳定剂量关系

若低位清零强度是主要有效因素，通常应看到从 P0 到 P01、P012 的大多数指标稳定上升。实际结果为：

- P0 -> P01：Clean 和 AA 上升，但 PGD-20/50、C&W略降。
- P01 -> P012：PGD/C&W 上升，但 Clean 和 AA 下降。

因此当前不能得出“清除低三位优于低两位”或“位数越多越鲁棒”的结论。三个配置的差异很可能混合了训练波动、量化效应和位清零效应。

### 5.3 未看到最直接的梯度遮蔽信号，但仍需加强验证

四组均满足：

```text
PGD-10 > PGD-20 > PGD-50 > AutoAttack
```

C&W 和 AutoAttack 也明显低于 PGD-10，说明简单增加攻击强度仍能降低准确率。AutoAttack 中包含 Square Attack，且完整攻击后的结果没有异常升高。这些现象支持当前 BPDA 评估比旧非自适应评估可信。

但 BPDA 使用的是 identity surrogate gradient，仍存在梯度近似偏差。正式论文前还应补充多重随机重启 PGD、不同步长、迁移攻击或额外黑盒攻击，不能仅凭一次标准 AutoAttack 排除全部 obfuscated-gradient 风险。

### 5.4 Robustness Gap 不是单独的主结论

Base 的 Clean-AA gap 为 `35.58` 点；P0、P01、P012 分别为 `35.03、35.29、35.24` 点。低位配置的 gap 略小，但 gap 可能因 Clean 降低而缩小，因此只能与 Clean、AA 绝对值联合解读，不能单独作为方法优越性的证据。

评估文件中名为 `Robustness Gap` 的数值实际计算的是 `Clean - PGD-10`，不是 `Clean - AutoAttack`。报告正文应避免只写“Robustness Gap”而不说明攻击口径。

## 6. 训练动态与运行完整性

四组训练均正常完成，best 均出现在第二次学习率衰减附近的 epoch 106--108，没有出现上一阶段辅助流实验中的中期坍塌。

| 实验 | final train loss | final train acc | final Clean | final PGD-10 | 训练时间 |
|---|---:|---:|---:|---:|---:|
| Base | 0.9292 | 63.38 | 84.52 | 52.63 | 1h 36m 52s |
| P0 | 0.9317 | 63.35 | 84.15 | 52.82 | 1h 37m 38s |
| P01 | 0.9291 | 63.26 | 84.36 | 53.06 | 1h 37m 37s |
| P012 | 0.9268 | 63.40 | 84.16 | 52.94 | 1h 46m 00s |

训练损失、训练准确率和最终指标都非常接近，表明低位包装没有破坏 PGD-AT 的基本优化过程。P012 本轮训练慢约 8--9 分钟，但位运算本身不足以解释全部差距，更可能含当时节点负载波动；在重复计时前不应把它写成稳定计算开销。

本轮 launcher 最终正常退出，四个 status 均为 `complete`，四组都存在 `best_model.pt`、`final_model.pt`、`metrics.csv` 和 `eval_results_best.txt`。总墙钟时间约 4 小时 45 分钟。

## 7. 与上一阶段结果的关系

上一阶段低位辅助流最好的 Align-only 候选仍比当时 Base 低 `1.66` AA 点，较大权重和 Mask-CE 还会导致严重坍塌。本轮去掉辅助目标、把低位变换纳入被攻击模型后，训练不再坍塌，并出现 `+0.20` 到 `+0.31` AA 信号。

这说明目前真正有价值的不是旧 Mask-CE/Align/Contrast，而是以下组合：

```text
低位输入变换 + 全流程自适应训练 + 全流程自适应评估
```

但是不能把两阶段差异完全归因于 BPDA 位平面，因为本轮还同时修正了 PGD 生成阶段的 BN 状态。本轮 Corrected Base 的 AA 为 `48.39%`，上一阶段 Base 为 `48.84%`；两者训练代码语义不同，不能把旧 Base 混入本轮增益表。本报告只使用同轮 Corrected Base 做公平对照。

## 8. 当前方法是否有效、哪些模块有用

### 8.1 可以确认

1. BPDA 低位包装是一个**可训练、可评估、未导致坍塌**的候选模块。
2. 相对同轮 Base，P0/P01/P012 的强攻击结果方向一致为正，值得继续验证。
3. P01 当前拥有最好的 Clean-AA 权衡；P012 当前拥有最好的 PGD/C&W 数值。
4. 自适应攻击是必要组成。旧的非自适应 masked evaluation 不能作为论文结果。

### 8.2 不能确认

1. 不能确认 `+0.31` AA 是统计显著提升，目前只有一个 seed。
2. 不能确认增益来自“位平面语义”而不是普通 8-bit 量化，因为当前没有 quantize-only 对照。
3. 不能确认低两位或低三位哪个更好，两者差距过小且指标排序不一致。
4. 不能据此证明原 BitCons 的 Mask-CE、Align、Contrast 有效，因为本轮三者均未启用。
5. 不能把当前确定性预处理本身作为足够强的论文创新；类似 bit-depth reduction 与 BPDA 防御已有成熟研究脉络。

## 9. 下一步最小验证方案

当前不应立即扩展到所有宿主方法和数据集。先用最小矩阵拆分“量化”和“位清零”的贡献：

| 组 | 变换 | 目的 |
|---|---|---|
| Corrected PGD Base | 无 | 基线复现 |
| Quantize-only | 只 round 到 8-bit，不清零 | 分离普通量化收益 |
| BPDA P01 | round + 清零 bit 0,1 | 当前 Clean/AA 候选 |
| BPDA P012 | round + 清零 bit 0,1,2 | 当前 PGD/C&W 候选 |

快速阶段仍可先跑单 seed 110 epoch，但进入论文主表前应对 Base、Quantize-only 和最终胜出的一个位平面配置补至少 3 seeds，报告 mean +/- std。推荐的继续/停止门槛为：

1. 最终候选在多 seed AutoAttack 上平均提高至少 `0.5` 点，且标准差不能覆盖全部增益。
2. Clean 平均下降不超过 `0.5` 点。
3. 候选必须稳定优于 Quantize-only，否则不能把收益归因于位平面设计。
4. PGD 多重重启、不同步长、C&W、AutoAttack 与额外黑盒/迁移攻击结论方向一致。

若上述门槛不能通过，应停止把“确定性低位清零”作为核心方向，转向可学习或随机化的位平面机制，例如样本自适应位选择、训练期随机低位重采样、不同位平面的可学习融合；同时必须保持攻击自适应，随机机制还需用 EOT 评估。

## 10. 论文定位建议

当前结果最多支持如下谨慎表述：

> 在 CIFAR-10/ResNet18、单 seed 的探索实验中，将低位清零作为模型的一部分并采用 BPDA 自适应对抗训练，相比同轮修正 PGD-AT 获得了最高 `+0.31` AutoAttack 和 `+0.47` PGD-50 的初步增益，且没有明显训练坍塌。

目前不支持“显著提升鲁棒性”“低三位最优”或“BitCons 整体有效”的表述。论文工作的下一核心任务不是继续堆叠对比实验，而是先完成 quantize-only 因果对照和多 seed 验证；只有确认收益来自位平面机制后，才值得扩展到 TRADES、MART、RPAT、其他数据集和模型。

## 11. 数据与日志位置

- 启动脚本：`run_bitplane_bpda_exploration.sh`
- 运行清单：`logs/bitplane_bpda_bpda_explore_20260828_123058/manifest.tsv`
- 状态目录：`logs/bitplane_bpda_bpda_explore_20260828_123058/status/`
- 训练与评估日志：`logs/bitplane_bpda_bpda_explore_20260828_123058/`
- 每组原始结果：manifest 中对应实验目录下的 `metrics.csv`、`training_report.txt`、`eval_results_best.txt`

本报告所有主结果均来自 best checkpoint 的 `eval_results_best.txt`，没有将 final checkpoint 指标混入主表。
