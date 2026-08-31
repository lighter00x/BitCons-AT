# BitCons SOTA 方法设计方案

## 1. 方法定位

论文主体继续使用 **BitCons**。BitMax、BPDA、量化对照都服务于 BitCons，不取代它。

建议把旧定义：

```text
固定 mask(adv) -> Mask-CE + detached-logit Align + Contrast
```

升级为：

```text
Worst-case BitCons
  -> 连续 PGD 生成标准 adversarial view
  -> BitMax 在同一 L_inf 球内生成最坏 bit-plane view
  -> 对真正产生增量风险的 bit view 做监督和一致性约束
  -> 错误参考、低价值样本和冲突梯度不强制对齐
  -> 推理仍是普通模型，不使用量化或清位预处理
```

这条路线保留 BitCons 的核心思想：**模型应对同一语义样本的 adversarial view 与 bit-plane view 保持一致**。创新不是重新命名 BitMax，而是把一致性约束从固定、无条件的辅助分支，改造成由最坏位风险驱动的条件鲁棒目标。

## 2. 为什么这比旧 BitCons 更合理

旧实验的负结果并不证明“bit consistency”原则一定无效，证明的是当前实现存在四个结构问题：

1. 固定 `[3,4,5]` 或 `[0,1,2]` mask 未必是每个样本的最坏位变化。
2. `mask(adv)` 曾越出威胁球；新 BitMax 候选会投影回 clean-centered `L_inf` 球。
3. detached adversarial teacher 可能已经预测错误，强制 Align 会复制错误。
4. 所有样本共享统一 alpha，且不检查一致性梯度是否破坏主 CE。

新 BitCons 分别用 worst-case view、合法投影、可靠参考门控、风险权重和冲突控制解决这些问题。

## 3. 主方法：RA-WC-BitCons

名称建议：**Risk-Adaptive Worst-Case Bit-Plane Consistency Adversarial Training**，简称 `RA-WC-BitCons`。

### 3.1 内层：BitMax 生成最坏位视图

现有 [src/attacks/bitmax.py](src/attacks/bitmax.py) 已经完成：

- 标准 PGD 候选；
- 低位全 0、全 1 和随机候选；
- clean-centered `L_inf` 投影；
- 离散跳转后的短步 PGD refine；
- 逐样本最大 CE 选择；
- selection rate、loss gain 和实际 `L_inf` 统计。

第一版只需扩展返回值：

```text
x_adv, logits_adv, ce_adv
x_bit, logits_bit, ce_bit
selected_index
gain_i = ce_bit_i - ce_adv_i
```

标准 PGD 必须始终留在候选集合，保证训练内攻不会按构造弱于 PGD。

第二版再借鉴论文 011 的稀疏 PGD：将离散 bit support 和 bit value 分开，采用真实离散前向、连续 relaxation 反向以及 support 停滞重启。这是最有算法创新潜力的 `BitMax-S`，但应在固定候选版本出现正信号后实现。

### 3.2 外层：最坏风险分类目标

不要把标准 PGD CE 和 bit CE 简单等权相加。先使用逐样本 hard maximum：

```text
L_rob_i = max(CE(f(x_adv_i), y_i), CE(f(x_bit_i), y_i))
L_rob   = mean_i L_rob_i
```

这等价于在当前连续/离散候选集合上做经验内层最大化。`BitMax-only` 就是 `L_rob`，它是必须保留的强消融。

### 3.3 BitCons：风险自适应一致性

计算标准 adversarial view 和最坏 bit view 的预测分布：

```text
p_adv = softmax(f(x_adv) / T)
p_bit = softmax(f(x_bit) / T)
```

基础一致性可用逐样本 JS：

```text
d_i = JS(p_bit_i, stopgrad(p_adv_i))
```

但不再对所有样本使用统一权重。定义：

```text
risk_i = clamp(relu(gain_i) / tau_gain, 0, 1)
reliable_i = 1[predict(x_adv_i) = y_i and margin_adv_i > m]
lambda_i = lambda_max * curriculum(epoch) * risk_i * reliable_i
L_cons = sum_i lambda_i * d_i / max(sum_i lambda_i, 1)
```

总损失：

```text
L_total = L_rob + L_cons
```

这个设计吸收论文 031 和 039 的样本自适应思想，但使用本项目特有的 `BitMax loss gain`，而不是教师 entropy 或一般 clean/adv confidence。它直接度量“位候选是否提供了标准 PGD 未覆盖的风险”。

### 3.4 错误教师保护

如果 `x_adv` 已经误分类，旧 Align 会把错误预测传给 bit view。第一版直接令该样本 `lambda_i=0`，但 `L_rob` 仍训练它。

第二版可使用标签混合参考：

```text
q_i = eta_i * one_hot(y_i) + (1 - eta_i) * stopgrad(p_adv_i)
```

参考 margin 越低，`eta_i` 越大。这样低置信样本更多依赖真标签，高置信正确样本保留软分布信息。必须与 binary correct gate 比较，不能默认复杂版本更好。

### 3.5 冲突安全 BitCons

旧结果表明 Align 标量很小也可能破坏主目标。建议分两阶段控制：

1. **低成本版本**：限制 `L_cons / L_rob` 的最大比例，并记录二者对最后分类层的 batch gradient cosine；cosine 为负时将本 batch 的一致性权重乘 0 或 0.1。
2. **增强版本**：只对最后 block 或 classifier 使用 PCGrad 式投影，不对全模型做昂贵的逐样本二阶计算。

该模块借鉴论文 015 的方向错位分析和论文 048 的梯度冲突筛选。它应作为 `Conflict-Safe BitCons` 独立消融，而不是隐藏在完整模型中。

### 3.6 BitCons Curriculum

借鉴论文 042 的 epsilon scheduling、论文 010 的多位宽课程和论文 039 的动态样本计算：

```text
epoch 0..19:     标准 PGD-AT，BitCons 关闭
epoch 20..59:    P0/P01 确定候选，refine=0，lambda 上限很小
epoch 60..89:    加入 P012，refine=1
epoch 90..end:   refine=2，保持 risk/reliable/conflict gate
```

课程控制的是候选难度和 `lambda` 上限；每个样本的实际权重仍由风险门控决定。不能再次只按 epoch 把所有样本的 alpha 推到同一个大值。

## 4. 可进一步加入的创新模块

### 4.1 Pixel-aware BitCons

借鉴论文 014 的 local worst-case 聚合和论文 016 的 pixel-aware inner attack，但修复 PART 缩小训练威胁集的问题：

- 所有候选仍投影到统一 `8/255` 球；
- 以 4x4 或 8x8 block 为单位计算低位翻转 loss gain；
- 优先搜索高增益 block 的 bit configurations；
- 与随机同面积、反向选择和全图 uniform 严格匹配候选数；
- 检查攻击是否转移到未选区域。

这可以形成 `Local-WC-BitCons`，但组合空间和训练成本较高，放在 RA-WC-BitCons 有正结果之后。

### 4.2 Layer-aware BitCons Repair

借鉴论文 029 和 041：

- 比较 `x_adv` 与 `x_bit` 在各层的 feature displacement ratio；
- 找到放大 bit discrepancy 最强的少数 critical layers；
- 仅对这些层加入小幅 AWP、feature consistency 或参数微调；
- 每 10 epoch 重选 critical layers。

现有仓库已经有 AWP/RWP，因而无需先实现新的权重扰动器。该模块适合在 BitCons 主方法有效但训练不稳定时加入。

### 4.3 EMA Teacher BitCons

如果在线 adversarial reference 波动较大，可以维护 EMA teacher：

- teacher 只提供可靠 `p_adv`；
- student 在 `x_bit` 上对齐 teacher；
- teacher 错误时仍关闭一致性；
- 与无 EMA、SWA-only 严格比较。

论文 031 表明教师的总体鲁棒率不是充分条件，因此仍必须保留逐样本可靠性门控。EMA 只作为稳定器，不作为 BitCons 的核心创新。

### 4.4 多半径与类别自适应分析

借鉴论文 042、045：

- 报告 `epsilon=2/4/6/8/10/255` 的鲁棒曲线或 AUC；
- 报告 worst-class AA；
- 统计每类 BitMax selection rate、loss gain 和 consistency weight；
- 检查平均增益是否由少数类别贡献，或是否牺牲困难类别。

这些首先是机制证据，不应未经验证直接变成 class-wise loss reweighting。

## 5. 对现有代码的落地映射

| 文件 | 建议改动 |
|---|---|
| `src/attacks/bitmax.py` | 返回逐样本 PGD/bit CE、candidate index、gain；支持 planes-family 候选和后续 relaxation/restart |
| `src/losses/bitcons.py` | 增加 per-sample JS、margin、risk weight、reliable gate、target mixing 和 loss-ratio cap |
| `src/training/methods/bitcons_at.py` | 新建主方法：PGD + BitMax + hard-max robust CE + gated BitCons；保留 legacy `pgd_at.py` 不动 |
| `src/training/methods/__init__.py` | 注册 `bitcons_at`，使论文主方法拥有清晰语义 |
| `configs/training/bitcons_at.yaml` | 集中定义 candidate family、gain temperature、margin gate、lambda、curriculum 和 conflict mode |
| `src/common/args.py` | 增加新 BitCons 参数；旧参数保留供历史实验重现 |
| `src/common/config.py` | 校验候选 planes、门控阈值、课程边界和互斥配置 |
| `src/utils/logger.py` | 记录 bit selection、gain 分布、gate rate、JS、gradient cosine、各 planes 选择率 |
| `tests/test_bitcons.py` | 覆盖 per-sample loss、错误教师 gate、无正 gain、归一化和数值稳定 |
| `tests/test_bitmax.py` | 覆盖 PGD 永远在候选集、投影合法、逐样本最大选择和返回统计 |

主方法应新建 `bitcons_at.py`，不要继续把新逻辑叠进已经承担 legacy 复现的 `pgd_at.py`。这样旧失败实验可复现，新 BitCons 的算法语义也更容易审计。

## 6. 必须做的因果消融

第一轮固定 CIFAR-10、ResNet18、seed 4243：

| 组 | 训练目标 | 回答的问题 |
|---|---|---|
| A. Corrected PGD Base | 标准 PGD CE | 基线 |
| B. Quantize-only | 8-bit round | 量化本身是否有效 |
| C. Fixed BitPlane P01/P012 | 测试时固定变换 + BPDA | 固定清位是否有效 |
| D. BitMax-only | `L_rob=max(CE_adv,CE_bit)` | 更强离散内攻是否有效 |
| E. BitMax + constant BitCons | D + 固定 JS | 旧式统一一致性是否仍有害 |
| F. RA-WC-BitCons | D + risk/reliable gated JS | 风险门控是否修复一致性 |
| G. F + Curriculum | 分阶段候选和权重 | 训练路径是否重要 |
| H. G + Conflict-Safe | 负 cosine 抑制/投影 | 梯度冲突是否是剩余瓶颈 |

最重要的比较是 `D vs F`：只有 F 稳定超过 D，才能证明论文贡献来自 BitCons，而不是仅来自 BitMax 更强内攻。

## 7. 筛选门槛

单 seed 快速阶段建议使用硬门槛：

- RA-WC-BitCons 相对 Corrected Base 的 AutoAttack 至少 `+0.5`；
- 相对 BitMax-only 的 AA 至少 `+0.3`，否则 BitCons 独立价值太弱；
- 必须超过 Quantize-only；
- Clean 下降不超过 `0.5`；
- PGD-50、C&W、AA 方向一致；
- gate rate 不能接近 0，consistency 也不能长期占总 loss 过高；
- 多 seed 后 paired gain 的标准差不能覆盖全部增益。

未过门槛时先检查 selection/gain/gate/gradient cosine，不扩大数据集和模型。

## 8. 从有效方法到 SOTA 的两条路线

### 8.1 方法 SOTA：同训练预算公平比较

先在相同 ResNet18/WRN34-10、相同 epoch、数据和攻击下，证明 BitCons 对 PGD-AT、TRADES 或 MART 有稳定增益。这是算法贡献的主要证据。

建议顺序：

1. CIFAR-10 ResNet18，3 seeds；
2. CIFAR-10 WRN34-10，3 seeds；
3. CIFAR-100 WRN34-10，3 seeds；
4. 最终候选接入 TRADES + AWP；
5. standard inference 下完整 AutoAttack。

### 8.2 Leaderboard SOTA：强训练底座

当前 ResNet18、110 epoch、无额外数据的 `48%` AA 级别无法仅靠一个小模块追到 CIFAR-10 全局 SOTA。要冲 leaderboard，BitCons 必须叠加到强底座：

- 更宽模型，至少 WRN34-10，最好补 WRN70-16 类配置；
- 400 epoch 级训练和 cosine schedule；
- TRADES/AWP 或经过验证的强鲁棒训练 recipe；
- EMA/SWA 强基线；
- 高质量额外或合成数据，论文 020 表明鲁棒 scaling 更依赖大量高质量 unique samples；
- 完整 validation/test 隔离和 3--5 seeds；
- 报告训练 FLOPs、数据量和推理成本，避免与不同预算结果直接混比。

正确顺序是先证明 `Strong Base + BitCons > Strong Base`，再谈全局排名。否则即使大模型数字高，也无法证明 BitCons 的贡献。

## 9. 可信评估底线

主方法推理期不做 bit masking，因此标准 AutoAttack 可以直接攻击普通模型，不依赖 BPDA。这应成为论文优势。

仍需执行：

- PGD-20/50/100，多 restart 和步长扫描；
- C&W 与 AutoAttack standard；
- Square/迁移攻击交叉检查；
- validation PGD 选 checkpoint，test 只做冻结后评估；
- 3 seeds mean/std 和 paired delta；
- final/best 差距；
- Clean、AA、worst-class AA、训练时间、显存；
- BitMax selection rate、loss gain、candidate family 和 gate rate 的完整分布。

测试时 BitPlane/BPDA 只作为对照，不应与训练期、普通推理的 RA-WC-BitCons 混为一个方法。

## 10. 最终实施顺序

```text
P0  完成 Quantize-only 和当前 BitMax 结果，补齐因果基线
P1  新建 bitcons_at.py，实现 BitMax-only hard-max robust CE
P2  实现 per-sample JS + risk/reliable gate，得到 RA-WC-BitCons
P3  加 BitCons curriculum，完成 D/E/F/G 消融
P4  记录梯度 cosine，必要时实现 Conflict-Safe BitCons
P5  有稳定正增益后实现连续 bit support 搜索 BitMax-S
P6  做 Pixel-aware/Layer-aware 扩展
P7  接入 WRN + TRADES/AWP + EMA/额外数据，冲击 SOTA
```

论文的核心创新应始终表述为 **worst-case、risk-adaptive、conflict-safe bit-plane consistency**。BitMax 是内层优化技术，量化和 BPDA 是因果/评估对照，BitCons 才是完整方法。
