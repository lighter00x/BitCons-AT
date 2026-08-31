# 文献方法借鉴与 BitCons 实验路线总结

> **方法定位修正：** BitCons 保留为论文核心概念。本文早期将 BitMax 置于主方法位置的表述不再作为最终设计；BitMax 应作为 BitCons 的内层最坏位视图生成器。最终可实施方案见 [BitCons_SOTA方法设计方案.md](BitCons_SOTA方法设计方案.md)。

> 阅读范围：`paper_screening/reading_notes` 中 48 篇解读报告。  
> 目标：筛选能扩展当前 BitCons/BitPlane-AT/BitMax 方法体系、并可能产生真实实验增益的机制。  
> 原则：优先考虑能够进入当前 PGD-AT 代码、威胁模型闭环、增量成本可控、可通过 AutoAttack 和独立攻击验证的方案。

## 1. 总结结论

现有文献最支持的方向不是继续恢复旧的 Mask-CE、Align、Contrast 三分支，而是把当前项目重构为一个明确的**连续像素扰动与离散低位扰动联合优化框架**：

```text
标准 PGD 连续内攻
  + 威胁球内离散低位候选搜索（BitMax）
  + 样本/像素自适应的低位操作强度
  + 按候选真实增益选择或加权训练样本
  + 自适应 surrogate、黑盒和长步攻击组成的评估协议
```

最值得优先实验的不是新的辅助损失，而是以下四个机制：

1. **BitMax-S：可学习/连续化的位候选搜索。** 借鉴 sPGD 的“离散 support 前向、连续 mask 反向、停滞重启”，把当前 BitMax 的固定全 0/全 1/random 低位候选改成更有效的离散内层最大化。
2. **BitMax-A：样本自适应位强度。** 用标准 PGD 与 BitMax 候选之间的 per-sample loss gain、margin drop 或 entropy change 决定是否启用 P0/P01/P012，而不是所有样本固定清相同位。
3. **BitMax-P：像素自适应低位搜索。** 以低位敏感度或梯度结构选择像素区域，但保持候选始终投影在同一个 `L_inf` 威胁球内；不缩小测试威胁模型。
4. **BitMax-C：从弱到强的课程。** 先普通 PGD-AT，再逐渐提高离散候选数、低位平面数或选择率，避免第二阶段辅助权重升高导致的训练坍塌。

其中第一项最有方法创新潜力，第二项最便宜、最适合快速筛选，第三项需要严格因果对照，第四项适合作为稳定器而非独立贡献。

## 2. 与当前项目问题的对应关系

| 当前问题 | 文献给出的启发 | 可执行改变 |
|---|---|---|
| 固定低位清零收益仅 `+0.31 AA` | 离散变换不是天然鲁棒，真正有效的是训练其边界或显式求离散最坏情况 | 以 BitMax 为主方法，BitPlane-AT 退为固定变换基线 |
| `P0/P01/P012` 无单调关系 | 样本和像素的脆弱度不均，固定强度会混合有益和有害样本 | 依据 loss gain/margin/entropy 动态选 planes |
| 旧 Align/Mask-CE 高权重坍塌 | 辅助目标可能与主鲁棒方向错位；标量 loss 小不代表梯度无害 | 不先恢复辅助流；先测梯度夹角、位响应和候选选择率 |
| Contrast 明确有害 | 普通表示拉近/推远未必对应最坏风险；迁移梯度分离也不等于单模型鲁棒 | Contrast 保持停用，只将结构化梯度指标用于诊断 |
| 量化与清位贡献混杂 | 离散前端必须有量化因果对照 | Base、Quantize-only、BitPlane、BitMax 同轮比较 |
| BPDA 只用 identity surrogate | 不可微系统的 surrogate 是攻击算法的一部分 | identity、soft-round、piecewise、candidate-aware surrogate 取最坏结果 |
| 单次 PGD/C&W 与单 seed | 自研白盒攻击可能在自家模型上失效 | 多 restart、长步、Square/黑盒、迁移、surrogate sweep；至少 3 seeds |
| test set 逐 epoch 选 best | 训练动态和鲁棒过拟合会造成选择偏差 | 固定 validation split，validation PGD 选模，test 一次性评估 |

## 3. 推荐的方法体系

### 3.1 BitMax-S：连续与离散联合内层最大化

当前 BitMax 已经能生成低位全 0、全 1、随机候选，并在 clean-centered `L_inf` 球内投影，然后逐样本选择最大 CE。下一步可将候选搜索形式化为：

```text
min_theta E[max_{delta in B_inf, z in Z_bit} L(f_theta(T_z(x + delta)), y)]
```

其中 `z` 不只是 planes 集合，还可包含每像素低位取值或有限候选索引。借鉴论文 011：

- 真实前向始终使用离散 8-bit 候选，保证训练样本合法；
- 反向对离散位 mask 使用 sigmoid/softmax relaxation；
- 位值和像素 support 分开更新；
- 离散 support 连续若干步不变时随机重启；
- 训练使用短步搜索，评估使用更长搜索加独立随机/黑盒方法；
- 标准 PGD 候选必须始终包含在候选集合中，保证 BitMax 内攻不会按构造弱于 PGD。

最小版本不必立刻实现全像素 Gumbel 搜索。可以先对 `{PGD, P0-low, P0-high, P01-low, P01-high, P012-low, P012-high}` 做逐样本最大选择，再观察 `bitmax_selection_rate` 和 `bitmax_loss_gain` 是否持续大于零。

### 3.2 BitMax-A：按样本决定是否需要位攻击

对每个样本定义低位脆弱度：

```text
s_i = L_bitmax_i - L_pgd_i
```

可选辅助统计包括 true-class margin drop、预测 entropy change、标准 PGD 与位候选的梯度余弦。训练策略：

- `s_i <= 0`：保留标准 PGD，避免无收益的额外离散操作；
- `s_i > 0`：使用最坏位候选；
- 可用连续权重 `w_i = sigmoid((s_i - tau)/T)`，但必须保留 hard-max 作为主基线；
- 周期统计 `s_i` 分布，检验收益是否集中在少数样本、类别或 epoch；
- 不依据 clean confidence 单独决定难度，优先使用“位候选相对 PGD 的增量风险”。

这比直接借鉴论文 031 的教师 entropy 更贴合本项目，因为不需要额外教师，信号直接回答“低位候选是否提供了 PGD 没覆盖的风险”。

### 3.3 BitMax-P：像素级位敏感区域

论文 014、016 提示全局平均会掩盖局部最坏区域，输入结构可进入内层攻击。但 PART 的主要缺陷是缩小了训练威胁集，容易把 Clean 提升误判成机制收益。本项目应采用不同设计：

- 所有候选仍受同一个 `8/255` clean-centered 投影约束；
- 不降低非关键像素允许的测试扰动预算；
- 用像素级 `|grad_x L|`、低位翻转 loss gain、局部 margin drop 或块级 CVaR 选择位操作区域；
- 比较 gradient/sensitivity、随机同面积、反向选择、uniform 全图四种 mask；
- 匹配候选数、被操作像素比例和计算预算，隔离“选择策略”贡献；
- 检查攻击是否转移到未选区域，防止形成空间 shortcut。

第一版建议使用 4x4 或 8x8 block，而不是逐像素搜索，以控制组合空间和训练成本。

### 3.4 BitMax-C：离散难度课程

借鉴 epsilon scheduling、BACT 和动态采样：

```text
epoch 0..E1:       标准 PGD-AT
epoch E1..E2:      PGD + P0/P01 两个确定候选
epoch E2..end:     增加 P012、随机候选或短步离散 refine
```

课程变量可以是候选数量、planes 数、refine steps 或 BitMax 样本比例。不要再次采用一个随 epoch 单调升高但不看实际风险的辅助 loss alpha。课程是否有效必须与“从 epoch 0 全量 BitMax”和“相同平均候选预算的随机启用”比较。

### 3.5 诊断与轻量修复

以下机制更适合作为诊断或二阶段增强：

- **梯度/方向诊断（015、019、010）**：测 clean-adv 方向、位候选方向与 margin gradient 的夹角；测不同 planes 的输入梯度余弦、HOG/edge 结构和局部 Jacobian 响应。
- **层级诊断（029、041）**：定位 BitMax 加入后哪些层放大位候选特征差，必要时只对少数 critical layers 加 AWP 或 feature consistency。
- **权重平均（038）**：SWA/EMA 是低成本 robust-overfitting 基线，任何新稳定化方法都应与其比较。
- **类别统计（045）**：报告 worst-class AA 和各类位候选选择率，检查平均提升是否牺牲困难类别。
- **半径曲线（042）**：除 `8/255` 外报告多个测试 epsilon 的曲线/AUC，但论文主指标仍以固定威胁模型 AA 为准。

## 4. 逐篇可借鉴判断

| 编号 | 论文 | 可借鉴思想 | 在本项目中的用途 | 优先级 |
|---:|---|---|---|---|
| 001 | DRIFT | 多变换响应/梯度差异、VJP 近似 | 诊断不同 bit transforms 的迁移与梯度共识；不可把分歧当鲁棒保证 | B |
| 002 | Keep It Real | 多 surrogate BPDA、长步攻击、取攻击网格最小值 | 直接升级 BitPlane/Quantize 的自适应评估协议 | A |
| 003 | Randomized Feature Squeezing | 随机二值/量化与 fixed-randomness 诊断 | 可做随机低位重采样探索；必须 EOT 收敛和固定随机对照 | C |
| 004 | DiffBreak | 精确随机路径重放、worst/majority 风险定义 | 若未来加入随机位变换，借鉴 EOT、随机决策和最坏路径协议 | B |
| 005 | Noise + Bilateral Filters | 组合预处理可能互补 | 仅提示“量化+局部平滑”消融；评估问题严重，不宜做主线 | D |
| 006 | Discrete Image Tokenizers | 对离散层前表示做无监督最坏一致性训练 | 可将位变换前后 margin/feature 稳定作为后续轻量分支；需 code-cell/梯度审计 | B |
| 007 | NIC-RobustBench | 多目标、worst-over-objectives、标准化工具链 | 扩展评估维度和配置记录，不直接提供训练增益 | B |
| 008 | Diffusion Compressing Image Space | 条件 Jacobian/局部压缩率，区分随机漂移与输入响应 | 作为 BitPlane 机制指标：平均方向收缩不能替代最坏方向 | B |
| 009 | Reliable SNN Evaluation | 自适应 surrogate family 与优化器耦合 | 为 round/bitwise 层做 surrogate sweep，而非固定 identity STE | A |
| 010 | TriQDef | 跨量化分支的梯度/边缘/HOG结构诊断 | 只借鉴诊断；反对齐降低迁移不等于提高单模型 AA | C |
| 011 | Efficient `l0` Training/Evaluation | 连续变量与离散 support 分解、代理反向、停滞重启、独立黑盒 | BitMax-S 的最核心算法来源 | A+ |
| 012 | Fast `l0` AT | 离散 support 探索、软监督、噪声和 curriculum | 借鉴离散搜索资源分配及课程；不直接照搬软标签三件套 | B |
| 013 | Single Diffusion Classifier | 把动态变换写进完整决策模型与内层优化 | 强化 BitMax 数学定义；生成式路径成本过高，不实现 | C |
| 014 | Modular NIC Attack | global-to-local、block max/CVaR 风险聚合 | BitMax-P：局部最坏低位区域与分块候选 | A |
| 015 | Feature Compression | margin gradient 与类间/扰动方向夹角 | 解释旧 Align 失败与不同 planes 响应，适合作为机制图 | A- |
| 016 | PART | 输入结构化预算、动态像素 mask | 借鉴像素自适应，但不得通过缩小训练威胁集制造 Clean 收益 | A |
| 017 | Adversary-Aware Optimization | 显式建模攻击残差分布 | 可统计低位残差先验；训练扩散残差模型离当前主线过远 | D |
| 018 | Dictionary Structure | 可学习混合保真项、结构层先验 | 可能作为长期网络结构方向，当前实现成本高、理论不稳 | D |
| 019 | Compressibility and Robustness | dominant directions、top-k spread、层间 alignment | 分析低位扰动是否集中到少数方向，辅助设计正则 | B |
| 020 | Scaling Law | 高质量多样数据和 compute allocation | 正式放大实验时使用，不解决当前机制有效性 | C |
| 021 | Contrastive Guidance | trajectory 自构造 positive | 旧 Contrast 已负面，不建议恢复；只保留“同轨迹局部正样本”思想 | D |
| 022 | Frequency-domain Purification | 幅度/相位分别约束、频率非均匀 data consistency | 做位域与频域相关性诊断，不做扩散净化主线 | C |
| 023 | Distributional Discrepancy | 一个统计量兼作训练信号和路径路由 | 可用位响应 discrepancy 决定走 PGD 或 BitMax；先做逐样本而非批路由 | B |
| 024 | ACR Is Poor | 检查平均指标的样本贡献和完整分布 | 报告 bitmax loss gain/selection rate 分布，不能只报均值 | B |
| 025 | MAE-Pure | 自监督重建误差作为输入能量 | 可作位候选筛选信号，但完整净化评估风险高，不优先 | D |
| 026 | Coarse-to-Fine Tensor Purification | 多尺度与容量限制可避免拟合扰动 | 启发 block-level coarse-to-fine 位搜索；不实现张量净化 | C |
| 027 | Progressive Residual TN | 分尺度残差和递减容量 | 可做从 P0 到 P012 的残差课程；原防御评估不足 | C |
| 028 | BEYOND | 自增强邻域、标签和表示双信号 | 可做检测/诊断，但拒绝式指标不适合当前鲁棒分类主表 | D |
| 029 | Layer-Aware CO | 前层先恶化、前强后弱 AWP | 若 BitMax 训练不稳，加入 layer-aware AWP 作为二阶段修复 | B+ |
| 030 | Pro-Trans | 注意力局部平滑、渐进容量 | 仅启发像素 mask；测试时优化过重且攻击不充分 | D |
| 031 | SAAD | 样本级 entropy/传递性权重、把无效样本转给 clean 目标 | 借鉴样本自适应，但以 BitMax 相对 PGD loss gain 替代教师 entropy | A |
| 032 | OODRobustBench | 分布/威胁偏移、AER 与绝对鲁棒并报 | 最终验证低位机制是否只拟合 `L_inf 8/255` | C |
| 033 | SINAI | 模块级选择性随机噪声、clean-drop 约束 | 随机低位机制的远期参考；白盒/EOT 风险高 | D |
| 034 | Transduction + Rejection | 攻击整个算法、拒绝需专门 loss | 不适合当前固定分类器主线；可作为未来检测扩展 | D |
| 035 | Error Amplification | confidence-aware 样本预算、错误类抑制 | BitMax-A 的备选权重信号；主要针对 fast AT，不优先 | B |
| 036 | Multimodal Purification | 生成模型 residual 作为能量 | 成本与评估风险过高，不适合当前代码路线 | D |
| 037 | Probabilistic Robustness | 平均风险和最坏风险同时报告 Pareto | 随机位变换时必须采用；当前确定性版本仅作扩展 | C |
| 038 | Uniform Stability | Moreau/参数平均缓解 robust overfitting | 加 SWA/EMA 强基线，防止把普通平均收益归给 BitMax | B+ |
| 039 | VDAT | 动态脆弱度采样、计算只花在有价值样本 | BitMax-A 的直接工程模板，可降低候选搜索开销 | A |
| 040 | Certified Robustness Position | 局部保证与系统安全分离 | 规范论文表述，不提供当前训练增益 | C |
| 041 | CLAT | 动态 critical-layer 选择、少参数鲁棒修复 | BitMax 若只在少数层放大，可做低成本 feature repair | B+ |
| 042 | Epsilon Scheduling | 从弱到强引入鲁棒目标 | BitMax-C：逐步增加 planes/候选/refine 强度 | A- |
| 043 | Clean Generalization/Robust Overfitting | 局部鲁棒记忆不等于全局规则 | 强制 validation、final/best 与 OOD/威胁偏移验证 | B |
| 044 | SAM-AT Duality | 平坦优化可能改善鲁棒特征偏好 | AWP/SAM 作为组合基线，不作为位平面创新点 | C |
| 045 | Class-wise Disparity | head-only 修复、worst-class robustness | 增加类别级位脆弱度与 worst-class AA；必要时轻量 head 调整 | B |
| 046 | Nasty AT | 辅助模型作为“远离的错误方向” | 可对旧 teacher 错误传播做反向对照，但额外模型会稀释主线 | C |
| 047 | Benign Overfitting | 插值、clean 泛化、robust 泛化可分离 | 理论边界参考，不直接改代码 | D |
| 048 | DataFreeShield | 梯度冲突筛选、符号一致聚合 | 可用于分析/过滤 BitCons 辅助梯度；主方法已不依赖辅助流 | C |

优先级含义：`A+` 核心方法来源；`A` 建议近期实现；`B` 重要诊断或二阶段增强；`C` 远期/论文分析；`D` 不建议当前投入。

## 5. 最小实验路线

### 阶段 0：补齐当前因果缺口

同一代码语义、同一 seed、同一 checkpoint 规则：

| 组 | 目的 |
|---|---|
| Corrected PGD Base | 基线 |
| Quantize-only | 分离 round 收益 |
| BitPlane P01 | 固定变换候选 |
| BitPlane P012 | 固定变换强度对照 |
| Current BitMax | 判断已有离散最坏选择是否超过固定变换 |

若 Quantize-only 已等于或优于 P01/P012，固定低位清零不再作为主创新；BitMax 仍可成立，因为它研究的是威胁球内离散最坏情况，而不是固定预处理。

### 阶段 1：低成本筛选 BitMax-A/C

固定 CIFAR-10/ResNet18/seed 4243：

1. Current BitMax hard-max。
2. 只在 `loss_gain > 0` 样本启用 BitMax，行为应与 hard-max 等价，用于验证实现。
3. Top-25%/50% loss-gain 样本运行 refine，其余只 PGD，测试计算-鲁棒 Pareto。
4. 从 epoch 0 全量 BitMax vs 20 epoch PGD burn-in 后启用。
5. 固定 P012 vs sample-wise 在 P0/P01/P012 中选择。

筛选门槛：AA 相对 Corrected Base 至少 `+0.5`，且超过 Quantize-only；Clean 下降不超过 `0.5`；BitMax 候选选择率不能长期趋近 0，否则离散搜索没有独立价值。

### 阶段 2：BitMax-S/P 方法实验

只在阶段 1 有正信号后投入：

- 固定候选 hard-max；
- 连续低位 relaxation；
- relaxation + support restart；
- block sensitivity mask；
- 随机同面积 mask；
- 反 sensitivity mask；
- 全图 uniform。

所有方案严格匹配候选数、refine steps、训练 epoch 和 threat ball。逐步加入组件，不做大笛卡尔积。

### 阶段 3：可信评估

- 3 seeds，报告 mean/std 和 paired delta；
- validation PGD 选 best，test 只评估冻结后的 checkpoint；
- PGD-20/50/100，多 restart，步长扫描；
- C&W、AutoAttack standard；
- identity BPDA、soft-round、candidate-aware surrogate 取最坏；
- Square/黑盒和迁移攻击；
- Clean、AA、worst-class AA、训练时间、显存、BitMax selection rate/loss gain 分布；
- Base、Quantize-only、BitPlane、BitMax 必须用同一完整协议。

## 6. 论文创新叙事建议

如果 BitMax-S/A 获得稳定增益，论文主线可以写成：

> 固定位深压缩和位平面清零把离散输入变换当作静态预处理，容易产生威胁模型错配，且没有针对样本依赖的低位脆弱性。我们将连续像素扰动与离散低位配置统一为联合内层最大化，并通过逐样本最坏候选、可松弛的低位搜索和自适应训练预算，让模型学习抵抗 PGD 未覆盖的离散像素风险。

可形成的贡献层次：

1. **问题发现**：固定清位、辅助一致性和非自适应评估为什么失败。
2. **方法**：连续 `L_inf` 扰动与离散低位变量的联合 adversarial training。
3. **自适应机制**：按样本/局部低位增量风险分配搜索与训练预算。
4. **评估协议**：多 surrogate BPDA、独立黑盒和 quantize-only 因果对照。
5. **机制证据**：位候选选择率、loss gain、方向/层级响应和类别分布。

如果 BitMax 仍没有超过 Quantize-only/Base，则更可信的论文方向应转为系统性负结果与评估论文：

> 位平面辅助正则和静态离散预处理在标准对抗鲁棒性下为何失效，以及非自适应评估如何产生虚假收益。

## 7. 不建议近期投入的方向

1. **恢复原 Contrast。** 当前项目已有直接负证据，外部 contrastive purification 论文在强自适应攻击下也多不稳定。
2. **随机低位防御作为主要结果。** 它会立即引入 EOT 收敛、决策聚合和有限查询安全问题，实验成本远高于当前确定性 BitMax。
3. **扩散/张量网络净化。** 与现有训练代码边界差异过大，且所读论文普遍存在完整链路攻击不足。
4. **检测与拒绝。** 指标语义会从 robust classification 改成 coverage/rejection risk，无法直接修复当前主结论。
5. **仅加入新的 feature alignment loss。** 第二阶段已经显示辅助梯度可能严重冲突；没有先做方向诊断前，不应重新堆损失。
6. **只追求 PGD 数字。** P012 已出现 PGD 更高但 AA 不占优；后续选择必须以 AA 和最坏攻击为主。

## 8. 最终优先级

```text
P0  补跑 Quantize-only，完成 Base/Quantize/BitPlane 的因果闭环
P1  完整评估已有 BitMax，确认是否提供 PGD 之外的真实离散风险
P2  实现 sample-wise planes/hard candidate selection 与课程启用
P3  实现连续离散搜索和停滞重启 BitMax-S
P4  做 block-level pixel-adaptive BitMax-P 与严格随机/反向对照
P5  加层级、方向、类别机制分析和多 seed/多模型正式验证
```

这条路线最大程度利用当前已经实现的 `QuantizeBPDA`、`BitPlaneBPDA`、BitMax candidate selection、PGD 和 AutoAttack 管线，同时避开已被本项目实验否证的辅助分支。
