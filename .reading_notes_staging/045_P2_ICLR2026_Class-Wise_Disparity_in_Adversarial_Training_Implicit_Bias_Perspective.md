# Class-Wise Disparity in Adversarial Training: Implicit Bias Perspective

## 1. 基本信息
匿名作者（double-blind稿）；ICLR 2026；PDF 26页。CIFAR-10/100、STL-10、OfficeHome、ImageNet-100；WRN、RN50、ViT；RTX3090。代码称在supplementary提供，无公开URL。
## 2. 一句话结论
AT使分类头范数出现类间不均，简单归一化或冻结特征后用类自适应SAM微调1 epoch可大幅提高最差类鲁棒率，但平均AA常略降，且“头范数导致公平差距”的因果链尚未完全建立。
## 3. 研究问题
平衡数据上为何各类robust accuracy差异巨大，能否无需验证集修复。例子：同一共享特征下，每类最后一条线性决策向量尺度不同，导致部分类边界余量更差。边界为分类头层面的robust fairness，不解决feature-level bias。
## 4. 研究动机
作者测得robust model头范数与类鲁棒率相关约0.95，且AT放大范数比。理论把clean/adv gradient gap连到class hardness；但正文“hard类范数增长更快”与“低norm类需更大SAM半径”的叙事存在张力。
## 5. 核心贡献
新机制分析、post-hoc HWNwB和DecoSAM防御。核心经验贡献是只改head提升worst-class；HWNwB可去掉但论文仍成立，去掉head干预则核心方法消失。组件为weight normalization与SAM的针对性组合。
## 6. 方法总览
```text
完成AT -> 统计每类head norm
方案A: 每类W归一化，保留bias
方案B: 冻结feature extractor和bias，仅1 epoch更新head
       低norm类分配更大SAM扰动 -> 输出普通模型
```
## 7. 方法细节
`psi`为冻结特征，`W_k,b_k`为类k head。HWNwB只规范`W_k`。DecoSAM按`nu_k=softmax(-tau||W_k||)`分配类扰动半径，内层扰动head，外层更新head；bias固定。无推理开销。
## 8. 关键公式
Definition 1/2（PDF第4页）定义gradient gap`Delta_k`与hardness`H_k`；Proposition 1在强假设下`Delta_k=mu_Z H_k`。Theorem 1给`E||W_k^T||=||W_k^0||+eta T Delta_k`，近似忽略向量方向/随机梯度复杂性。式(7)为class-wise SAM内层；式(8)的`nu_k`使低norm类半径更大。`tau`越大分配越尖锐。
## 9. 算法伪代码
```text
load AT theta=(psi,W,b); freeze psi,b
for one epoch: compute class radii from W norms; perturb each W_k by SAM inner max;
               backprop outer CE only to W
return psi,W,b; inference unchanged
```
## 10. 理论分析
命题依赖附录的特征/梯度分布假设，给期望增长关系而非公平性充分条件；高相关系数也非因果。需要直接交换/缩放norm并控制bias、feature geometry来验证机制。
## 11. 实验设置
baseline AT 100 epoch，PGD-10 `8/255`、步长`2/255`，LR里程碑90/95，final checkpoint；评估PGD-20与AA。DecoSAM 1 epoch；RTX3090。seed数量/均值方差未充分报告。
## 12. 核心实验结果
| C10 WRN28-5 PGD-AT | AA | worst-class AA | 来源 |
|---|---:|---:|---|
| baseline | 49.50 | 17.60 | 表2，PDF第8页 |
| HWNwB | 48.25 | 29.10 | 表2 |
| DecoSAM | 49.09 | 30.70 | 表2 |
DecoSAM最差类+13.10pp但平均AA-0.41pp，支持公平性改善而非整体鲁棒提升。
## 13. 消融实验
表5显示保留bias优于同时归一bias；表3跨STL，表4与FRL/FAT/CFA/FAAL/WAT组合。缺完整Pareto曲线、多seed、feature-level控制与类频率/语义混杂分析。
## 14. 威胁模型与评估可信度
白盒`l_inf 8/255` PGD20和AA直接攻击最终确定性模型，无BPDA/EOT。按类样本少使worst-class方差较高。**可信度：中高（攻击），中（公平统计）**。
## 15. 可复现性
Algorithm 1、冻结策略和训练设置清楚；匿名代码、seed和部分超参不足。**可复现性：中。**
## 16. 局限与失败模式
作者承认仅处理head bias。低norm未必等于hard类；类别多/样本少时估计噪声大；平均鲁棒性可下降，规范化可能破坏校准。公平概念仅为类间最差准确率，不覆盖群体属性。
## 17. 批判性评价
优点是极低成本且最差类提升大；缺点是机制叙事矛盾和无方差。应补norm因果干预、5 seed置信区间、平均-最差-clean-calibration Pareto。可信的是post-hoc head修复有效；“implicit bias根因”仍需验证。
## 18. 相关工作定位
位于robust fairness、class-wise disparity、last-layer retraining、SAM；比较FRL、FAT、CFA、FAAL、WAT。关键词：`worst-class adversarial robustness`、`classifier head norm bias`。
## 19. 阅读后的关键启发
鲁棒公平可作为单独优化轴；最后一层往往提供低成本干预，但不可把相关代理直接称根因。
## 20. 尚未解决的问题
feature bias如何分解？norm与bias/角度谁主导？是否影响校准和clean worst-class？
## 21. 证据索引
| 结论 | PDF页 | 位置 | 证据 |
|---|---:|---|---|
| hardness与norm增长 | 4–6 | Prop.1, Thm.1 | 假设下期望关系 |
| 类自适应SAM | 7 | 式(7)(8) | 低norm分配大扰动 |
| worst-class提升 | 8 | 表2 | 17.60→30.70 |
| head-level限制 | 10 | Limitation | feature bias未解决 |

### 最终评分
问题14/15；新颖性15/20；技术14/20；实验16/20；可信度12/15；复现7/10；**78/100**。阅读优先级：建议精读；适合robust fairness研究者。已读全文附录并核对算法/表格。
