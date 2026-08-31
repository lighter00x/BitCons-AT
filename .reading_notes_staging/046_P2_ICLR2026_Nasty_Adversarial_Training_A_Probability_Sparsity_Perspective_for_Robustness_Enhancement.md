# Nasty Adversarial Training: A Probability Sparsity Perspective for Robustness Enhancement

## 1. 基本信息
Yuhang Zhou、Zhongyun Hua、Zhaoquan Gu、Keke Tang、Rushi Lan、Yushu Zhang等；ICLR 2026；PDF 22页。CIFAR-10/100、ImageNet-100，另ReID/检测；RTX4090，PyTorch1.12.1。代码地址论文未明确给出。
## 2. 一句话结论
NAT在PGD-AT目标中加入“远离辅助模型输出”的负KL项，诱导目标模型更稀疏地分配非真类概率，在PGD/C&W上显著提升且clean较高；但AA并非总胜最强baseline，概率稀疏主要是相关机制而非严格因果定理。
## 3. 研究问题
能否利用Nasty Training中模型间输出排斥增强AT，而不是只靠更强攻击/数据。例子：目标分类器不仅答对“猫”，还主动避免复制普通模型把概率分给飞机等易受骗类别。边界为监督分类AT及其扩展，不是认证防御。
## 4. 研究动机
作者观察Nasty模型输出概率更稀疏、决策距离更大，假设这种概率结构促进robust features。图表、显式正则对照支持NAT独特性，但未排除负KL带来的logit/梯度尺度等替代机制。
## 5. 核心贡献
新防御NAT、概率稀疏解释、跨架构/任务经验。核心是辅助模型负KL；Taylor/几何解释可移除而算法仍成立。方法继承PGD-AT与Nasty Training，属于新组合及扩展。
## 6. 方法总览
```text
预先训练辅助adversary model f_a（默认同结构自然模型，冻结）
 -> target生成PGD adversarial x' -> CE(target(x'),y)
 -> 减去lambda*KL(target(x') || adversary(x'))
 -> 只更新target；推理丢弃辅助模型
```
## 7. 方法细节
`theta_t`更新、`theta_a`固定；温度`tau_a`控制KL分布，`omega_a/lambda`控制排斥。辅助模型可换架构/状态/ensemble。随机性来自初始化、batch和PGD。额外模型仅训练期使用，但其预训练成本未计入每epoch表。
## 8. 关键公式
式(5)（PDF第4页）总目标为AT交叉熵减加权KL，内层仍最大化目标模型攻击loss；负号使target远离辅助输出。式(6)(7) Taylor展开声称高阶项诱导概率稀疏，依赖局部近似，不能推出全局鲁棒。自适应攻击式(14，PDF第16页)直接最大化完整NAT目标。`lambda`过小退化AT，过大可能牺牲AA/稳定性；消融最佳约0.06。
## 9. 算法伪代码
```text
train/load frozen auxiliary f_a; initialize target f_t
for batch: PGD inner max on f_t; compute CE(f_t(x_adv),y)-lambda KL(f_t(x_adv),f_a(x_adv)); update theta_t
return f_t; ordinary inference
```
## 10. 理论分析
无正式鲁棒保证。Taylor展开、head weight gap、超平面距离和可视化为机制解释，多属相关性。证明概率稀疏因果性需要匹配logit norm/entropy的干预和中介分析。
## 11. 实验设置
PGD训练，CIFAR WRN34-10/RN18，ImageNet100 ViT-small；SGD momentum .9、WD`5e-4`、300 epoch、LR .1在160/240降；batch 512/128；三次独立运行，表给最佳值且括号均值±std。评估PGD10–100、C&W、AA、迁移/Square和自适应APGD。
## 12. 核心实验结果
| C10 RN18 | clean | PGD100 | AA | 来源 |
|---|---:|---:|---:|---|
| PGD-AT | 84.25 | 44.76 | 41.69 | 表2，PDF第8页 |
| AGAIN-AWP | 86.52 | 58.85 | 51.89 | 表2 |
| NAT best | 90.86 | 59.91 | 50.18 | 表2 |
| NAT last | 90.28 | 58.89 | 48.96 | 表2 |
NAT兼顾clean/PGD，但AA低于AGAIN-AWP 1.71pp。ImageNet100 AA 37.12→45.44（表4，第15页）。
## 13. 消融实验
lambda最佳.06；表7/8换辅助架构/参数状态仍有效但AA与PGD权衡；表9 RN18每epochAT108.6s、NAT129.8s且不含辅助预训练；表11 entropy/negative-norm/LASSO/mixup不及NAT。消融不能单独证明“自适应类相似概率”机制。
## 14. 威胁模型与评估可信度
白盒`l_inf`攻击完整target；推理无辅助/随机/不可微，BPDA/EOT不适用。AA、长PGD、Square/迁移与针对负KL的自适应APGD均有；APGD步数增加准确率合理下降。**可信度：高**，但best-of-three及AA不总领先需诚实解读。
## 15. 可复现性
目标、训练日程、硬件、三次运行和攻击较全；辅助模型预训练recipe、代码URL与若干温度细节仍是障碍。**可复现性：中高。**
## 16. 局限与失败模式
需额外模型和约20%训练时延，未计预训练成本；辅助错误偏差可能传递；大lambda造成过排斥。作者跨任务扩展较浅。概率稀疏不等于参数稀疏，也不足以单独提供鲁棒表示。
## 17. 批判性评价
优点是攻击评估完整、clean/PGD收益强；缺点是机制因果不足和选择best。应补同均值统计的最强baseline复现、logit-scale matched干预、总GPUh/显存与辅助预训练摊销。可信结论是负KL正则可增强若干AT配置，不是全面SOTA。
## 18. 相关工作定位
连接PGD-AT、Nasty Training、robust distillation、entropy/logit regularization、AWP和EDM增强；正文比较AGAIN、LAS-AT等。关键词：`negative KL adversarial training`、`probability sparsity robustness`。
## 19. 阅读后的关键启发
辅助模型可作为“应避免的偏差方向”，蒸馏不必总是模仿；多攻击指标会揭示PGD与AA排序差异。
## 20. 尚未解决的问题
何种辅助分布最优？稀疏是原因还是结果？自然模型预训练成本如何公平计入？对校准/OOD有何影响？
## 21. 证据索引
| 结论 | PDF页 | 位置 | 证据 |
|---|---:|---|---|
| 负KL目标 | 4 | 式(5) | target远离冻结辅助输出 |
| 主结果/统计 | 7–8 | 表1,2 | best及三次mean±std |
| 自适应攻击 | 16 | 式(14),表6 | 直接攻击完整目标 |
| 成本 | 18 | 表9 | RN18 108.6→129.8s/epoch |

### 最终评分
问题13/15；新颖性15/20；技术15/20；实验18/20；可信度14/15；复现8/10；**83/100**。阅读优先级：建议精读；适合AT正则/蒸馏研究者。全文附录已读，表2乱码已按行列核对。
