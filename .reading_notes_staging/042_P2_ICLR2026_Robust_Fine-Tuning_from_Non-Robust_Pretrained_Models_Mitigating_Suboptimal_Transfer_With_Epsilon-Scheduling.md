# Robust Fine-Tuning from Non-Robust Pretrained Models: Mitigating Suboptimal Transfer With Epsilon-Scheduling

## 1. 基本信息
Jonas Ngnawe等；ICLR 2026；PDF 21页。六种非鲁棒backbone、五个细粒度数据集及Imagenette；代码/硬件/seed在论文中未完整报告，无新数据。
## 2. 一句话结论
从非鲁棒预训练模型直接用目标扰动预算微调会阻碍早期任务适配；先零预算适配、再线性升至目标预算可显著避免迁移失败，但优化的是区间平均鲁棒性而非保证目标预算处总是最优。
## 3. 研究问题
研究非鲁棒pretraining能否可靠转为下游鲁棒模型。现有固定epsilon RFT在困难任务/大预算会崩。例子：新员工尚未学会识别鸟种就立刻要求抵抗最难伪装，学习停滞；先学任务再逐步加难。边界为全参数视觉fine-tuning与`l_inf`攻击。
## 4. 研究动机
训练曲线显示早期robust objective妨碍task adaptation。作者假设epsilon curriculum可解耦适配与鲁棒化；跨30个model-task pair结果支持效果，但机制仍是优化相关证据。
## 5. 核心贡献
新经验现象“suboptimal transfer”、epsilon调度方法和Expected Robustness指标。核心是调度；指标可移除而方法仍成立。调度属简单curriculum组合，指标是对攻击半径AUC的规范化。
## 6. 方法总览
```text
non-robust pretrained model -> epsilon=0自然适配(T1)
 -> T1..T2线性升至epsilon_g -> 固定epsilon_g收尾
 -> final checkpoint；测试多个epsilon并算AUC
```
## 7. 方法细节
全模型更新，无教师/辅助分支。50 epoch默认`T1=12,T2=37`。训练APGD-7；推理普通前向，评估APGD-10/AA。随机性来自batch和攻击初始化。
## 8. 关键公式
调度式（PDF第3页）：`alpha(t)=0,(t-T1)/(T2-T1),1`分段，`epsilon(t)=alpha(t)epsilon_g`。增大T1利于适配但压缩鲁棒训练；增大T2使升温更慢。Expected Robustness（第4页）为`[0,epsilon_g]`准确率均匀积分除以区间长度，隐含各半径等概率的人工prior。
## 9. 算法伪代码
```text
load pretrained theta
for t=1..50: compute epsilon(t); APGD-7 inner max; minimize CE; update all theta
return final theta; evaluate accuracy curve and normalized AUC
```
## 10. 理论分析
无正式理论。训练动态与随机epsilon对照说明“逐步”而非仅混合半径重要，但不建立因果机制；需优化景观、梯度冲突和多seed轨迹实验。
## 11. 实验设置
ViT、Swin、ConvNeXt、RN50、CLIP-ViT/ConvNeXt；Aircraft、Caltech、Cars、CUB、Dogs；`4/255,8/255`。final checkpoint，作者称robust overfitting可忽略。30配置paired t-test是跨任务样本，不是seed不确定性；优化器、增强、硬件见附录，部分信息未报告。
## 12. 核心实验结果
| 设置 | fix→schedule | 变化 | 来源 |
|---|---:|---:|---|
| 4/255 Swin-Aircraft adv | 4.80→32.00 | +27.20pp | 表1，PDF第4页 |
| 8/255 Swin-Cars adv | 5.60→23.50 | +17.90pp | 表2，第6页 |
| 4/255 Swin-Aircraft AA | 3.30→31.40 | +28.10pp | 表6，附录 |
| 8/255 Swin-Cars AA | 2.40→22.90 | +20.50pp | 表6 |
8/255下30配置中28个target adversarial accuracy提高，30个Expected Robustness全提高；但robust pretrained backbone上有时牺牲目标epsilon鲁棒性换AUC。
## 13. 消融实验
表7显示T1/T2敏感，硬切换可崩；表8随机均匀epsilon不如顺序调度；表9跨pair显著。缺多seed方差、cosine/cyclic curriculum和相同AUC但不同终点对照。
## 14. 威胁模型与评估可信度
白盒`l_inf 4/255,8/255`，攻击完整确定性模型；AA仅少数关键配置，无BPDA/EOT需求。**可信度：中**：大幅结果由AA确认，但大多数表仅APGD-10、restart/seed不足。
## 15. 可复现性
核心调度完整，数据/模型公开；训练细节较多但seed、完整命令和部分checkpoint信息不足。**可复现性：中高。**
## 16. 局限与失败模式
依赖非鲁棒初始化、50 epoch和选定T1/T2；Expected Robustness会掩盖目标预算退化；只测视觉分类与`l_inf`。困难度、数据规模或预训练域改变时固定日程可能失效。
## 17. 批判性评价
优点是发现影响实际RFT的强失败模式；缺点是启发式日程和不充分seed。应补自适应梯度冲突调度、多seed、目标epsilon与AUC Pareto及更强AA全表。可信的是避免若干灾难性失败，不可信的是普遍改善worst-case鲁棒性。
## 18. 相关工作定位
位于robust fine-tuning、curriculum AT、pretraining transfer与robustness AUC。正文比较固定RFT、随机epsilon及robust pretrained transfer。后续关键词：`epsilon curriculum adversarial training`、`robust transfer failure`、`robustness curve AUC`。
## 19. 阅读后的关键启发
鲁棒目标的引入时机与强度和目标本身同等重要；跨半径AUC与单一worst-case回答不同问题。
## 20. 尚未解决的问题
能否根据梯度冲突自动选T1/T2？对部分参数微调、prompt/LoRA和更大预算是否成立？uniform threat prior是否合理？
## 21. 证据索引
| 结论 | PDF页 | 位置 | 证据 |
|---|---:|---|---|
| 分段调度 | 3 | 方法公式 | 0→线性→目标epsilon |
| 大规模迁移失败被修复 | 4,6 | 表1,2 | Swin细粒度任务大幅提升 |
| AA确认 | 附录 | 表6 | 4/255与8/255关键案例 |
| 日程敏感 | 附录 | 表7,8 | 硬切换/随机半径较差 |

### 最终评分
问题14/15；新颖性14/20；技术16/20；实验17/20；可信度12/15；复现8/10；**81/100**。阅读优先级：建议精读。适合鲁棒迁移研究者；前置APGD/迁移学习；后续curriculum AT、robust fine-tuning。已读全文/附录并核对主表。
