# 三篇指定论文的全文交叉审查

审查日期：2026-09-26。审查对象是用户指定的三份本地 PDF（均为各自 PDF 首页所示的 arXiv v1）；下文按全文的方法、数据、实验、结果、附录与局限交叉核对，摘要只用于确认论文身份。

| 论文 | 版本 / 篇幅 | 本项目中的用途 |
|---|---:|---|
| [From paper to benchmark: agentic, framework-based reproduction of under-specified methods in machine health intelligence](https://arxiv.org/abs/2605.28371) | arXiv:2605.28371v1，37 页 | 复现实验的假设记录、槽位映射与分层验证流程 |
| [Picid: A Modular Evaluation Infrastructure for Reproducible PHM Across Tasks and Domains](https://arxiv.org/abs/2605.28345) | arXiv:2605.28345v1，67 页 | PICID 数据、标签、分割、变换、窗口和评价契约 |
| [Towards Unified and Data-Efficient Prognostics and Health Management with Tabular Foundation Models](https://arxiv.org/abs/2606.05481) | arXiv:2606.05481v1，43 页 | TabPFN/TabDPT 的最近基线与本次复现协议 |

## 逐篇结论

### From paper to benchmark

论文研究“如何把未充分说明的 PHM 论文转换为共享框架中可执行、可比较的实验”，贡献在复现工作流，而不在新的 PHM 预测器。其流程为论文索引、概念/算法分析、六类框架槽位绑定、实现、静态/模型健全性/结果匹配验证和报告；每个未说明选择都要成为有证据或明确缺证的 assumption record。

实验覆盖 16 篇 PHM 论文。框架绑定完整率为 AiF 7/16、AiF-D 9/16、FCA 9/16；FCA 执行率 14/16。但“能运行”不等于“忠实复现”：LLM 评分与实际执行相互矛盾；基准表还出现例如 DePater N-CMAPSS 实现 nMAE 3137.38 及诊断任务映射错误。作者承认 judge 评分可靠性不足，部分论文规格和数据不足，并报告 FCA 的 token 成本很高。对本项目的直接启示是分开记录配置可组合、代码可运行、结果匹配三个结论，不能把测试通过或成功启动写成复现成功。

文中六个“槽位”的命名并非完全一致：§3.3 写 task/datasource/transform/sequencer/model/evaluator，实验/附录部分又将 split、model configuration、experiment 拆分为槽位。实现 PICID 时应以本仓库实际接口作映射表，不照搬名称后声称完全对应。

### PICID evaluation infrastructure

论文把 PHM 样本形式化为特征变换 (G)、目标变换与对齐 (H)、窗口构造 (S)，分别定义设备间整机分割与设备内时间分割，并规定所有可拟合的特征/目标统计只用训练分区。ICL 上下文只来自训练集；验证集选参；测试集只做最终评估。窗口级误差按窗口加权，per-unit 误差先按设备算再等权平均，二者是不同估计对象。PICID 的源码/YAML 是这些规则的执行面，不只是模型包装层。

论文对本次两个数据域提供了两个易混淆的结果：通用 PICID 论文的 XJTU-SY 采用 in-domain fold-1（9 train/3 val/3 test）；TFM-PHM 论文的复现实验则采用 PHMD split（8/3/4）。这两个方案不可混用。论文还明确解释了 N-CMAPSS scaler 的硬编码常量是 benchmark-defined fixed constants，不在当前训练分区拟合；所以它们不会在每次本地运行中由测试分区拟合，但常量最初由哪些设备/文件估计，论文与源码均未给出可追溯来源。

论文覆盖 13 个模型、12 个数据集，讨论了协议一致性带来的模型排名变化。作者明确列出边界：协议统一无法修复数据质量、标签噪声、域偏移或故障定义异质性；当前评测仍是固定离线监督协议，不覆盖流式部署、在线适应和不确定性决策。故它支撑严谨复现，但不证明真实维修预警有效。

### TFM-PHM benchmark

论文将设备记录切成监督窗口，序列模型读取窗口张量，TabPFN/TabDPT读取时间优先展平的表格行；基础模型参数冻结，通过带标签训练上下文作 ICL。上下文样本严格来自训练分区；fit-predict 对照也使用同一特征、分割、窗口和指标。论文报告 13 个模型、12 个 PHM 任务、五个 seed；fit-predict 的 context/stride 组合为 (1,1)、(5,1)、(10,5)、(20,5)、(50,50)，验证集选择。TabPFN 与 TabDPT 均使用 8 路集成；当前锁定的 TabPFN 2.2.1 默认 `n_estimators=8`，TabDPT 1.1.13 默认 8 个推理集成，代码 wrapper 与此吻合。

N-CMAPSS 的 NC-P 是 DS01/04/05/07 多源预后任务，每源 units 1–5 训练、6 验证、7–10 测试；NC-DS02 是另一独立协议（训练 2/5/10/16/20，验证 18，测试 11/14/15）。NC-P 采用非重叠 60 步聚合、RUL 乘 0.01、固定标准 scaler；TabDPT 在 NC-P 的归一化 MAE 最佳，但 NC-DS02 由 STF 最佳。XJTU-SY 的 PHMD split 是 8/3/4，目标是按总寿命归一化的健康指数；该任务的 LSTM 最佳（归一化 MAE 21.89±0.40），TabPFN 22.27±0.35，TabDPT 23.24±0.45。TFM 的跨域评价是每个数据集分别提供任务训练上下文，不是发动机预训练/适配后零样本预测轴承。

上下文抽样/data-efficiency 主实验仅在 PHME20、Unibo 和 MZVAV 做随机与分块比例实验；没有在 NC-P 或 XJTU-SY证明新的检索机制。XJTU 的时间窗口研究已有结果。TabPFN 概率输出只作定性展示，论文没有多时域故障预警、校准误差或固定误报率的评估。论文因此已经覆盖“通用 TFM + PHM tabularization + context subsampling”，这些不能再作为本项目的独立创新点。

## 对代码的裁决

| 核查项 | 论文 / 数据源证据 | PICID 副本证据 | 当前结论 |
|---|---|---|---|
| 聚合参数 | PICID/TMF YAML 声明 `aggregation: mean/last` | 原类仅收 `agg`，其余进入 `**kwargs` 后被静默忽略；旧代码实际总用 mean | 已修复：类显式支持旧 alias；正式配置审计会把声明解析成显式 `agg` |
| XJTU 原始目标 | uv.lock 固定 PHMD commit `512426b4cee87290fb2dfdd24918a438cb3f2f83`；reader 对每个 32,768 行 acquisition 赋反向 RUL `N−1,...,0` | datasource 取列 `rul`；旧参数名 `runtime_key` 容易误读成 elapsed time；HI 公式使用 `input/total_life` | `HI=RUL/N` 对应 `1−one_based_elapsed/N`；若 elapsed 使用零起点则差一个 acquisition interval `1/N`。数据审计确认各轴承 CSV 行数、寿命表与该标签公式一致；已改用显式 `rul_key` 并更正注释/测试 |
| XJTU 评价单位 | TFM 论文报告归一化 HI 与反归一化原工程单位；per-unit 先设备内计算，再跨测试设备平均 | HealthIndex inverse transform 将 HI 乘该设备表中 total life | 归一化 HI 指标与 inverse-RUL 分钟指标分别报告；后者只用于离线 benchmark 的工程单位还原，模型输入没有总寿命 |
| N-CMAPSS 缩放 | TFM 附录 C.2.3、PICID 附录 F.3 称固定标准 scaler；PICID 称常量 benchmark-defined、非当前 split 拟合 | `N_CMAPSSFeaturesScaler` / `N_CMAPSSDescriptorsScaler` 中硬编码 1457 常量 | 按标准固定 scaler 复现；统计量最初的单位/文件来源未披露，已新增按 NC-P 训练 units 1–5 流式计算并对照的 CPU 审计。TFM 附录 A.3 的“min-max”与详尽 schema/代码冲突，后两者一致，复现以详尽 schema+代码为准并报告冲突 |
| XJTU 分割 | TFM-PHM PHMD split 8/3/4；PICID infrastructure fold-1 9/3/3 | 两套配置均存在 | 主复现固定 TFM 论文 PHMD split；另一个 split 只作为独立验证，结果分别命名 |
| NC-P 复现入口 | NC-P 是四源多源预后 | 原 `fit_predict.sh` 清单只列 `concepts_n_cmapss` 和 DS02，没有 `concepts_n_cmapss_multi` | 用已有 `concepts_n_cmapss_multi/prognostics/*` 配置显式生成实验；不把单源/DS02冒称 NC-P |
| XGBoost 基线实现 | TFM-PHM 正文/附录明确将该基线描述为 XGBoost（正则化决策树梯度提升） | `xgboost_fit_predict` 指向 `FitPredictXGBoostWrapper`，但 wrapper 实际构造 `sklearn.ensemble.GradientBoostingRegressor/Classifier`，并非 `xgboost.XGBRegressor/XGBClassifier`；锁文件也没有 xgboost 依赖 | 发布代码路径只能称为 sklearn GradientBoosting 的 code-faithful control，不能声称复现论文的 XGBoost。正式论文对照需另行实现真实 XGBoost 并记录参数/调参规则；两者结果分开报告 |

## 对后续研究主张的边界

三篇论文共同支持的结论是：论文协议必须落到可执行配置，标签/窗口/设备划分决定了任务含义，TFM-PHM 的特征表格化与上下文抽样已经是 prior work。它们均不使用真实维修事件标签，也未验证多时域告警校准。因此，后续告警实验只能先作为基于公开 run-to-failure RUL 的派生预警任务；必须另查预警/survival/选择性风险文献并加入事件时间、设备级拆分、提前量与误报的强对照后，才能提出方法贡献。
