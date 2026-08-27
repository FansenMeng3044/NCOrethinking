# Set-XML100 正式评测审计

## 用户指出的五项缺口

| 缺口 | 处理结果 |
|---|---|
| 仓库未附带 10,000 实例和最优值 | 从 CVRPLIB 官方 `XML.7z` 下载；固定 SHA-256；校验 10,000 个 `.vrp`、10,000 个 `.sol`、378 组及 26/27 的组规模。 |
| 没有 XML100 专用汇总 | `summarize_xml100.py` 输出总体、4 类属性、378 组、增强前后、尾部 Gap、置信区间及配对检验。 |
| 只有 no-aug/aug 距离，没有路线、复验和 Gap | 每个模型—实例写原子 JSON，保存 `routes` 与 `no_aug_routes`；独立整数复验；`.sol Cost` 与 `OptimalCosts.ods` 双源锁定最优值并计算 Gap。 |
| 距离取整不够严格 | 正式结果全部按每边 `EUC_2D = floor(sqrt(dx²+dy²)+0.5)` 后求和。需纠正：固定版本 MVMoE 的 CVRPEnv 在 `loc_scaler=1000` 时也已逐边 `torch.round`，并非只在总和后取整；本工具的改进是整数精确实现和可审计复验。 |
| MVMoE checkpoint 与本地模型不兼容 | 不加载 MVMoE 权重，只移植 greedy、8-fold、POMO-size=n 和固定 1000 缩放协议；分别严格加载本地 POMO、POMO-Split、AM、AM-Split。正式运行前四类模型全部预加载验型。 |

## 额外发现并处理的问题

1. **官方参考路线文本并非全部可复验。** 固定哈希的官方包中有 91 个 `.sol` 路线文本存在重复/缺失客户；全部 10,000 个 `Cost` 仍与 `OptimalCosts.ods` 完全一致。异常逐项写入 `xml100_reference_route_anomalies.json`，官方文件不做修补，模型路线也不享受豁免。
2. **Direct 动作序列的隐含仓库。** AM 等解码器可能省略开头/结尾仓库；原始序列若直接 `roll`，会错误加入“末客户→首客户”。现显式按 `depot -> actions -> depot` 计分，并有回归测试。
3. **增强前结果不能由增强后结果反推。** 分别在 augmentation index 0 的候选池和完整 8-fold 候选池中按同一官方目标选择并保存路线。
4. **测试集泄漏。** 10,000 个公开实例、特征分组、最优值和结果只能用于最终一次测试，不得用于训练、验证、checkpoint/超参数/模型选择。生成器可用于另建训练与验证集。
5. **比较预算不相同。** POMO 的 8-fold 候选数为 800，greedy AM 为 8；必须同时报告候选数和耗时，不能把质量差异全归因于网络结构。
6. **Split 是额外算法。** Split 结果包括严格整数目标下的精确 Bellman 容量分段；Direct/Split 比较是完整算法比较，不是纯 encoder 消融。
7. **失败不能静默删除。** `no_feasible_candidate` 保留为空路线结果；汇总首先报告成功率。成功率不足 100% 时，平均 Gap 明确是条件均值，不能借少量成功子集获得优势。
8. **配对统计的样本定义。** 增强前后各自输出 wins/ties/losses、Wilcoxon 双侧检验与 Holm 校正，并记录共同成功实例数。组级置信区间使用 Student-t；无 SciPy 才退化为正态近似并标注。
9. **可复现身份。** 正式运行锁定数据哈希、最优值表哈希、checkpoint SHA-256/epoch、模型架构、源码 branch/commit/clean 状态、评测脚本哈希、PyTorch/CUDA/GPU、batch size、CUDA 可见设备和确定性设置。
10. **恢复运行污染。** `--resume` 会核对实例集合、模型配置、checkpoint、代码、数据、硬件和 batch 身份，并逐个检查已有 JSON；混入额外/损坏结果会失败，不会跳过。
11. **车辆数不是 XML100 目标。** XML100 是距离最小化 CVRP，不设置 Solomon 式最大车辆数，也不做车辆数优先；车辆数只作为描述指标。
12. **均值原始距离不能跨异质组排序。** 正式主质量指标是逐实例 Gap；原始距离均值只作描述。另给等权 378 组宏平均和尾部 Gap。
13. **计时边界。** 路线后处理计入 Split/复验相关时间；模型 forward 时间在同步后计量；模型预加载、整次 wall time、硬件和 batch 均写 manifest。跨硬件的秒数不能直接比较。
14. **严格输入解析。** 强制 `TYPE=CVRP`、`DIMENSION=101`、`EDGE_WEIGHT_TYPE=EUC_2D`、depot ID=1、连续节点 ID、坐标 `[0,1000]`、正需求和容量合法；不依赖文件顺序的隐含假设。
15. **“官方”边界。** CVRPLIB 官方规定的是数据、CVRP 约束、EUC_2D 目标和最优值，不规定神经网络必须 greedy/8-fold/POMO-size=n；后者是移植的 MVMoE 方法协议，论文中必须单独命名。

## 论文中至少应报告

- checkpoint 的训练数据分布、训练预算、epoch 与 SHA-256；
- greedy、augmentation=1/8、POMO size、候选数以及禁止使用的搜索/repair/TTO；
- 成功数/10,000、均值/中位数/P95/P99/最大 Gap、最优命中数；
- 3×3×7×6 属性/组汇总和配对检验；
- Direct 与 Split 的完整算法定义；
- GPU/CPU、PyTorch/CUDA、batch size、forward 与后处理/总 wall time；
- 使用逐边整数 EUC_2D，以及官方最优值双源交叉确认；
- 官方参考路线文本的 91 项异常和本工具不修改官方文件的处理原则。
