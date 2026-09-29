# Pick Mango 资产推理验证（2026-09-29）

本文件保留资产验证证据；后续环境和 Dobot 软件通路结果见 [GPU 环境验证](GPU_ENV_VALIDATION_STATUS.md)，安装及启动见仓库 [README](../../../README.md#installation)。验证用的一次性 `scripts/check_*.py` 已清理，原始结果仅保留在验证机器的 `outputs/pick_mango_asset_check/`，不随仓库发布。

结论：ctx10 / ctx20 的当前三件套均可被 dsrl_pi0 shaped reward 通路加载并完成图像到 reward 的计算。按用户指定，正式默认 OOD 阈值为 **0.5**。早期使用 0.05 的探索结果不作为本报告结论。

## 修复

- dsrl_pi0 环境原来缺少 torchvision；已安装 `torchvision==0.21.0+cpu`，匹配已有 `torch==2.6.0+cpu`。
- `shaped_reward/encoder.py` 对齐 FineProg 的 channels_last / channels_last_3d 卷积布局，消除同输入因布局不同导致的数值差异。未修改模型结构、权重或资产。
- `shaped_reward/dobot.py` 和 `examples/launch_train_real_dobot.py` 的 OOD 默认阈值统一为 0.5；已验证 CLI 默认参数。

## 测试覆盖

使用真实 `DobotShapedReward.apply` 和 `DobotDataWrapper.prepare_progress_frame`，无 encoder mock，无机器人动作或远端策略请求。

- ctx10 / ctx20，query_freq=10；历史上下文分别取前 1 / 2 个 query frame。
- 每路抽取 valid、fail、真实 rollout 数据集的首条视频，最多前 200 帧，按 10 帧间隔查询。H5 中真实的 224×224 crop 被放回人工 640×480 画布，以验证 wrapper 裁剪和时序通路。
- 另用原始 topFullImg JPEG 的 8 个 query frame 验证真实完整图像裁剪。
- 增加固定随机噪声序列；默认 0.5 下进行实际 OOD 判定。
- 单独用仅测试用阈值 ctx10=0.35、ctx20=0.60 覆盖 ID→OOD→ID 的保持和恢复，未改动正式默认值。
- 每条序列均测试 dense/PBRS × success/failure 四种组合，最后一个动作块仅 3 个控制步；验证 discount**实际步数、terminal mask、成功奖励、episode reset、有限值和 float32 reward 输出。
- 重新验证六个资产文件的 SHA-256。

## 结果

| 环境 | 集成用例 | 与同设备 FineProg encoder 最大误差 | progress 与独立 NumPy 最大误差 |
| --- | ---: | ---: | ---: |
| dsrl_pi0 / torch 2.6.0+cpu | 48 / 48 通过 | 0 | 2.96e-8 |
| fineprog / torch 2.11.0+cu128 / RTX 5090 | 48 / 48 通过 | 0 | 2.91e-8 |
| dsrl_pi0_reward_cuda / torch 2.7.1+cu128 / RTX 5090 | 48 / 48 通过 | 0 | 2.98e-8 |

第二轮 GPU 测试使用 dsrl_pi0 的代码和 assets，Python/CUDA 运行环境来自 fineprog。后续独立训练环境直接运行同一份 dsrl_pi0 代码；对应结果为 `isolated_cuda.json`。

默认阈值 0.5 下的 OOD 帧数（CPU/GPU 一致）：

| 输入 | ctx10 | ctx20 |
| --- | ---: | ---: |
| valid 首条，15 个 query frame | 3 | 1 |
| fail 首条，14 个 query frame | 5 | 6 |
| rollout 首条，20 个 query frame | 0 | 9 |
| 原始 topFullImg，8 个 query frame | 3 | 0 |
| 混合随机噪声序列，8 个 query frame | 7 | 5 |

FineProg 已保存的 embedding 来自离线 CUDA 批量提取，在线单帧推理不要求逐位相等；本次最大逐元素差异约 0.00261，各比较序列的最小余弦相似度均大于 0.9999。独立构造上下文后，同设备 FineProg 原始 encoder 与 DSRL 在线输出完全一致，排除了上下文错配。

第二轮已有回归测试为 **23 passed**。后续加入训练入口接线测试后，原环境和训练副本均 **31 passed、1 skipped**；LeRobot 模块在独立服务环境 CPU/GPU 各 **6 passed**。`git diff --check` 通过。

完整逐例结果：仓库根目录 `outputs/pick_mango_asset_check/cpu.json`、`cuda.json` 和 `isolated_cuda.json`。

## 复制前仓库核对

2026-09-29 已 fetch 两个项目的远端。dsrl_pi0 本地 `main` 与 `v373/main` 同为 `fc5e5e26424d406e1ec7c4f7603684f44b3f5557`，比 `origin/main` (`7f48937d4553e95244cd81c79236a3256df80597`) 多 7 个提交；复制前无 tracked 文件修改，仅有未跟踪的 `outputs/`。FineProg HEAD `e9f6844` 与 `origin/main` 一致，但已有未提交修改和未跟踪文件；这些本地产物不是远端已提交资产。

## 使用边界

保留的原 dsrl_pi0 环境需使用 `reward_device=cpu`；新建的独立 dsrl_pi0_reward_cuda 环境可使用 `reward_device=cuda`。原启动脚本仍为 sparse，三个资产路径仍需按 README 配置；复制及验证资产不自动启用 shaped reward。原环境依赖没有被 CUDA 升级覆盖。

本次确认资产兼容性和离线 reward 数值计算，不代表已完成真机、策略服务器、SAC 学习效果或全数据集 OOD 质量评估。测试没有启动机器人、commit 或 push。
