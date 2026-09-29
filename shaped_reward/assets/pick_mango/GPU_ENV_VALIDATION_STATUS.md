# GPU 环境验证已完成（2026-09-29）

安装和启动命令见仓库 [README](../../../README.md#installation)。下列为 2026-09-29 的验证记录；原始结果仅保留在验证机器的 `outputs/pick_mango_asset_check/`，不随仓库发布，一次性测试脚本已清理。

- 原 `conda dsrl_pi0` 的 254 行包清单与 CUDA 升级前快照一致，未覆盖升级。
- 独立训练环境 `/home/user/zhangzk/envs/dsrl_pi0_reward_cuda`：ctx10/20 GPU reward 48 用例通过；实际 JAX GPU SAC 更新通过（包括 batch=256、模拟 prefix=2048）。
- 独立服务环境 `/home/user/zhangzk/envs/dobot_pi0_service`：LeRobot 0.6.1，CPU/GPU 各 6 个真实 API/processor/checkpoint/WebSocket 测试通过。
- 两个 Dobot 入口已串联验证：各 43 个模拟控制步 → 真实 LeRobot WebSocket → 真实 CUDA asset reward → 真实 replay → 首个真实 SAC GPU update。
- 原环境和训练副本回归均 31 passed、1 skipped；LeRobot 模块在服务环境单独验证。
- 官方 OpenPI 在原环境和副本均可推理；prefix 一致，动作存在最大约 0.00781 的数值差异，所以不承诺依赖升级后逐位一致，也不替换原环境。
- 默认 OOD 阈值 0.5；启动脚本仍 sparse，未自动启用 shaped reward。
- 测试服务已停止；没有启动机器人、commit 或 push。

真实 Dobot checkpoint 与 SDK 尚未提供。测试 π₀ 是缩小维度的随机模型，只验证软件通路，不能用于机器人控制；完整模型并发显存、长期稳定性和策略效果仍需实际系统验证。

证据文件：`isolated_cuda.json`、`isolated_sac_reward_gpu_batch256.json`、`full_integration_ctx10.json`、`full_integration_ctx20.json`、`environment_validation.json`、`openpi_environment_comparison.json`；各环境的包清单和测试日志也在上述 outputs 目录中。
