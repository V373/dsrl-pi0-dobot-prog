# Pick Mango 推理资产

2026-09-29 从本地 FineProg 复制。安装和 PBRS 启动命令见仓库 [README](../../../README.md#training-real-dobot)。

`.pt` / `.h5` 文件由 Git LFS 管理，共约 235 MiB。克隆后按根目录 README 安装环境并执行 `git lfs pull`，确认下载到完整文件后再启动训练。

ctx10 / ctx20 指 **context_stride**，不是 context_size；两路均为 context_size=2、epoch 20000。每路必须使用同一目录的三个文件：

- `encoder_epoch020000.pt`：encoder checkpoint。
- `calibration_embeddings.h5`：同一 encoder 提取的 18 条 calibration 视频 embedding。
- `gaussian_progress_model.h5`：同一 encoder 的 36 条 train 视频 embedding 拟合的 Gaussian 模型。

| 目录 | FineProg run 后缀 | Gaussian 拟合时间 | reward_context_stride |
| --- | --- | --- | ---: |
| `ctx10` | `20260925-050342-ctxstride10` | `20260929-160153` | 10 |
| `ctx20` | `20260925-090936-ctxstride20` | `20260929-163251` | 20 |

每个目录的 `manifest.json` 记录完整源路径、大小和 SHA-256；六个资产文件已验证复制前后哈希一致。共同参数：L2 normalization、128 维 embedding、PCA 32 维、20 bins、independent covariance。H5 中的源路径无需存在于部署机器上；Gaussian 文件的 `enable_calibration=false` 表示运行时从单独的 calibration H5 构建校准分布。

`external_rgb` 必须是原始 **480×640 RGB uint8 topFullImg**；wrapper 使用 `[168:392, 256:480]` 裁剪为 224×224。`query_freq=10` 时，ctx10 / ctx20 分别间隔 1 / 2 个 query frame。OOD 阈值默认为 **0.5**。放入 assets 不会自动启用 shaped reward，启动时需指定 `pbrs` 或 `dense` 及三个文件路径。

当前资产已通过 CPU/GPU 推理和模拟 Dobot 软件通路验证。保留的审计记录见 [资产推理验证](INFERENCE_TEST_REPORT.md) 与 [GPU 环境验证](GPU_ENV_VALIDATION_STATUS.md)；不代表已完成真机或策略效果验证。
