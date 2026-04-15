# Docker 部署

LunaVox 提供一份 multi-stage `Dockerfile` + `compose.yml`，用户
不用自己装 CMake、不用本机编译 C++ 引擎就能一行起 HTTP/WebSocket
服务层。

> [!NOTE]
> 本 Dockerfile 当前构建的是 **纯 CPU 镜像** —— 捆绑的是
> `linux_cpu` 的 ONNX Runtime 和 llama.cpp 运行库。CUDA 镜像在路线
> 图上但还没出（欢迎贡献，改动面是 builder 阶段
> `lunavox build libs --platform` 换参数 + runtime 阶段基础镜像改
> 成 `nvidia/cuda`）。

## 前置条件

- Docker 24+（`Dockerfile` 用了 `# syntax=docker/dockerfile:1.7`）
- 首次构建至少 6 GB 空闲磁盘（builder 阶段要下 ONNX Runtime +
  llama.cpp 运行库，还要编译 C++ 引擎）
- 宿主机 `./models/` 下已经有一个拉好的模型（镜像本身**不**自己
  下模型 —— 体积太大不适合烧进镜像，而且部署时用户常常想现选
  模型变体）

## 1. 在宿主机拉模型

```bash
pip install lunavox
lunavox model pull --model base_small
```

完成后宿主机上会有 `./models/base_small/`，里面包含 `*.gguf`、
`*.onnx`、`tokenizer.json` 以及 `embeddings/` 目录。

## 2. 构建镜像

```bash
docker build -t lunavox:2.2.0 .
```

首次构建耗时约 8–15 分钟，主要花在 CMake 和 C++ 编译上。后续
重建会复用缓存 —— 仅 Python 改动约 1 分钟以内，触及 C++ 的改动
约 3 分钟。

## 3. 用 `docker compose` 起服务（推荐）

```bash
docker compose up
```

这会在 `http://localhost:8000` 启动服务，自动：
- 把 `./models/` 只读挂载到 `/app/models`
- 把 `./ref/` 只读挂载到 `/app/ref`
- 把 `./output/` 读写挂载到 `/app/output`
- 用 `--batch-size auto`（探测空闲显存；CPU 环境下回落到 4）
- 每 30 秒对 `/health` 做一次健康检查

通过环境变量覆盖端口或批大小：

```bash
LUNAVOX_PORT=9000 docker compose up
LUNAVOX_BATCH_SIZE=2 docker compose up
```

## 4. 不用 compose 直接 `docker run`

```bash
docker run --rm \
    -p 8000:8000 \
    -v "$(pwd)/models:/app/models:ro" \
    -v "$(pwd)/ref:/app/ref:ro" \
    -v "$(pwd)/output:/app/output" \
    lunavox:2.2.0
```

`lunavox serve` 的所有 flag 都能透传：

```bash
docker run --rm \
    -p 8000:8000 \
    -v "$(pwd)/models:/app/models:ro" \
    lunavox:2.2.0 \
    lunavox serve --host 0.0.0.0 --port 8000 --batch-size 2 --model base_small
```

## 镜像内部结构

Dockerfile 是两阶段构建：

**Stage 1 — builder**（`python:3.11-slim-bookworm`）
- 装 `cmake`、`ninja`、`g++`、`libgomp1`
- 把 LunaVox 仓库拷到 `/src`
- 跑 `lunavox build libs --platform linux_cpu` 拉 ONNX Runtime
  和 llama.cpp 二进制到 `/src/lib/`
- 跑 `lunavox build --clean` 编译 C++ 引擎，把 `liblunavox.so`
  和 `lunavox-cli` 编到 `/src/build/`

**Stage 2 — runtime**（`python:3.11-slim-bookworm`）
- 只装 `libgomp1`、`libstdc++6`、`dumb-init`
- 创建非 root 用户 `lunavox`（UID 10001）
- 从 PyPI 安装 `lunavox[serve]==2.2.0`
- 从 stage 1 拷 `/src/build/` 和 `/src/lib/` 过来
- 创建 `.lunavox-root` 部署布局标记文件，让
  `lunavox.core.project.resolve_project_root()` 认可 `/app` 为
  合法 root（因为容器里没有 `CMakeLists.txt` / `src/`）
- 设置 `LUNAVOX_PROJECT_ROOT=/app`、
  `LUNAVOX_LIB_PATH=/app/build/liblunavox.so`
- `EXPOSE 8000`，默认 `CMD` 跑 `lunavox serve`

最终镜像约 500 MB —— 基础 Python 镜像 150 MB、pip 依赖（uvicorn、
fastapi、pydantic、numpy、prometheus-client、typer、rich、
huggingface-hub）约 200 MB、编译出的 C++ 引擎加 ONNX Runtime 和
llama.cpp 二进制约 150 MB。

## 生产注意事项

- **非 root 用户**。容器以 UID 10001 运行，宿主机 bind mount 要
  保证目录能被 10001 或匹配的 GID 读写。
- **健康检查**。`compose.yml` 每 30 秒对 `GET /health` 做一次
  健康探针；Kubernetes 用户可以直接用同一个端点作为 liveness /
  readiness 探针。
- **Prometheus 抓取**。`GET /metrics` 在同端口可用，Prometheus
  配 `http://<container>:8000/metrics` 就行。
- **信号处理**。`dumb-init` 把 `docker stop` 送来的 `SIGTERM`
  转给 uvicorn，正在进行的合成请求有机会完成后再退出。
- **batch size 权衡**。`--batch-size auto` 在没有 pynvml 的 CPU
  环境下回落到 4。每个 pool 槽位各自持有 KV cache 和 ONNX
  decoder 状态，所以 0.6B 模型 `batch_size=4` 约 800 MB 额外
  RAM，基础引擎本身还要 ~1.5 GB。内存紧张的部署建议设
  `--batch-size 1`。

## 相关文档

- [服务层指南](serve.md) —— HTTP/WebSocket 端点完整参考
- [CLI 参考](cli_reference.md) —— `lunavox serve` 支持的所有
  flag
