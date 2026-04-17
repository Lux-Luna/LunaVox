# Docker 部署

LunaVox 提供一份 multi-stage `Dockerfile` + `compose.yml`，宿主机不用
装 CMake、不用编译 C++ 引擎就能一行起 HTTP/WebSocket 服务层。

当前镜像是**纯 CPU 镜像**（捆绑 `linux_cpu` 的 ONNX Runtime + llama.cpp
运行库）。CUDA 镜像在路线图上 —— builder 阶段
`lunavox build libs --platform` 换参数 + runtime 基础镜像改成
`nvidia/cuda` 即可。

安装：`pip install lunavox`（详见 [CLI 参考](cli_reference.md)）。

## 前置条件

- Docker 24+（`Dockerfile` 用了 `# syntax=docker/dockerfile:1.7`）
- 首次构建至少 6 GB 空闲磁盘
- 宿主机 `./models/` 下已有拉好的模型 —— 镜像本身**不**下模型

## 1. 在宿主机拉模型

```bash
pip install lunavox
lunavox model pull --model base_small
```

完成后 `./models/base_small/` 下应包含 `*.gguf`、`*.onnx`、
`tokenizer.json` 以及 `embeddings/`。

## 2. 构建镜像

```bash
docker build -t lunavox:2.2.0 .
```

首次构建耗时 ~8–15 分钟，主要花在 CMake 和 C++ 编译。缓存命中后：
仅 Python 改动 < 1 分钟，触及 C++ 约 3 分钟。

## 3. 用 `docker compose` 起服务（推荐）

```bash
docker compose up
```

在 `http://localhost:8000` 启动服务，自动：
- 把 `./models/` 只读挂载到 `/app/models`
- 把 `./ref/` 只读挂载到 `/app/ref`
- 把 `./output/` 读写挂载到 `/app/output`
- 用 `--batch-size auto`（探测空闲显存；CPU 环境回落到 4）
- 每 30 秒对 `/health` 做一次健康检查

通过环境变量覆盖：

```bash
LUNAVOX_PORT=9000 docker compose up
LUNAVOX_BATCH_SIZE=2 docker compose up
```

## 4. 不用 compose 直接 `docker run`

```bash
# 基本用法
docker run --rm \
    -p 8000:8000 \
    -v "$(pwd)/models:/app/models:ro" \
    -v "$(pwd)/ref:/app/ref:ro" \
    -v "$(pwd)/output:/app/output" \
    lunavox:2.2.0

# 透传任意 lunavox serve flag
docker run --rm \
    -p 8000:8000 \
    -v "$(pwd)/models:/app/models:ro" \
    lunavox:2.2.0 \
    lunavox serve --host 0.0.0.0 --port 8000 --batch-size 2 --model base_small
```

## 镜像内部结构

两阶段构建：

**Stage 1 — builder**（`python:3.11-slim-bookworm`）。装 `cmake` /
`ninja` / `g++` / `libgomp1`，把仓库拷到 `/src`，跑
`lunavox build libs --platform linux_cpu` 拉运行库，然后
`lunavox build --clean` 把 `liblunavox.so` 和 `lunavox-cli` 编到
`/src/build/`。

**Stage 2 — runtime**（`python:3.11-slim-bookworm`）。只装 `libgomp1`
/ `libstdc++6` / `dumb-init`，创建非 root 用户 `lunavox`（UID 10001），
从 PyPI 安装 `lunavox==2.2.0`，从 stage 1 拷 `/src/build/` 和
`/src/lib/` 过来。写 `.lunavox-root` 部署布局标记，让
`lunavox.core.project.resolve_project_root()` 认可 `/app` 为合法 root
（容器里没有 `CMakeLists.txt` / `src/`）。设置
`LUNAVOX_PROJECT_ROOT=/app` 和
`LUNAVOX_LIB_PATH=/app/build/liblunavox.so`；`EXPOSE 8000`，默认
`CMD` 跑 `lunavox serve`。

最终镜像约 500 MB：基础 Python 镜像 ~150 MB、pip 依赖（uvicorn、
fastapi、pydantic、numpy、prometheus-client、typer、rich、
huggingface-hub）~200 MB、C++ 引擎 + 运行库 ~150 MB。

## 生产注意事项

- **非 root 用户**。UID 10001 运行，bind mount 目录须被该 UID 或
  匹配 GID 可读写。
- **健康检查**。`compose.yml` 每 30 秒对 `GET /health` 探针；
  Kubernetes 直接用同一端点作为 liveness / readiness。
- **Prometheus 抓取**。`GET /metrics` 在同端口，Prometheus 配
  `http://<container>:8000/metrics`。
- **信号处理**。`dumb-init` 把 `SIGTERM` 转给 uvicorn，正在进行
  的合成请求有机会完成后再退出。
- **batch size**。VRAM / 吞吐取舍详见
  [服务层指南 —— 并发模型](serve.md#并发模型)。内存紧张时设
  `--batch-size 1`。
