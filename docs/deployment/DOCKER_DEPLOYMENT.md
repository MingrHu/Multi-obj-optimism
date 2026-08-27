# Docker 构建、发布与部署

本方案将仓库中的 API、代理模型、优化、提取和 DEFORM 批处理 Python 代码统一打入一个
Linux 镜像。容器启动后自动启动 `mobo-api`，运行数据和日志通过 Docker volume 持久化。

> Linux 容器不包含也不能运行 Windows DEFORM 的 `DEF_PRE_64.exe` 和
> `DEF_ARM_CTL.COM`。批处理代码仍在镜像内，可执行采样、文本处理和 dry-run；真实求解需
> 由装有 DEFORM 的 Windows 主机或独立服务完成。

## 1. WSL2 本地构建与启动

在 Docker Desktop 中启用 Ubuntu 22.04 的 WSL Integration，然后进入 WSL 仓库目录：

```bash
docker version
docker compose version
docker compose up --build -d
docker compose ps
curl http://127.0.0.1:5000/health
```

健康检查预期返回：

```json
{"code":0,"data":{"service":"mobo-doe"},"message":"ok"}
```

首次构建需要下载 CPU 版 PyTorch、TensorFlow 等完整运行依赖，耗时和镜像体积会明显大于
普通 Flask 服务。后续依赖文件不变时会复用构建缓存。

常用管理命令：

```bash
docker compose logs -f mobo-api
docker compose restart mobo-api
docker compose down
docker compose down -v   # 同时删除任务数据和日志卷，仅在确认不再需要数据时执行
```

默认把宿主机 `5000` 映射到容器 `5000`。需要修改宿主机端口时：

```bash
MOBO_HTTP_PORT=8000 docker compose up -d
curl http://127.0.0.1:8000/health
```

完整 HTTP 流程可从宿主机仓库环境运行：

```bash
MOBO_API_URL=http://127.0.0.1:5000 python -m mobo.api.demo
```

## 2. 数据持久化

Compose 创建两个命名卷：

- `mobo-data` → `/app/data`：DOE、模型、优化结果和任务状态；
- `mobo-logs` → `/app/logs`：服务与业务日志。

重新构建或替换容器不会删除命名卷。查看实际卷名：

```bash
docker volume ls
docker inspect mobo-api
```

如果端上要求把数据保存到明确的宿主机目录，可把 `compose.yaml` 中的命名卷改为绑定挂载，
例如 `/srv/mobo/data:/app/data` 和 `/srv/mobo/logs:/app/logs`。

## 3. 发布镜像

为每次发布使用不可变版本标签：

```bash
MOBO_IMAGE_TAG=1.0.0 docker compose build
docker image inspect mobo-api:1.0.0
docker save mobo-api:1.0.0 -o mobo-api-1.0.0.tar
```

将 `mobo-api-1.0.0.tar` 和 `compose.yaml` 传到 CentOS Stream 8。服务器导入并启动：

```bash
docker load -i mobo-api-1.0.0.tar
MOBO_IMAGE_TAG=1.0.0 docker compose up -d --no-build
docker compose ps
curl http://127.0.0.1:5000/health
```

若服务器不使用 Compose：

```bash
docker volume create mobo-data
docker volume create mobo-logs
docker run -d \
  --name mobo-api \
  --restart unless-stopped \
  -p 5000:5000 \
  -v mobo-data:/app/data \
  -v mobo-logs:/app/logs \
  mobo-api:1.0.0
```

## 4. 更新与回滚

开发阶段修改代码后重新构建新标签，不在运行中的容器内修改或提交代码：

```bash
MOBO_IMAGE_TAG=1.0.1 docker compose build
docker save mobo-api:1.0.1 -o mobo-api-1.0.1.tar
```

服务器加载新版镜像后修改标签并重新创建容器，命名卷会继续复用。需要回滚时把
`MOBO_IMAGE_TAG` 改回旧标签再执行 `docker compose up -d --no-build`。

## 5. 运行模型

容器由 Gunicorn 承载 Flask 应用。当前后台训练、优化和停止控制使用进程内协调，因此固定
为一个 worker，并通过多个线程处理 HTTP 查询。容器应只运行一个副本；若将来需要横向扩容，
需先把后台任务和中止状态迁移到独立任务队列。容器进程使用非 root 的 `mobo` 用户运行。
