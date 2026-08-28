# Docker 构建、发布与部署

本文面向一台新 Linux 服务器，说明为什么需要克隆仓库、如何检查或安装 Docker、如何构建并
启动 `mobo-api`，以及宿主机和远程客户端如何通过 HTTP 调用容器服务。WSL2 仅是本地开发
环境的一种选择，不是服务器部署的前提。

## 1. 背景与总体流程

本仓库依赖 Python、TensorFlow、CPU 版 PyTorch、代理模型和多目标优化库。直接在服务器上
逐项安装容易产生 Python 版本、二进制库和依赖版本差异。Docker 将这些运行依赖和仓库代码
构建为同一个 Linux 镜像，使开发机、测试机和服务器运行相同的软件环境。

完整流程为：

```text
准备 Linux 宿主机
  → 克隆仓库（取得源码、Dockerfile、Compose 和脚本）
  → 检查/安装 Docker Engine 与 Compose v2
  → 从仓库构建 mobo-api 镜像
  → Compose 创建容器、网络和持久卷
  → Gunicorn 自动启动 HTTP API
  → 宿主机通过 127.0.0.1:5000 调用
  → 远程客户端通过 <服务器IP或域名>:5000 调用
```

镜像包括以下 Python 能力：

- DOE HTTP API；
- 代理模型训练、评价和推理；
- NSGA-II 和强化学习优化；
- 数据提取与 DEFORM 批处理代码。

Linux 容器不能执行 Windows DEFORM 的 `DEF_PRE_64.exe` 和 `DEF_ARM_CTL.COM`。真实 DEFORM
求解仍需由装有 DEFORM 的 Windows 主机或独立服务完成；这不影响采样、训练、推理、优化和
HTTP Demo。

## 2. 为什么必须先克隆仓库

当前采用“从源码构建镜像”的发布方式。克隆仓库不是为了在服务器的 Python 环境中直接运行
算法，而是为了向 Docker 提供以下构建和编排材料：

| 仓库内容 | 部署作用 |
|---|---|
| `src/mobo/` | 被复制进镜像的 API、算法和批处理源码 |
| `requirements/` | 镜像中安装的锁定运行依赖 |
| `deploy/docker/Dockerfile` | 定义 Python 基础镜像、依赖安装、非 root 用户和启动命令 |
| `deploy/docker/gunicorn.conf.py` | 定义容器内 HTTP 服务的进程模型 |
| `compose.yaml` | 定义端口、环境变量、持久卷和重启策略 |
| `.dockerignore` | 防止 `.git`、`.venv`、本地数据和日志进入镜像 |
| `scripts/` | 宿主机检查、Docker 安装和开发环境脚本 |

在新机器执行：

```bash
# CentOS 尚未安装 Git 时先执行
sudo dnf install -y git

git clone https://github.com/MingrHu/Multi-obj-optimism.git
cd Multi-obj-optimism
git checkout main
```

后续的 `docker compose build` 必须在仓库根目录执行，因为 `compose.yaml` 以当前仓库作为
Docker 构建上下文。若采用已经导出的镜像 tar 包，则服务器可以不保留完整源码，但仍建议
保留同版本的 `compose.yaml` 和部署文档用于运维。

## 3. 检查宿主机环境

### 3.1 建议资源

| 项目 | 要求或建议 |
|---|---|
| 操作系统 | 仍在维护的 64 位 Linux；CentOS 建议 Stream 9/10 |
| 架构 | 当前完整验证为 `x86_64` / `linux/amd64` |
| CPU | 建议至少 4 个逻辑核 |
| 内存 | 建议至少 8 GiB |
| 磁盘 | 首次构建建议至少预留 12 GiB |
| 网络 | 在线构建需访问 GitHub、Docker Hub、PyPI 和 PyTorch CPU wheel 源 |
| 端口 | 默认使用宿主机 TCP 5000 |

仓库提供只读检查脚本，不会安装软件、修改用户组或开放端口：

```bash
bash scripts/check_docker_host.sh
```

脚本检查操作系统、架构、CPU、内存、磁盘、Git、curl、Docker daemon、Compose v2 和
5000 端口。最终显示 `0 个失败` 才表示可以直接进入构建阶段。

也可以人工检查：

```bash
cat /etc/os-release
uname -m
docker version
docker compose version
docker info
```

必须使用 Compose v2，即命令形式为 `docker compose`，不是旧的 `docker-compose`。

### 3.2 CentOS Stream 8 的处理

最初提供的服务器截图是 CentOS Stream 8。该版本已于 2024-05-31 结束维护，不再获得安全
或功能更新；Docker 官方当前的 CentOS 安装说明只列出仍受支持的 CentOS Stream 9 和 10。
参考：[CentOS Stream 8 生命周期](https://www.centos.org/centos-linux/)、
[Docker CentOS 安装要求](https://docs.docker.com/engine/install/centos/)。

因此分两种情况处理：

- **服务器已经安装并能正常运行 Docker Engine 和 Compose v2**：可以执行检查脚本，在明确
  接受宿主机 EOL 风险的前提下部署本容器；容器内部环境不依赖 CentOS 用户空间。
- **服务器尚未安装 Docker**：不要在生产环境继续套用历史 CentOS 8 仓库命令。优先把宿主机
  迁移到 CentOS Stream 9/10、RHEL 受支持版本或其他仍受维护的 Linux，再安装 Docker；如果
  服务器不能迁移，应由运维团队提供其正式支持的容器运行时和安全维护方案。

仓库安装脚本会拒绝在 CentOS Stream 8 上自动安装，避免制造一个无法持续安全更新的生产
环境。

## 4. 安装 Docker Engine

### 4.1 CentOS Stream 9/10 自动安装

在确认脚本内容符合服务器运维规范后执行：

```bash
bash scripts/install_docker_centos.sh --add-current-user
```

脚本按照 Docker 官方 RPM 仓库方式安装：

- `docker-ce` 和 Docker CLI；
- `containerd.io`；
- Docker Buildx；
- Docker Compose v2 插件；
- 启动 Docker daemon 并设置开机自启；
- 运行 `hello-world` 验证；
- 可选地把当前登录用户加入 `docker` 组。

加入 `docker` 组后必须注销并重新登录。Docker 官方提醒该用户组具备接近 root 的宿主机控制
能力（参见 [Linux 安装后配置](https://docs.docker.com/engine/install/linux-postinstall/)）；
不允许普通用户使用 Docker 时，可以不传 `--add-current-user`，后续命令统一由管理员使用
`sudo docker ...` 执行。

脚本发现可能冲突的旧 Docker 软件包时只会停止并报告，不会自动卸载，避免影响服务器上已有
容器和数据。

### 4.2 CentOS Stream 9/10 人工安装

与脚本等价的官方仓库命令为：

```bash
sudo dnf -y install dnf-plugins-core git curl
sudo dnf config-manager --add-repo \
  https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf -y install \
  docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker run --rm hello-world
sudo docker compose version
```

生产服务器建议使用 RPM 仓库方式，便于后续通过包管理器升级。Docker 官方的
`get.docker.com` 便利脚本主要面向开发和测试环境，不建议直接作为生产部署方案。

安装后重新执行：

```bash
bash scripts/check_docker_host.sh
```

## 5. 构建并启动服务

### 5.1 在线源码构建

在仓库根目录执行：

```bash
docker compose build mobo-api
docker compose up -d --no-build
docker compose ps
```

也可以合并为：

```bash
docker compose up --build -d
```

首次构建会下载 Python 3.12 基础镜像、CPU PyTorch、TensorFlow 和其余锁定依赖，耗时较长，
最终镜像约 3.5 GB。Dockerfile 使用 BuildKit pip 缓存；依赖不变时，后续代码构建会直接复用
缓存。

Compose 启动时会：

1. 创建 `mobo-api` 容器；
2. 创建数据卷和日志卷；
3. 把宿主机 `5000` 映射到容器 `5000`；
4. 由 Gunicorn 自动启动 Flask API；
5. 配置容器异常退出和宿主机重启后的自动恢复；
6. 持续执行 `/health` 健康检查。

查看状态和日志：

```bash
docker compose ps
docker compose logs -f mobo-api
```

状态显示 `healthy` 后才开始发送业务请求。

如果 5000 端口已被占用：

```bash
MOBO_HTTP_PORT=8000 docker compose up -d
curl http://127.0.0.1:8000/health
```

### 5.2 离线服务器部署

若 CentOS 服务器不能访问外网，应在可联网且架构相同的机器构建镜像：

```bash
MOBO_IMAGE_TAG=1.1.0 docker compose build mobo-api
docker save mobo-api:1.1.0 -o mobo-api-1.1.0.tar
```

将镜像 tar、`compose.yaml` 和仓库中 `deploy/` 目录传到服务器，然后执行：

```bash
docker load -i mobo-api-1.1.0.tar
MOBO_IMAGE_TAG=1.1.0 docker compose up -d --no-build
```

使用 Compose 启动时仍建议保留完整仓库，以保证 Compose 路径、版本说明和运维脚本完全一致；
`--no-build` 会确保离线服务器只使用已经导入的镜像。

## 6. 宿主机如何发送 HTTP 请求

Compose 的端口映射为：

```text
宿主机 0.0.0.0:5000  →  容器 mobo-api:5000
```

因此在同一台 CentOS 宿主机上不需要安装服务发现组件，也不需要知道容器 IP，直接请求
`127.0.0.1:5000`：

```bash
curl http://127.0.0.1:5000/health
```

预期返回：

```json
{"code":0,"data":{"service":"mobo-doe"},"message":"ok"}
```

创建一个 DOE：

```bash
curl -X POST http://127.0.0.1:5000/api/v1/doe/add \
  -H 'Content-Type: application/json' \
  -d '{"id":"doe_test_001","name":"Docker测试"}'
```

查询 DOE：

```bash
curl http://127.0.0.1:5000/api/v1/doe/list
```

端上完整接口、参数和响应以 [DOE_HTTP_API.md](../api/DOE_HTTP_API.md) 为准。

## 7. 运行仓库完整 Demo

推荐直接从宿主机命令行让容器运行仓库自带 Demo，这样宿主机不需要另外安装 Python 和机器
学习依赖：

```bash
docker exec \
  -e MOBO_API_URL=http://127.0.0.1:5000 \
  mobo-api \
  python -m mobo.api.demo
```

该 Demo 会实际调用正在运行的 API，依次执行 DOE 创建、LHS 采样、按字段取数、训练数据生成、
随机森林训练与评价、推理、NSGA-II 优化和优化结果字段查询。

若必须让 Demo 的 Python 客户端进程运行在 CentOS 宿主机，而不是容器内，则先安装仓库环境：

```bash
bash scripts/setup_env.sh
MOBO_API_URL=http://127.0.0.1:5000 .venv/bin/python -m mobo.api.demo
```

这会在宿主机额外安装完整 Python 依赖，通常没有必要；日常连通性验证使用 `curl`，完整流程
验证使用 `docker exec` 即可。

## 8. 远程客户端如何调用

远程客户端不需要发现容器，只需要知道宿主机 IP 或域名及公开端口：

```text
http://<CentOS服务器IP>:5000/health
```

如服务器使用 firewalld，并且确实要允许可信内网直接访问：

```bash
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --reload
```

然后在客户端执行：

```bash
curl http://<CentOS服务器IP>:5000/health
```

当前 API 本身没有 TLS 和身份认证，不应把 5000 端口直接暴露到公网。正式环境建议由现有
Nginx、网关或端上服务提供 HTTPS、认证和访问控制，再反向代理到
`http://127.0.0.1:5000`。

不同调用位置使用的地址如下：

| 调用位置 | 地址 | 是否需要服务发现 |
|---|---|---|
| 同一宿主机 | `http://127.0.0.1:5000` | 不需要 |
| 远程端上客户端 | `http://<宿主机IP或域名>:5000` | 不需要；可使用已有 DNS/网关 |
| 同一 Compose 网络中的其他容器 | `http://mobo-api:5000` | Compose 自带 DNS，以服务名解析 |

## 9. 数据持久化

Compose 创建两个命名卷：

- `multi-obj-optimism_mobo-data` → `/app/data`：DOE、模型、优化结果和任务状态；
- `multi-obj-optimism_mobo-logs` → `/app/logs`：服务与业务日志。

查看卷：

```bash
docker volume ls
docker inspect mobo-api
```

以下命令删除容器但保留数据：

```bash
docker compose down
```

不要在没有备份和确认的情况下执行 `docker compose down -v`，因为 `-v` 会同时删除 DOE、
模型和优化结果卷。

如果运维要求数据位于明确宿主机目录，可以把 `compose.yaml` 的卷改为绑定挂载，例如：

```text
/srv/mobo/data:/app/data
/srv/mobo/logs:/app/logs
```

修改前应由运维人员创建目录并确保容器内 `mobo` 用户具有写权限。

## 10. 更新与回滚

首先回顾一下整体服务核心部署流程：

```bash
# 1. 安装 Git（CentOS）
sudo dnf install -y git

# 2. 克隆仓库
git clone https://github.com/MingrHu/Multi-obj-optimism.git
cd Multi-obj-optimism

# 3. 检查 Docker、Compose、端口和系统资源
bash scripts/check_docker_host.sh

# 4. 仅在 CentOS 9/10 未安装 Docker 时执行
bash scripts/install_docker_centos.sh --add-current-user

# 5. 首次加入 docker 用户组后，注销并重新登录，再次检查
bash scripts/check_docker_host.sh

# 6. 构建镜像并启动 mobo-api
docker compose up --build -d

# 7. 查看容器状态
docker compose ps

# 8. 从宿主机验证服务
curl http://127.0.0.1:5000/health
```

运行一个HTTP请求（基于curl工具）

```bash
docker exec \
  -e MOBO_API_URL=http://127.0.0.1:5000 \
  mobo-api \
  python -m mobo.api.demo
```

查看日志

```bash
# 必须在含 compose.yaml 的仓库根目录执行
docker compose logs -f mobo-api

# 可以在宿主机任意目录执行
docker logs -f mobo-api
```

更新代码并构建新版本，分两种情况：

1 再宿主机上直接更新代码&仓库（不推荐，建议华科方统一更新服务）

```bash
# 即使还没有提交，当前工作区内容也会参与镜像构建
docker compose up --build -d

docker compose ps
curl http://127.0.0.1:5000/health
```

2 宿主机通过git拉取开发机的更新并更新docker服务

```bash
# 开发机
git add -A
git commit -m "MingrHu: 更新代码与部署文档"
git push

# 部署服务器
cd Multi-obj-optimism
git pull --ff-only
docker compose up --build -d
docker compose ps

# 检查新容器是否已经进入 healthy 状态 后端服务是否就绪
curl http://127.0.0.1:5000/health
```

需要关注的点：

```bash
git pull 只更新宿主机仓库，不会修改已经运行的容器
Compose 将仓库根目录作为构建上下文
Dockerfile负责构建镜像，文件中的 COPY . . 把代码、README、docs、脚本和配置复制进镜像的 /app
pip install . --no-deps 把 src/mobo 安装进镜像里的独立 Python 环境
docker compose up --build -d 生成新镜像，并用新镜像重新创建容器
docker compose restart 只会重启旧容器，不能更新代码或文档
data 和 logs 使用 Docker 命名卷，重新创建容器不会删除已有任务、模型、结果和日志
```

**每次发布新功能或者升级时直接按照上述流程进行即可，命名卷与镜像分离，因此重新创建容器不会覆盖运行数据（包括日志、用户数据、优化数据）**

```bash
# 原理如下
旧容器 ──挂载──> mobo-data
                  ↑
新容器 ──挂载────———┘
```

回滚：

```bash
# 把服务切回仍保留在宿主机上的 1.0.0 镜像
MOBO_IMAGE_TAG=1.0.0 docker compose up -d --no-build
```



## 11. 日常运维、进入容器与日志

### 11.1 查看容器状态和标准输出日志

`docker compose` 默认在当前目录查找 `compose.yaml`，因此 Compose 命令应先进入仓库根目录：

```bash
# 进入包含 compose.yaml 的仓库根目录；请替换为服务器实际路径
cd /path/to/Multi-obj-optimism

# 查看 mobo-api 的运行状态、健康状态和端口映射
docker compose ps

# 持续查看 Gunicorn、HTTP 访问和程序标准输出日志；Ctrl+C 只退出查看，不停止容器
docker compose logs -f mobo-api

# 只显示最近 200 行 Compose 日志后退出
docker compose logs --tail 200 mobo-api

# 重启 mobo-api 容器，命名卷中的数据不会删除
docker compose restart mobo-api

# 停止 mobo-api，但保留容器和数据卷
docker compose stop mobo-api

# 重新启动已经停止的 mobo-api
docker compose start mobo-api

# 停止并删除本项目容器和网络，但保留命名卷
docker compose down
```

只要知道容器名，`docker logs` 就不依赖 `compose.yaml`，可以在宿主机任意目录执行：

```bash
# 在任意目录持续查看 mobo-api 的标准输出日志
docker logs -f mobo-api

# 在任意目录查看最近 200 行标准输出日志
docker logs --tail 200 mobo-api
```

Docker 标准输出日志由 Docker daemon 管理。在普通 Linux Docker Engine 上，其内部文件通常
位于 `/var/lib/docker/containers/<容器ID>/<容器ID>-json.log`；日常不要直接修改该文件，
统一使用 `docker logs`。

### 11.2 进入和退出容器

以下命令在宿主机执行：

```bash
# 以默认的非 root 用户 mobo 进入容器 Bash
docker exec -it mobo-api bash

# 仅在排障确有需要时，以 root 用户进入容器 Bash
docker exec -u root -it mobo-api bash

# 如果某个精简镜像没有 Bash，可退回使用 POSIX sh
docker exec -it mobo-api sh
```

进入容器后可执行：

```bash
# 显示当前目录；mobo 用户的工作目录通常是 /app
pwd

# 显示当前用户和用户组，正常运行用户应为 mobo
id

# 查看复制进镜像的仓库文件
ls -lah /app

# 查看持久化 DOE、模型和优化数据
ls -lah /app/data

# 查看应用文件日志
ls -lah /app/logs

# 退出容器交互终端，不会停止容器服务
exit
```

只执行一条命令时不需要先进入交互终端：

```bash
# 从宿主机直接列出容器数据目录
docker exec mobo-api ls -lah /app/data

# 从宿主机直接持续查看当天的应用文件日志；文件名按实际日期调整
docker exec mobo-api tail -f /app/logs/run_20260827.log
```

### 11.3 查看 CPU、内存、进程和健康状态

生产镜像基于 `python:3.12-slim`，默认不安装 `top`、`ps`、`vim`、`ping` 等调试工具。
优先在宿主机使用 Docker 自带的监控命令：

```bash
# 实时显示 mobo-api 的 CPU、内存、网络和磁盘 I/O；Ctrl+C 退出查看
docker stats mobo-api

# 查看容器内进程，不要求容器自身安装 ps 或 top
docker top mobo-api

# 输出容器的完整配置、挂载、网络、日志路径和运行状态
docker inspect mobo-api

# 只输出 Docker 保存的健康检查状态和最近检查记录
docker inspect mobo-api --format '{{json .State.Health}}'
```

如果必须在容器内临时使用 `top`，以下操作会修改当前容器，但重新创建容器后会消失。

先在宿主机执行：

```bash
# 临时以 root 用户进入当前容器
docker exec -u root -it mobo-api bash
```

再在容器内执行：

```bash
# 更新当前容器的 Debian 软件包索引
apt-get update

# 临时安装 top、ps、free 和 vmstat 所在的 procps 包
apt-get install -y procps

# 交互查看容器内进程和资源占用
top

# 退出临时 root 终端
exit
```

生产环境通常不把调试工具永久装入业务镜像，以减少镜像体积和攻击面。

### 11.4 查看和导出应用数据与文件日志

应用文件日志与 Docker 标准输出日志不是同一类日志：

| 类型 | 容器内位置 | 推荐查看方式 |
|---|---|---|
| Gunicorn、HTTP 访问、stdout/stderr | Docker daemon 管理 | `docker logs` / `docker compose logs` |
| mobo 应用文件日志 | `/app/logs` | `docker exec` 或 `docker cp` |
| DOE、模型和优化结果 | `/app/data` | HTTP 字段接口、`docker exec` 或备份时 `docker cp` |

```bash
# 查看应用日志目录中的文件名和大小
docker exec mobo-api ls -lh /app/logs

# 查看应用日志文件最后 100 行；文件名按实际日期调整
docker exec mobo-api tail -n 100 /app/logs/run_20260827.log

# 查找数据目录三层以内的文件，了解 DOE 和模型落盘情况
docker exec mobo-api find /app/data -maxdepth 3 -type f

# 把整个应用日志目录复制到宿主机当前目录，生成 mobo-logs-backup
docker cp mobo-api:/app/logs ./mobo-logs-backup

# 把整个数据目录复制到宿主机当前目录，生成 mobo-data-backup
docker cp mobo-api:/app/data ./mobo-data-backup

# 查看本项目创建的 Docker 命名卷
docker volume ls --filter name=multi-obj-optimism

# 查看数据卷的 Docker 内部挂载位置和元数据
docker volume inspect multi-obj-optimism_mobo-data

# 查看日志卷的 Docker 内部挂载位置和元数据
docker volume inspect multi-obj-optimism_mobo-logs
```

在普通 CentOS Docker Engine 上，命名卷通常位于 `/var/lib/docker/volumes/.../_data`；在
Windows Docker Desktop 上，该路径属于 Docker Desktop 内部 Linux 虚拟机，不是普通的
`C:\...` 目录，应通过 `docker exec`、`docker cp` 或 Docker Desktop 界面访问。

## 12. WSL2 与 Docker Desktop

WSL2 只用于 Windows 开发机的本地构建和接口测试。启用 Docker Desktop 的 WSL Integration
后，在 Ubuntu 仓库目录执行的仍是相同命令：

```bash
docker compose up --build -d
curl http://127.0.0.1:5000/health
```

服务器端使用 Docker Engine，不需要 WSL2，也不需要 Docker Desktop。
