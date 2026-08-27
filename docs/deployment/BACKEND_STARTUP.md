# DOE 优化后端启动与完整 Demo 测试

本文用于从空环境启动 Flask 后端，并运行仓库自带的完整优化流程 Demo

完整 Demo 会依次执行以下步骤

```text
创建 DOE
→ 生成 LHS 样本并按字段读取
→ 生成合成训练数据集
→ 按字段读取训练数据集
→ 训练随机森林代理模型
→ 三折交叉验证评价
→ 基于代理模型按字段推理
→ 查询最近一次推理结果
→ 提交 NSGA2 多目标优化
→ 等待优化完成并按字段查询结果
```

合成数据只用于验证 HTTP 接口和完整流程，不代表真实 DEFORM 仿真结果

## 1 环境要求

- Python 3.11 或 3.12，推荐 Python 3.12
- Windows PowerShell 或 Linux/macOS Bash
- Demo 不需要安装 DEFORM
- 默认需要访问 Python 包源和 PyTorch CPU 包源

## 2 安装环境

### Windows PowerShell

在仓库根目录执行

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

如果 `.venv` 来自已经卸载或移动的 Python，使用重新创建参数

```powershell
.\scripts\setup_env.ps1 -Recreate
```

如果系统中的 `py` 没有注册 Python 3.11 或 3.12，可直接指定解释器

```powershell
.\scripts\setup_env.ps1 -Recreate -PythonPath "C:\Python312\python.exe"
```

### Linux 或 macOS

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

需要重新创建虚拟环境时执行

```bash
bash scripts/setup_env.sh --recreate
source .venv/bin/activate
```

安装脚本会检查 Flask、requests、代理模型、NSGA2、强化学习及 PyTorch 依赖

## 3 启动后端服务

`mobo-api` 使用 Flask 内置服务，适用于本地开发、接口联调和完整 Demo 验证
正式生产部署时应由生产级 WSGI 服务承载 `mobo.api.app:create_app`

本仓库提供的 Docker 方案已使用 Gunicorn 承载，并在容器启动时自动启动 API：

```bash
docker compose up --build -d
curl http://127.0.0.1:5000/health
```

完整说明见 [`DOCKER_DEPLOYMENT.md`](DOCKER_DEPLOYMENT.md)。

打开第一个终端，激活虚拟环境后执行

```powershell
mobo-api
```

默认监听地址

```text
http://127.0.0.1:5000
```

日志中出现以下内容表示服务已经启动

```text
Running on http://127.0.0.1:5000
```

可以在另一个 PowerShell 中检查健康状态

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

预期返回

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "service": "mobo-doe"
  }
}
```

如果 `mobo-api` 命令不可用，可以使用模块入口

```powershell
python -c "from mobo.api.app import main; main()"
```

## 4 修改监听地址或端口

Windows PowerShell

```powershell
$env:MOBO_API_HOST = "127.0.0.1"
$env:MOBO_API_PORT = "8000"
mobo-api
```

Linux 或 macOS

```bash
export MOBO_API_HOST=127.0.0.1
export MOBO_API_PORT=8000
mobo-api
```

修改端口后，运行 Demo 前需要设置相同的服务地址

```powershell
$env:MOBO_API_URL = "http://127.0.0.1:8000"
```

## 5 运行完整优化 Demo

保持后端终端运行，打开第二个终端并激活同一个虚拟环境

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python -m mobo.api.demo
```

Linux 或 macOS

```bash
source .venv/bin/activate
python -m mobo.api.demo
```

Demo 会为本次运行生成唯一 DOE 标识，因此可以重复执行

每个 HTTP 请求都会打印状态码和 JSON 响应，最终优化状态应为

```json
{
  "status": "finished"
}
```

训练和优化是后台任务，Demo 会每两秒查询一次进度，默认最长等待十分钟

Demo 中的字段查询使用 GET，例如：

```text
GET /api/v1/hust/doe/data/get?id=<demo_doe_id>&data_type=sample&fields=temperature
```

查询多个字段时重复传递 `fields`。端上无需打开响应中的服务端文件路径。

## 6 Demo 产物位置

完整流程产物位于

```text
data/doe_tasks/<demo_doe_id>/
├── doe.json
├── samples/
├── models/
├── training/
│   └── demo_training_dataset.tsv
└── optimization/
    └── pareto_solutions.tsv
```

其中

- `doe.json` 保存 DOE 元数据、训练状态和优化状态
- 样本、`demo_training_dataset.tsv` 和 `pareto_solutions.tsv` 均无表头；列名顺序记录在
  `doe.json`
- `models` 保存代理模型及标准化器快照
- `pareto_solutions.tsv` 保存 NSGA2 推荐参数和目标值

## 7 常见问题

### 虚拟环境中的 Python 无法启动

说明 `.venv` 绑定的基础 Python 已失效，重新创建即可

```powershell
.\scripts\setup_env.ps1 -Recreate
```

### 端口已被占用

切换端口并同步设置 Demo 地址

```powershell
$env:MOBO_API_PORT = "8000"
$env:MOBO_API_URL = "http://127.0.0.1:8000"
```

### 训练或优化返回 failed

先查询对应响应中的 `error` 字段，再检查服务终端的异常日志

```text
GET /api/hust/v1/doe/train/progress?id=<doe_id>
GET /api/v1/hust/doe/optimize/getById?id=<doe_id>
```

### Demo 无法连接服务

确认健康检查成功，并确认 `MOBO_API_URL` 与后端实际地址一致

## 8 相关文档

- HTTP 接口定义见 [`DOE_HTTP_API.md`](../api/DOE_HTTP_API.md)
- 分层结构见 [`ARCHITECTURE.md`](../ARCHITECTURE.md)
- 环境及项目总览见 [`README.md`](../../README.md)
- Docker 构建和部署见 [`DOCKER_DEPLOYMENT.md`](DOCKER_DEPLOYMENT.md)
