"""Gunicorn configuration for the Docker API process."""

import os


bind = f"{os.environ.get('MOBO_API_HOST', '0.0.0.0')}:{os.environ.get('MOBO_API_PORT', '5000')}"

# 训练、优化及停止信号目前由进程内运行时协调。保持单 worker，避免多进程状态割裂；
# 并发 HTTP 查询由线程承担。
workers = 1
worker_class = "gthread"
threads = int(os.environ.get("MOBO_GUNICORN_THREADS", "4"))
timeout = int(os.environ.get("MOBO_GUNICORN_TIMEOUT", "300"))
graceful_timeout = 30
accesslog = "-"
errorlog = "-"
capture_output = True
