#!/usr/bin/env bash
# 检查 Linux 宿主机是否具备构建和运行 mobo-api Docker 服务的条件。
# 脚本只读取系统状态，不安装软件、不修改用户组、不开放防火墙端口。

set -uo pipefail

failures=0
warnings=0

ok() {
    printf '[OK] %s\n' "$1"
}

warn() {
    printf '[WARN] %s\n' "$1" >&2
    warnings=$((warnings + 1))
}

fail() {
    printf '[FAIL] %s\n' "$1" >&2
    failures=$((failures + 1))
}

printf '==> mobo-api Docker 宿主机检查\n'

if [ "$(uname -s)" != "Linux" ]; then
    fail "该脚本面向 Linux 宿主机；当前系统为 $(uname -s)"
fi

if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    printf '系统: %s\n' "${PRETTY_NAME:-${ID:-unknown} ${VERSION_ID:-unknown}}"
    if [ "${ID:-}" = "centos" ] && [ "${VERSION_ID%%.*}" = "8" ]; then
        warn "CentOS Stream 8 已结束维护，Docker 官方也不再支持在该版本上新装 Docker"
    fi
else
    warn "无法读取 /etc/os-release"
fi

arch="$(uname -m)"
printf '架构: %s\n' "$arch"
if [ "$arch" = "x86_64" ]; then
    ok "宿主机架构与当前已验证的 linux/amd64 镜像一致"
else
    warn "当前架构尚未由本仓库验证；建议在发布前单独构建并运行完整 Demo"
fi

cpu_count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '0')"
if [ "$cpu_count" -ge 4 ] 2>/dev/null; then
    ok "CPU 逻辑核数: $cpu_count"
else
    warn "CPU 逻辑核数为 $cpu_count；模型训练和优化建议至少 4 核"
fi

if [ -r /proc/meminfo ]; then
    memory_kib="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
    memory_gib=$((memory_kib / 1024 / 1024))
    if [ "$memory_gib" -ge 8 ]; then
        ok "内存约 ${memory_gib} GiB"
    else
        warn "内存约 ${memory_gib} GiB；完整机器学习环境建议至少 8 GiB"
    fi
fi

disk_kib="$(df -Pk . | awk 'NR==2 {print $4}')"
disk_gib=$((disk_kib / 1024 / 1024))
if [ "$disk_gib" -ge 12 ]; then
    ok "当前文件系统可用空间约 ${disk_gib} GiB"
else
    warn "当前文件系统仅剩约 ${disk_gib} GiB；首次构建建议至少预留 12 GiB"
fi

if command -v git >/dev/null 2>&1; then
    ok "$(git --version)"
else
    fail "缺少 git，无法克隆或更新仓库"
fi

if command -v curl >/dev/null 2>&1; then
    ok "curl 已安装"
else
    fail "缺少 curl，无法执行健康检查和 HTTP 冒烟请求"
fi

if ! command -v docker >/dev/null 2>&1; then
    fail "缺少 docker 命令"
else
    ok "$(docker --version)"
    if docker info >/dev/null 2>&1; then
        ok "当前用户可以访问 Docker daemon"
    else
        fail "Docker daemon 未运行或当前用户无权访问；检查 systemctl status docker 和 docker 用户组"
    fi

    if docker compose version >/dev/null 2>&1; then
        ok "$(docker compose version)"
    else
        fail "缺少 Docker Compose v2 插件（需要使用 docker compose，而不是旧版 docker-compose）"
    fi
fi

if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq '(^|:)5000$'; then
    warn "宿主机 TCP 5000 端口已被占用；部署时请设置 MOBO_HTTP_PORT"
else
    ok "未发现 TCP 5000 端口占用"
fi

printf '\n检查结果: %d 个失败，%d 个警告\n' "$failures" "$warnings"
if [ "$failures" -ne 0 ]; then
    exit 1
fi

printf '宿主机具备运行 mobo-api Docker 服务的基础条件。\n'
