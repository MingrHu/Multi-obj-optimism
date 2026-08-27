#!/usr/bin/env bash
# 按照 Docker 官方 RPM 仓库流程，在受支持的 CentOS Stream 9/10 上安装 Docker Engine。
# 用法：bash scripts/install_docker_centos.sh [--add-current-user]

set -euo pipefail

add_current_user=0
if [ "${1:-}" = "--add-current-user" ]; then
    add_current_user=1
elif [ "$#" -ne 0 ]; then
    printf '未知参数: %s\n' "$1" >&2
    exit 2
fi

if [ "$(uname -s)" != "Linux" ] || [ ! -r /etc/os-release ]; then
    printf '该脚本只能在可识别的 Linux 宿主机上运行。\n' >&2
    exit 1
fi

# shellcheck disable=SC1091
. /etc/os-release
major="${VERSION_ID%%.*}"
if [ "${ID:-}" != "centos" ]; then
    printf '当前系统为 %s；请使用 Docker 官方对应发行版安装文档。\n' "${PRETTY_NAME:-unknown}" >&2
    exit 1
fi
if [ "$major" = "8" ]; then
    printf '%s\n' \
        '拒绝在 CentOS Stream 8 上自动安装 Docker：该系统已于 2024-05-31 结束维护，' \
        'Docker 官方当前只支持 CentOS Stream 9/10。请先迁移宿主机或由运维团队提供受支持的容器运行时。' >&2
    exit 1
fi
if [ "$major" != "9" ] && [ "$major" != "10" ]; then
    printf 'Docker 官方安装流程不支持当前 CentOS 主版本: %s\n' "$major" >&2
    exit 1
fi

if [ "$(uname -m)" != "x86_64" ]; then
    printf '警告：本仓库当前只在 linux/amd64 上完成过完整镜像验证。\n' >&2
fi

if [ "$EUID" -eq 0 ]; then
    sudo_cmd=()
elif command -v sudo >/dev/null 2>&1; then
    sudo_cmd=(sudo)
else
    printf '安装 Docker 需要 root 权限或 sudo。\n' >&2
    exit 1
fi

legacy_packages=(
    docker docker-client docker-client-latest docker-common docker-latest
    docker-latest-logrotate docker-logrotate docker-engine
)
installed_legacy=()
for package in "${legacy_packages[@]}"; do
    if rpm -q "$package" >/dev/null 2>&1; then
        installed_legacy+=("$package")
    fi
done
if [ "${#installed_legacy[@]}" -ne 0 ]; then
    printf '检测到可能冲突的旧软件包: %s\n' "${installed_legacy[*]}" >&2
    printf '请先确认旧容器和数据的迁移方案，再按 Docker 官方文档人工卸载；脚本不会自动删除。\n' >&2
    exit 1
fi

printf '==> 安装仓库管理、Git 和 HTTP 工具\n'
"${sudo_cmd[@]}" dnf -y install dnf-plugins-core git curl

if [ ! -f /etc/yum.repos.d/docker-ce.repo ]; then
    printf '==> 添加 Docker 官方 RPM 仓库\n'
    "${sudo_cmd[@]}" dnf config-manager --add-repo \
        https://download.docker.com/linux/centos/docker-ce.repo
fi

printf '==> 安装 Docker Engine、Buildx 和 Compose v2\n'
"${sudo_cmd[@]}" dnf -y install \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

printf '==> 启动 Docker 并设置开机自启\n'
"${sudo_cmd[@]}" systemctl enable --now docker

printf '==> 验证 Docker Engine\n'
"${sudo_cmd[@]}" docker version
"${sudo_cmd[@]}" docker compose version
"${sudo_cmd[@]}" docker run --rm hello-world

if [ "$add_current_user" -eq 1 ]; then
    login_user="${SUDO_USER:-${USER:-}}"
    if [ -n "$login_user" ] && [ "$login_user" != "root" ]; then
        "${sudo_cmd[@]}" usermod -aG docker "$login_user"
        printf '已将 %s 加入 docker 组；请注销并重新登录后再执行无 sudo 的 docker 命令。\n' "$login_user"
        printf '注意：docker 组具备等同 root 的宿主机控制能力。\n'
    else
        printf '未找到可加入 docker 组的非 root 登录用户。\n' >&2
    fi
fi

printf 'Docker 安装完成。下一步运行: bash scripts/check_docker_host.sh\n'
