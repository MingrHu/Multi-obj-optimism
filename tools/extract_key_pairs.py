#  @author MingrHu
#  @date   2026-11-07
"""从独立输入目录和结果目录按工况编号配对复制 DEFORM 初态与终态 KEY"""

from os import name as platform_name
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory


#  @brief  读取终端路径并接受带引号的粘贴内容
#  @param  prompt 终端提示
#  @return 用户指定的绝对路径
#  @author MingrHu
#  @date   2026-11-07
def read_path(prompt: str) -> Path:
    value = input(prompt).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        value = value[1:-1]
    if not value.strip():
        raise ValueError("目录路径不能为空")
    return Path(value).expanduser().resolve()


#  @brief  按文件检查并补齐工况的输入初态和求解终态 KEY
#  @param  initial 直接包含工件名称加编号 KEY 文件的 input_keys 目录
#  @param  source 直接包含数字工况子目录的终态 result_keys 目录
#  @param  output 保存数字工况目录的目标根目录
#  @return 已复制工况数和跳过工况数
#  @author MingrHu
#  @date   2026-11-07
def extract_key_pairs(initial: Path, source: Path, output: Path) -> tuple[int, int]:
    initial, source, output = initial.resolve(), source.resolve(), output.resolve()
    for directory in (initial, source):
        if not directory.is_dir():
            raise ValueError(f"源目录不存在或不是目录：{directory}")
        if output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValueError("源目录和输出目录必须分开，不能相同或互相包含")

    cases = sorted(
        (path for path in source.iterdir() if path.is_dir() and path.name.isascii() and path.name.isdecimal()),
        key=lambda path: (int(path.name), path.name),
    )
    if not cases:
        raise ValueError(f"源目录中没有数字命名的工况子目录：{source}")
    output.mkdir(parents=True, exist_ok=True)

    copied = 0
    for case in cases:
        terminals = [path for path in case.iterdir()
                     if path.is_file() and path.name.endswith("_terminal.KEY")]
        if len(terminals) != 1:
            print(f"跳过工况 {case.name}，需要唯一的 _terminal.KEY 文件，实际 {len(terminals)} 个")
            continue
        terminal = terminals[0]
        # 从终态文件名直接移除后缀 不猜测材料名称或数字前缀
        prefix = terminal.name.removesuffix("_terminal.KEY")
        if not prefix.endswith(case.name):
            print(f"跳过工况 {case.name}，终态文件名与工况编号不匹配：{terminal.name}")
            continue
        files = (initial / f"{prefix}.KEY", terminal)
        destination = output / case.name
        if destination.is_symlink() or (
            destination.exists() and (not destination.is_dir() or destination.resolve() != destination)
        ):
            print(f"跳过工况 {case.name}，目标不是普通目录：{destination}")
            continue
        # 仅清理输出工况目录内的旧初态文件 源目录中的文件保持不变
        try:
            for old_step in destination.glob("*_step_1.KEY"):
                if old_step.is_file() or old_step.is_symlink():
                    old_step.unlink()
                    print(f"已删除旧初态文件：{old_step}")
        except OSError as error:
            print(f"跳过工况 {case.name}，旧初态清理失败：{error}")
            continue
        pending = [path for path in files if not (destination / path.name).is_file()]
        if not pending:
            print(f"跳过工况 {case.name}，两个 KEY 已存在")
            continue
        missing = [path.name for path in pending if not path.is_file()]
        if missing:
            print(f"跳过工况 {case.name}，源目录缺少 {', '.join(missing)}")
            continue
        try:
            # 复制完成后再发布文件 防止中断留下同名的残缺 KEY
            with TemporaryDirectory(prefix=".key_pair_", dir=output) as temporary:
                staged = Path(temporary) / case.name
                staged.mkdir()
                for path in pending:
                    copy2(path, staged / path.name)
                if not destination.exists():
                    staged.rename(destination)
                else:
                    for path in pending:
                        target = destination / path.name
                        # Windows 重命名拒绝覆盖 POSIX 使用独占硬链接发布
                        try:
                            if platform_name == "nt":
                                (staged / path.name).rename(target)
                            else:
                                target.hardlink_to(staged / path.name)
                        except FileExistsError:
                            if not target.is_file():
                                raise
        except OSError as error:
            print(f"跳过工况 {case.name}，复制失败：{error}")
            continue
        copied += 1
        print(f"工况 {case.name} 的两个 KEY 已齐全")
    return copied, len(cases) - copied


#  @brief  交互读取输入输出目录并显示复制汇总
#  @return 退出码 0 表示所有工况复制成功 1 表示失败或存在跳过
#  @author MingrHu
#  @date   2026-11-07
def main() -> int:
    try:
        initial = read_path("请输入初态 input_keys 目录（直接包含 KEY 文件）：")
        source = read_path("请输入终态 result_keys 目录（包含数字工况子目录）：")
        output = read_path("请输入 KEY 输出目录：")
        copied, skipped = extract_key_pairs(initial, source, output)
    except (EOFError, KeyboardInterrupt):
        print("\n已取消提取")
        return 1
    except (ValueError, OSError) as error:
        print(f"提取失败：{error}")
        return 1
    print(f"完成：处理 {copied} 个工况，跳过 {skipped} 个工况。")
    print(f"输出目录：{output}")
    return int(skipped != 0)


if __name__ == "__main__":
    raise SystemExit(main())
