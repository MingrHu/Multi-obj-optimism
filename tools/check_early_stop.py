"""读取一组收敛值，计算变化率并判断是否满足5%早停条件。"""

import sys

from mobo.optimization.convergence import calculate_change_rate


THRESHOLD = 0.05


def main():
    first_line = sys.stdin.readline().strip()
    try:
        count = int(first_line)
    except ValueError:
        print("错误：第一行必须是正整数n。", file=sys.stderr)
        return 2
    if count <= 0:
        print("错误：n必须是正整数。", file=sys.stderr)
        return 2

    tokens = []
    while len(tokens) < count:
        line = sys.stdin.readline()
        if not line:
            print(f"错误：需要输入{count}个数值，实际输入{len(tokens)}个。", file=sys.stderr)
            return 2
        tokens.extend(line.split())
    if len(tokens) > count:
        print(f"错误：需要输入{count}个数值，实际输入{len(tokens)}个。", file=sys.stderr)
        return 2
    try:
        values = [float(token) for token in tokens]
    except ValueError:
        print("错误：输入中包含非数值内容。", file=sys.stderr)
        return 2

    change_rate = calculate_change_rate(values)
    print(f"变化率：{change_rate:.6%}")
    if change_rate <= THRESHOLD:
        print("满足停止条件")
    else:
        print("不满足停止条件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
