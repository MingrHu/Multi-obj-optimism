#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""碾环圆度提取命令行入口。

对应原 ``Geo/deform_ring_roundness.py`` 的 ``main()``，仅将 argparse 默认 KEY 文件
路径从 macOS 硬编码改为集中式 :data:`mobo.common.paths.KEY_FILE_DIR`，算法调用逻辑
不变。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mobo.common.paths import KEY_FILE_DIR
from mobo.extraction.ring_roundness import (
    analyze_profile,
    build_section_components,
    component_perimeter,
    determine_default_plane_z,
    determine_default_tolerances,
    extract_boundary_faces,
    extract_section_segments,
    order_component_as_loop,
    polygon_signed_area,
    print_summary,
    read_deform_key,
    save_plot,
    write_results_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="读取 DEFORM KEY 网格，提取 Ring 的平面截面并计算内外轮廓圆度。"
    )
    parser.add_argument("key_file", type=Path,
                        nargs="?",
                        default=KEY_FILE_DIR / "RINGROLL.KEY",
                        help="DEFORM .KEY 文件路径")
    parser.add_argument(
        "--object-name",
        default="Workpiece - Ring",
        help='对象名称，默认："Workpiece - Ring"',
    )
    parser.add_argument(
        "--object-id",
        type=int,
        default=None,
        help="直接指定对象编号；给出后优先于 --object-name",
    )
    parser.add_argument(
        "--plane-z",
        type=float,
        default=None,
        help="截面平面的 z 值，默认使用工件 Z 向包围盒中心",
    )
    parser.add_argument(
        "--z-tol",
        type=float,
        default=None,
        help="判断节点是否位于截面平面的绝对容差；默认按模型尺度自动确定",
    )
    parser.add_argument(
        "--merge-tol",
        type=float,
        default=None,
        help="合并截面端点的 XY 绝对容差；默认按模型尺度自动确定",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3000,
        help="每条闭合轮廓按弧长均匀重采样的点数，默认 3000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认与 KEY 文件同目录",
    )
    args = parser.parse_args()

    if not args.key_file.exists():
        parser.error(f"文件不存在：{args.key_file}")

    output_dir = args.output_dir or args.key_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = read_deform_key(
        args.key_file,
        object_id=args.object_id,
        object_name=args.object_name,
    )

    default_z_tol, default_merge_tol = determine_default_tolerances(mesh)
    plane_z = determine_default_plane_z(mesh) if args.plane_z is None else args.plane_z
    z_tol = args.z_tol if args.z_tol is not None else default_z_tol
    merge_tol = args.merge_tol if args.merge_tol is not None else default_merge_tol

    if z_tol <= 0 or merge_tol <= 0:
        raise ValueError("z-tol 和 merge-tol 必须为正数。")

    boundary_faces = extract_boundary_faces(mesh.elements)
    segments = extract_section_segments(
        mesh,
        boundary_faces,
        plane_z=plane_z,
        z_tol=z_tol,
        merge_tol=merge_tol,
    )

    point_map, components = build_section_components(segments, merge_tol)

    component_infos = []
    for component in components:
        if len(component) < 3:
            continue
        perimeter = component_perimeter(component, point_map)
        loop, exact_graph_loop = order_component_as_loop(component, point_map)
        if len(loop) < 3:
            continue
        area = abs(polygon_signed_area(loop))
        component_infos.append(
            {
                "component": component,
                "perimeter": perimeter,
                "loop": loop,
                "area": area,
                "exact_graph_loop": exact_graph_loop,
            }
        )

    if len(component_infos) < 2:
        raise ValueError(
            "未获得两个有效的闭合截面轮廓。"
            f"当前有效连通分量数：{len(component_infos)}。"
            "请检查 z=0 是否确实与 Ring 相交，并适当调整 --z-tol 和 --merge-tol。"
        )

    # 截面可能包含极小伪分量，优先取周长最大的两个主轮廓。
    component_infos.sort(key=lambda item: item["perimeter"], reverse=True)
    selected = component_infos[:2]

    if len(component_infos) > 2:
        ignored = len(component_infos) - 2
        print(
            f"警告：截面检测到 {len(component_infos)} 个有效分量，"
            f"仅使用周长最大的两个，忽略 {ignored} 个较小分量。",
            file=sys.stderr,
        )

    for item in selected:
        if not item["exact_graph_loop"]:
            print(
                "警告：某截面分量的节点度数不全为 2，已采用绕拟合圆心的角度排序。",
                file=sys.stderr,
            )

    # 按包围面积区分内圈和外圈。
    selected.sort(key=lambda item: item["area"])
    inner_loop = selected[0]["loop"]
    outer_loop = selected[1]["loop"]

    inner_result = analyze_profile("内圈", inner_loop, args.samples)
    outer_result = analyze_profile("外圈", outer_loop, args.samples)
    results = [inner_result, outer_result]

    stem = args.key_file.stem
    csv_path = output_dir / f"{stem}_z{plane_z:g}_roundness.csv"
    png_path = output_dir / f"{stem}_z{plane_z:g}_section.png"

    write_results_csv(
        csv_path,
        mesh,
        plane_z,
        len(boundary_faces),
        len(segments),
        results,
    )
    save_plot(
        png_path,
        plane_z,
        [
            ("Inner", inner_loop, inner_result),
            ("Outer", outer_loop, outer_result),
        ],
    )

    print_summary(
        mesh,
        len(boundary_faces),
        len(segments),
        len(components),
        results,
    )
    print(f"结果表：{csv_path}")
    print(f"截面图：{png_path}")
    print(f"使用容差：z_tol={z_tol:.6g}, merge_tol={merge_tol:.6g}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
